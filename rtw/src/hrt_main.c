#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <errno.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>     // calloc()
#include <alloca.h>     // alloca()

#include <signal.h>

#include "rtmodel.h"
#include "rtwtypes.h"
#include "rt_nonfinite.h"
#include "rt_sim.h"

#include "ext_work.h"


#define SET_RT_PRIO 1


/****************************************************************************/

#define MAX_SAFE_STACK (8 * 1024) /** The maximum stack size which is
                                    guranteed safe to access without faulting.
                                   */

/* To quote a string */
#define _STR(x) #x
#define QUOTE(x) _STR(x)

#define EXPAND_CONCAT(name1, name2) name1 ## name2
#define CONCAT(name1, name2) EXPAND_CONCAT(name1, name2)
#define MODEL_VERSION        CONCAT(MODEL, _version)
#define RT_MODEL             CONCAT(MODEL, _rtModel)

#ifndef max
#define max(x1, x2) ((x1) > (x2) ? (x1) : (x2))
#endif

#ifndef min
#define min(x1, x2) ((x2) > (x1) ? (x1) : (x2))
#endif

#ifndef RT
# error "must define RT"
#endif

#ifndef MODEL
# error "must define MODEL"
#endif

#ifndef NUMST
# error "must define number of sample times, NUMST"
#endif

#ifndef NCSTATES
# error "must define NCSTATES"
#endif

/*====================*
 * External functions *
 *====================*/

#ifdef __cplusplus
extern "C" {
#endif

extern RT_MODEL *MODEL(void);

extern void MdlInitializeSizes(void);
extern void MdlInitializeSampleTimes(void);
extern void MdlStart(void);
extern void MdlOutputs(int_T tid);
extern void MdlUpdate(int_T tid);
extern void MdlTerminate(void);

extern const char *MODEL_VERSION;

#if NCSTATES > 0
extern void rt_ODECreateIntegrationData(RTWSolverInfo *si);
extern void rt_ODEUpdateContinuousStates(RTWSolverInfo *si);

# define rt_CreateIntegrationData(S) \
    rt_ODECreateIntegrationData(rtmGetRTWSolverInfo(S));
# define rt_UpdateContinuousStates(S) \
    rt_ODEUpdateContinuousStates(rtmGetRTWSolverInfo(S));
# else
# define rt_CreateIntegrationData(S)  \
      rtsiSetSolverName(rtmGetRTWSolverInfo(S),"FixedStepDiscrete");
# define rt_UpdateContinuousStates(S) \
      rtmSetT(S, rtsiGetSolverStopTime(rtmGetRTWSolverInfo(S)));
#endif



#ifdef __cplusplus
}
#endif


#ifdef EXT_MODE
#define rtExtModeSingleTaskUpload(S)                          \
   {                                                            \
        int stIdx;                                              \
        rtExtModeUploadCheckTrigger(rtmGetNumSampleTimes(S));   \
        for (stIdx=0; stIdx<NUMST; stIdx++) {                   \
            if (rtmIsSampleHit(S, stIdx, 0 /*unused*/)) {       \
                rtExtModeUpload(stIdx,rtmGetTaskTime(S,stIdx)); \
            }                                                   \
        }                                                       \
   }
#else
#define rtExtModeSingleTaskUpload(S)	/* Do nothing */
#endif


const char* rt_OneStepMain(RT_MODEL *s);

struct thread_task {
    RT_MODEL *S;
    uint_T sl_tid;
    unsigned int *running;
    const char *err;
    double sample_time;
    struct timespec monotonic_time;
    struct timespec world_time;
    pthread_t thread;
};

# if TID01EQ == 1
#  define FIRST_TID 1
# else
#  define FIRST_TID 0
# endif

#define NSEC_PER_SEC (1000000000)

#undef timeradd
inline void timeradd(struct timespec *t, unsigned int dt)
{
    t->tv_nsec += dt;
    while (t->tv_nsec >= NSEC_PER_SEC) {
        t->tv_nsec -= NSEC_PER_SEC;
        t->tv_sec++;
    }
}

// return time difference t_B-t_A in nanoseconds
#define DIFF_NS(A, B) (((long long) (B).tv_sec - (A).tv_sec) * NSEC_PER_SEC + \
        (B).tv_nsec - (A).tv_nsec)

#define ABS(X) ((X) >= 0 ? (X) : -(X))

/****************************************************************************/


static struct thread_task task;


/** Perform one step of the model.
 *
 * This function is modeled such that it could be called from an interrupt
 * service routine (ISR) with minor modifications.
 */
const char *
rt_OneStepMain(RT_MODEL *S)
{
    real_T tnext;

    tnext = rt_SimGetNextSampleHit();
    rtsiSetSolverStopTime(rtmGetRTWSolverInfo(S),tnext);

    MdlOutputs(0);
    
    rtExtModeSingleTaskUpload (S);
    
    MdlUpdate(0);

    rt_SimUpdateDiscreteTaskSampleHits(rtmGetNumSampleTimes(S),
                                       rtmGetTimingData(S),
                                       rtmGetSampleHitPtr(S),
                                       rtmGetTPtr(S));

    if (rtmGetSampleTime(S,0) == CONTINUOUS_SAMPLE_TIME) {
        rt_UpdateContinuousStates(S);
    }
    
    rtExtModeCheckEndTrigger ();

    return rtmGetErrorStatusFlag(S);
}

void *run_task(void *p)
{
    struct thread_task *thread = p;
    unsigned int dt = 1.0e9 * thread->sample_time + 0.5; // sample time in nanoseconds

    int32_t overruns = 0, toolates = 0;

    struct timespec start_time, end_time = thread->monotonic_time;

    
    #ifdef SET_RT_PRIO

    struct sched_param param = { };
    param.sched_priority = sched_get_priority_max (SCHED_FIFO) - 1;
    if (sched_setscheduler (0, SCHED_FIFO, &param) == -1)
    {
        fprintf (stderr, "Setting SCHED_FIFO" " with priority %i failed: %s\n",
                param.sched_priority, strerror (errno));
    }

    printf ("Set task prio to %d\n", param.sched_priority);

    #endif

    clock_gettime (CLOCK_MONOTONIC, &task.monotonic_time);
    timeradd(&thread->monotonic_time, dt);
    unsigned int iloop = 0;
    while (!thread->err && *thread->running
            && !clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME,
                &thread->monotonic_time, 0)) {

        iloop++;
        // get starttime and check if it is more than 0.5ms too late
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        long long diffns1 = DIFF_NS(thread->monotonic_time, start_time);
        if (diffns1 > 500000){
            toolates++;
            printf("Loop %d. TOO LATE (%d). Threadtime %ld.%.9ld s, starttime %ld.%.9ld s, diff %lld ns\n",
              iloop, toolates, thread->monotonic_time.tv_sec, thread->monotonic_time.tv_nsec, start_time.tv_sec, start_time.tv_nsec, diffns1);
        }

        clock_gettime(CLOCK_REALTIME, &thread->world_time);
        // One step of the simulink model
        thread->err = rt_OneStepMain(thread->S);

        // increase timestruct by model sample time dt
        timeradd(&thread->monotonic_time, dt);

        clock_gettime(CLOCK_MONOTONIC, &end_time); // write CLOCK_MONOTONIC in end_time

        // check if the current loop iteration took too long
        long long diffns2 = DIFF_NS(end_time, thread->monotonic_time);
        if (diffns2 < 0){
            overruns++;
            printf("Loop %d. OVERRUN (%d). Threadtime %ld.%.9ld s, endtime %ld.%.9ld s, diff %lld ns\n",
              iloop, overruns, thread->monotonic_time.tv_sec, thread->monotonic_time.tv_nsec, start_time.tv_sec, start_time.tv_nsec, diffns2);
        }
    }

    *thread->running = 0;

    return 0;
}

const char *init_application(RT_MODEL *S)
{
    const char *errmsg;

    /************************
     * Initialize the model *
     ************************/
    MdlInitializeSizes();
    MdlInitializeSampleTimes();

    if ((errmsg = rt_SimInitTimingEngine(
                    rtmGetNumSampleTimes(S),
                    rtmGetStepSize(S),
                    rtmGetSampleTimePtr(S),
                    rtmGetOffsetTimePtr(S),
                    rtmGetSampleHitPtr(S),
                    rtmGetSampleTimeTaskIDPtr(S),
                    rtmGetTStart(S),
                    &rtmGetSimTimeStep(S),
                    &rtmGetTimingData(S)))) {
        return errmsg;
    }

    rtExtModeCheckInit (rtmGetNumSampleTimes (S));
    rtExtModeWaitForStartPkt (rtmGetRTWExtModeInfo (S), rtmGetNumSampleTimes (S), (boolean_T *) & rtmGetStopRequested(S));

    MdlStart();
    if ((errmsg = rtmGetErrorStatus(S))) {
        MdlTerminate();
        return errmsg;
    }

    return NULL;
}

/****************************************************************************/

/** Cause a stack fault before entering cyclic operation.
 */
void stack_prefault(void)
{
    unsigned char dummy[MAX_SAFE_STACK];

    memset(dummy, 0, MAX_SAFE_STACK);
}

static volatile int keepRunning = 1;

void
intHandler (int dummy)
{
    printf ("EXIT!\n");
    keepRunning = 0;
}

/** Process main function.
 */
int main(int argc, char **argv)
{
    RT_MODEL *S;
    unsigned int dt_main;
    unsigned int running = 1;
    const char *err = NULL;

    rtExtModeParseArgs (argc, argv, NULL);
      
    signal (SIGINT, intHandler);
    
    /* Initialize model */
    S = MODEL();
  
    task.S = S;
    task.sample_time = rtmGetSampleTime (S, task.sl_tid);
    printf ("Sample-Time: %f\n", task.sample_time);
  
    /* Lock all memory forever. */
    if (mlockall(MCL_CURRENT | MCL_FUTURE))
        fprintf(stderr, "mlockall() failed: %s\n", strerror(errno));

    /* Provoke the first stack fault before cyclic operation. */
    stack_prefault();

    if ((err = init_application(S)))
        goto out;


    
    task.running = &running;
    task.err = 0;
    pthread_create (&task.thread, 0, run_task, &task);

    dt_main = 0.02 * 1.0e9 + 0.5; // 20ms interval for external mode
    struct timespec main_time;
    clock_gettime (CLOCK_MONOTONIC, &main_time);
    /* Main thread running here */
    do
    {
        clock_gettime (CLOCK_REALTIME, &task.world_time);

        rtExtModePauseIfNeeded (rtmGetRTWExtModeInfo (S), rtmGetNumSampleTimes (S), (boolean_T *) & rtmGetStopRequested (S));

        /* external mode */
        rtExtModeOneStep (rtmGetRTWExtModeInfo (S), rtmGetNumSampleTimes (S), (boolean_T *) & rtmGetStopRequested (S));

        // set timer for external mode
        timeradd (&main_time, dt_main);
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME,&main_time, NULL);
    }
    while (!err && running && keepRunning);

    /* external mode */
    rtExtModeOneStep (rtmGetRTWExtModeInfo (S), rtmGetNumSampleTimes (S),
            (boolean_T *) & rtmGetStopRequested (S));

    running = 0;

    rtExtModeShutdown (rtmGetNumSampleTimes (S));

    pthread_join (task.thread, 0);

    /* Clean up */
    MdlTerminate();

out:
    if (err) {
        fprintf(stderr, "Fatal error: %s\n", err);
        return 1;
    }
    else {
        return 0;
    }
}

/****************************************************************************/
