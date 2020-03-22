/*
 * $RCSfile: el31xx.c,v $
 * $Revision$
 * $Date$
 *
 *
 * Copyright (c) 2008, Richard Hacker
 * License: GPLv3+
 */


#define S_FUNCTION_NAME  master_state
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"

#define MASTER             mxGetScalar(ssGetSFcnParam(S,0))
#define DEVICES            mxGetScalar(ssGetSFcnParam(S,1))
#define RESET              mxGetScalar(ssGetSFcnParam(S,2))
#define REFCLOCK_DEC       mxGetScalar(ssGetSFcnParam(S,3))
#define TSAMPLE            mxGetScalar(ssGetSFcnParam(S,4))
#define PARAM_COUNT                                     5


/*====================*
 * S-function methods *
 *====================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    uint_T i;
    int_T num_devices = DEVICES;

    ssSetNumSFcnParams(S, PARAM_COUNT);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        /* Return if number of expected != number of actual parameters */
        return;
    }
    for( i = 0; i < PARAM_COUNT; i++) 
        ssSetSFcnParamTunable(S,i,SS_PRM_NOT_TUNABLE);

    if (num_devices <= 0) {
        ssSetErrorStatus(S, "Number of devices must be a positive integer");
        return;
    }
    
    if (RESET) {
        ssSetNumInputPorts(S, 1);
        ssSetNumIWork(S, 1);
        ssSetInputPortDataType(S, 0, DYNAMICALLY_TYPED);
        ssSetInputPortWidth(S, 0, 1);
    }
    else {
        ssSetNumInputPorts(S, 0);
    }

    if (!ssSetNumOutputPorts(S, 3)) return;
    ssSetOutputPortWidth(S, 0, num_devices);
    ssSetOutputPortDataType(S, 0, SS_UINT32);
    ssSetOutputPortWidth(S, 1, num_devices);
    ssSetOutputPortDataType(S, 1, SS_UINT8);
    ssSetOutputPortWidth(S, 2, num_devices);
    ssSetOutputPortDataType(S, 2, SS_BOOLEAN);

    ssSetNumSampleTimes(S, 1);
    ssSetNumPWork(S, 1);

    ssSetOptions(S, 
            SS_OPTION_WORKS_WITH_CODE_REUSE | 
            SS_OPTION_RUNTIME_EXCEPTION_FREE_CODE);
}

/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    This function is used to specify the sample time(s) for your
 *    S-function. You must register the same number of sample times as
 *    specified in ssSetNumSampleTimes.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, TSAMPLE);
    ssSetOffsetTime(S, 0, 0.0);
}

/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block. Generally outputs are placed in the input vector, ssGetY(S).
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
}

/* Function: mdlTerminate =====================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was
 *    allocated in mdlStart, this is the place to free it.
 */
static void mdlTerminate(SimStruct *S)
{
}

#define MDL_RTW
static void mdlRTW(SimStruct *S)
{
    int32_T master = MASTER;
    uint32_T refclock_dec = REFCLOCK_DEC >= 1.0 ? REFCLOCK_DEC : 0;

    if (!ssWriteRTWScalarParam(S, "MasterId", &master, SS_INT32))
        return;
    if (!ssWriteRTWScalarParam(S, "RefClkSyncDec", &refclock_dec, SS_UINT32))
        return;
    if (!ssWriteRTWWorkVect(S, "PWork", 1, "MasterPtr", 1))
        return;

    if (ssGetNumInputPorts(S)) {
        if (!ssWriteRTWWorkVect(S, "IWork", 1, "ResetState", 1))
            return;
    }
}



/*======================================================*
 * See sfuntmpl_doc.c for the optional S-function methods *
 *======================================================*/

/*=============================*
 * Required S-function trailer *
 *=============================*/

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
