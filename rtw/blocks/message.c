/*
* $Id$
*
* This SFunction implements an event generator
*
* Copyright (c) 2012, Richard Hacker
* License: GPL
*
*/

#define S_FUNCTION_NAME  message
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include <string.h>

#define PRIORITY      (uint_T)(mxGetScalar(ssGetSFcnParam(S,0)))
#define EXTRESET   (boolean_T)(mxGetScalar(ssGetSFcnParam(S,1)))
#define INVERT     (boolean_T)(mxGetScalar(ssGetSFcnParam(S,2)))
#define CONFIRM                           (ssGetSFcnParam(S,3))
#define TSAMPLE               (mxGetScalar(ssGetSFcnParam(S,4)))
#define PARAM_COUNT                                         5

/* Function: mdlInitializeSizes =============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    int_T i;

    ssSetNumSFcnParams(S, PARAM_COUNT);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        /* Return if number of expected != number of actual parameters */
        return;
    }

    for( i = 0; i < PARAM_COUNT; i++)
        ssSetSFcnParamTunable(S, i, SS_PRM_NOT_TUNABLE);

    /* Make CONFIRM tunable */
    if (mxGetNumberOfElements(CONFIRM))
        ssSetSFcnParamTunable(S, 3, SS_PRM_TUNABLE);

    /* Process input ports */
    if (!ssSetNumInputPorts(S, 1 + EXTRESET))
        return;

    ssSetInputPortWidth(   S, 0, DYNAMICALLY_SIZED);
    ssSetInputPortDataType(S, 0, DYNAMICALLY_TYPED);
    ssSetInputPortDirectFeedThrough(S, 0, 1);

    if (ssGetNumInputPorts(S) > 1) {
        /* External reset input */
        ssSetInputPortWidth(   S, 1, DYNAMICALLY_SIZED);
        ssSetInputPortDataType(S, 1, DYNAMICALLY_TYPED);
        ssSetInputPortDirectFeedThrough(S, 1, 1);
    }

    if (!ssSetNumOutputPorts(S, 1))
        return;
    /* State output port */
    ssSetOutputPortWidth(   S, 0, DYNAMICALLY_SIZED);
    ssSetOutputPortDataType(S, 0, SS_DOUBLE);

    ssSetNumSampleTimes(S, 1);
    ssSetNumPWork(S, 1);

    ssSetOptions(S,
            SS_OPTION_WORKS_WITH_CODE_REUSE
            | SS_OPTION_RUNTIME_EXCEPTION_FREE_CODE
            | SS_OPTION_CALL_TERMINATE_ON_EXIT);
}

/* Function: mdlInitializeSampleTimes =======================================
 * Abstract:
 *    This function is used to specify the sample time(s) for your
 *    S-function. You must register the same number of sample times as
 *    specified in ssSetNumSampleTimes.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, TSAMPLE);
}

#define MDL_SET_INPUT_PORT_WIDTH
static void mdlSetInputPortWidth(SimStruct *S, int_T port, int_T width)
{
    ssSetInputPortWidth(S, port, width);

    if (port == 0) {
        int_T confirmCount = mxGetNumberOfElements(CONFIRM);

        ssSetOutputPortWidth(S, 0, width);

        confirmCount = mxGetNumberOfElements(CONFIRM);
        if (confirmCount
                && ((confirmCount != 1 && confirmCount != width)
                    || !mxGetPr(CONFIRM))) {
            ssSetErrorStatus(S,
                    "Confirm is size is incompatible with input");
            return;
        }
    }

    if (ssGetNumInputPorts(S) > 1
            && ssGetInputPortWidth(S, 0) > 0
            && ssGetInputPortWidth(S, 1) > 1
            && ssGetInputPortWidth(S, 0) != ssGetInputPortWidth(S, 1))
        ssSetErrorStatus(S,
                "Reset port must be scalar or same dimension as input port");
}

#define MDL_SET_OUTPUT_PORT_WIDTH
static void mdlSetOutputPortWidth(SimStruct *S, int_T port, int_T width)
{
    ssSetOutputPortWidth(S,port,width);
}

#define MDL_SET_WORK_WIDTHS
static void mdlSetWorkWidths(SimStruct *S)
{
    ssParamRec p; /* Initialize an ssParamRec structure */
    int_T dim;
    real_T *pr;
    boolean_T *enable;
    int_T i, dlg;
    uint32_T *confirm;

    if (!ssSetNumRunTimeParams(S, 1 + (mxGetNumberOfElements(CONFIRM) > 0)))
        return;

    /* First parameter is "Enable" */
    dim = ssGetInputPortWidth(S, 0);

    enable = mxCalloc(dim, sizeof(boolean_T));
    mexMakeMemoryPersistent(enable);
    for (i = 0; i < dim; ++i)
        enable[i] = 1;

    p.name             = "Enable";
    p.nDimensions      = 1;
    p.dimensions       = &dim;
    p.dataTypeId       = SS_BOOLEAN;
    p.complexSignal    = COMPLEX_NO;
    p.data             = enable;
    p.dataAttributes   = NULL;
    p.nDlgParamIndices = 0;
    p.dlgParamIndices  = NULL;
    p.transformed      = RTPARAM_TRANSFORMED;
    p.outputAsMatrix   = false;   
    
    /* Set run-time parameter information */
    if (!ssSetRunTimeParamInfo(S, 0, &p)
            || ssGetNumRunTimeParams(S) < 2)
        return;

    /* Second parameter is "Confirm" */
    ssSetNumRWork(S, ssGetInputPortWidth(S,0));

    ssRegDlgParamAsRunTimeParam(S, 3, 1, "Confirm", SS_DOUBLE);

    return;
}

/* Function: mdlOutputs =====================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block. Generally outputs are placed in the output vector, ssGetY(S).
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
}

/* Function: mdlTerminate ===================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was
 *    allocated in mdlStart, this is the place to free it.
 */
static void mdlTerminate(SimStruct *S)
{
    if (ssGetNumRunTimeParams(S))
        mxFree(ssGetRunTimeParamInfo(S,0)->data);
}

#define MDL_RTW
static void mdlRTW(SimStruct *S)
{
    boolean_T invert = INVERT;
    uint32_T priority = PRIORITY;

    if (!ssWriteRTWWorkVect(S, "PWork", 1, "EventPtr", 1))
        return;

    if (ssGetNumRWork(S)
            && !ssWriteRTWWorkVect(S, "RWork", 1,
                "ConfirmTime", ssGetNumRWork(S)))
        return;

    if (!ssWriteRTWParamSettings(S, 2,

                SSWRITE_VALUE_DTYPE_NUM,
                "Priority", &priority, DTINFO(SS_UINT32,0),

                SSWRITE_VALUE_DTYPE_NUM,
                "Invert", &invert, DTINFO(SS_BOOLEAN,0)))
        return;
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
