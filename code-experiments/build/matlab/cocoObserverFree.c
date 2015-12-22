#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "coco.h"
#include "coco.c"

#include "mex.h"
//#include "matrix.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    coco_observer_t *observer;
    long long *ref;
    
    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoObserverFree:nrhs","One input required.");
    }
    /* get the observer */
    ref = (long long *)mxGetData(prhs[0]);
    observer = (coco_observer_t *)(*ref);
    /* call coco_observer_free() */
    coco_observer_free(observer);
}