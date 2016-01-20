#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "coco.h"
#include "coco.c"

#include "mex.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *observer_name;
    char *observer_options;
    coco_observer_t *observer = NULL;
    size_t *res;
    
    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoObserver:nrhs","Two inputs required.");
    }
    /* get observer_name */
    observer_name = mxArrayToString(prhs[0]);
    /* get observer_options */
    observer_options = mxArrayToString(prhs[1]);
    /* call coco_observer() */
    observer = coco_observer(observer_name, observer_options);
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    res = (size_t *)mxGetData(plhs[0]);
    *res = (size_t)observer;
}
