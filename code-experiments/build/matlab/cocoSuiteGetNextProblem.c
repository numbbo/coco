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
    coco_suite_t *suite;
    coco_observer_t *observer;
    coco_problem_t *problem;
    size_t *ref, *ref2;
    
    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoSuiteGetNextProblem:nrhs","Two inputs required.");
    }
    
    /* get the suite */
    ref = (mwSize *)mxGetData(prhs[0]);
    suite = (coco_suite_t *)(*ref);
    /* get the observer */
    ref2 = (mwSize *)mxGetData(prhs[1]);
    observer = (coco_observer_t *)(*ref2);
    
    /* call coco_suite_get_next_problem() */
    
    problem = coco_suite_get_next_problem(suite, observer);
    
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    ref = (size_t *)mxGetData(plhs[0]);
    *ref = (size_t)problem;
}