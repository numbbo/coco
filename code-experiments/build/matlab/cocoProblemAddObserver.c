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
    coco_problem_t *problem;
    coco_observer_t *observer;
    long long *ref, *ref2;

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoProblemAddObserver:nrhs","Two inputs required.");
    }
    /* get the problem */
    ref = (long long *)mxGetData(prhs[0]);
    problem = (coco_problem_t *)(*ref);
    /* get the observer */
    ref2 = (long long *)mxGetData(prhs[1]);
    observer = (coco_observer_t *)(*ref2);
    
    /* call coco_problem_add_observer() */
    
    problem = coco_problem_add_observer(problem, observer);
    
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    ref = (long long *)mxGetData(plhs[0]);
    *ref = (long long)problem;
}

