#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "coco.h"
#include "coco.c"

#include "mex.h"
#include "matrix.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *observer;
    coco_problem_t *problem;
    char *options;
    long long *ref;

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("cocoProblemAddObserver:nrhs","Three inputs required.");
    }
    /* get the observer */
    observer = mxArrayToString(prhs[0]);
    /* get the problem */
    ref = (long long *)mxGetData(prhs[1]);
    problem = (coco_problem_t *)(*ref);
    /* get the options */
    options = mxArrayToString(prhs[2]);
    /* call deprecated__coco_problem_add_observer() */
    problem = deprecated__coco_problem_add_observer(problem, observer, options);
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    ref = (long long *)mxGetData(plhs[0]);
    *ref = (long long)problem;
}

