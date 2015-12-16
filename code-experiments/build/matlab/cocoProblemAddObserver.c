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
    coco_problem_t *problem;
    coco_observer_t *observer;
    char *observer_options;
    long long *ref, *ref2;
    observer_biobj_t *data;

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("cocoProblemAddObserver:nrhs","Three inputs required.");
    }
    /* get the observer */
    observer_name = mxArrayToString(prhs[0]);
    /* get the problem */
    ref = (long long *)mxGetData(prhs[1]);
    problem = (coco_problem_t *)(*ref);
    /* get the observer options */
    observer_options = mxArrayToString(prhs[2]);
    
    /* call deprecated__coco_problem_add_observer() */
    /*problem = deprecated__coco_problem_add_observer(problem, observer_name, observer_options);*/
    
    observer = coco_observer(observer_name, observer_options);
    problem = coco_problem_add_observer(problem, observer);
    
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    ref = (long long *)mxGetData(plhs[0]);
    *ref = (long long)problem;
    plhs[1] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    ref2 = (long long *)mxGetData(plhs[1]);
    *ref2 = (long long)observer;
}

