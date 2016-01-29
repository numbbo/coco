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
    coco_suite_t *problem_suite;
    size_t findex;
    coco_problem_t *pb = NULL;
    size_t *res; 
    size_t *ref;

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoSuiteGetProblem:nrhs","Two inputs required.\n Try \'help cocoSuiteGetProblem.m\'.");
    }
    /* get problem_suite */   
    ref = (size_t *)mxGetData(prhs[0]);
    problem_suite = (coco_suite_t *)(*ref);
 
    /* get function_index */
    findex = (size_t)mxGetScalar(prhs[1]);
    /* call coco_suite_get_problem() */
    pb = coco_suite_get_problem(problem_suite, findex);
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    res = (size_t *)mxGetData(plhs[0]);
    *res = (size_t)pb;   
}
