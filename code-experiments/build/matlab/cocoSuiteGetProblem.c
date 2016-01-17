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
    char *problem_suite;
    int findex;
    coco_problem_t *pb = NULL;
    long long *res; 

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoSuiteGetProblem:nrhs","Two inputs required.");
    }
    /* get problem_suite */
    problem_suite = mxArrayToString(prhs[0]);
    /* get function_index */
    findex = (int)mxGetScalar(prhs[1]);
    /* call coco_suite_get_problem() */
    pb = coco_suite_get_problem(problem_suite, findex);
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    res = (long long *)mxGetData(plhs[0]);
    *res = (long long)pb;   
}
