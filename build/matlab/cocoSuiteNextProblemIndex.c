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
    long long *ref;
    coco_problem_t *problem = NULL;
    const char *problem_suite = NULL;
    const char *select_options = NULL;
    int problem_index;
    int *res;
    const mwSize dims[2] = {1, 1};

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("cocoSuiteNextProblemIndex:nrhs","Three inputs required.");
    }
    /* get problem_suite */
    problem_suite = mxArrayToString(prhs[0]);
    /* get problem_index */
    problem_index = (int)mxGetScalar(prhs[1]);
    /* get select_options */
    select_options = mxArrayToString(prhs[2]);
    plhs[0] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    res = (int *)mxGetData(plhs[0]);
    res[0] = coco_suite_next_problem_index(problem_suite, problem_index, select_options);
}
