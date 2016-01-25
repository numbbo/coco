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
    size_t *ref;
    coco_problem_t *problem = NULL;
    const mwSize dims[2] = {1, 1};
    int *res;

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoProblemGetDimension:nrhs","One input required.");
    }
    /* get the problem */
    ref = (size_t *)mxGetData(prhs[0]);
    problem = (coco_problem_t *)(*ref);
    /* prepare the return value */
    plhs[0] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    res = (int *)mxGetData(plhs[0]);
    res[0] = coco_problem_get_dimension(problem);
}
