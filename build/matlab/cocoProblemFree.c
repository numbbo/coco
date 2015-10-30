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
    char *problem_suite;
    coco_problem_t *problem = NULL;
    long long *ref;

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoProblemFree:nrhs","One input required.");
    }
    /* get the problem */
    ref = (long long *)mxGetData(prhs[0]);
    problem = (coco_problem_t *)(*ref);
    /* call coco_problem_free() */
    coco_problem_free(problem);
}
