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
    coco_problem_t *pb = NULL;

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("problemIsValid:nrhs","One input required.");
    }
    /* get the problem */
    ref = (long long *)mxGetData(prhs[0]);
    pb = (coco_problem_t *)(*ref);
    plhs[0] = mxCreateLogicalScalar(pb != NULL);
}

