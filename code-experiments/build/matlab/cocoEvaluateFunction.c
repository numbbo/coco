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
    mwSize *ref;
    mxArray *problem_prop;
    coco_problem_t *problem = NULL;
    /* const char *class_name = NULL; */
    int nb_objectives;
    double *x;
    double *y;

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:nrhs","Two inputs required.");
    }
    /* get the problem */
    ref = (mwSize *) mxGetData(prhs[0]);
    problem = (coco_problem_t *)(*ref);
    /* make sure the second input argument is array of doubles */
    if(!mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:notDoubleArray","Input x must be aan array of doubles.");
    }
    /* get the x vector */
    x = mxGetPr(prhs[1]);
    /* prepare the return value */
    nb_objectives = coco_problem_get_number_of_objectives(problem);
    plhs[0] = mxCreateDoubleMatrix(1, (mwSize)nb_objectives, mxREAL);
    y = mxGetPr(plhs[0]);
    /* call coco_evaluate_function(...) */
    coco_evaluate_function(problem, x, y);
}
