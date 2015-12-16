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
    long long *ref;
    mxArray *problem_prop;
    coco_problem_t *problem = NULL;
    const char *class_name = NULL;
    int nb_objectives;
    double *x;
    double *y;

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:nrhs","Two inputs required.");
    }
    /*if(nlhs!=1) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:nlhs","One output required.");
    }*/
    /* make sure the first input argument is Problem */
    class_name = mxGetClassName(prhs[0]); /* may be replaced by mxIsClass */
    if(strcmp(class_name, "Problem") != 0) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:notProblem","Input problem must be a Problem object.");
    }
    /* make sure the second input argument is array of doubles */
    if(!mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:notDoubleArray","Input x must be aan array of doubles.");
    }
    /* get Problem.problem */
    problem_prop = mxGetProperty(prhs[0], 0, "problem");
    ref = (long long *)mxGetData(problem_prop);
    problem = (coco_problem_t *)(*ref);
    /* get the x vector */
    x = mxGetPr(prhs[1]);
    /* prepare the return value */
    nb_objectives = coco_problem_get_number_of_objectives(problem);
    plhs[0] = mxCreateDoubleMatrix(1, (mwSize)nb_objectives, mxREAL);
    y = mxGetPr(plhs[0]);
    /* call coco_evaluate_function(...) */
    coco_evaluate_function(problem, x, y);
}
