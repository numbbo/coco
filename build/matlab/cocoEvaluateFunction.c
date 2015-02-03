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
    char *problem_suit;
    mxArray *problem_suit_prop;
    int findex;
    mxArray *findex_prop;
    char *observer = NULL;
    mxArray *observer_prop;
    char *options = NULL;
    mxArray *options_prop;
    coco_problem_t *pb = NULL;
    const char *class_name = NULL;
    int nb_objectives;
    double *x;
    double *y;

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:nlhs","One output required.");
    }
    /* make sure the first input argument is Problem */
    class_name = mxGetClassName(prhs[0]); /* may be replaced by mxIsClass */
    if(strcmp(class_name, "Problem") != 0) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:notProblem","Input problem must be a Problem object.");
    }
    /* make sure the second input argument is array of doubles */
    if(!mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt("cocoEvaluateFunction:notProblem","Input x must be aan array of doubles.");
    }
    /* get the properties of the Problem object */
    problem_suit_prop = mxGetProperty(prhs[0], 0, "problem_suit");
    problem_suit = mxArrayToString(problem_suit_prop); /* mxFree(problem_suit) */
    findex_prop = mxGetProperty(prhs[0], 0, "function_index");
    findex = (int)mxGetScalar(findex_prop);
    observer_prop = mxGetProperty(prhs[0], 0, "observer_name");
    observer = mxArrayToString(observer_prop); /* mxFree(observer) */
    options_prop = mxGetProperty(prhs[0], 0, "options");
    options = mxArrayToString(options_prop); /* mxFree(options) */
    /* get the problem */
    pb = coco_get_problem(problem_suit, findex);
    pb = coco_observe_problem(observer, pb, options);
    /* prepare the return value */
    nb_objectives = coco_get_number_of_objectives(pb);
    plhs[0] = mxCreateDoubleMatrix(1, (mwSize)nb_objectives, mxREAL);
    y = mxGetPr(plhs[0]);
    /* get the x vector */
    x = mxGetPr(prhs[1]);
    /* call coco_evaluate_function(...) */
    coco_evaluate_function(pb, x, y);
    /* free resources */
    coco_free_problem(pb);
    mxFree(problem_suit);
    mxFree(observer);
    mxFree(options);
}
