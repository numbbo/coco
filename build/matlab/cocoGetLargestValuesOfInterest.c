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
    int nb_variables;
    const double *res;

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("cocoGetLargestValuesOfInterest:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("cocoGetLargestValuesOfInterest:nlhs","One output required.");
    }
    /* make sure the first input argument is Problem */
    class_name = mxGetClassName(prhs[0]); /* may be replaced by mxIsClass */
    if(strcmp(class_name, "Problem") != 0) {
        mexErrMsgIdAndTxt("cocoGetLargestValuesOfInterest:notProblem","Input problem must be a Problem object.");
    }
    /* get the properties of the Problem object */
    problem_suit_prop = mxGetProperty(prhs[0], 0, "problem_suit");
    problem_suit = mxArrayToString(problem_suit_prop); /* mxFree(problem_suit) */
    findex_prop = mxGetProperty(prhs[0], 0, "function_index");
    findex = (int)mxGetScalar(findex_prop);
    /* get the problem */
    pb = coco_get_problem(problem_suit, findex);
    /* prepare the return value */
    nb_variables = coco_get_number_of_variables(pb);
    plhs[0] = mxCreateDoubleMatrix(1, (mwSize)nb_variables, mxREAL);
    res = mxGetPr(plhs[0]);
    /* call coco_get_smallest_values_of_interest(...) */
    res = coco_get_largest_values_of_interest(pb);
    /* free resources */
    coco_free_problem(pb);
    mxFree(problem_suit);
}
