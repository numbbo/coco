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
    const char *res;

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoGetProblemId:nrhs","One input required.");
    }
    /* make sure the first input argument is Problem */
    class_name = mxGetClassName(prhs[0]); /* may be replaced by mxIsClass */
    if(strcmp(class_name, "Problem") != 0) {
        mexErrMsgIdAndTxt("cocoGetProblemId:notProblem","Input problem must be a Problem object.");
    }
    /* get the properties of the Problem object */
    problem_suit_prop = mxGetProperty(prhs[0], 0, "problem_suit");
    problem_suit = mxArrayToString(problem_suit_prop); /* mxFree(problem_suit) */
    findex_prop = mxGetProperty(prhs[0], 0, "function_index");
    findex = (int)mxGetScalar(findex_prop);
    /* get the problem */
    pb = coco_get_problem(problem_suit, findex);
    /* call coco_get_problem_id(...) */
    res = coco_get_problem_id(pb);
    /* prepare the return value */
    plhs[0] = mxCreateString(res);
    /* free resources */
    coco_free_problem(pb);
    mxFree(problem_suit);
}