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
    coco_problem_t *pb = NULL;
    const char *class_name = NULL;
    int *res;
    const mwSize dims[2] = {1, 1};

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoGetNumberOfObjectives:nrhs","One input required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("cocoGetNumberOfObjectives:nlhs","One output required.");
    }
    /* make sure the first input argument is Problem */
    class_name = mxGetClassName(prhs[0]); /* may be replaced by mxIsClass */
    if(strcmp(class_name, "Problem") != 0) {
        mexErrMsgIdAndTxt("cocoGetNumberOfObjectives:notProblem","Input problem must be a Problem object.");
    }
    /* get the properties of the Problem object */
    problem_suit_prop = mxGetProperty(prhs[0], 0, "problem_suit");
    problem_suit = mxArrayToString(problem_suit_prop); /* mxFree(problem_suit) */
    findex_prop = mxGetProperty(prhs[0], 0, "function_index");
    findex = (int)mxGetScalar(findex_prop);
    /* get the problem */
    pb = coco_get_problem(problem_suit, findex);
    /* prepare the return value */

    plhs[0] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    res = (int *)mxGetData(plhs[0]);
    res[0] = coco_get_number_of_objectives(pb);
    /* free resources */
    coco_free_problem(pb);
    mxFree(problem_suit);
}
