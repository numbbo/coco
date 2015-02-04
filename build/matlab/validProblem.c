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
    const char *problem_suit;   /* input string */
    int findex;                 /* input integer */
    
    coco_problem_t *pb = NULL;  /* intern variable */

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("validProblem:nrhs","Two inputs required.");
    }
    /* make sure the first input argument is string */
    if(!mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("validProblem:notString","Input problem_suit must be a string.");
    }
    /* make sure the second input argument is integer */
    /*if(!mxIsInt8(prhs[1]) && !mxIsInt16(prhs[1]) && !mxIsInt32(prhs[1]) && !mxIsInt64(prhs[1])) {
        mexErrMsgIdAndTxt("validProblem:notInteger","Input findex must be an integer.");
    }*/ /* TODO : doesn't work... */
    /* get the value of the string input */
    problem_suit = mxArrayToString(prhs[0]);
    if(problem_suit == NULL) 
      mexErrMsgIdAndTxt( "validProblem:conversionFailed", "Could not convert input to string.");
    /* get the value of the integer input */
    findex = mxGetScalar(prhs[1]);

    /* call the computational routine */
    pb = coco_get_problem(problem_suit, (mwSize)findex);
    plhs[0] = mxCreateLogicalScalar(pb != NULL);
    coco_free_problem(pb);
}

