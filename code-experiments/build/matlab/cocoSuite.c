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
    char *suite_name;
    char *suite_instance;
    char *suite_options;
    coco_suite_t *suite = NULL;
    long long *res;
    
    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("cocoSuite:nrhs","Three inputs required.");
    }
    /* get suite_name */
    suite_name = mxArrayToString(prhs[0]);
    /* get suite_instance */
    suite_instance = mxArrayToString(prhs[1]);
    /* get suite_options */
    suite_options = mxArrayToString(prhs[2]);
    /* call coco_suite() */
    suite = coco_suite(suite_name, suite_instance, suite_options);
    /* prepare the return value */
    plhs[0] = mxCreateNumericMatrix(1, 1 ,mxINT64_CLASS, mxREAL);
    res = (long long *)mxGetData(plhs[0]);
    *res = (long long)suite;
}