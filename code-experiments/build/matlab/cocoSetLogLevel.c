#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "coco.h"
#include "coco.c"

#include "mex.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
    char *level;
    const char *res;

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoSetLogLevel:nrhs","One input required.");
    }
    /* get the problem */
    level = mxArrayToString(prhs[0]);
    /* call coco_set_log_level(...) */
    res = coco_set_log_level(level);
    /* prepare the return value */
    plhs[0] = mxCreateString(res);

    coco_warning("This Coco functionality is not yet supported in Matlab.");
}
