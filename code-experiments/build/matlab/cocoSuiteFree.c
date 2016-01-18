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
    coco_suite_t *suite = NULL;
    size_t *ref;
    
    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoSuiteFree:nrhs","One input required.");
    }
    /* get the suite */
    ref = (size_t *)mxGetData(prhs[0]);
    suite = (coco_suite_t *)(*ref);
    /* call coco_suite_free() */
    coco_suite_free(suite);
}