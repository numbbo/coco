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
    size_t *ref;
    coco_problem_t *problem = NULL;
    size_t nb_dim;
    const double *res;
    int i;
    double *v; /* intermediate variable that allows to set plhs[0] */

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoProblemGetLargestValuesOfInterest:nrhs","One input required.");
    }
    /* get the problem */
    ref = (size_t *)mxGetData(prhs[0]);
    problem = (coco_problem_t *)(*ref);
    
    nb_dim = coco_problem_get_dimension(problem);
    plhs[0] = mxCreateDoubleMatrix(1, nb_dim, mxREAL);
    v = mxGetPr(plhs[0]);
    /* call coco_problem_get_largest_values_of_interest(...) */
    res = coco_problem_get_largest_values_of_interest(problem);
    for (i = 0; i < nb_dim; i++){
        v[i] = res[i];
    }
}
