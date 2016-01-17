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
    coco_problem_t *problem = NULL;
    int nb_dim, i;
    const double *res;
    double *v; /* intermediate variable that aloows to set plhs[0] */

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("cocoProblemGetSmallestValuesOfInterest:nrhs","One input required.");
    }
    /* get the problem */
    ref = (long long *)mxGetData(prhs[0]);
    problem = (coco_problem_t *)(*ref);
    nb_dim = coco_problem_get_dimension(problem);
    plhs[0] = mxCreateDoubleMatrix(1, (mwSize)nb_dim, mxREAL);
    v = mxGetPr(plhs[0]);
    /* call coco_problem_get_smallest_values_of_interest(...) */
    res = coco_problem_get_smallest_values_of_interest(problem);
    for (i = 0; i < nb_dim; i++){
        v[i] = res[i];
    }
}
