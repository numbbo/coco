#include "mex.h"

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h> /* for isspace() */

/* #define VARIANT 4 */

int stop_dimension = 1; /* default: stop on dimension 2 */


double hypervolume(double* points, double* referencePoint, unsigned int noObjectives, unsigned int noPoints);

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	double * volume;
	double * data, * reference;
	unsigned int row, col, row_r, col_r;
	const int  *dims, *dims_r;
    
	if(nrhs != 2 || nlhs != 1)
	{
	    printf("\nsynopsis:   volume = hv(paretofront', reference_point)");
	    plhs[0]    = mxCreateDoubleMatrix(0 , 0 ,  mxREAL);
	    return;
	}
	
    data = mxGetPr(prhs[0]);
	dims = mxGetDimensions(prhs[0]);
    /* ATTENTION: please transpose the pf matrix!!! */
	row = dims[1];
	col = dims[0];

    reference = mxGetPr(prhs[1]);
	dims_r = mxGetDimensions(prhs[1]);
	row_r = dims_r[0];
	col_r = dims_r[1];
    
    if((col != col_r) || (row_r != 1))
    {
	    printf("\n size-pf: %d x %d  size-ref: %d x %d", row, col, row_r, col_r);
	    printf("\n ATTENTION: please transpose the pf matrix!!!");
	    printf("\nsynopsis:   volume = hv(paretofront', reference_point)");
	    plhs[0]    = mxCreateDoubleMatrix(0 , 0 ,  mxREAL);
	    return;
    }
    
    /* ----- output ----- */

	plhs[0]    = mxCreateDoubleMatrix (1 , 1, mxREAL);
	volume = (double *) mxGetPr(plhs[0]);
	
	
	/* main call */
	(*volume) = hypervolume(data, reference, col, row);
}

