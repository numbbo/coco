/*
 * =============================================================
 * ParetoFilter.cpp - Select nondominated points from a given population
 *
 * Copyright (c) 2006 Aimin Zhou
 * Dept. of Computer Science
 * Univ. of Essex
 * Colchester, CO4 0DY, U.K
 * azhou@essex.ac.uk
 * =============================================================
 */

#include "mex.h"
#include <vector>
#include <cmath>

int 	XDim,	//dimension of X
		FDim,	//dimension of F
		NData,	//number of data
		SData;	//number of selected data
double 	**pX,	//pointer to X
		**pF,	//pointer to F
		*pV;	//pointer to V
std::vector<int> existV;	//exist indicator

// Dominate
// point1 dominates point2	: 1
// point2 dominates point1	: -1
// non-dominates each other	: 0
int Dominate(int iA, int iB)
{
	int strictBetter = 0;
	int strictWorse  = 0;
	int better		 = 0;
	int worse		 = 0;
	int i;

	for(i=0; i<FDim; i++)
	{
		if(pF[iA][i]<=pF[iB][i])
		{
			better++;
			strictBetter += pF[iA][i]<pF[iB][i] ? 1:0;
		}
		if(pF[iA][i]>=pF[iB][i])
		{
			worse++;
			strictWorse += pF[iA][i]>pF[iB][i] ? 1:0;
		}
	}

	if(better == FDim && strictBetter > 0) return 1;
	if(worse  == FDim && strictWorse  > 0) return -1;
	return 0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[])
{
	int i,j,k,dom;
	// Check for proper number of arguments.
	if(nrhs != 3)
	{
		mexErrMsgTxt("Three input required.");
	} else if(nlhs != 3)
	{
		mexErrMsgTxt("Three output arguments");
	}

	// Check for data number
	FDim 	= mxGetM(prhs[0]);
	XDim 	= mxGetM(prhs[1]);
	NData	= mxGetN(prhs[0]);
	if(NData != mxGetN(prhs[1]))
	{
		mexErrMsgTxt("Input F and X must have the same number.");
	}

	// Copy data.
	double *pf = mxGetPr(prhs[0]);
	double *px = mxGetPr(prhs[1]);
	double *pv = mxGetPr(prhs[2]);
	pF = new double*[NData];
	pX = new double*[NData];
	pV = new double[NData];
	for(i=0; i<NData; i++)
	{
		pF[i] = new double[FDim];
		pX[i] = new double[XDim];
		pV[i] = pv[i];
		for(j=0; j<FDim; j++) pF[i][j] = pf[i*FDim+j];
		for(j=0; j<XDim; j++) pX[i][j] = px[i*XDim+j];
	}


	// Set existV vector
	existV.resize(NData);

	for(i=0; i<NData; i++) if(pV[i]>0.00001) existV[i] = 0.0; else existV[i] = 1.0;

	for(i=0; i<NData; i++) if(existV[i]>0)
	{
		for(j=i+1; j<NData; j++) if(existV[j]>0)
		{
			dom = Dominate(i,j);
			if(dom>0) 		existV[j] = 0;
			else if(dom<0) 	existV[i] = 0;
		}
	}

	SData = 0;
	for(i=0; i<NData; i++) if(existV[i] > 0) SData++;


	// Create matrix for the return arguments.
	plhs[0] = mxCreateDoubleMatrix(FDim,SData, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(XDim,SData, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1,SData, mxREAL);
	pf 		= mxGetPr(plhs[0]);
	px 		= mxGetPr(plhs[1]);
	pv 		= mxGetPr(plhs[2]);

	// Copy data to return arguments.
	k=0;
	for(i=0; i<NData; i++)
		if(existV[i]>0)
		{
			for(j=0; j<FDim; j++) pf[k*FDim+j] = pF[i][j];
			for(j=0; j<XDim; j++) px[k*XDim+j] = pX[i][j];
			pv[k] = pV[i];
			k++;
		}

	// Free space
	for(i=0; i<NData; i++)
	{
		delete []pF[i];
		delete []pX[i];
	}
	delete []pF;
	delete []pX;
	delete []pV;
}
