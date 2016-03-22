/*
 * =============================================================
 * MOSelector.c - Select some points out from a given population
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
		SData,	//number of selected data
		FData;	//number of feasible data
double 	**pX,	//pointer to X
		**pF,	//pointer to F
		*pV;	//pointer to V
std::vector<int> rankV;	//rank vector

//sort population by constraint vialation
void SortFeasible()
{
	int i,j;
	double *p, p1;
	for(i=0; i<NData; i++)
		for(j=i+1; j<NData; j++)
			if(pV[j]<pV[i])
			{
				p = pF[i]; 		pF[i] 	= pF[j]; 	pF[j] 	= p;
				p = pX[i]; 		pX[i] 	= pX[j]; 	pX[j] 	= p;
				p1= pV[i]; 		pV[i]   = pV[j];	pV[j]   = p1;
			}
	FData = 0;
	while(FData<NData && pV[FData]<0.00001) FData++;
}

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

// Sort population by rank value
void SortRank()
{
	int i,j,dom;
	std::vector< std::vector<int> > domM;	//dominate matrix
	std::vector<int> domV;					//dominate vector

	domM.resize(FData);
	domV.resize(FData);
	rankV.resize(FData);
	for(i=0; i<FData; i++)
	{
		domM[i].resize(FData);
		domV[i] 	= 0;
		rankV[i]	= -1;
	}

	// Set dominate matrix
	for(i=0; i<FData; i++)
	{
		domM[i][i] = 0;
		for(j=i+1; j<FData; j++)
		{
			dom = Dominate(i,j);
			if(dom>0) {domM[j][i] = 1; domV[j]++;}
			else if(dom<0) {domM[i][j] = 1; domV[i]++;}
		}
	}

	// Assign rank
	int NAssign = 0;
	int CRank   = 0;
	int MRank;
	while(NAssign < FData)
	{
		CRank   = CRank + 1;
		MRank	= FData + 10;
		for(i=0; i<FData; i++) if(rankV[i]<0 && domV[i]<MRank ) MRank = domV[i];
		for(i=0; i<FData; i++) if(rankV[i]<0 && domV[i]==MRank) {rankV[i]= CRank; NAssign++;}
		for(i=0; i<FData; i++)
			if(rankV[i] == CRank)
				for(j=0; j<FData; j++) if(rankV[j]<0 && domM[j][i]>0) domV[j]--;
	}

	// Sort by rank
	double *p; int r;
	for(i=0; i<FData; i++)
		for(j=i+1; j<FData; j++) if(rankV[i]>rankV[j])
		{
			r = rankV[i]; 	rankV[i]= rankV[j]; rankV[j]= r;
			p = pF[i]; 		pF[i] 	= pF[j]; 	pF[j] 	= p;
			p = pX[i]; 		pX[i] 	= pX[j]; 	pX[j] 	= p;
		}
}

// Contribution to the density
double FDen(double dis)
{
	if(dis<1.0E-10) return 1.0E10;
	else return 1.0/dis;
}

// Sort population by density
void SortDensity(int iS, int iE)
{
	if(iE == SData || iS == SData-1) return;

	std::vector< std::vector<double> > denM;
	std::vector< double > denV;
	std::vector< int > staV;
	int N = iE-iS+1;
	int i,j,k,index;
	double dis;

	denM.resize(N);
	denV.resize(N);
	staV.resize(N);
	for(i=0; i<N; i++)
	{
		denM[i].resize(N);
		denV[i] = 0;
		staV[i] = 1;
	}
	for(i=0; i<N; i++)
	{
		denM[i][i] = 0;
		for(j=i+1; j<N; j++)
		{
			dis = 0;
			for(k=0; k<FDim; k++) dis += (pF[i+iS][k]-pF[j+iS][k])*(pF[i+iS][k]-pF[j+iS][k]);
			denM[i][j] = denM[j][i] = FDen(sqrt(dis));
			denV[i] += denM[i][j];
			denV[j] += denM[j][i];
		}
	}

	// Remove one by one
	for(k=0; k<iE-SData+1; k++)
	{
		index = -1;
		for(i=0; i<N; i++) if(staV[i]>0 && (index<0 || denV[i]>denV[index])) index = i;
		staV[index] = 0;
		for(i=0; i<N; i++) if(staV[i]>0) denV[i] -= denM[i][index];
	}

	// Sort
    i=iS; j=iE;
    int s; double *p;
    while(i<j)
    {
        while(i<=iE && staV[i-iS]>0) i = i+1;
        while(j>=iS && staV[j-iS]<1) j = j-1;
        if(i<j)
        {
            s = staV[i-iS]; staV[i-iS] = staV[j-iS]; staV[j-iS] = s;
            p = pF[i]; pF[i] = pF[j]; pF[j] = p;
            p = pX[i]; pX[i] = pX[j]; pX[j] = p;
            i  = i + 1;
            j  = j - 1;
		}
	}
}

// Select operator
void Select()
{
	//mexPrintf("rank start\n");
	// Sort by constraint vialation
	SortFeasible();

	if(FData<=SData) return;

	// Sort population by rank value
	SortRank();

	//mexPrintf("rank over\n");

	// Find the subpopulation to sort by density
	int iS, iE;
	iS = iE = 0;
	while(iE < FData && rankV[iE] == rankV[iS]) iE++;
	iE--;
	while(iE < SData-1)
	{
    	iS = iE + 1; iE = iE + 1;
    	while(iE < FData && rankV[iE] == rankV[iS]) iE++;
    	iE--;
	}

	//mexPrintf("iS=%d, iE=%d\n",iS,iE);

	// Sort by density
	SortDensity(iS, iE);

	//mexPrintf("density over\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[])
{
	int i,j;
	// Check for proper number of arguments.
	if(nrhs != 4)
	{
		mexErrMsgTxt("Four input required.");
	} else if(nlhs != 3)
	{
		mexErrMsgTxt("Three output arguments");
	}

	// Check for data number
	FDim 	= mxGetM(prhs[0]);
	XDim 	= mxGetM(prhs[1]);
	NData	= mxGetN(prhs[0]);
	SData	=  (int)mxGetScalar(prhs[3]);
	if(SData > NData) SData = NData;
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

	// Select
	Select();

	// Create matrix for the return arguments.
	plhs[0] = mxCreateDoubleMatrix(FDim,SData, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(XDim,SData, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1,SData, mxREAL);
	pf 		= mxGetPr(plhs[0]);
	px 		= mxGetPr(plhs[1]);
	pv 		= mxGetPr(plhs[2]);
	// Copy data to return arguments.
	for(i=0; i<SData; i++)
	{
		pv[i] = pV[i];
		for(j=0; j<FDim; j++) pf[i*FDim+j] = pF[i][j];
		for(j=0; j<XDim; j++) px[i*XDim+j] = pX[i][j];
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
