// paretofront returns the logical Pareto membership of a set of points
// synopsis:  frontFlag = paretofront(objMat)
// Created by Yi Cao: y.cao@cranfield.ac.uk
// for compiling type:
//   mex paretofront.c

#include <stdio.h>
#include <stdlib.h>  // memory, e.g. malloc
#include <stdbool.h> // to use the bool datatype, required C99
#include <math.h>
#include "mo_recorder.h"
#include <stdbool.h> // to use the bool datatype, required C99

#ifdef __cplusplus
extern "C" {
#endif


void mococo_pareto_front(bool *frontFlag, double *obj, unsigned nrow, unsigned ncol);


void mococo_pareto_filtering(struct mococo_solutions_archive *archive) {
    // Create the objective vectors and frontFlag of appropriate format for paretofront()
    size_t len = archive->size;
    size_t nObjs = archive->numobj;
    bool *frontFlag = (bool*) malloc(len * sizeof(bool));
    double *obj = (double*) malloc(len * nObjs * sizeof(double));
    size_t i;
    for (i=0; i < len; i++) {
        size_t k;
        for (k=0; k < nObjs; k++) {
            obj[i + k*len] = archive->active[i]->obj[k];
        }
        frontFlag[i] = false;
    }
    
    // Call the non-dominated sorting engine
    mococo_pareto_front(frontFlag, obj, len, nObjs);
    
    // Mark non-dominated solutions and filter out dominated ones
    size_t s = 0;
    for (i=0; i < len; i++) {
        if (frontFlag[i] == true) {
            archive->active[i]->status = 1;
            if (i != s)
                archive->active[s] = archive->active[i];
            s++;
        } else {
            archive->active[i]->status = 0; // filter out dominated solutions
        }
    }
    archive->size = s;
    
    free(obj);
    free(frontFlag);
}


void mococo_pareto_front(bool *frontFlag, double *obj, unsigned nrow, unsigned ncol) {
    unsigned t, s, i, j, j1, j2;
    bool *checklist, colDominatedFlag;
    
    checklist = (bool*)malloc(nrow*sizeof(bool));
    
    for(t=0; t<nrow; t++)
        checklist[t] = true;
    for(s=0; s<nrow; s++) {
        t = s;
        if (!checklist[t])
            continue;
        checklist[t] = false;
        colDominatedFlag = true;
        for(i=t+1; i<nrow; i++) {
            if (!checklist[i])
                continue;
            checklist[i] = false;
            for (j=0,j1=i,j2=t; j<ncol; j++,j1+=nrow,j2+=nrow) {
                if (obj[j1] < obj[j2]) {
                    checklist[i] = true;
                    break;
                }
            }
            if (!checklist[i])
                continue;
            colDominatedFlag = false;
            for (j=0,j1=i,j2=t; j<ncol; j++,j1+=nrow,j2+=nrow) {
                if (obj[j1] > obj[j2]) {
                    colDominatedFlag = true;
                    break;
                }
            }
            if (!colDominatedFlag) { //swap active index continue checking
                frontFlag[t] = false;
                checklist[i] = false;
                colDominatedFlag = true;
                t = i;
            }
        }
        frontFlag[t] = colDominatedFlag;
        if (t>s) {
            for (i=s+1; i<t; i++) {
                if (!checklist[i])
                    continue;
                checklist[i] = false;
                for (j=0,j1=i,j2=t; j<ncol; j++,j1+=nrow,j2+=nrow) {
                    if (obj[j1] < obj[j2]) {
                        checklist[i] = true;
                        break;
                    }
                }
            }
        }
    }
    free(checklist); 
}

#ifdef __cplusplus
}
#endif
