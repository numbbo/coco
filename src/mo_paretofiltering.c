/* paretofront returns the logical Pareto membership of a set of points
   synopsis:  frontFlag = paretofront(objMat)
   Created by Yi Cao: y.cao@cranfield.ac.uk
   for compiling type:
   mex paretofront.c

   adapted to fit ANSI C89 standard by the BBOBies
 * no '//' comments
 * no use of C++ bool datatype anymore
 */

#include <stdio.h>
#include <stdlib.h>  /* memory, e.g. malloc */
#include <math.h>
#include "mo_recorder.h"

void mococo_pareto_front(int *frontFlag, double *obj, size_t nrow, size_t ncol);


void mococo_pareto_filtering(struct mococo_solutions_archive *archive) {
  /* Create the objective vectors and frontFlag of appropriate format for paretofront() */
  size_t len = archive->size;
  size_t nObjs = archive->numobj;
  int *frontFlag = (int*) malloc(len * sizeof(int));
  double *obj = (double*) malloc(len * nObjs * sizeof(double));
  size_t i;
  size_t s;

  for (i=0; i < len; i++) {
    size_t k;
    for (k=0; k < nObjs; k++) {
      obj[i + k*len] = archive->active[i]->obj[k];
    }
    frontFlag[i] = 0; /* set to false */
  }

  /* Call the non-dominated sorting engine */
  mococo_pareto_front(frontFlag, obj, len, nObjs);

  /* Mark non-dominated solutions and filter out dominated ones */
  s = 0;
  for (i=0; i < len; i++) {
    if (frontFlag[i]) {
      archive->active[i]->status = 1;
      if (i != s)
        archive->active[s] = archive->active[i];
      s++;
    } else {
      archive->active[i]->status = 0; /* filter out dominated solutions */
    }
  }
  archive->size = s;

  free(obj);
  free(frontFlag);
}


void mococo_pareto_front(int *frontFlag, double *obj, size_t nrow, size_t ncol) {
  size_t t, s, i, j, j1, j2;
  int *checklist, colDominatedFlag;

  checklist = (int*) malloc(nrow * sizeof(int));

  for (t = 0; t < nrow; t++)
    checklist[t] = 1; /* set to true */
  for (s = 0; s < nrow; s++) {
    t = s;
    if (!checklist[t])
      continue;
    checklist[t] = 0; /* set to false */
    colDominatedFlag = 1; /* set to true */
    for (i = t + 1; i < nrow; i++) {
      if (!checklist[i])
        continue;
      checklist[i] = 0;
      for (j = 0, j1 = i, j2 = t; j < ncol; j++, j1 += nrow, j2 += nrow) {
        if (obj[j1] < obj[j2]) {
          checklist[i] = 1; /* set to true */
          break;
        }
      }
      if (!checklist[i])
        continue;
      colDominatedFlag = 0; /* set to false */
      for (j = 0, j1 = i, j2 = t; j < ncol; j++, j1 += nrow, j2 += nrow) {
        if (obj[j1] > obj[j2]) {
          colDominatedFlag = 1; /* set to true */
          break;
        }
      }
      if (!colDominatedFlag) { /* swap active index continue checking */
        frontFlag[t] = 0; /* set to false */
        checklist[i] = 0; /* set to false */
        colDominatedFlag = 1; /**/
        t = i;
      }
    }
    frontFlag[t] = colDominatedFlag;
    if (t > s) {
      for (i = s + 1; i < t; i++) {
        if (!checklist[i])
          continue;
        checklist[i] = 0; /* set to false */
        for (j = 0, j1 = i, j2 = t; j < ncol; j++, j1 += nrow, j2 +=
            nrow) {
          if (obj[j1] < obj[j2]) {
            checklist[i] = 1; /* set to true */
            break;
          }
        }
      }
    }
  }
  free(checklist);
}
