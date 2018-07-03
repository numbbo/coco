/*
 * rw_top_trump_test.c
 *
 *  Created on: 29. jun. 2018
 *      Author: Tea Tusar
 */
#include <stdio.h>
#include "rw_top_trumps.h"

int main(void) {

  size_t function = 1;
  size_t instance = 2;
  size_t size_x = 88;
  size_t size_y = 1;
  size_t i;
  double *x;
  double *y;
  double * lb;
  double * ub;

  top_trumps_test();

  x = (double *) malloc(size_x * sizeof(double));
  y = (double *) malloc(size_y * sizeof(double));

  for (i = 0; i < size_x; i++)
    x[i] = 12;

  top_trumps_evaluate(function, instance, size_x, x, size_y, y);

  printf("function = %lu, instance = %lu, objectives = %lu\n",
      (unsigned long)function, (unsigned long)instance, (unsigned long)size_y);
  printf("x = ");
  for (i = 0; i < size_x; i++)
    printf("%.0f\t", x[i]);
  printf("\n");
  printf("y = ");
  for (i = 0; i < size_y; i++)
    printf("%.4f\n", y[i]);
  printf("\n");
  fflush(stdout);

  function=2;
  top_trumps_evaluate(function, instance, size_x, x, size_y, y);

  printf("function = %lu, instance = %lu, objectives = %lu\n",
      (unsigned long)function, (unsigned long)instance, (unsigned long)size_y);
  printf("y = ");
  for (i = 0; i < size_y; i++)
    printf("%.4f\n", y[i]);
  printf("\n");
  fflush(stdout);

  function=3;
  top_trumps_evaluate(function, instance, size_x, x, size_y, y);

  printf("function = %lu, instance = %lu, objectives = %lu\n",
      (unsigned long)function, (unsigned long)instance, (unsigned long)size_y);
  printf("y = ");
  for (i = 0; i < size_y; i++)
    printf("%.4f\n", y[i]);
  printf("\n");
  fflush(stdout);

  function=4;
  top_trumps_evaluate(function, instance, size_x, x, size_y, y);

  printf("function = %lu, instance = %lu, objectives = %lu\n",
      (unsigned long)function, (unsigned long)instance, (unsigned long)size_y);
  printf("y = ");
  for (i = 0; i < size_y; i++)
    printf("%.4f\n", y[i]);
  printf("\n");
  fflush(stdout);

  function=5;
  top_trumps_evaluate(function, instance, size_x, x, size_y, y);

  printf("function = %lu, instance = %lu, objectives = %lu\n",
      (unsigned long)function, (unsigned long)instance, (unsigned long)size_y);
  printf("y = ");
  for (i = 0; i < size_y; i++)
    printf("%.4f\n", y[i]);
  printf("\n");
  fflush(stdout);

  free(y);

  function = 6;
  size_y = 2;
  y = (double *) malloc(size_y * sizeof(double));
  

  lb = (double *) malloc(size_x* sizeof(double));
  ub= (double *) malloc(size_x* sizeof(double));
  top_trumps_bounds(function, instance, size_x, lb, ub);
  
  
  top_trumps_evaluate(function, instance, size_x, x, size_y, y);

  printf("function = %lu, instance = %lu, objectives = %lu\n",
      (unsigned long)function, (unsigned long)instance, (unsigned long)size_y);
  printf("x = ");
  for (i = 0; i < size_x; i++)
    printf("%.0f\t", x[i]);
  printf("\n");
  printf("y = ");
  for (i = 0; i < size_y; i++)
    printf("%.4f\n", y[i]);
  printf("\n");
  fflush(stdout);

  free(x);
  free(y);

  return 0;
}

