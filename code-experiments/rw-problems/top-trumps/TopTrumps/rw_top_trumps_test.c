/*
 * rw_top_trump_test.c
 *
 *  Created on: 29. jun. 2018
 *      Author: Tea Tusar
 */
#include <stdio.h>
#include "rw_top_trumps.h"

int main(void) {

  size_t function = 2;
  size_t instance = 1;
  size_t size_x = 88;
  size_t size_y = 1;
  size_t i;
  double *x;
  double *y;

  top_trumps_test();

  x = (double *) malloc(size_x * sizeof(double));
  y = (double *) malloc(size_y * sizeof(double));

  for (i = 0; i < size_x; i++)
    x[i] = i;

  top_trumps_evaluate(function, instance, size_x, x, size_y, y);

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

