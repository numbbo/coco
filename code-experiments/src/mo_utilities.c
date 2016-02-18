/**
 * @file mo_utilities.c
 * @brief Definitions of miscellaneous functions used for multi-objective problems.
 */

#include <stdlib.h>
#include <stdio.h>
#include "coco.h"

/**
 * @brief Checks the dominance relation in the unconstrained minimization case between objectives1 and
 * objectives2.
 *
 * @return
 *  1 if objectives1 dominates objectives2 <br>
 *  0 if objectives1 and objectives2 are non-dominated <br>
 * -1 if objectives2 dominates objectives1 <br>
 * -2 if objectives1 is identical to objectives2
 */
static int mo_get_dominance(const double *objectives1, const double *objectives2, const size_t num_obj) {
  /* TODO: Should we care about comparison precision? */
  size_t i;

  int flag1 = 0;
  int flag2 = 0;

  for (i = 0; i < num_obj; i++) {
    if (objectives1[i] < objectives2[i]) {
      flag1 = 1;
    } else if (objectives1[i] > objectives2[i]) {
      flag2 = 1;
    }
  }

  if (flag1 && !flag2) {
    return 1;
  } else if (!flag1 && flag2) {
    return -1;
  } else if (flag1 && flag2) {
    return 0;
  } else { /* (!flag1 && !flag2) */
    return -2;
  }
}

/**
 * @brief Computes and returns the Euclidean norm of two dim-dimensional points first and second.
 */
static double mo_get_norm(const double *first, const double *second, const size_t dim) {

  size_t i;
  double norm = 0;

  for (i = 0; i < dim; i++) {
    norm += pow(first[i] - second[i], 2);
  }

  return sqrt(norm);
}

/**
 * @brief Computes and returns the minimal normalized distance from the point y to the ROI.
 *
 * @note Assumes the point is dominated by the ideal point and the dimension equals 2.
 */
static double mo_get_distance_to_ROI(const double *y,
                                     const double *ideal,
                                     const double *nadir,
                                     const size_t dimension) {

  double distance = 0;

  assert(dimension == 2);
  assert(mo_get_dominance(ideal, y, 2) == 1);

  /* y is weakly dominated by the nadir point */
  if (mo_get_dominance(y, nadir, 2) <= -1) {
    distance = mo_get_norm(y, nadir, 2);
  }
  else if (y[0] < nadir[0])
    distance = y[1] - nadir[1];
  else if (y[1] < nadir[1])
    distance = y[0] - nadir[0];
  else {
    coco_error("mo_get_distance_to_ROI(): unexpected exception");
    return 0; /* Never reached */
  }

  return distance / ((nadir[1] - ideal[1]) * (nadir[0] - ideal[0]));

}
