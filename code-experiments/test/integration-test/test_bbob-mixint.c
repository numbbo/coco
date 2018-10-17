/**
 * Tests that some of the bbob-mixint problems with dimension 2 have been correctly discretized.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"
#include "coco.c"

/* Tests whether the discretization of a single-objective suite was implemented correctly by
 * checking if the optima of the continuous and mixint problems match. This is not a comprehensive
 * test, because it disregards problems with dimension != 5 and those that have more than 20,000
 * discrete points and checks only the optima (even if the optima match, the transformation could
 * be wrong).
 */
void check_discretization_single(char *suite_name, char *suite_options) {

  coco_suite_t *suite;
  coco_problem_t *problem_cont, *problem_disc;
  size_t j, k1, k2, k3, k4;
  double *xopt_cont, *fopt_cont, fopt_disc;
  double *x, *f;
  size_t num_points[5];
  size_t all_points = 1;
  size_t matching_roi, counter = 0;
  const size_t MAX_NUM = 20000;

  x = coco_allocate_vector(5);
  f = coco_allocate_vector(1);

  suite = coco_suite(suite_name, NULL, suite_options);

  while ((problem_disc = coco_suite_get_next_problem(suite, NULL)) != NULL) {

    /* Skip problems of dimension != 5, those with more than MAX_NUM discrete points or those that
     * whose continuous ROIs do not match */
    if (problem_disc->number_of_variables != 5)
      continue;

    /* Compute the number of points and check that the continuous ROIs match */
    matching_roi = 1;
    for (j = 0; j < 5; j++) {
      if (j < problem_disc->number_of_integer_variables) {
        num_points[j] = (unsigned long)(problem_disc->largest_values_of_interest[j] - problem_disc->smallest_values_of_interest[j] + 1);
        all_points *= num_points[j];
      }
      else {
        num_points[j] = 1;
        matching_roi = matching_roi && coco_double_almost_equal(problem_disc->largest_values_of_interest[j],
          problem_disc->largest_values_of_interest[j], 1e-10);
      }
    }
    if ((all_points > MAX_NUM) || (!matching_roi))
      continue;

    /* Compute and compare the optima */
    problem_cont = coco_problem_transformed_get_inner_problem(problem_disc);
    xopt_cont = problem_cont->best_parameter;
    fopt_cont = problem_cont->best_value;
    fopt_disc = DBL_MAX;
    for (j = 0; j < num_points[0]; j++) {
      x[0] = ((num_points[0] == 1) ? xopt_cont[0] : (double)j);
      for (k1 = 0; k1 < num_points[1]; k1++) {
        x[1] = ((num_points[1] == 1) ? xopt_cont[1] : (double)k1);
        for (k2 = 0; k2 < num_points[2]; k2++) {
          x[2] = ((num_points[2] == 1) ? xopt_cont[2] : (double)k2);
          for (k3 = 0; k3 < num_points[3]; k3++) {
            x[3] = ((num_points[3] == 1) ? xopt_cont[3] : (double)k3);
            for (k4 = 0; k4 < num_points[4]; k4++) {
              x[4] = ((num_points[4] == 1) ? xopt_cont[4] : (double)k4);
              coco_evaluate_function(problem_disc, x, f);
              if (f[0] < fopt_disc) {
                fopt_disc = f[0];
              }
            }
          }
        }
      }
    }
    if (!coco_double_almost_equal(fopt_cont[0], fopt_disc, 1e-10)) {
      coco_suite_free(suite);
      coco_free_memory(x);
      coco_free_memory(f);
      coco_error("The optima of the original and discretized problem %lu do not match (%f, %f)!",
          (unsigned long)coco_problem_get_suite_dep_index(problem_disc), fopt_cont[0], fopt_disc);
    }
    counter++;
  }

  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(f);

  if (counter == 0)
    coco_error("No tests of %s were performed!", suite_name);
  else
    printf("Performed %lu tests on %s\n", (unsigned long)counter, suite_name);
  printf("DONE!\n");
  fflush(stdout);
}

/* Tests whether the discretization of a bi-objective suite was implemented correctly by checking
 * if the extreme solutions of the continuous and mixint problems match. This is not a comprehensive
 * test, because it disregards problems with dimension != 5 and those that have more than 20,000
 * discrete points and checks only the extreme solutions (even if they match, the transformation
 * could be wrong).
 *
 * Note also that this test assumes that the discretization was the last transformation, i.e. that
 * the discretized problems are constructed with the following 'wrapping' of problems:
 * discretize(biobjective(problem1, problem2))
 */
void check_discretization_bi(char *suite_name, char *suite_options) {

  coco_suite_t *suite;
  coco_problem_t *problem_cont, *problem_disc;
  coco_problem_t *problem1_cont, *problem2_cont;
  size_t j, k1, k2, k3, k4;
  double *x_ext_cont;
  double f_ext1_cont, f_ext2_cont;
  double f_ext1_disc, f_ext2_disc;
  double *x, *f;
  size_t num_points[5];
  size_t all_points = 1;
  size_t matching_roi, counter = 0;
  const size_t MAX_NUM = 20000;

  x = coco_allocate_vector(5);
  f = coco_allocate_vector(2);

  suite = coco_suite(suite_name, NULL, suite_options);

  while ((problem_disc = coco_suite_get_next_problem(suite, NULL)) != NULL) {

    /* Skip problems of dimension != 5, those with more than MAX_NUM discrete points or those that
     * whose continuous ROIs do not match */
    if (problem_disc->number_of_variables != 5)
      continue;

    /* Compute the number of points and check that the continuous ROIs match */
    matching_roi = 1;
    for (j = 0; j < 5; j++) {
      if (j < problem_disc->number_of_integer_variables) {
        num_points[j] = (unsigned long)(problem_disc->largest_values_of_interest[j] - problem_disc->smallest_values_of_interest[j] + 1);
        all_points *= num_points[j];
      }
      else {
        num_points[j] = 1;
        matching_roi = matching_roi && coco_double_almost_equal(problem_disc->largest_values_of_interest[j],
          problem_disc->largest_values_of_interest[j], 1e-10);
      }
    }
    if ((all_points > MAX_NUM) || (!matching_roi))
      continue;

    /* Check whether the two extreme points are equal */
    problem_cont = coco_problem_transformed_get_inner_problem(problem_disc);
    f_ext1_cont = problem_cont->best_value[0];
    f_ext2_cont = problem_cont->best_value[1];
    problem1_cont = ((coco_problem_stacked_data_t *) problem_cont->data)->problem1;
    problem2_cont = ((coco_problem_stacked_data_t *) problem_cont->data)->problem2;

    /* The first extreme point */
    x_ext_cont = problem1_cont->best_parameter;
    f_ext1_disc = DBL_MAX;
    f_ext2_disc = DBL_MAX;
    for (j = 0; j < num_points[0]; j++) {
      x[0] = ((num_points[0] == 1) ? x_ext_cont[0] : (double)j);
      for (k1 = 0; k1 < num_points[1]; k1++) {
        x[1] = ((num_points[1] == 1) ? x_ext_cont[1] : (double)k1);
        for (k2 = 0; k2 < num_points[2]; k2++) {
          x[2] = ((num_points[2] == 1) ? x_ext_cont[2] : (double)k2);
          for (k3 = 0; k3 < num_points[3]; k3++) {
            x[3] = ((num_points[3] == 1) ? x_ext_cont[3] : (double)k3);
            for (k4 = 0; k4 < num_points[4]; k4++) {
              x[4] = ((num_points[4] == 1) ? x_ext_cont[4] : (double)k4);
              coco_evaluate_function(problem_disc, x, f);
              if (f[0] < f_ext1_disc) {
                f_ext1_disc = f[0];
                f_ext2_disc = f[1];
              }
            }
          }
        }
      }
    }
    if (!coco_double_almost_equal(f_ext1_cont, f_ext1_disc, 1e-10) ||
        !coco_double_almost_equal(f_ext2_cont, f_ext2_disc, 1e-10)) {
      coco_suite_free(suite);
      coco_free_memory(x);
      coco_free_memory(f);
      coco_error("The extreme point of the original and discretized problem %lu do not match (%f, %f) != (%f, %f)!",
          (unsigned long)coco_problem_get_suite_dep_index(problem_disc), f_ext1_cont, f_ext2_cont,
          f_ext1_disc, f_ext2_disc);
    }

    /* The second extreme point */
    x_ext_cont = problem2_cont->best_parameter;
    f_ext1_disc = DBL_MAX;
    f_ext2_disc = DBL_MAX;
    for (j = 0; j < num_points[0]; j++) {
      x[0] = ((num_points[0] == 1) ? x_ext_cont[0] : (double)j);
      for (k1 = 0; k1 < num_points[1]; k1++) {
        x[1] = ((num_points[1] == 1) ? x_ext_cont[1] : (double)k1);
        for (k2 = 0; k2 < num_points[2]; k2++) {
          x[2] = ((num_points[2] == 1) ? x_ext_cont[2] : (double)k2);
          for (k3 = 0; k3 < num_points[3]; k3++) {
            x[3] = ((num_points[3] == 1) ? x_ext_cont[3] : (double)k3);
            for (k4 = 0; k4 < num_points[4]; k4++) {
              x[4] = ((num_points[4] == 1) ? x_ext_cont[4] : (double)k4);
              coco_evaluate_function(problem_disc, x, f);
              if (f[1] < f_ext2_disc) {
                f_ext2_disc = f[1];
                f_ext1_disc = f[0];
              }
            }
          }
        }
      }
    }
    if (!coco_double_almost_equal(f_ext1_cont, f_ext1_disc, 1e-10) ||
        !coco_double_almost_equal(f_ext2_cont, f_ext2_disc, 1e-10)) {
      coco_suite_free(suite);
      coco_free_memory(x);
      coco_free_memory(f);
      coco_error("The extreme point of the original and discretized problem %lu do not match (%f, %f) != (%f, %f)!",
          (unsigned long)coco_problem_get_suite_dep_index(problem_disc), f_ext1_cont, f_ext2_cont,
          f_ext1_disc, f_ext2_disc);
    }
    counter++;
  }

  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(f);

  if (counter == 0)
    coco_error("No tests of %s were performed!", suite_name);
  else
    printf("Performed %lu tests on %s\n", (unsigned long)counter, suite_name);
  printf("DONE!\n");
  fflush(stdout);
}

int main(void)  {

  check_discretization_bi("bbob-biobj-mixint", "dimensions: 5");
  check_discretization_single("bbob-mixint-1", "dimensions: 5");

  coco_remove_directory("exdata");
  return 0;
}
