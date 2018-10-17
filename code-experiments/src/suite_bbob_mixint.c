/**
 * @file suite_bbob_mixint.c
 * @brief Implementation of a suite with mixed-integer bbob problems (based on the 24 bbob functions,
 * but using their large-scale implementations instead of the original ones).
 */

#include "coco.h"
#include "suite_bbob.c"
#include "transform_vars_discretize.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob-mixint suite.
 */
static coco_suite_t *suite_bbob_mixint_initialize(const char *suite_name) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 5, 10, 20, 40, 80, 160 };
  /* TODO: Use also dimensions 80 and 160 (change the 4 below into a 6) */
  suite = coco_suite_allocate(suite_name, 24, 4, dimensions, "year: 2019");

  return suite;
}

/**
 * @brief Returns the problem from the bbob-mixint suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_bbob_mixint_get_problem(coco_suite_t *suite,
                                                     const size_t function_idx,
                                                     const size_t dimension_idx,
                                                     const size_t instance_idx) {

  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  /* The cardinality of variables (0 = continuous variables should always come last) */
  const size_t variable_cardinality_1[] = { 2, 4, 8, 16, 0 };
  const size_t variable_cardinality_2[] = { 2, 6, 18, 0, 0 };

  double *smallest_values_of_interest = coco_allocate_vector(dimension);
  double *largest_values_of_interest = coco_allocate_vector(dimension);
  char *inner_problem_id;

  size_t i, j;
  size_t cardinality = 0;
  size_t num_integer = dimension;
  if (dimension % 5 != 0)
    coco_error("suite_bbob_mixint_get_problem(): dimension %lu not supported for suite_bbob_mixint", dimension_idx);

  /* Sets the ROI according to the given cardinality of variables */
  for (i = 0; i < dimension; i++) {
    j = i / (dimension / 5);
    if (strcmp(suite->suite_name, "bbob-mixint-1") == 0)
      cardinality = variable_cardinality_1[j];
    else
      cardinality = variable_cardinality_2[j];
    if (cardinality == 0) {
      smallest_values_of_interest[i] = -5;
      largest_values_of_interest[i] = 5;
      if (num_integer == dimension)
        num_integer = i;
    }
    else {
      smallest_values_of_interest[i] = 0;
      largest_values_of_interest[i] = (double)cardinality - 1;
    }
  }

  /* TODO: Use the large scale versions of the bbob problems */
  problem = coco_get_bbob_problem(function, dimension, instance);
  assert(problem != NULL);
  inner_problem_id = problem->problem_id;

  problem = transform_vars_discretize(problem,
      smallest_values_of_interest,
      largest_values_of_interest,
      num_integer);

  coco_problem_set_id(problem, "bbob-mixint_f%03lu_i%02lu_d%02lu", function, instance, dimension);
  coco_problem_set_name(problem, "mixint(%s)", inner_problem_id);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}
