/**
 * @file suite_bbob_mixint.c
 * @brief Test implementation of a suite with mixed-integer BBOB problems.
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
static coco_suite_t *suite_bbob_mixint_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  suite = coco_suite_allocate("bbob-mixint", 24, 6, dimensions, "year: 2018");

  return suite;
}

/**
 * @brief Sets the ROI (smallest_values_of_interest, largest_values_of_interest)
 * in a deterministic way depending on the values of dimension and instance.
 * This is just a temporary implementation for testing purposes.
 */
static void suite_bbob_mixint_set_ROI(const size_t dimension, const size_t instance,
    double *smallest_values_of_interest, double *largest_values_of_interest) {

  size_t i;
  double num;

  /* The last variable is continuous */
  smallest_values_of_interest[dimension - 1] = -5;
  largest_values_of_interest[dimension - 1] = 5;

  for (i = 0; i < dimension - 1; i++) {
    smallest_values_of_interest[i] = 0;
    num = (double)((dimension + instance + i) % 5);
    largest_values_of_interest[i] = coco_double_round(pow(10, num));
  }
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

  double *smallest_values_of_interest = coco_allocate_vector(dimension);
  double *largest_values_of_interest = coco_allocate_vector(dimension);

  suite_bbob_mixint_set_ROI(dimension, instance, smallest_values_of_interest,
      largest_values_of_interest);

  problem = coco_get_bbob_problem(function, dimension, instance);
  assert(problem != NULL);

  problem = transform_vars_discretize(problem,
      smallest_values_of_interest,
      largest_values_of_interest,
      dimension-1);

  coco_problem_set_id(problem, "bbob-mixint_f%03lu_i%02lu_d%02lu", function, instance, dimension);
  coco_problem_set_name(problem, "mixed-integer bbob suite problem f%lu instance %lu in %luD", function, instance, dimension);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}
