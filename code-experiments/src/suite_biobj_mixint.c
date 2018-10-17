/**
 * @file suite_biobj_mixint.c
 * @brief Implementation of a bi-objective mixed-integer suite. The functions are the same as those
 * in the bbob-biobj-ext suite with 92 functions, but the large-scale implementations of the
 * functions are used instead of the original ones.
 */

#include "coco.h"
#include "mo_utilities.c"
#include "suite_biobj_utilities.c"
#include "transform_vars_discretize.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj-mixint suite.
 */
static coco_suite_t *suite_biobj_mixint_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 5, 10, 20, 40, 80, 160 };

  /* TODO: Use also dimensions 80 and 160 (change the 4 below into a 6) */
  suite = coco_suite_allocate("bbob-biobj-mixint", 92, 4, dimensions, "instances: 1-15");
  suite->data_free_function = suite_biobj_new_inst_free;

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj-mixint suites.
 *
 * @note The instances of the bi-objective suites generally do not changes with years.
 */
static const char *suite_biobj_mixint_get_instances_by_year(const int year) {

  if ((year == 2016) || (year == 0000)) { /* test case */
    return "1-10";
  }
  else
    return "1-15";
}

/**
 * @brief Returns the problem from the bbob-biobj-mixint suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_biobj_mixint_get_problem(coco_suite_t *suite,
                                                      const size_t function_idx,
                                                      const size_t dimension_idx,
                                                      const size_t instance_idx) {

  coco_problem_t *problem = NULL;
  suite_biobj_new_inst_t *new_inst_data = (suite_biobj_new_inst_t *) suite->data;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  /* TODO: use the correct cardinality!
   * The cardinality of variables (0 = continuous variables should always come last) */
  const size_t variable_cardinality[] = { 2, 4, 8, 16, 0 };

  double *smallest_values_of_interest = coco_allocate_vector(dimension);
  double *largest_values_of_interest = coco_allocate_vector(dimension);
  char *inner_problem_name;

  size_t i, j;
  size_t num_integer = dimension;
  if (dimension % 5 != 0)
    coco_error("suite_bbob_mixint_get_problem(): dimension %lu not supported for suite_bbob_mixint", dimension_idx);

  /* Sets the ROI according to the given cardinality of variables */
  for (i = 0; i < dimension; i++) {
    /* Sets the largest_values_of_interest and smallest_values_of_interest outside the if clause to show
     * valgrind that they are indeed always initialized */
    smallest_values_of_interest[i] = -5;
    largest_values_of_interest[i] = 5;
    j = i / (dimension / 5);
    if (variable_cardinality[j] == 0) {
      if (num_integer == dimension)
        num_integer = i;
    }
    else {
      smallest_values_of_interest[i] = 0;
      largest_values_of_interest[i] = (double)variable_cardinality[j] - 1;
    }
  }

  /* TODO: Use the large scale versions of the bbob-biobj problems */
  problem = coco_get_biobj_problem(function, dimension, instance, &new_inst_data,
      suite->number_of_instances, suite->dimensions, suite->number_of_dimensions);
  assert(problem != NULL);
  inner_problem_name = problem->problem_name;

  problem = transform_vars_discretize(problem,
      smallest_values_of_interest,
      largest_values_of_interest,
      num_integer);

  coco_problem_set_id(problem, "bbob-biobj-mixint_f%03lu_i%02lu_d%02lu", (unsigned long) function,
      (unsigned long) instance, (unsigned long) dimension);
  coco_problem_set_name(problem, "mixint(%s)", inner_problem_name);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}

