/**
 * @file suite_test_mixint.c
 * @brief Test implementation of a suite with mixed-integer problems.
 */

#include "coco.h"

#include "mi_f_sphere.c"
#include "suite_bbob.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the test-mixint suite.
 */
static coco_suite_t *suite_test_mixint_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  suite = coco_suite_allocate("test-mixint", 2, 6, dimensions, "year: 2018");

  return suite;
}

/**
 * @brief Creates and returns a test-mixint problem.
 */
static coco_problem_t *coco_get_test_mixint_problem(const size_t function,
                                                    const size_t dimension,
                                                    const size_t instance) {
  coco_problem_t *problem = NULL;
  size_t i;

  const char *problem_id_template = "test-mixint_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "test-mixint suite problem f%lu instance %lu in %luD";

  const long rseed = (long) (function + 10000 * instance);

  double *smallest_values_of_interest = coco_allocate_vector(dimension);
  double *largest_values_of_interest = coco_allocate_vector(dimension);
  int *are_variables_integer = coco_allocate_vector_int(dimension);

  if (function == 1) {
  	for (i = 0; i < dimension; i++) {
      smallest_values_of_interest[i] = -5;
      largest_values_of_interest[i] = 5;
  		if (i < dimension / 2)
  			are_variables_integer[i] = 0;
  		else
  			are_variables_integer[i] = 1;
  	}
    problem = mi_f_sphere_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template, smallest_values_of_interest,
        largest_values_of_interest, are_variables_integer);
  } else if (function == 2) {
  	for (i = 0; i < dimension; i++) {
      smallest_values_of_interest[i] = -5;
      largest_values_of_interest[i] = 5;
  		if (i % 3 == 0)
  			are_variables_integer[i] = 0;
  		else
  			are_variables_integer[i] = 1;
  	}
    problem = mi_f_sphere_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template, smallest_values_of_interest,
        largest_values_of_interest, are_variables_integer);
  } else {
    coco_error("coco_get_test_mixint_problem(): cannot retrieve problem f%lu instance %lu in %luD",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension);
    return NULL; /* Never reached */
  }

  coco_free_memory(are_variables_integer);
  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}

/**
 * @brief Returns the problem from the bbob-discrete suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_test_mixint_get_problem(coco_suite_t *suite,
		                                               const size_t function_idx,
		                                               const size_t dimension_idx,
		                                               const size_t instance_idx) {

  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = coco_get_test_mixint_problem(function, dimension, instance);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
