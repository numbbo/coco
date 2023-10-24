/**
 * @file suite_biobj_mixint.c
 * @brief Implementation of a bi-objective mixed-integer suite. The functions are the same as those
 * in the bbob-biobj-ext suite with 92 functions, but the large-scale implementations of the
 * functions are used instead of the original ones for dimensions over 40.
 */

#include "coco.h"
#include "mo_utilities.c"
#include "suite_biobj_utilities.c"
#include "suite_largescale.c"
#include "transform_vars_discretize.c"
#include "transform_obj_scale.c"
#include "suite_bbob_mixint.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances,
                                         const int known_optima);
static void suite_biobj_new_inst_free(void *stuff);

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj-mixint suite.
 */
static coco_suite_t *suite_biobj_mixint_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 5, 10, 20, 40, 80, 160 };
  const size_t num_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);

  suite = coco_suite_allocate("bbob-biobj-mixint", 92, num_dimensions, dimensions, "instances: 1-15", 0);
  suite->data_free_function = suite_biobj_new_inst_free;

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj-mixint suites.
 *
 * @note The instances of the bi-objective suites generally do not changes with years.
 */
static const char *suite_biobj_mixint_get_instances_by_year(const int year) {

  (void) year; /* To get rid of compiler warnings */
  return "1-15";
}

/**
 * @brief Creates and returns a mixed-integer bi-objective bbob problem without needing the actual
 * bbob-mixint suite.
 *
 * The problem is constructed by first finding the underlying single-objective continuous problems,
 * then discretizing the problems, then scaling them to adjust their difficulty and finally stacking
 * them to get a bi-objective mixed-integer problem.
 *
 * @param function Function
 * @param dimension Dimension
 * @param instance Instance
 * @param coco_get_problem_function The function that is used to access the single-objective problem.
 * @param new_inst_data Structure containing information on new instance data.
 * @param num_new_instances The number of new instances.
 * @param dimensions An array of dimensions to take into account when creating new instances.
 * @param num_dimensions The number of dimensions to take into account when creating new instances.
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *coco_get_biobj_mixint_problem(const size_t function,
                                                     const size_t dimension,
                                                     const size_t instance,
                                                     const coco_get_problem_function_t coco_get_problem_function,
                                                     suite_biobj_new_inst_t **new_inst_data,
                                                     const size_t num_new_instances,
                                                     const size_t *dimensions,
                                                     const size_t num_dimensions) {

  coco_problem_t *problem_cont = NULL, *problem = NULL;
  coco_problem_t *problem1, *problem2;
  coco_problem_t *problem1_mixint, *problem2_mixint;
  coco_problem_t *problem1_cont, *problem2_cont;

  double *smallest_values_of_interest = coco_allocate_vector(dimension);
  double *largest_values_of_interest = coco_allocate_vector(dimension);

  size_t i, j;
  size_t num_integer = dimension;
  /* The cardinality of variables (0 = continuous variables should always come last) */
  const size_t variable_cardinality[] = { 2, 4, 8, 16, 0 };
  size_t function1, function2;

  if (dimension % 5 != 0)
    coco_error("coco_get_biobj_mixint_problem(): dimension %lu not supported for suite_bbob_mixint", dimension);

  /* First, find the underlying single-objective continuous problems */
  problem_cont = coco_get_biobj_problem(function, dimension, instance, coco_get_problem_function, new_inst_data,
      num_new_instances, dimensions, num_dimensions);
  assert(problem_cont != NULL);
  problem1_cont = ((coco_problem_stacked_data_t *) problem_cont->data)->problem1;
  problem2_cont = ((coco_problem_stacked_data_t *) problem_cont->data)->problem2;
  problem1 = coco_problem_duplicate(problem1_cont);
  problem2 = coco_problem_duplicate(problem2_cont);
  assert(problem1);
  assert(problem2);
  /* Copy also the data of the underlying problems and set all pointers in such a way that
   * problem_cont can be safely freed */
  problem1->data = problem1_cont->data;
  problem2->data = problem2_cont->data;
  problem1_cont->data = NULL;
  problem2_cont->data = NULL;
  problem1_cont->problem_free_function = NULL;
  problem2_cont->problem_free_function = NULL;
  coco_problem_free(problem_cont);

  /* Set the ROI of the outer problem according to the given cardinality of variables and the ROI of the
   * inner problems to [-4, 4] for variables that will be discretized */
  for (i = 0; i < dimension; i++) {
    j = i / (dimension / 5);
    if (variable_cardinality[j] == 0) {
      /* Continuous variables */
      /* Outer problem */
      smallest_values_of_interest[i] = -100;
      largest_values_of_interest[i] = 100;
      if (num_integer == dimension)
        num_integer = i;
    }
    else {
      /* Outer problem */
      smallest_values_of_interest[i] = 0;
      largest_values_of_interest[i] = (double)variable_cardinality[j] - 1;
      /* Inner problems */
      problem1->smallest_values_of_interest[i] = -4;
      problem1->largest_values_of_interest[i] = 4;
      problem2->smallest_values_of_interest[i] = -4;
      problem2->largest_values_of_interest[i] = 4;
    }
  }

  /* Second, discretize the single-objective problems */
  problem1_mixint = transform_vars_discretize(problem1, smallest_values_of_interest,
      largest_values_of_interest, num_integer);
  problem2_mixint = transform_vars_discretize(problem2, smallest_values_of_interest,
      largest_values_of_interest, num_integer);

  /* Third, scale the objective values */
  function1 = coco_problem_get_suite_dep_function(problem1);
  function2 = coco_problem_get_suite_dep_function(problem2);
  problem1_mixint = transform_obj_scale(problem1_mixint, suite_bbob_mixint_scaling_factors[function1 - 1]);
  problem2_mixint = transform_obj_scale(problem2_mixint, suite_bbob_mixint_scaling_factors[function2 - 1]);

  /* Fourth, combine the problems in a bi-objective mixed-integer problem */
  problem = coco_problem_stacked_allocate(problem1_mixint, problem2_mixint, smallest_values_of_interest,
      largest_values_of_interest);

  /* Use the standard stacked problem_id as problem_name and construct a new problem_id */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "bbob-biobj-mixint_f%03lu_i%02lu_d%03lu", (unsigned long) function,
      (unsigned long) instance, (unsigned long) dimension);

  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem1->problem_type, problem2->problem_type);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}
/**
 * @brief Returns the problem from the bbob-biobj-mixint suite that corresponds to the given parameters.
 *
 * Uses large-scale bbob functions if dimension is equal or larger than the hard-coded dim_large_scale
 * value (50).
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
  const size_t dim_large_scale = 50; /* Switch to large-scale functions for dimensions over 50 */

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  if (dimension < dim_large_scale)
    problem = coco_get_biobj_mixint_problem(function, dimension, instance, coco_get_bbob_problem,
        &new_inst_data, suite->number_of_instances, suite->dimensions, suite->number_of_dimensions);
  else
    problem = coco_get_biobj_mixint_problem(function, dimension, instance, coco_get_largescale_problem,
        &new_inst_data, suite->number_of_instances, suite->dimensions, suite->number_of_dimensions);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}

