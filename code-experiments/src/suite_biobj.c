/**
 * @file suite_biobj.c
 * @brief Implementation of two bi-objective suites created by combining two single-objective problems
 * from the bbob suite:
 * - bbob-biobj contains 55 functions and 6 dimensions
 * - bbob-biobj-ext contains 55 + 37 functions and 6 dimensions
 *
 * The 55 functions of the bbob-biobj suite are created by combining any two single-objective bbob functions
 * i,j (where i<j) from a subset of 10 functions.
 *
 * The first 55 functions of the bbob-biobj-ext suite are the same as in the original bbob-biobj test suite
 * to which 37 functions are added. Those additional functions are constructed by combining all not yet
 * contained in-group combinations (i,j) of single-objective bbob functions i and j such that i<j (i.e. in
 * particular not all combinations (i,i) are included in this bbob-biobj-ext suite), with the exception of
 * the Weierstrass function (f16) for which the optimum is not unique and thus a nadir point is difficult
 * to compute, see http://numbbo.github.io/coco-doc/bbob-biobj/functions/ for details.
 *
 * @note See file suite_biobj_utilities.c for the implementation of the bi-objective problems and the handling
 * of new instances.
 */

#include "coco.h"
#include "mo_utilities.c"
#include "suite_biobj_utilities.c"
#include "suite_bbob.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances,
                                         const int known_optima);

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj suites.
 */
static coco_suite_t *suite_biobj_initialize(const char *suite_name) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };
  const size_t num_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);

  if (strcmp(suite_name, "bbob-biobj") == 0) {
    suite = coco_suite_allocate("bbob-biobj", 55, num_dimensions, dimensions, "instances: 1-15", 1);
  } else if (strcmp(suite_name, "bbob-biobj-ext") == 0) {
    suite = coco_suite_allocate("bbob-biobj-ext", 55+37, num_dimensions, dimensions, "instances: 1-15", 1);
  } else {
    coco_error("suite_biobj_initialize(): unknown problem suite");
    return NULL;
  }

  suite->data_free_function = suite_biobj_new_inst_free;

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj suites.
 *
 * @note The instances of the bi-objective suites generally do not changes with years.
 */
static const char *suite_biobj_get_instances_by_year(const int year) {

  if ((year == 2016) || (year == 0000)) { /* test case */
    return "1-10";
  }
  else
    return "1-15";
}

/**
 * @brief Returns the problem from the bbob-biobj suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_biobj_get_problem(coco_suite_t *suite,
                                               const size_t function_idx,
                                               const size_t dimension_idx,
                                               const size_t instance_idx) {

  coco_problem_t *problem = NULL;
  suite_biobj_new_inst_t *new_inst_data = (suite_biobj_new_inst_t *) suite->data;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = coco_get_biobj_problem(function, dimension, instance, coco_get_bbob_problem, &new_inst_data,
      suite->number_of_instances, suite->dimensions, suite->number_of_dimensions);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);
  problem->is_opt_known = 1;

  return problem;
}

