/**
 * @file suite_biobj_ext.c
 * @brief Implementation of the extended biobjective bbob-biobj-ext suite containing 92 functions and 6 dimensions.
 *
 * The bbob-biobj-ext suite is created by combining two single-objective problems from the bbob suite.
 * The first 55 functions are the same as in the original bbob-biobj test suite to which 37 functions are added.
 * Those additional functions are constructed by combining all not yet contained in-group combinations (i,j) of
 * single-objective bbob functions i and j such that i<j (i.e. in particular not all combinations (i,i) are
 * included in this bbob-biobj-ext suite), with the exception of the Weierstrass function (f16) for which
 * the optimum is not unique and thus a nadir point is difficult to compute, see
 * http://numbbo.github.io/coco-doc/bbob-biobj/functions/ for details.
 *
 * @note See file suite_biobj_utilities.c for the implementation of the bi-objective problems and the handling
 * of new instances.
 *
 * @note This file is based on the original suite_biobj.c and extends it by 37 functions in 6 dimensions.
 */

#include "coco.h"
#include "mo_utilities.c"
#include "suite_biobj_utilities.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj-ext suite.
 */
static coco_suite_t *suite_biobj_ext_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  suite = coco_suite_allocate("bbob-biobj-ext", 55+37, 6, dimensions, "instances: 1-15");

  return suite;
}

/**
 * @brief Returns the problem from the bbob-biobj-ext suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_biobj_ext_get_problem(coco_suite_t *suite,
                                                   const size_t function_idx,
                                                   const size_t dimension_idx,
                                                   const size_t instance_idx) {
  
  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  suite_biobj_new_inst_t *data = (suite_biobj_new_inst_t *) suite->data;
  suite->data_free_function = suite_biobj_new_inst_free;
  
  problem = coco_get_biobj_problem(function, dimension, instance, data, suite->number_of_instances,
      suite->dimensions, suite->number_of_dimensions);
    
  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}

