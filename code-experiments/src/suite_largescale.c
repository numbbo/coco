/**
 * @file suite_largescale.c
 * @brief Implementation of the bbob large-scale suite containing 1 function in 6 large dimensions.
 */

#include "coco.h"

#include "f_ellipsoid.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob large-scale suite.
 */
static coco_suite_t *suite_largescale_initialize(void) {
  
  coco_suite_t *suite;
  /*const size_t dimensions[] = { 8, 16, 32, 64, 128, 256,512,1024};*/
  const size_t dimensions[] = { 40, 80, 160, 320, 640, 1280};
  suite = coco_suite_allocate("bbob-largescale", 1, 6, dimensions, "instances:1-15");
  return suite;
}

/**
 * @brief Creates and returns a large-scale problem without needing the actual large-scale suite.
 */
static coco_problem_t *coco_get_largescale_problem(const size_t function,
                                                   const size_t dimension,
                                                   const size_t instance) {
  coco_problem_t *problem = NULL;

  const char *problem_id_template = "bbob_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "BBOB suite problem f%lu instance %lu in %luD";

  const long rseed = (long) (function + 10000 * instance);
  /*const long rseed_3 = (long) (3 + 10000 * instance);*/
  /*const long rseed_17 = (long) (17 + 10000 * instance);*/
  if (function == 1) {
    problem = f_ellipsoid_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else {
    coco_error("coco_get_largescale_problem(): cannot retrieve problem f%lu instance %lu in %luD",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension);
    return NULL; /* Never reached */
  }

  return problem;
}

/**
 * @brief Returns the problem from the bbob large-scale suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_largescale_get_problem(coco_suite_t *suite,
                                                    const size_t function_idx,
                                                    const size_t dimension_idx,
                                                    const size_t instance_idx) {
  
  coco_problem_t *problem = NULL;
  
  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];
  
  problem = coco_get_largescale_problem(function, dimension, instance);
  
  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);
  
  return problem;
}
