/**
 * @file suite_rw_gan.c
 *
 * @brief Implementation of a single-objective suite containing real-world problems of unsupervised
 * learning of a Generative Adversarial Network (GAN) that understands the structure of Super Mario
 * Bros. levels. A bi-objective version can be found in the file suite_rw_gan_biobj.c (TODO)
 *
 * The suite contains 30 problems with dimensions 10, 20, 30, 40 and one instance (at the moment).
 */

#include "coco.h"
#include "rw_gan.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the suite_rw_gan suite.
 */
static coco_suite_t *suite_rw_gan_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 10 };
  suite = coco_suite_allocate("rw-gan", 13, 1, dimensions, "instances: 1");

  return suite;
}

/**
 * @brief Returns the problem from the rw-gan suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_rw_gan_get_problem(coco_suite_t *suite,
                                                const size_t function_idx,
                                                const size_t dimension_idx,
                                                const size_t instance_idx) {

  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = rw_gan_problem_allocate(function, dimension, instance);
  assert(problem != NULL);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
