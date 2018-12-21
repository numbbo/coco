/**
 * @file suite_rw_gan_mario.c
 *
 * @brief Implementation of a single-objective suite containing real-world problems of unsupervised
 * learning of a Generative Adversarial Network (GAN) that understands the structure of Super Mario
 * Bros. levels. A bi-objective version can be found in the file suite_rw_gan_mario_biobj.c
 *
 * The suite contains 84 problems with dimensions 10, 20, 30, 40 and 15 instances.
 */

#include "coco.h"
#include "rw_gan_mario.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the rw-gan-mario suite.
 */
static coco_suite_t *suite_rw_gan_mario_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 10, 20, 30, 40 };

  suite = coco_suite_allocate("rw-gan-mario", 84, 4, dimensions, "instances: 1-7");

  return suite;
}

/**
 * @brief Returns the problem from the rw-gan-mario suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_rw_gan_mario_get_problem(coco_suite_t *suite,
                                                      const size_t function_idx,
                                                      const size_t dimension_idx,
                                                      const size_t instance_idx) {

  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = rw_gan_mario_problem_allocate(suite->suite_name, 1, function, dimension, instance);
  assert(problem != NULL);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
