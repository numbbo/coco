/**
 * @file rw_gan.c
 *
 * @brief Implementation of the real-world problems of unsupervised learning of a Generative
 * Adversarial Network (GAN) that understands the structure of Super Mario Bros. levels.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Implements the sphere function without connections to any COCO structures.
 */
static double rw_gan_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
    
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

  /* TODO external evaluation */
  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    result += x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void rw_gan_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = rw_gan_raw(x, problem->number_of_variables);
}

/**
 * @brief Creates a rw-gan problem.
 */
static coco_problem_t *rw_gan_problem_allocate(const size_t function,
                                               const size_t dimension,
                                               const size_t instance) {

  coco_problem_t *problem = coco_problem_allocate(dimension, 1, 0);
  size_t i;

  for (i = 0; i < dimension; ++i) {
    problem->smallest_values_of_interest[i] = 0; /* TODO */
    problem->largest_values_of_interest[i] = 1;  /* TODO */
  }
  problem->are_variables_integer = NULL;
  problem->evaluate_function = rw_gan_evaluate;

  coco_problem_set_id(problem, "rw-gan_f%03lu_i%02lu_d%02lu", function, instance, dimension);
  coco_problem_set_name(problem, "real-world GAN problem f%lu instance %lu in %luD", function, instance, dimension);
  coco_problem_set_type(problem, "rw-gan-mario-single");

  /* The best parameter and value are not known */
  if (problem->best_parameter != NULL) {
    coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
  }
  if (problem->best_value != NULL) {
    coco_free_memory(problem->best_value);
    problem->best_value = NULL;
  }

  return problem;
}
