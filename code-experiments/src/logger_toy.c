/**
 * @file logger_toy.c
 * @brief Implementation of the toy logger.
 *
 * Logs the evaluation number and function value each time a target has been hit.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_string.c"
#include "observer_toy.c"

/**
 * @brief The toy logger data type.
 */
typedef struct {
  coco_observer_t *observer;   /**< @brief Pointer to the COCO observer. */
  size_t next_target;          /**< @brief The next target. */
  long number_of_evaluations;  /**< @brief The number of evaluations performed so far. */
} logger_toy_data_t;

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information based on the
 * targets that have been hit.
 */
static void logger_toy_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_toy_data_t *logger;
  observer_toy_data_t *observer_toy;
  double *targets;

  logger = coco_problem_transformed_get_data(problem);
  observer_toy = (observer_toy_data_t *) logger->observer->data;
  targets = observer_toy->targets;

  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  logger->number_of_evaluations++;

  /* Add a line for each target that has been hit */
  while (y[0] <= targets[logger->next_target] && logger->next_target < observer_toy->number_of_targets) {
    fprintf(observer_toy->log_file, "%e\t%5ld\t%.5f\n", targets[logger->next_target],
        logger->number_of_evaluations, y[0]);
    logger->next_target++;
  }
  /* Flush output so that impatient users can see the progress */
  fflush(observer_toy->log_file);
}

/**
 * @brief Initializes the toy logger.
 */
static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *inner_problem) {

  logger_toy_data_t *logger_toy;
  coco_problem_t *problem;
  FILE *output_file;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_toy(): The toy logger shouldn't be used to log a problem with %d objectives", inner_problem->number_of_objectives);
  }

  logger_toy = coco_allocate_memory(sizeof(*logger_toy));
  logger_toy->observer = observer;
  logger_toy->next_target = 0;
  logger_toy->number_of_evaluations = 0;

  output_file = ((observer_toy_data_t *) logger_toy->observer->data)->log_file;
  fprintf(output_file, "\n%s, %s\n", coco_problem_get_id(inner_problem), coco_problem_get_name(inner_problem));

  problem = coco_problem_transformed_allocate(inner_problem, logger_toy, NULL);
  problem->evaluate_function = logger_toy_evaluate;
  return problem;
}
