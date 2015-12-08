#include <stdio.h>
#include <assert.h>

#include "coco.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"
#include "observer_toy.c"

/**
 * This is a toy logger that logs the evaluation number and function value each time a target has been hit.
 */

typedef struct {
  coco_observer_t *observer;
  size_t next_target;
  long number_of_evaluations;
} logger_toy_t;

/**
 * Evaluates the function, increases the number of evaluations and outputs information based on the targets
 * that have been hit.
 */
static void logger_toy_evaluate(coco_problem_t *self, const double *x, double *y) {

  logger_toy_t *logger;
  observer_toy_t *observer_toy;
  double *targets;

  logger = coco_transformed_get_data(self);
  observer_toy = (observer_toy_t *) logger->observer->data;
  targets = observer_toy->targets;

  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
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
 * Initializes the toy logger.
 */
static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *problem) {

  logger_toy_t *logger;
  coco_problem_t *self;
  FILE *output_file;

  if (problem->number_of_objectives != 1) {
    coco_warning("logger_toy(): The toy logger shouldn't be used to log a problem with %d objectives", problem->number_of_objectives);
  }

  logger = coco_allocate_memory(sizeof(*logger));
  logger->observer = observer;
  logger->next_target = 0;
  logger->number_of_evaluations = 0;

  output_file = ((observer_toy_t *) logger->observer->data)->log_file;
  fprintf(output_file, "\n%s, %s\n", coco_problem_get_id(problem), coco_problem_get_name(problem));

  self = coco_transformed_allocate(problem, logger, NULL);
  self->evaluate_function = logger_toy_evaluate;
  return self;
}
