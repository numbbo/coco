/**
 * @file logger_toy.c
 * @brief Implementation of the toy logger.
 *
 * Logs the evaluation number, function value the target hit and all the variables each time a target has
 * been hit.
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
  FILE *log_file;                        /**< @brief Pointer to the file already prepared for logging. */
  coco_observer_log_targets_t *targets;  /**< @brief Triggers based on logarithmic target values. */
  size_t number_of_evaluations;          /**< @brief The number of evaluations performed so far. */
  int precision_x;                       /**< @brief Precision for outputting decision values. */
  int precision_f;                       /**< @brief Precision for outputting objective values. */
} logger_toy_data_t;

/**
 * @brief Frees the memory of the given toy logger.
 */
static void logger_toy_free(void *stuff) {

  logger_toy_data_t *logger;

  assert(stuff != NULL);
  logger = (logger_toy_data_t *) stuff;

  if (logger->targets != NULL){
    coco_free_memory(logger->targets);
    logger->targets = NULL;
  }

}

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information based on the
 * targets that have been hit.
 */
static void logger_toy_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_toy_data_t *logger = (logger_toy_data_t *) coco_problem_transformed_get_data(problem);
  size_t i;

  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  logger->number_of_evaluations++;

  /* Output the solution when a new target that has been hit */
  if (coco_observer_log_targets_trigger(logger->targets, y[0])) {
    fprintf(logger->log_file, "%lu\t%.*e\t%.*e", (unsigned long) logger->number_of_evaluations,
    		logger->precision_f, y[0], logger->precision_f, logger->targets->value);
    for (i = 0; i < problem->number_of_variables; i++) {
      fprintf(logger->log_file, "\t%.*e", logger->precision_x, x[i]);
    }
    fprintf(logger->log_file, "\n");
  }

  /* Flush output so that impatient users can see the progress */
  fflush(logger->log_file);
}

/**
 * @brief Initializes the toy logger.
 */
static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *inner_problem) {

  logger_toy_data_t *logger_data;
  coco_problem_t *problem;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_toy(): The toy logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }

  /* Initialize the logger_toy_data_t object instance */
  logger_data = (logger_toy_data_t *) coco_allocate_memory(sizeof(*logger_data));
  logger_data->number_of_evaluations = 0;
  logger_data->targets = coco_observer_log_targets(observer->number_target_triggers, observer->log_target_precision);
  logger_data->log_file = ((observer_toy_data_t *) observer->data)->log_file;
  logger_data->precision_x = observer->precision_x;
  logger_data->precision_f = observer->precision_f;

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_toy_free, observer->observer_name);
  problem->evaluate_function = logger_toy_evaluate;

  /* Output initial information */
  assert(coco_problem_get_suite(inner_problem));
  fprintf(logger_data->log_file, "\n");
  fprintf(logger_data->log_file, "suite = '%s', problem_id = '%s', problem_name = '%s', coco_version = '%s'\n",
          coco_problem_get_suite(inner_problem)->suite_name, coco_problem_get_id(inner_problem),
          coco_problem_get_name(inner_problem), coco_version);
  fprintf(logger_data->log_file, "%% evaluation number | function value | target hit | %lu variables \n",
  		(unsigned long) inner_problem->number_of_variables);

  return problem;
}
