/**
 * @file logger_rw.c
 * @brief Implementation of the real-world logger.
 *
 * Can be used to log all (or just those that are better than the preceding) solutions with information
 * about objectives, decision variables (optional) and constraints (optional). See observer_rw() for
 * more information on the options. Produces one "txt" file for each problem function, dimension and
 * instance.
 *
 * @note This logger can be used with single- and multi-objective problems, but in the multi-objective
 * case, all solutions are always logged.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_string.c"
#include "observer_rw.c"

/**
 * @brief The rw logger data type.
 *
 * @note Some fields from the observers (coco_observer as well as observer_rw) need to be copied here
 * because the observers can be deleted before the logger is finalized and we need these fields for
 * finalization.
 */
typedef struct {
  FILE *out_file;                /**< @brief File for logging. */
  size_t num_func_evaluations;   /**< @brief The number of function evaluations performed so far. */
  size_t num_cons_evaluations;   /**< @brief The number of evaluations of constraints performed so far. */

  double best_value;             /**< @brief The best-so-far value. */
  double current_value;          /**< @brief The current value. */

  int log_vars;                  /**< @brief Whether to log the decision values. */
  int log_cons;                  /**< @brief Whether to log the constraints. */
  int log_only_better;           /**< @brief Whether to log only solutions that are better than previous ones. */
  int log_time;                  /**< @brief Whether to log evaluation time. */

  int precision_x;               /**< @brief Precision for outputting decision values. */
  int precision_f;               /**< @brief Precision for outputting objective values. */
  int precision_g;               /**< @brief Precision for outputting constraint values. */
  int log_discrete_as_int;       /**< @brief Whether to output discrete variables in int or double format. */
} logger_rw_data_t;

/**
 * @brief Evaluates the function and constraints and outputs the information according to the
 * observer options.
 */
static void logger_rw_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_rw_data_t *logger;
  coco_problem_t *inner_problem;
  double *constraints = NULL;
  size_t i;
  int log_this_time = 1;
  time_t start, end;

  logger = (logger_rw_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* Time the evaluations */
  if (logger->log_time)
    time(&start);

  /* Evaluate the objective(s) */
  coco_evaluate_function(inner_problem, x, y);
  logger->num_func_evaluations++;

  if (problem->number_of_objectives == 1)
    logger->current_value = y[0];

  /* Evaluate the constraints */
  if (problem->number_of_constraints > 0) {
    constraints = coco_allocate_vector(problem->number_of_constraints);
    inner_problem->evaluate_constraint(inner_problem, x, constraints, 0);
  }
  logger->num_cons_evaluations = problem->evaluations_constraints;

  /* Time the evaluations */
  if (logger->log_time)
    time(&end);

  /* Log to the output file */
  if ((problem->number_of_objectives == 1) && (logger->current_value < logger->best_value))
    logger->best_value = logger->current_value;
  else if (problem->number_of_objectives == 1)
    log_this_time = !logger->log_only_better;
  if ((logger->num_func_evaluations == 1) || log_this_time) {
    fprintf(logger->out_file, "%lu\t", (unsigned long) logger->num_func_evaluations);
    fprintf(logger->out_file, "%lu\t", (unsigned long) logger->num_cons_evaluations);
    for (i = 0; i < problem->number_of_objectives; i++)
      fprintf(logger->out_file, "%+.*e\t", logger->precision_f, y[i]);
    if (logger->log_vars) {
      for (i = 0; i < problem->number_of_variables; i++) {
        if ((i < problem->number_of_integer_variables) && (logger->log_discrete_as_int))
          fprintf(logger->out_file, "%d\t", coco_double_to_int(x[i]));
        else
          fprintf(logger->out_file, "%+.*e\t", logger->precision_x, x[i]);
      }
    }
    if (logger->log_cons) {
      for (i = 0; i < problem->number_of_constraints; i++)
        fprintf(logger->out_file, "%+.*e\t", logger->precision_g, constraints[i]);
    }
    /* Log time in seconds */
    if (logger->log_time)
      fprintf(logger->out_file, "%.0f\t", difftime(end, start));
    fprintf(logger->out_file, "\n");
  }
  fflush(logger->out_file);

  if (problem->number_of_constraints > 0)
    coco_free_memory(constraints);

}

/**
 * @brief Frees the memory of the given rw logger.
 */
static void logger_rw_free(void *stuff) {

  logger_rw_data_t *logger;

  assert(stuff != NULL);
  logger = (logger_rw_data_t *) stuff;

  if (logger->out_file != NULL) {
    fclose(logger->out_file);
    logger->out_file = NULL;
  }
}

/**
 * @brief Initializes the rw logger.
 *
 * Copies all observer field values that are needed after initialization into logger field values for two
 * reasons:
 * - If the observer is deleted before the suite, the observer is not available anymore when the logger
 * is finalized.
 * - This reduces function calls.
 */
static coco_problem_t *logger_rw(coco_observer_t *observer, coco_problem_t *inner_problem) {

  coco_problem_t *problem;
  logger_rw_data_t *logger_data;
  observer_rw_data_t *observer_data;
  char *path_name, *file_name = NULL;

  logger_data = (logger_rw_data_t *) coco_allocate_memory(sizeof(*logger_data));
  logger_data->num_func_evaluations = 0;
  logger_data->num_cons_evaluations = 0;

  observer_data = (observer_rw_data_t *) observer->data;
  /* Copy values from the observes that you might need even if they do not exist any more */
  logger_data->precision_x = observer->precision_x;
  logger_data->precision_f = observer->precision_f;
  logger_data->precision_g = observer->precision_g;
  logger_data->log_discrete_as_int = observer->log_discrete_as_int;

  if (((observer_data->log_vars_mode == LOG_LOW_DIM) &&
      (inner_problem->number_of_variables > observer_data->low_dim_vars))
      || (observer_data->log_vars_mode == LOG_NEVER))
    logger_data->log_vars = 0;
  else
    logger_data->log_vars = 1;

  if (((observer_data->log_cons_mode == LOG_LOW_DIM) &&
      (inner_problem->number_of_constraints > observer_data->low_dim_cons))
      || (observer_data->log_cons_mode == LOG_NEVER)
      || (inner_problem->number_of_constraints == 0))
    logger_data->log_cons = 0;
  else
    logger_data->log_cons = 1;

  logger_data->log_only_better = (observer_data->log_only_better) &&
      (inner_problem->number_of_objectives == 1);
  logger_data->log_time = observer_data->log_time;

  logger_data->best_value = DBL_MAX;
  logger_data->current_value = DBL_MAX;

  /* Construct file name */
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_create_directory(path_name);
  file_name = coco_strdupf("%s_rw.txt", coco_problem_get_id(inner_problem));
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);

  /* Open and initialize the output file */
  logger_data->out_file = fopen(path_name, "a");
  if (logger_data->out_file == NULL) {
    coco_error("logger_rw() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(path_name);
  coco_free_memory(file_name);

  /* Output header information */
  fprintf(logger_data->out_file, "\n%% suite = '%s', problem_id = '%s', problem_name = '%s', coco_version = '%s'\n",
          coco_problem_get_suite(inner_problem)->suite_name, coco_problem_get_id(inner_problem),
          coco_problem_get_name(inner_problem), coco_version);
  fprintf(logger_data->out_file, "%% f-evaluations | g-evaluations | %lu objective",
      (unsigned long) inner_problem->number_of_objectives);
  if (inner_problem->number_of_objectives > 1)
    fprintf(logger_data->out_file, "s");
  if (logger_data->log_vars)
    fprintf(logger_data->out_file, " | %lu variable",
        (unsigned long) inner_problem->number_of_variables);
  if (inner_problem->number_of_variables > 1)
    fprintf(logger_data->out_file, "s");
  if (logger_data->log_cons)
    fprintf(logger_data->out_file, " | %lu constraint",
        (unsigned long) inner_problem->number_of_constraints);
  if (inner_problem->number_of_constraints > 1)
    fprintf(logger_data->out_file, "s");
  if (logger_data->log_time)
    fprintf(logger_data->out_file, " | evaluation time (s)");
  fprintf(logger_data->out_file, "\n");

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_rw_free, observer->observer_name);
  problem->evaluate_function = logger_rw_evaluate;

  return problem;
}
