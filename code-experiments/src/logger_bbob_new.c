/**
 * @file logger_bbob.c
 * @brief Implementation of the bbob logger.
 *
 * Logs the performance of a single-objective optimizer on noisy or noiseless problems with or without constraints.
 * It produces four kinds of files:
 * - The "info" files ...
 * - The "dat" files ...
 * - The "tdat" files ...
 * - The "rdat" files ...
 */

/* TODOs:
 * - Document this file in doxygen style!
 */

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <errno.h>

#include "coco.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_string.c"
#include "observer_bbob_new.c"

/** @brief Factor used in logged indicator (f-f*)^+ + sum_i g_i^+ in front of the sum */
#define LOGGER_BBOB_WEIGHT_CONSTRAINTS 1e0

/**
 * @brief The bbob logger data type.
 *
 * TODO: Are any of the fields superfluous?
 */
typedef struct {
  coco_observer_t *observer;                 /**< @brief Pointer to the observer (might be NULL at the end) */
  char *suite_name;                          /**< @brief The suite name */
  int is_initialized;                        /**< @brief Whether the logger was already initialized */

  FILE *info_file;                           /**< @brief Index file */
  FILE *dat_file;                            /**< @brief File with function value aligned data */
  FILE *tdat_file;                           /**< @brief File with number of function evaluations aligned data */
  FILE *rdat_file;                           /**< @brief File with restart information */

  size_t num_func_evaluations;               /**< @brief The number of function evaluations performed so far. */
  size_t num_cons_evaluations;               /**< @brief The number of evaluations of constraints performed so far. */
  int last_logged_evaluation;                /**< @brief Whether the last evaluation was logged (needed for finalization) */

  double *best_solution;                     /**< @brief The best found solution to this problem. */
  double best_value;                         /**< @brief The best found value for this problem. */
  double optimal_value;                      /**< @brief The optimal value for this problem (might be unknown). TODO */
  double current_value;                      /**< @brief The current value for this problem. */

  size_t function;                           /**< @brief Suite-dependent function number */
  size_t instance;                           /**< @brief Suite-dependent instance number */
  size_t number_of_variables;                /**< @brief Number of all variables */
  size_t number_of_integer_variables;        /**< @brief Number of integer variables */
  int log_discrete_as_int;                   /**< @brief Whether to output discrete variables in integer or double format. */

  coco_observer_targets_t *targets;          /**< @brief Triggers based on target values. */
  coco_observer_evaluations_t *evaluations;  /**< @brief Triggers based on the number of evaluations. */

} logger_bbob_new_data_t;

/**
 * @brief The data format used by the logger
 *
 * Back to 5 columns, 5-th column writes single digit constraint values.
 *
 * Previous formats:
 *
 * -> "old":
 * function evaluation |
 * noise-free fitness - Fopt (7.948000000000e+01) |
 * best noise-free fitness - Fopt |
 * measured fitness |
 * best measured fitness |
 * x1 | x2...
 *
 * -> "bbob-new":
 * f evaluations |
 * g evaluations |
 * best noise-free fitness - Fopt |
 * noise-free fitness - Fopt (%13.12e) |
 * measured fitness |
 * best measured fitness |
 * x1 | x2...
 */
static const char *logger_bbob_new_data_format = "bbob-new2"; /*  */

/**
 * @brief The header used by the logger in .dat, .tdat and .rdat files
 *
 * See also logger_bbob_new_data_format
 */
static const char *logger_bbob_new_header = "%% "
    "f evaluations | "
    "g evaluations | "
    "best noise-free fitness - Fopt (%13.12e) + sum g_i+ | "
    "measured fitness | "
    "best measured fitness or single-digit g-values | "
    "x1 | "
    "x2...\n";

/**
 * @brief Discretized constraint value, ~8 + log10(c), in a single digit.
 *
 * -\infty..0 -> 0
 *    0..1e-7 -> 1
 * 1e-7..1e-6 -> 2
 *    ...
 * 1e-1..1    -> 8
 *   >1       -> 9
 */
static int logger_bbob_new_single_digit_constraint(const double c) {
  const double limits[9] = {0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1};
  int i;

  for (i = 0; i < 9; ++i)
    if (c <= limits[i])
      return i;
  return 9;
}

/**
 * @brief Decides whether a new line in the info file should be started
 *
 * A new line is started when he current dimension differs from the last logged dimension for this
 * function.
 */
static int logger_bbob_new_start_new_line(observer_bbob_new_data_t *observer_data,
                                          size_t current_dimension,
                                          size_t current_function) {
  /* This is complicated, because the function indices are not directly accessible */
  size_t i, last_dimension = 0;
  size_t function_idx = observer_data->num_functions;
  for (i = 0; i < observer_data->num_functions; i++) {
    if (current_function == observer_data->functions_array[i]) {
      function_idx = i;
      break;
    }
  }
  if (function_idx >= observer_data->num_functions)
    coco_error("logger_bbob_new_start_new_line(): Cannot find function %lu", current_function);

  last_dimension = observer_data->last_dimensions[function_idx];
  /* Finally, update current dimension */
  observer_data->last_dimensions[function_idx] = current_dimension;

  return (current_dimension != last_dimension);
}

/**
 * @brief Outputs a formated line to a data file
 */
static void logger_bbob_new_output(FILE *data_file,
                                   size_t num_func_evaluations,
                                   size_t num_cons_evaluations,
                                   double current_value,
                                   double best_value,
                                   double optimal_value,
                                   const double *x,
                                   size_t number_of_variables,
                                   size_t number_of_integer_variables,
                                   const double *constraints,
                                   size_t number_of_constraints,
                                   const int log_discrete_as_int) {
  /* This function contains many hard-coded values (10.9, 22, 5.4) that could be read through
   * observer options */
  size_t i;

  fprintf(data_file, "%lu %lu %+10.9e %+10.9e ", (unsigned long) num_func_evaluations,
      (unsigned long) num_cons_evaluations, best_value - optimal_value, current_value);

  if (number_of_constraints > 0) {
    for (i = 0; i < number_of_constraints; ++i) {
      /* print 01234567890123..., may happen in the last line of .tdat */
      fprintf(data_file, "%d",
          constraints ? logger_bbob_new_single_digit_constraint(constraints[i]) : (int) (i % 10));
    }
  } else {
    fprintf(data_file, "%+10.9e", best_value);
  }

  if ((number_of_variables - number_of_integer_variables) < 22) {
    for (i = 0; i < number_of_variables; i++) {
      if ((i < number_of_integer_variables) && (log_discrete_as_int))
        fprintf(data_file, " %d", coco_double_to_int(x[i]));
      else
        fprintf(data_file, " %+5.4e", x[i]);
    }
  }
  fprintf(data_file, "\n");

  /* Flush output so that impatient users can see progress.
   * Otherwise it can take a long time until the output appears.
   */
  fflush(data_file);
}

/**
 * @brief Opens the file in append mode
 */
static void logger_bbob_new_open_file(FILE **file, const char *file_path) {
  if (*file == NULL) {
    *file = fopen(file_path, "a");
    if (*file == NULL) {
      coco_error("logger_bbob_new_open_file(): Error opening file: %s\nError: %d", file_path, errno);
    }
  }
}

/**
 * @brief Creates the data file (if it didn't exist before) and opens it
 */
static void logger_bbob_new_open_data_file(FILE **data_file,
                                           const char *path,
                                           const char *file_name,
                                           const char *file_extension) {
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  char relative_file_path[COCO_PATH_MAX + 2] = { 0 };
  strncpy(relative_file_path, file_name, COCO_PATH_MAX - strlen(relative_file_path) - 1);
  strncat(relative_file_path, file_extension, COCO_PATH_MAX - strlen(relative_file_path) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_file_path, NULL);
  logger_bbob_new_open_file(data_file, file_path);
}

/**
 * @brief Creates the info file (if it didn't exist before) and opens it
 */
static void logger_bbob_new_open_info_file(logger_bbob_new_data_t *logger,
                                           const char *folder,
                                           const char *function_string,
                                           const char *data_file_name,
                                           const char *suite_name) {
  char data_file_path[COCO_PATH_MAX + 2] = { 0 };
  int start_new_line = 0;
  char file_name[COCO_PATH_MAX + 2] = { 0 };
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  FILE **info_file;
  observer_bbob_new_data_t *observer_data;

  coco_debug("Started logger_bbob_new_open_info_file()");

  assert(logger != NULL);
  assert(logger->observer != NULL);
  observer_data = ((observer_bbob_new_data_t *)((coco_observer_t *)logger->observer)->data);
  assert (observer_data != NULL);

  strncpy(data_file_path, data_file_name, COCO_PATH_MAX - strlen(data_file_path) - 1);

  info_file = &(logger->info_file);

  strncpy(file_name, observer_data->prefix, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, "_f", COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, function_string, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
  coco_join_path(file_path, sizeof(file_path), folder, file_name, NULL);

  if (*info_file == NULL) {
    logger_bbob_new_open_file(info_file, file_path);
    start_new_line = logger_bbob_new_start_new_line(observer_data, logger->number_of_variables, logger->function);
    if (start_new_line) {
      fprintf(*info_file, "\n");
    }
    fprintf(*info_file,
            "suite = '%s', funcId = %lu, DIM = %lu, Precision = %.3e, algId = '%s', coco_version = '%s', logger = '%s', data_format = '%s'\n",
            suite_name,
            (unsigned long) logger->function,
            (unsigned long) logger->number_of_variables,
            pow(10, -8),
            logger->observer->algorithm_name,
            coco_version,
            ((coco_observer_t *)logger->observer)->observer_name,
            logger_bbob_new_data_format);

      fprintf(*info_file, "%%\n");
      fprintf(*info_file, "%s.dat", data_file_path); /* data_file_path does not have the extension */
  }

  coco_debug("Ended   logger_bbob_new_open_info_file()");
}

/**
 * @brief Generates the files and folders needed by the logger if they don't already exist
 */
static void logger_bbob_new_initialize(logger_bbob_new_data_t *logger) {

  char relative_path[COCO_PATH_MAX + 2] = { 0 }; /* Relative path to the .dat file from where the .info file is */
  char folder_path[COCO_PATH_MAX + 2] = { 0 };
  char *function_string;
  char *dimension_string;
  observer_bbob_new_data_t *observer_data;

  coco_debug("Started logger_bbob_new_initialize()");

  assert(logger != NULL);
  assert(logger->observer != NULL);
  observer_data = (observer_bbob_new_data_t *)((coco_observer_t *)logger->observer)->data;
  assert (observer_data != NULL);

  function_string = coco_strdupf("%lu", (unsigned long) logger->function);
  dimension_string = coco_strdupf("%lu", (unsigned long) logger->number_of_variables);

  /* Prepare paths and names */
  strncpy(relative_path, "data_f", COCO_PATH_MAX);
  strncat(relative_path, function_string, COCO_PATH_MAX - strlen(relative_path) - 1);
  coco_join_path(folder_path, sizeof(folder_path), logger->observer->result_folder, relative_path, NULL);
  coco_create_directory(folder_path);
  strncat(relative_path, "/bbobexp_f", COCO_PATH_MAX - strlen(relative_path) - 1);
  strncat(relative_path, function_string, COCO_PATH_MAX - strlen(relative_path) - 1);
  strncat(relative_path, "_DIM", COCO_PATH_MAX - strlen(relative_path) - 1);
  strncat(relative_path, dimension_string, COCO_PATH_MAX - strlen(relative_path) - 1);

  /* info file */
  logger_bbob_new_open_info_file(logger, logger->observer->result_folder, function_string,
      relative_path, logger->suite_name);
  fprintf(logger->info_file, ", %lu", (unsigned long) logger->instance);

  /* data files */
  logger_bbob_new_open_data_file(&(logger->dat_file), logger->observer->result_folder, relative_path, ".dat");
  fprintf(logger->dat_file, logger_bbob_new_header, logger->optimal_value);

  logger_bbob_new_open_data_file(&(logger->tdat_file), logger->observer->result_folder, relative_path, ".tdat");
  fprintf(logger->tdat_file, logger_bbob_new_header, logger->optimal_value);

  logger_bbob_new_open_data_file(&(logger->rdat_file), logger->observer->result_folder, relative_path, ".rdat");
  fprintf(logger->rdat_file, logger_bbob_new_header, logger->optimal_value);

  logger->is_initialized = 1;
  coco_free_memory(dimension_string);
  coco_free_memory(function_string);

  coco_debug("Ended   logger_bbob_new_initialize()");
}

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information according to
 * observer options.
 */
static void logger_bbob_new_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double y_logged, max_value, sum_constraints;
  double *constraints = NULL;
  logger_bbob_new_data_t *logger = (logger_bbob_new_data_t *) coco_problem_transformed_get_data(problem);
  coco_problem_t *inner_problem = coco_problem_transformed_get_inner_problem(problem);
  const int is_feasible = problem->number_of_constraints <= 0 || coco_is_feasible(inner_problem, x, NULL);

  coco_debug("Started logger_bbob_new_evaluate()");

  if (!logger->is_initialized) {
    logger_bbob_new_initialize(logger);
  }
  if ((coco_log_level >= COCO_DEBUG) && logger->num_func_evaluations == 0) {
    coco_debug("%4lu: ", (unsigned long) inner_problem->suite_dep_index);
    coco_debug("on problem %s ... ", coco_problem_get_id(inner_problem));
  }

  /* Fulfill contract of a COCO evaluate function */
  coco_evaluate_function(inner_problem, x, y);

  logger->num_cons_evaluations = coco_problem_get_evaluations_constraints(problem);
  logger->num_func_evaluations++;
  logger->last_logged_evaluation = 0;
  logger->current_value = y[0];

  y_logged = y[0];
  if (coco_is_nan(y_logged))
    y_logged = NAN_FOR_LOGGING;
  else if (coco_is_inf(y_logged))
    y_logged = INFINITY_FOR_LOGGING;
  /* Do sanity check */
  if (is_feasible) {
    /* Infeasible solutions can have much better y0 values */
    assert(y_logged + 1e-13 >= logger->optimal_value);
  }

  /* Evaluate the constraints */
  if (problem->number_of_constraints > 0) {
    constraints = coco_allocate_vector(problem->number_of_constraints);
    inner_problem->evaluate_constraint(inner_problem, x, constraints);
  }

  /* Compute the sum of positive constraint values */
  sum_constraints = 0;
  for (i = 0; i < problem->number_of_constraints; ++i) {
    if (constraints[i] > 0)
        sum_constraints += constraints[i];
  }
  sum_constraints *= LOGGER_BBOB_WEIGHT_CONSTRAINTS;  /* do this before the checks */
  if (coco_is_nan(sum_constraints))
    sum_constraints = NAN_FOR_LOGGING;
  else if (coco_is_inf(sum_constraints))
    sum_constraints = INFINITY_FOR_LOGGING;

  max_value =  y_logged > logger->optimal_value ? y_logged : logger->optimal_value;

  /* Update logger state
   * At logger->number_of_evaluations == 1 the logger->best_value is not initialized,
   * also compare to y_logged to not potentially be thrown off by weird values in y[0]
   */
  if (logger->num_func_evaluations == 1 || (max_value + sum_constraints < logger->best_value)) {
    logger->best_value = max_value + sum_constraints;
    for (i = 0; i < problem->number_of_variables; i++)
      logger->best_solution[i] = x[i]; /* May well be infeasible */

    /* Add a line in the .dat file for each logging target reached
     * by a feasible solution and always at evaluation one */
    if (logger->num_func_evaluations == 1 ||
        coco_observer_targets_trigger(logger->targets, logger->best_value - logger->optimal_value)) {
      logger_bbob_new_output(
          logger->dat_file,
          logger->num_func_evaluations,
          logger->num_cons_evaluations,
          y_logged,
          logger->best_value,
          logger->optimal_value,
          x,
          problem->number_of_variables,
          problem->number_of_integer_variables,
          constraints,
          problem->number_of_constraints,
          logger->log_discrete_as_int);
    }
  }

  /* Add a line in the .tdat file each time an fevals trigger is reached.*/
  if (coco_observer_evaluations_trigger(logger->evaluations, logger->num_func_evaluations + logger->num_cons_evaluations)) {
    logger_bbob_new_output(
        logger->tdat_file,
        logger->num_func_evaluations,
        logger->num_cons_evaluations,
        y_logged,
        logger->best_value,
        logger->optimal_value,
        x,
        problem->number_of_variables,
        problem->number_of_integer_variables,
        constraints,
        problem->number_of_constraints,
        logger->log_discrete_as_int);
    logger->last_logged_evaluation = 1;
  }

  /* Free allocated memory */
  if (problem->number_of_constraints > 0)
    coco_free_memory(constraints);

  coco_debug("Ended   logger_bbob_new_evaluate()");

}  /* end logger_bbob_evaluate */

/**
 * @brief Takes care of some final output and frees the logger data
 */
static void logger_bbob_new_free(void *stuff) {

  logger_bbob_new_data_t *logger = (logger_bbob_new_data_t *) stuff;
  coco_observer_t *observer; /* The observer might not exist at this point */

  coco_debug("Started logger_bbob_new_free()");

  if ((coco_log_level >= COCO_DEBUG) && logger && logger->num_func_evaluations > 0) {
    coco_debug("best f=%e after %lu fevals (done observing)\n", logger->best_value,
    		(unsigned long) logger->num_func_evaluations);
  }
  if (logger->info_file != NULL) {
    fprintf(logger->info_file, ":%lu|%.1e",
            (unsigned long) logger->num_func_evaluations,
            logger->best_value - logger->optimal_value);
    fclose(logger->info_file);
    logger->info_file = NULL;
  }
  if (logger->dat_file != NULL) {
    fclose(logger->dat_file);
    logger->dat_file = NULL;
  }
  if (logger->tdat_file != NULL) {
    if (!logger->last_logged_evaluation) {
      logger_bbob_new_output(logger->tdat_file,
          logger->num_func_evaluations,
          logger->num_cons_evaluations,
          logger->best_value,
          logger->best_value,
          logger->optimal_value,
          logger->best_solution,
          logger->number_of_variables,
          logger->number_of_integer_variables,
          NULL,
          0,
          logger->log_discrete_as_int);
	}
    fclose(logger->tdat_file);
    logger->tdat_file = NULL;
  }

  if (logger->rdat_file != NULL) {
    fclose(logger->rdat_file);
    logger->rdat_file = NULL;
  }

  if (logger->best_solution != NULL) {
    coco_free_memory(logger->best_solution);
    logger->best_solution = NULL;
  }

  if (logger->targets != NULL){
    coco_free_memory(logger->targets);
    logger->targets = NULL;
  }

  if (logger->evaluations != NULL){
    coco_observer_evaluations_free(logger->evaluations);
    logger->evaluations = NULL;
  }

  observer = logger->observer;
  if ((observer != NULL) && (observer->data != NULL)) {
    /* If the observer still exists (if it does not, logger_is_used does not matter any longer) */
    ((observer_bbob_new_data_t *)observer->data)->logger_is_used = 0;
  }

  coco_debug("Ended   logger_bbob_new_free()");
}

static coco_problem_t *logger_bbob_new(coco_observer_t *observer, coco_problem_t *inner_problem) {
  coco_problem_t *problem;
  logger_bbob_new_data_t *logger_data;
  observer_bbob_new_data_t *observer_data;
  coco_suite_t *suite;
  size_t i;

  coco_debug("Started logger_bbob_new()");

  assert(inner_problem);
  assert(inner_problem->suite);
  suite = (coco_suite_t *)inner_problem->suite;
  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_bbob_new(): The bbob logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }

  assert(observer != NULL);
  observer_data = (observer_bbob_new_data_t *) observer->data;
  assert(observer_data != NULL);
  if (observer_data->logger_is_used) {
    coco_error("logger_bbob_new(): The current logger (observer) must be closed before a new one is opened");
  }

  /* These values are initialized only the first time a bbob_new logger is allocated */
  if (observer_data->num_functions == 0) {
    observer_data->num_functions = suite->number_of_functions;
  }
  if (observer_data->last_dimensions == NULL) {
    observer_data->last_dimensions = coco_allocate_vector_size_t(observer_data->num_functions);
    for (i = 0; i < observer_data->num_functions; i++)
      observer_data->last_dimensions[i] = 0;
  }
  if (observer_data->functions_array == NULL) {
    observer_data->functions_array = coco_allocate_vector_size_t(observer_data->num_functions);
    for (i = 0; i < observer_data->num_functions; i++)
      observer_data->functions_array[i] = suite->functions[i];
  }

  observer_data->logger_is_used = 1;

  logger_data = (logger_bbob_new_data_t *) coco_allocate_memory(sizeof(*logger_data));
  logger_data->observer = observer;
  logger_data->suite_name = coco_problem_get_suite(inner_problem)->suite_name;
  logger_data->is_initialized = 0;

  logger_data->info_file = NULL;
  logger_data->dat_file = NULL;
  logger_data->tdat_file = NULL;
  logger_data->rdat_file = NULL;

  logger_data->num_func_evaluations = 0;
  logger_data->num_cons_evaluations = 0;
  logger_data->last_logged_evaluation = 0;

  logger_data->best_solution = coco_allocate_vector(inner_problem->number_of_variables);
  logger_data->best_value = DBL_MAX;
  if (inner_problem->best_value == NULL) {
    /* TODO!!! coco_error("Optimal f value must be defined for each problem in order for the logger to work properly"); */
    /* Setting the value to 0 results in the assertion y>=optimal_value being susceptible to failure */
    coco_warning("undefined optimal f value. Set to 0");
    logger_data->optimal_value = 0;
  } else {
    logger_data->optimal_value = *(inner_problem->best_value);
  }
  logger_data->current_value = DBL_MAX;

  logger_data->function = coco_problem_get_suite_dep_function(inner_problem);
  logger_data->instance = coco_problem_get_suite_dep_instance(inner_problem);
  logger_data->number_of_variables = inner_problem->number_of_variables;
  logger_data->number_of_integer_variables = inner_problem->number_of_integer_variables;
  logger_data->log_discrete_as_int = observer->log_discrete_as_int;
    
  /* Initialize triggers based on target values and number of evaluations */
  logger_data->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  logger_data->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, inner_problem->number_of_variables);

  coco_debug("Ended   logger_bbob_new()");

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_bbob_new_free, observer->observer_name);
  problem->evaluate_function = logger_bbob_new_evaluate;
  return problem;
}
