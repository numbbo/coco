/**
 * @file logger_bbob_old.c
 * @brief Old implementation of the bbob logger.
 *
 * Logs the performance of a single-objective optimizer on noisy or noiseless problems.
 * It produces four kinds of files:
 * - The "info" files ...
 * - The "dat" files ...
 * - The "tdat" files ...
 * - The "rdat" files ...
 */

/* TODO: Document this file in doxygen style! */

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
#include "observer_bbob_old.c"

static const double fvalue_logged_for_infinite = 3e21;   /* value used for logging try */
static const double fvalue_logged_for_nan = 2e21;
/* static const double fvalue_logged_for_infeasible = 1e21;  only in first evaluation */
static const double weight_constraints = 1e0;  /* factor used in logged indicator (f-f*)^+ + sum_i g_i^+ in front of the sum */

/*static const size_t bbob_old_nbpts_nbevals = 20; Wassim: tentative, are now observer options with these default values*/
/*static const size_t bbob_old_nbpts_fval = 5;*/
static size_t bbob_old_current_dim = 0;
static size_t bbob_old_current_funId = 0;
static size_t bbob_old_infoFile_firstInstance = 0;
char *bbob_old_infoFile_firstInstance_char;
/* a possible solution: have a list of dims that are already in the file, if the ones we're about to log
 * is != bbob_old_current_dim and the funId is currend_funId, create a new .info file with as suffix the
 * number of the first instance */
static const int bbob_old_number_of_dimensions = 6;
static size_t bbob_old_dimensions_in_current_infoFile[6] = { 0, 0, 0, 0, 0, 0 }; /* TODO should use dimensions from the suite */

/* The current_... mechanism fails if several problems are open.
 * For the time being this should lead to an error.
 *
 * A possible solution: bbob_old_logger_is_open becomes a reference
 * counter and as long as another logger is open, always a new info
 * file is generated.
 * TODO: Shouldn't the new way of handling observers already fix this?
 */
static int bbob_old_logger_is_open = 0; /* this could become lock-list of .info files */

/* TODO: add possibility of adding a prefix to the index files (easy to do through observer options) */

/**
 * @brief The bbob_old logger data type.
 */
typedef struct {
  coco_observer_t *observer;
  int is_initialized;
  /*char *path;// relative path to the data folder. //Wassim: now fetched from the observer */
  /*const char *alg_name; the alg name, for now, temporarily the same as the path. Wassim: Now in the observer */
  FILE *index_file; /* index file */
  FILE *fdata_file; /* function value aligned data file */
  FILE *tdata_file; /* number of function evaluations aligned data file */
  FILE *rdata_file; /* restart info data file */
  size_t number_of_evaluations;
  size_t number_of_evaluations_constraints;
  double best_fvalue;
  double last_fvalue;
  short written_last_eval; /* allows writing the data of the final fun eval in the .tdat file if not already written by the t_trigger*/
  double *best_solution;
  /* The following are to only pass data as a parameter in the free function. The
   * interface should probably be the same for all free functions so passing the
   * problem as a second parameter is not an option even though we need info
   * form it.*/
  size_t function_id; /*TODO: consider changing name*/
  size_t instance_id;
  size_t number_of_variables;
  size_t number_of_integer_variables;
  int log_discrete_as_int;            /**< @brief Whether to output discrete variables in int or double format. */
  double optimal_fvalue;
  char *suite_name;

  coco_observer_log_targets_t *targets;      /**< @brief Triggers based on target values. */
  coco_observer_evaluations_t *evaluations;  /**< @brief Triggers based on the number of evaluations. */

} logger_bbob_old_data_t;

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
static int single_digit_constraint_value(const double c) {
  const double limits[9] = {0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1};
  int i;

  for (i = 0; i < 9; ++i)
    if (c <= limits[i])
      return i;
  return 9;
}

/* was (old):
 * function evaluation |
 * noise-free fitness - Fopt (7.948000000000e+01) |
 * best noise-free fitness - Fopt |
 * measured fitness |
 * best measured fitness |
 * x1 | x2...
 was (bbob-new):
    "f evaluations | "
    "g evaluations | "
    "best noise-free fitness - Fopt | "
    "noise-free fitness - Fopt (%13.12e) | "
    "measured fitness | "
    "best measured fitness | "
    "x1 | "
    "x2...\n";
 */
static const char *bbob_old_file_header_str = "%% "
    "f evaluations | "
    "g evaluations | "
    "best noise-free fitness - Fopt (%13.12e) + sum g_i+ | "
    "measured fitness | "
    "best measured fitness or single-digit g-values | "
    "x1 | "
    "x2...\n";

static const char *logger_name = "bbob_old";
static const char *data_format = "bbob-new2"; /* back to 5 columns, 5-th column writes single digit constraint values */

/**
 * adds a formated line to a data file
 */
static void logger_bbob_old_write_data(FILE *target_file,
                                       size_t number_of_f_evaluations,
                                       size_t number_of_cons_evaluations,
                                       double fvalue,
                                       double best_fvalue,
                                       double best_value,
                                       const double *x,
                                       size_t number_of_variables,
                                       size_t number_of_integer_variables,
                                       const double *constraints,
                                       size_t number_of_constraints,
                                       const int log_discrete_as_int) {
  size_t i;
  /* for some reason, it's %.0f in the old code instead of the 10.9e
   * in the documentation
   */
  fprintf(target_file, "%lu %lu %+10.9e %+10.9e ",
          (unsigned long) number_of_f_evaluations,
	  (unsigned long) number_of_cons_evaluations,
          best_fvalue - best_value,
	  fvalue);

  if (number_of_constraints > 0)
    for (i = 0; i < number_of_constraints; ++i)
      fprintf(target_file, "%d",
              constraints ? single_digit_constraint_value(constraints[i])
                          : (int) (i % 10)); /* print 01234567890123..., may happen in last line of .tdat */
  else
    fprintf(target_file, "%+10.9e", best_fvalue);

  if ((number_of_variables - number_of_integer_variables) < 22) {
    for (i = 0; i < number_of_variables; i++) {
      if ((i < number_of_integer_variables) && (log_discrete_as_int))
        fprintf(target_file, " %d", coco_double_to_int(x[i]));
      else
        fprintf(target_file, " %+5.4e", x[i]);
    }
  }
  fprintf(target_file, "\n");

  /* Flush output so that impatient users can see progress.
   * Otherwise it can take a long time until the output appears.
   */
  fflush(target_file);
}

/**
 * Error when trying to create the file "path"
 */
static void logger_bbob_old_error_io(FILE *path, int errnum) {
  const char *error_format = "Error opening file: %s\n ";
  coco_error(error_format, strerror(errnum), path);
}

/**
 * Creates the data files or simply opens it
 */

/*
 calling sequence:
 logger_bbob_old_open_dataFile(&(logger->fdata_file), logger->observer->output_folder, dataFile_path,
 ".dat");
 */

static void logger_bbob_old_open_dataFile(FILE **target_file,
                                          const char *path,
                                          const char *dataFile_path,
                                          const char *file_extension) {
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  char relative_filePath[COCO_PATH_MAX + 2] = { 0 };
  int errnum;
  strncpy(relative_filePath, dataFile_path,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  strncat(relative_filePath, file_extension,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
  if (*target_file == NULL) {
    *target_file = fopen(file_path, "a+");
    errnum = errno;
    if (*target_file == NULL) {
      logger_bbob_old_error_io(*target_file, errnum);
    }
  }
}

/*
static void logger_bbob_old_open_dataFile(FILE **target_file,
                                          const char *path,
                                          const char *dataFile_path,
                                          const char *file_extension) {
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  char relative_filePath[COCO_PATH_MAX + 2] = { 0 };
  int errnum;
  strncpy(relative_filePath, dataFile_path,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  strncat(relative_filePath, file_extension,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
  if (*target_file == NULL) {
    *target_file = fopen(file_path, "a+");
    errnum = errno;
    if (*target_file == NULL) {
      _bbob_old_logger_error_io(*target_file, errnum);
    }
  }
}*/

/**
 * Creates the index file fileName_prefix+problem_id+file_extension in
 * folder_path
 */
static void logger_bbob_old_openIndexFile(logger_bbob_old_data_t *logger,
                                          const char *folder_path,
                                          const char *indexFile_prefix,
                                          const char *function_id,
                                          const char *dataFile_path,
                                          const char *suite_name) {
  /* to add the instance number TODO: this should be done outside to avoid redoing this for the .*dat files */
  char used_dataFile_path[COCO_PATH_MAX + 2] = { 0 };
  int errnum, newLine = 0; /* newLine is at 1 if we need a new line in the info file */
  char *function_id_char; /* TODO: consider adding them to logger */
  char file_name[COCO_PATH_MAX + 2] = { 0 };
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  FILE **target_file;
  FILE *tmp_file;
  strncpy(used_dataFile_path, dataFile_path, COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
  if (bbob_old_infoFile_firstInstance == 0) {
    bbob_old_infoFile_firstInstance = logger->instance_id;
  }
  function_id_char = coco_strdupf("%lu", (unsigned long) logger->function_id);
  bbob_old_infoFile_firstInstance_char = coco_strdupf("%lu", (unsigned long) bbob_old_infoFile_firstInstance);
  target_file = &(logger->index_file);
  tmp_file = NULL; /* to check whether the file already exists. Don't want to use target_file */
  strncpy(file_name, indexFile_prefix, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, "_f", COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, function_id_char, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, "_i", COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, bbob_old_infoFile_firstInstance_char, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
  coco_join_path(file_path, sizeof(file_path), folder_path, file_name, NULL);
  if (*target_file == NULL) {
    tmp_file = fopen(file_path, "r"); /* to check for existence */
    if ((tmp_file) && (bbob_old_current_dim == logger->number_of_variables)
        && (bbob_old_current_funId == logger->function_id)) {
        /* new instance of current funId and current dim */
      *target_file = fopen(file_path, "a+");
      if (*target_file == NULL) {
        errnum = errno;
        logger_bbob_old_error_io(*target_file, errnum);
      }
      fclose(tmp_file);
    } else { /* either file doesn't exist (new funId) or new Dim */
      /* check that the dim was not already present earlier in the file, if so, create a new info file */
      if (bbob_old_current_dim != logger->number_of_variables) {
        int i, j;
        for (i = 0;
            i < bbob_old_number_of_dimensions && bbob_old_dimensions_in_current_infoFile[i] != 0
                && bbob_old_dimensions_in_current_infoFile[i] != logger->number_of_variables; i++) {
          ; /* checks whether dimension already present in the current infoFile */
        }
        if (i < bbob_old_number_of_dimensions && bbob_old_dimensions_in_current_infoFile[i] == 0) {
          /* new dimension seen for the first time */
          bbob_old_dimensions_in_current_infoFile[i] = logger->number_of_variables;
          newLine = 1;
        } else {
          if (i < bbob_old_number_of_dimensions) { /* dimension already present, need to create a new file */
            newLine = 0;
            file_path[strlen(file_path) - strlen(bbob_old_infoFile_firstInstance_char) - 7] = 0; /* truncate the instance part */
            bbob_old_infoFile_firstInstance = logger->instance_id;
            coco_free_memory(bbob_old_infoFile_firstInstance_char);
            bbob_old_infoFile_firstInstance_char = coco_strdupf("%lu", (unsigned long) bbob_old_infoFile_firstInstance);
            strncat(file_path, "_i", COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, bbob_old_infoFile_firstInstance_char, COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
          } else { /* we have all dimensions */
            newLine = 1;
          }
          for (j = 0; j < bbob_old_number_of_dimensions; j++) { /* new info file, reinitialize list of dims */
            bbob_old_dimensions_in_current_infoFile[j] = 0;
          }
          bbob_old_dimensions_in_current_infoFile[i] = logger->number_of_variables;
        }
      } else {
        if ( bbob_old_current_funId != logger->function_id ) {
          /*new function in the same file */
          newLine = 1;
        }
      }
      *target_file = fopen(file_path, "a+"); /* in any case, we append */
      if (*target_file == NULL) {
        errnum = errno;
        logger_bbob_old_error_io(*target_file, errnum);
      }
      if (tmp_file) { /* File already exists, new dim so just a new line. Also, close the tmp_file */
        if (newLine) {
          fprintf(*target_file, "\n");
        }
        fclose(tmp_file);
      }
      /* data_format = coco_strdup("bbob-constrained"); */
      fprintf(*target_file,
              "suite = '%s', funcId = %d, DIM = %lu, Precision = %.3e, algId = '%s', coco_version = '%s', logger = '%s', data_format = '%s'\n",
              suite_name,
              (int) strtol(function_id, NULL, 10),
              (unsigned long) logger->number_of_variables,
              pow(10, -8),
              logger->observer->algorithm_name,
              coco_version,
              logger_name,
              data_format);

      fprintf(*target_file, "%%\n");
      strncat(used_dataFile_path, "_i", COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      strncat(used_dataFile_path, bbob_old_infoFile_firstInstance_char,
      COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      fprintf(*target_file, "%s.dat", used_dataFile_path); /* dataFile_path does not have the extension */
      bbob_old_current_dim = logger->number_of_variables;
      bbob_old_current_funId = logger->function_id;
    }
  }
  coco_free_memory(function_id_char);
}

/**
 * Generates the different files and folder needed by the logger to store the
 * data if these don't already exist
 */
static void logger_bbob_old_initialize(logger_bbob_old_data_t *logger, coco_problem_t *inner_problem) {
  /*
   Creates/opens the data and index files
   */
  char dataFile_path[COCO_PATH_MAX + 2] = { 0 }; /* relative path to the .dat file from where the .info file is */
  char folder_path[COCO_PATH_MAX + 2] = { 0 };
  char *tmpc_funId; /* serves to extract the function id as a char *. There should be a better way of doing this! */
  char *tmpc_dim; /* serves to extract the dimension as a char *. There should be a better way of doing this! */
  char indexFile_prefix[10] = "bbobexp"; /* TODO (minor): make the prefix bbobexp a parameter that the user can modify */

  assert(logger != NULL);
  assert(inner_problem != NULL);
  assert(inner_problem->problem_id != NULL);

  tmpc_funId = coco_strdupf("%lu", (unsigned long) coco_problem_get_suite_dep_function(inner_problem));
  tmpc_dim = coco_strdupf("%lu", (unsigned long) inner_problem->number_of_variables);

  /* prepare paths and names */
  strncpy(dataFile_path, "data_f", COCO_PATH_MAX);
  strncat(dataFile_path, tmpc_funId,
  COCO_PATH_MAX - strlen(dataFile_path) - 1);
  coco_join_path(folder_path, sizeof(folder_path), logger->observer->result_folder, dataFile_path,
  NULL);
  coco_create_directory(folder_path);
  strncat(dataFile_path, "/bbobexp_f",
  COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_funId,
  COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, "_DIM", COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_dim, COCO_PATH_MAX - strlen(dataFile_path) - 1);

  /* index/info file */
  assert(coco_problem_get_suite(inner_problem));
  logger_bbob_old_openIndexFile(logger, logger->observer->result_folder, indexFile_prefix, tmpc_funId,
      dataFile_path, coco_problem_get_suite(inner_problem)->suite_name);
  fprintf(logger->index_file, ", %lu", (unsigned long) coco_problem_get_suite_dep_instance(inner_problem));
  /* data files */
  /* TODO: definitely improvable but works for now */
  strncat(dataFile_path, "_i", COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, bbob_old_infoFile_firstInstance_char,
  COCO_PATH_MAX - strlen(dataFile_path) - 1);

  logger_bbob_old_open_dataFile(&(logger->fdata_file), logger->observer->result_folder, dataFile_path, ".dat");
  fprintf(logger->fdata_file, bbob_old_file_header_str, logger->optimal_fvalue);

  logger_bbob_old_open_dataFile(&(logger->tdata_file), logger->observer->result_folder, dataFile_path, ".tdat");
  fprintf(logger->tdata_file, bbob_old_file_header_str, logger->optimal_fvalue);

  logger_bbob_old_open_dataFile(&(logger->rdata_file), logger->observer->result_folder, dataFile_path, ".rdat");
  fprintf(logger->rdata_file, bbob_old_file_header_str, logger->optimal_fvalue);
  logger->is_initialized = 1;
  coco_free_memory(tmpc_dim);
  coco_free_memory(tmpc_funId);
  coco_free_memory(bbob_old_infoFile_firstInstance_char);
}

/**
 * Layer added to the transformed-problem evaluate_function by the logger
 */
static void logger_bbob_old_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double y_logged, max_fvalue, sum_cons;
  double *cons = NULL;
  logger_bbob_old_data_t *logger = (logger_bbob_old_data_t *) coco_problem_transformed_get_data(problem);
  coco_problem_t *inner_problem = coco_problem_transformed_get_inner_problem(problem);
  const int is_feasible = problem->number_of_constraints <= 0
                            || coco_is_feasible(inner_problem, x, NULL);

  if (!logger->is_initialized) {
    logger_bbob_old_initialize(logger, inner_problem);
  }
  if ((coco_log_level >= COCO_DEBUG) && logger->number_of_evaluations == 0) {
    coco_debug("%4lu: ", (unsigned long) inner_problem->suite_dep_index);
    coco_debug("on problem %s ... ", coco_problem_get_id(inner_problem));
  }

  coco_evaluate_function(inner_problem, x, y); /* fulfill contract as "being" a coco evaluate function */

  logger->number_of_evaluations_constraints = coco_problem_get_evaluations_constraints(problem);
  logger->number_of_evaluations++; /* could be != coco_problem_get_evaluations(problem) for non-anytime logging? */
  logger->written_last_eval = 0; /* flag whether the current evaluation was logged? */
  logger->last_fvalue = y[0]; /* asma: should be: max(y[0], logger->optimal_fvalue) */

  y_logged = y[0];
  if (coco_is_nan(y_logged))
    y_logged = fvalue_logged_for_nan;
  else if (coco_is_inf(y_logged))
    y_logged = fvalue_logged_for_infinite;
  /* do sanity check */
  if (is_feasible)  /* infeasible solutions can have much better y0 values */
    assert(y_logged + 1e-13 >= logger->optimal_fvalue);

  /* Evaluate the constraints */
  if (problem->number_of_constraints > 0) {
    cons = coco_allocate_vector(problem->number_of_constraints);
    inner_problem->evaluate_constraint(inner_problem, x, cons, 0);
  }

  /* Compute the sum of positive constraint values */
  sum_cons = 0;
  for (i = 0; i < problem->number_of_constraints; ++i) {
    if (cons[i] > 0)
        sum_cons += cons[i];
  }
  sum_cons *= weight_constraints;  /* do this before the checks */
  if (coco_is_nan(sum_cons))
    sum_cons = fvalue_logged_for_nan;
  else if (coco_is_inf(sum_cons))
    sum_cons = fvalue_logged_for_infinite;

  max_fvalue =  y_logged > logger->optimal_fvalue ? y_logged : logger->optimal_fvalue;

  /* Update logger state.
   *   at logger->number_of_evaluations == 1 the logger->best_fvalue is not initialized,
   *   also compare to y_logged to not potentially be thrown off by weird values in y[0]
   */
  if (logger->number_of_evaluations == 1 || (max_fvalue + sum_cons < logger->best_fvalue)) {
    logger->best_fvalue = max_fvalue + sum_cons;
    for (i = 0; i < problem->number_of_variables; i++)
      logger->best_solution[i] = x[i]; /* may well be infeasible */

    /* Add a line in the .dat file for each logging target reached
     * by a feasible solution and always at evaluation one
     */
    if (logger->number_of_evaluations == 1 || coco_observer_log_targets_trigger(logger->targets,
                                        logger->best_fvalue - logger->optimal_fvalue)) {
      logger_bbob_old_write_data(
          logger->fdata_file,
          logger->number_of_evaluations,
          logger->number_of_evaluations_constraints,
          y_logged,
          logger->best_fvalue,
          logger->optimal_fvalue,
          x,
          problem->number_of_variables,
          problem->number_of_integer_variables,
          cons,
          problem->number_of_constraints,
          logger->log_discrete_as_int);
    }
  }

  /* Add a line in the .tdat file each time an fevals trigger is reached.*/
  if (coco_observer_evaluations_trigger(logger->evaluations,
        logger->number_of_evaluations + logger->number_of_evaluations_constraints)) {
    logger_bbob_old_write_data(
        logger->tdata_file,
        logger->number_of_evaluations,
        logger->number_of_evaluations_constraints,
        y_logged,
        logger->best_fvalue,
        logger->optimal_fvalue,
        x,
        problem->number_of_variables,
        problem->number_of_integer_variables,
        cons,
        problem->number_of_constraints,
        logger->log_discrete_as_int);
    logger->written_last_eval = 1;
  }

  /* Free allocated memory */
  if (problem->number_of_constraints > 0)
    coco_free_memory(cons);

}  /* end logger_bbob_old_evaluate */

/**
 * Also serves as a finalize run method so. Must be called at the end
 * of Each run to correctly fill the index file
 *
 * TODO: make sure it is called at the end of each run or move the
 * writing into files to another function
 */
static void logger_bbob_old_free(void *stuff) {
  /* TODO: do all the "non simply freeing" stuff in another function
   * that can have problem as input
   */
  logger_bbob_old_data_t *logger = (logger_bbob_old_data_t *) stuff;

  if (stuff == NULL)
    return;

  if ((coco_log_level >= COCO_DEBUG) && logger && logger->number_of_evaluations > 0) {
    coco_debug("best f=%e after %lu fevals (done observing)\n", logger->best_fvalue,
		(unsigned long) logger->number_of_evaluations);
  }
  if (logger->index_file != NULL) {
    fprintf(logger->index_file, ":%lu|%.1e",
            (unsigned long) logger->number_of_evaluations,
            logger->best_fvalue - logger->optimal_fvalue);
    fclose(logger->index_file);
    logger->index_file = NULL;
  }
  if (logger->fdata_file != NULL) {
    fclose(logger->fdata_file);
    logger->fdata_file = NULL;
  }
  if (logger->tdata_file != NULL) {
    /* TODO: make sure it handles restarts well. i.e., it writes
     * at the end of a single run, not all the runs on a given
     * instance. Maybe start with forcing it to generate a new
     * "instance" of problem for each restart in the beginning
     */
    if (!logger->written_last_eval) {
      logger_bbob_old_write_data(logger->tdata_file,
          logger->number_of_evaluations,
          logger->number_of_evaluations_constraints,
          logger->best_fvalue,
          logger->best_fvalue,
          logger->optimal_fvalue,
          logger->best_solution,
          logger->number_of_variables,
          logger->number_of_integer_variables,
          NULL,
          0,
          logger->log_discrete_as_int);
	}
    fclose(logger->tdata_file);
    logger->tdata_file = NULL;
  }

  if (logger->rdata_file != NULL) {
    fclose(logger->rdata_file);
    logger->rdata_file = NULL;
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

  bbob_old_logger_is_open = 0;
}

static coco_problem_t *logger_bbob_old(coco_observer_t *observer, coco_problem_t *inner_problem) {
  logger_bbob_old_data_t *logger_data;
  coco_problem_t *problem;

  logger_data = (logger_bbob_old_data_t *) coco_allocate_memory(sizeof(*logger_data));
  logger_data->observer = observer;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_bbob_old(): The bbob_old logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }

  if (bbob_old_logger_is_open)
    coco_error("The current bbob_old_logger (observer) must be closed before a new one is opened");
  /* This is the name of the folder which happens to be the algName */
  /*logger->path = coco_strdup(observer->output_folder);*/
  logger_data->index_file = NULL;
  logger_data->fdata_file = NULL;
  logger_data->tdata_file = NULL;
  logger_data->rdata_file = NULL;
  logger_data->number_of_variables = inner_problem->number_of_variables;
  logger_data->number_of_integer_variables = inner_problem->number_of_integer_variables;
  if (inner_problem->best_value == NULL) {
    /* coco_error("Optimal f value must be defined for each problem in order for the logger to work properly"); */
    /* Setting the value to 0 results in the assertion y>=optimal_fvalue being susceptible to failure */
    coco_warning("undefined optimal f value. Set to 0");
    logger_data->optimal_fvalue = 0;
  } else {
    logger_data->optimal_fvalue = *(inner_problem->best_value);
  }

  logger_data->number_of_evaluations = 0;
  logger_data->number_of_evaluations_constraints = 0;
  logger_data->best_solution = coco_allocate_vector(inner_problem->number_of_variables);
  /* TODO: the following inits are just to be in the safe side and
   * should eventually be removed. Some fields of the bbob_old_logger struct
   * might be useless
   */
  logger_data->function_id = coco_problem_get_suite_dep_function(inner_problem);
  logger_data->instance_id = coco_problem_get_suite_dep_instance(inner_problem);
  logger_data->written_last_eval = 0;
  logger_data->last_fvalue = DBL_MAX;
  logger_data->is_initialized = 0;
  logger_data->log_discrete_as_int = observer->log_discrete_as_int;

  /* Initialize triggers based on target values and number of evaluations */
  logger_data->targets = coco_observer_log_targets(observer->number_target_triggers, observer->log_target_precision);
  logger_data->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_bbob_old_free, observer->observer_name);

  problem->evaluate_function = logger_bbob_old_evaluate;
  bbob_old_logger_is_open = 1;
  return problem;
}
