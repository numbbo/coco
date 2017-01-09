/**
 * @file logger_bbob.c
 * @brief Implementation of the bbob logger.
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
#include "observer_bbob.c"

/* The current_... mechanism fails if several problems are open.
 * For the time being this should lead to an error.
 *
 * A possible solution: bbob_logger_is_open becomes a reference
 * counter and as long as another logger is open, always a new info
 * file is generated.
 * TODO: Shouldn't the new way of handling observers already fix this? 
 *    logger_is_open still triggers when a python run from the python shell is interrupted and a new one is lunched
 */

/* TODO: add possibility of adding a prefix to the index files (easy to do through observer options) */

/**
 * @brief The bbob logger data type.
 */
typedef struct {
  coco_observer_t *observer;
  int is_initialized;
  FILE *index_file; /**< @brief index file */
  FILE *fdata_file; /**< @brief function value aligned data file */
  FILE *tdata_file; /**< @brief number of function evaluations aligned data file */
  FILE *rdata_file; /**< @brief restart info data file */
  size_t number_of_evaluations;
  double best_fvalue;
  double last_fvalue;
  short written_last_eval; /**< @brief allows writing the the data of the final fun eval in the .tdat file if not already written by the t_trigger*/
  double *best_solution;
  size_t function_id;
  size_t instance_id;
  size_t number_of_variables;
  double optimal_fvalue;

  coco_observer_targets_t *targets;          /**< @brief Triggers based on target values. */
  coco_observer_evaluations_t *evaluations;  /**< @brief Triggers based on the number of evaluations. */

} logger_bbob_data_t;

static const char *bbob_file_header_str = "%% function evaluation | "
    "noise-free fitness - Fopt (%13.12e) | "
    "best noise-free fitness - Fopt | "
    "measured fitness | "
    "best measured fitness | "
    "x1 | "
    "x2...\n";

/**
 * @brief adds a formated line to a data file
 */
static void logger_bbob_write_data(FILE *target_file,
                                   size_t number_of_evaluations,
                                   double fvalue,
                                   double best_fvalue,
                                   double best_value,
                                   const double *x,
                                   size_t number_of_variables,
                                   const int precision_f,
                                   const int precision_x) {
  
  fprintf(target_file, "%lu %.*e %.*e %.*e %.*e", number_of_evaluations, precision_f, fvalue - best_value,
          precision_f, best_fvalue - best_value, precision_f, fvalue, precision_f, best_fvalue);
  if (number_of_variables < 22) {
    size_t i;
    for (i = 0; i < number_of_variables; i++) {
      fprintf(target_file, " %.*e", precision_x, x[i]);
    }
  }
  fprintf(target_file, "\n");
}

/**
 * @brief Error when trying to create the file "path"
 */
static void logger_bbob_error_io(FILE *path, int errnum) {
  const char *error_format = "Error opening file: %s\n ";
  coco_error(error_format, strerror(errnum), path);
}

/**
 * @brief Creates the data file or simply opens it
 */
static void logger_bbob_open_dataFile(FILE **target_file,
                                      const char *path,
                                      const char *dataFile_path,
                                      const char *file_extension) {
  char file_path[COCO_PATH_MAX] = { 0 };
  char relative_filePath[COCO_PATH_MAX] = { 0 };
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
      logger_bbob_error_io(*target_file, errnum);
    }
  }
}

/**
 * @brief Creates the index file fileName_prefix+problem_id+file_extension in
 *        folde_path
 */
static void logger_bbob_openIndexFile(logger_bbob_data_t *logger,
                                      const char *folder_path,
                                      const char *indexFile_prefix,
                                      const char *function_id,
                                      const char *dataFile_path,
                                      const char *suite_name) {
  /* to add the instance number TODO: this should be done outside to avoid redoing this for the .*dat files */
  observer_bbob_data_t *observer_bbob;
  char bbob_infoFile_firstInstance_char[3];
  char used_dataFile_path[COCO_PATH_MAX] = { 0 };
  int errnum, newLine; /* newLine is at 1 if we need a new line in the info file */
  char function_id_char[3]; /* TODO: consider adding them to logger */
  char file_name[COCO_PATH_MAX] = { 0 };
  char file_path[COCO_PATH_MAX] = { 0 };
  FILE **target_file;
  FILE *tmp_file;
  observer_bbob = (observer_bbob_data_t *) logger->observer->data;
  strncpy(used_dataFile_path, dataFile_path, COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
  if (observer_bbob->info_file_first_instance == 0) {
    observer_bbob->info_file_first_instance = logger->instance_id;
  }
  sprintf(function_id_char, "%lu", (unsigned long) logger->function_id);
  sprintf(bbob_infoFile_firstInstance_char, "%lu", (unsigned long) observer_bbob->info_file_first_instance);
  target_file = &(logger->index_file);
  tmp_file = NULL; /* to check whether the file already exists. Don't want to use target_file */
  strncpy(file_name, indexFile_prefix, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, "_f", COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, function_id_char, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, "_i", COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, bbob_infoFile_firstInstance_char, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
  coco_join_path(file_path, sizeof(file_path), folder_path, file_name, NULL);
  if (*target_file == NULL) {
    tmp_file = fopen(file_path, "r"); /* to check for existence */
    if ((tmp_file) && (observer_bbob->current_dim == logger->number_of_variables)
        && (observer_bbob->current_fun_id == logger->function_id)) {
        /* new instance of current funId and current dim */
      newLine = 0;
      *target_file = fopen(file_path, "a+");
      if (*target_file == NULL) {
        errnum = errno;
        logger_bbob_error_io(*target_file, errnum);
      }
      fclose(tmp_file);
    } else { /* either file doesn't exist (new fun_id) or new Dim */
      /* check that the dim was not already present earlier in the file, if so, create a new info file */
      if (observer_bbob->current_dim != logger->number_of_variables) {
        int i, j;
        for (i = 0;
            i < observer_bbob->number_of_dimensions && observer_bbob->dimensions_in_current_info_file[i] != 0
                && observer_bbob->dimensions_in_current_info_file[i] != logger->number_of_variables; i++) {
          ; /* checks whether dimension already present in the current infoFile */
        }

        if (i < observer_bbob->number_of_dimensions && observer_bbob->dimensions_in_current_info_file[i] == 0) {
          /* new dimension seen for the first time */
          observer_bbob->dimensions_in_current_info_file[i] = logger->number_of_variables;
          newLine = 1;
        } else {
          if (i < observer_bbob->number_of_dimensions) { /* dimension already present, need to create a new file */
            newLine = 0;
            file_path[strlen(file_path) - strlen(bbob_infoFile_firstInstance_char) - 7] = 0; /* truncate the instance part */
            observer_bbob->info_file_first_instance = logger->instance_id;
            sprintf(bbob_infoFile_firstInstance_char, "%lu", (unsigned long) observer_bbob->info_file_first_instance);
            strncat(file_path, "_i", COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, bbob_infoFile_firstInstance_char, COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
          } else {/*we have all dimensions*/
            newLine = 1;
          }
          for (j = 0; j < observer_bbob->number_of_dimensions; j++) { /* new info file, reinitialize list of dims */
            observer_bbob->dimensions_in_current_info_file[j] = 0;
          }
          observer_bbob->dimensions_in_current_info_file[i] = logger->number_of_variables;
        }
      } else {
        if ( observer_bbob->current_fun_id != logger->function_id ) {
          /*new function in the same file */
          newLine = 1;
        }
      }
      *target_file = fopen(file_path, "a+"); /* in any case, we append */
      if (*target_file == NULL) {
        errnum = errno;
        logger_bbob_error_io(*target_file, errnum);
      }
      if (tmp_file) { /* File already exists, new dim so just a new line. Also, close the tmp_file */
        if (newLine) {
          fprintf(*target_file, "\n");
        }
        fclose(tmp_file);
      }


      fprintf(*target_file,
          "suite = '%s', funcId = %d, DIM = %lu, Precision = %.3e, algId = '%s', coco_version = '%s'\n",
          suite_name, (int) strtol(function_id, NULL, 10), (unsigned long) logger->number_of_variables,
          pow(10, -8), logger->observer->algorithm_name, coco_version);

      fprintf(*target_file, "%%\n");
      strncat(used_dataFile_path, "_i", COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      strncat(used_dataFile_path, bbob_infoFile_firstInstance_char,
      COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      fprintf(*target_file, "%s.dat", used_dataFile_path); /* dataFile_path does not have the extension */
      observer_bbob->current_dim = logger->number_of_variables;
      observer_bbob->current_fun_id = logger->function_id;
    }
  }
}

/**
 * Generates the different files and folder needed by the logger to store the
 * data if these don't already exist
 */
static void logger_bbob_initialize(logger_bbob_data_t *logger, coco_problem_t *inner_problem) {
  /*
   Creates/opens the data and index files
   */
  observer_bbob_data_t *observer_bbob;
  char bbob_infoFile_firstInstance_char[3];
  char dataFile_path[COCO_PATH_MAX] = { 0 }; /* relative path to the .dat file from where the .info file is */
  char folder_path[COCO_PATH_MAX] = { 0 };
  char *tmpc_funId; /* serves to extract the function id as a char *. There should be a better way of doing this! */
  char *tmpc_dim; /* serves to extract the dimension as a char *. There should be a better way of doing this! */
  char indexFile_prefix[10] = "bbobexp"; /* TODO (minor): make the prefix bbobexp a parameter that the user can modify */
  size_t str_length_funId, str_length_dim;
  observer_bbob = (observer_bbob_data_t *)logger->observer->data;
  str_length_funId = coco_double_to_size_t(bbob2009_fmax(1, ceil(log10((double) coco_problem_get_suite_dep_function(inner_problem)))));
  str_length_dim = coco_double_to_size_t(bbob2009_fmax(1, ceil(log10((double) inner_problem->number_of_variables))));
  tmpc_funId = coco_allocate_string(str_length_funId);
  tmpc_dim = coco_allocate_string(str_length_dim);

  assert(logger != NULL);
  assert(inner_problem != NULL);
  assert(inner_problem->problem_id != NULL);

  sprintf(tmpc_funId, "%lu", (unsigned long) coco_problem_get_suite_dep_function(inner_problem));
  sprintf(tmpc_dim, "%lu", (unsigned long) inner_problem->number_of_variables);

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
  logger_bbob_openIndexFile(logger, logger->observer->result_folder, indexFile_prefix, tmpc_funId,
      dataFile_path, coco_problem_get_suite(inner_problem)->suite_name);
  fprintf(logger->index_file, ", %lu", (unsigned long) coco_problem_get_suite_dep_instance(inner_problem));
  /* data files */
  /* TODO: definitely improvable but works for now */
  strncat(dataFile_path, "_i", COCO_PATH_MAX - strlen(dataFile_path) - 1);
  sprintf(bbob_infoFile_firstInstance_char, "%ld", observer_bbob->info_file_first_instance);
  strncat(dataFile_path, bbob_infoFile_firstInstance_char,
          COCO_PATH_MAX - strlen(dataFile_path) - 1);
  logger_bbob_open_dataFile(&(logger->fdata_file), logger->observer->result_folder, dataFile_path, ".dat");
  fprintf(logger->fdata_file, bbob_file_header_str, logger->optimal_fvalue);

  logger_bbob_open_dataFile(&(logger->tdata_file), logger->observer->result_folder, dataFile_path, ".tdat");
  fprintf(logger->tdata_file, bbob_file_header_str, logger->optimal_fvalue);

  logger_bbob_open_dataFile(&(logger->rdata_file), logger->observer->result_folder, dataFile_path, ".rdat");
  fprintf(logger->rdata_file, bbob_file_header_str, logger->optimal_fvalue);
  logger->is_initialized = 1;
  coco_free_memory(tmpc_dim);
  coco_free_memory(tmpc_funId);
}

/**
 * Layer added to the transformed-problem evaluate_function by the logger
 */
static void logger_bbob_evaluate(coco_problem_t *problem, const double *x, double *y) {
  logger_bbob_data_t *logger = (logger_bbob_data_t *) coco_problem_transformed_get_data(problem);
  coco_problem_t * inner_problem = coco_problem_transformed_get_inner_problem(problem);

  size_t i;

  if (!logger->is_initialized) {
    logger_bbob_initialize(logger, inner_problem);
  }
  if ((coco_log_level >= COCO_DEBUG) && logger->number_of_evaluations == 0) {
    coco_debug("%4lu: ", (unsigned long) inner_problem->suite_dep_index);
    coco_debug("on problem %s ... ", coco_problem_get_id(inner_problem));
  }
  coco_evaluate_function(inner_problem, x, y);
  logger->last_fvalue = y[0];
  logger->written_last_eval = 0;
  if (logger->number_of_evaluations == 0 || y[0] < logger->best_fvalue) {
    logger->best_fvalue = y[0];
    for (i = 0; i < problem->number_of_variables; i++)
      logger->best_solution[i] = x[i];
  }
  logger->number_of_evaluations++;

  /* Add sanity check for optimal f value */
  assert(y[0] + 1e-13 >= logger->optimal_fvalue);

  /* Add a line in the .dat file for each logging target reached. */
  if (coco_observer_targets_trigger(logger->targets, y[0] - logger->optimal_fvalue)) {

      logger_bbob_write_data(logger->fdata_file, logger->number_of_evaluations, y[0], logger->best_fvalue,
        logger->optimal_fvalue, x, problem->number_of_variables, logger->observer->precision_f, logger->observer->precision_x);
  }

  /* Add a line in the .tdat file each time an fevals trigger is reached.*/
  if (coco_observer_evaluations_trigger(logger->evaluations, logger->number_of_evaluations)) {
    logger->written_last_eval = 1;
    logger_bbob_write_data(logger->tdata_file, logger->number_of_evaluations, y[0], logger->best_fvalue,
        logger->optimal_fvalue, x, problem->number_of_variables, logger->observer->precision_f, logger->observer->precision_x);
  }

  /* Flush output so that impatient users can see progress. */
  fflush(logger->fdata_file);
}

/**
 * @brief Finalize function of the logger, writes final data to the index file
 */
static void logger_bbob_finalize(const logger_bbob_data_t *logger){

  if ((coco_log_level >= COCO_DEBUG) && logger && logger->number_of_evaluations > 0) {
    coco_debug("best f=%e after %ld fevals (done observing)\n", logger->best_fvalue,
              (unsigned long) logger->number_of_evaluations);
  }
  /* log the final information of the run in the info file*/
  /*fprintf(logger->index_file, ":%ld|%.1e", logger->number_of_evaluations,
          logger->best_fvalue - logger->optimal_fvalue);*/

  /* log the last evaluation (if not logged) in the *.tdata file*/
  if (!logger->written_last_eval) {
    logger_bbob_write_data(logger->tdata_file, logger->number_of_evaluations, logger->last_fvalue,
                           logger->best_fvalue, logger->optimal_fvalue, logger->best_solution, logger->number_of_variables, logger->observer->precision_f, logger->observer->precision_x);
  }

  /* let the observer know that the logger is closed */
  if (logger->observer->data != NULL) {
    /*the observer data seems to be freed before the logger in the last run!*/
    ((observer_bbob_data_t *) logger->observer->data)->logger_is_open = 0;
  }
}


/**
 * @brief calls the finalize functions then frees the logger
 */
static void logger_bbob_free(void *stuff) {

  logger_bbob_data_t *logger = (logger_bbob_data_t *) stuff;

  logger_bbob_finalize(logger);
  if (logger->index_file != NULL) {

    fprintf(logger->index_file, ":%lu|%.1e", (unsigned long) logger->number_of_evaluations,
        logger->best_fvalue - logger->optimal_fvalue);
    fclose(logger->index_file); /*Wassim: now done in logger_bbob_finalize*/
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

}

static coco_problem_t *logger_bbob(coco_observer_t *observer, coco_problem_t *inner_problem) {
  logger_bbob_data_t *logger_bbob;
  coco_problem_t *problem;

  logger_bbob = (logger_bbob_data_t *) coco_allocate_memory(sizeof(*logger_bbob));
  logger_bbob->observer = observer;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_bbob(): The bbob logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }
  if (((observer_bbob_data_t *) observer->data)->logger_is_open)
    coco_error("The current bbob_logger (observer) must be closed before a new one is opened");

  logger_bbob->index_file = NULL;
  logger_bbob->fdata_file = NULL;
  logger_bbob->tdata_file = NULL;
  logger_bbob->rdata_file = NULL;
  logger_bbob->number_of_variables = inner_problem->number_of_variables;
  if (inner_problem->best_value == NULL) {
    coco_warning("undefined optimal f value. Set to 0");
    logger_bbob->optimal_fvalue = 0;
  } else {
    logger_bbob->optimal_fvalue = *(inner_problem->best_value);
  }

  logger_bbob->number_of_evaluations = 0;
  logger_bbob->best_solution = coco_allocate_vector(inner_problem->number_of_variables);
  logger_bbob->function_id = coco_problem_get_suite_dep_function(inner_problem);
  logger_bbob->instance_id = coco_problem_get_suite_dep_instance(inner_problem);
  logger_bbob->written_last_eval = 1;
  logger_bbob->last_fvalue = DBL_MAX;
  logger_bbob->is_initialized = 0;

  /* Initialize triggers based on target values and number of evaluations */
  logger_bbob->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  logger_bbob->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, logger_bbob, logger_bbob_free, observer->observer_name);

  problem->evaluate_function = logger_bbob_evaluate;
  ((observer_bbob_data_t *) observer->data)->logger_is_open = 1;
  return problem;
}

