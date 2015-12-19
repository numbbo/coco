#include "coco.h"
#include "coco_internal.h"

/*#include "observer_bbob2009.c"*/
#include "logger_biobj.c"
#include "logger_toy.c"
#include "logger_bbob2009.c" /* Wassim: are these includes needed?! */

/**
 * Allocates memory for a coco_observer_t instance.
 */
static coco_observer_t *coco_observer_allocate(const char *output_folder,
                                               const char *algorithm_name,
                                               const char *algorithm_info,
                                               const int precision_x,
                                               const int precision_f) {

  coco_observer_t *observer;
  observer = (coco_observer_t *) coco_allocate_memory(sizeof(*observer));
  /* Initialize fields to sane/safe defaults */
  observer->output_folder = coco_strdup(output_folder);
  observer->algorithm_name = coco_strdup(algorithm_name);
  observer->algorithm_info = coco_strdup(algorithm_info);
  observer->precision_x = precision_x;
  observer->precision_f = precision_f;
  observer->data = NULL;
  observer->data_free_function = NULL;
  observer->logger_initialize_function = NULL;
  observer->is_active = 1;
  return observer;
}

/**
 * Frees memory for the given coco_observer_t instance.
 */
void coco_observer_free(coco_observer_t *self) {

  if (self != NULL) {
    self->is_active = 0;
    if (self->output_folder != NULL)
      coco_free_memory(self->output_folder);
    if (self->algorithm_name != NULL)
      coco_free_memory(self->algorithm_name);
    if (self->algorithm_info != NULL)
      coco_free_memory(self->algorithm_info);

    if (self->data != NULL) {
      if (self->data_free_function != NULL) {
        self->data_free_function(self->data);
      }
      coco_free_memory(self->data);
      self->data = NULL;
    }

    self->logger_initialize_function = NULL;
    coco_free_memory(self);
    self = NULL;
  }
}

/**
 * Initializes the observer. If observer_name is no_observer, no observer is used.
 * Possible observer_options:
 * - result_folder : string (the name of the result_folder is used to create a unique folder; default value
 * is "results")
 * - algorithm_name : string (to be used in logged output and plots; default value is "ALG")
 * - algorithm_info : string (to be used in logged output; default value is "")
 * - log_level : error (only error messages are output)
 * - log_level : warning (only error and warning messages are output; default value)
 * - log_level : info (only error, warning and info messages are output)
 * - log_level : debug (all messages are output)
 * - precision_x : integer value (precision used when outputting variables; default value is 8)
 * - precision_f : integer value (precision used when outputting f values; default value is 16)
 * - any option specified by the specific observers
 */
coco_observer_t *coco_observer(const char *observer_name, const char *observer_options) {

  coco_observer_t *observer;
  char *result_folder, *algorithm_name, *algorithm_info;
  char *log_level;
  int precision_x, precision_f;

  if (0 == strcmp(observer_name, "no_observer")) {
    return NULL;
  } else if (strlen(observer_name) == 0) {
    coco_warning("Empty observer_name has no effect. To prevent this warning use 'no_observer' instead");
    return NULL;
  }

  result_folder = (char *) coco_allocate_memory(COCO_PATH_MAX);
  algorithm_name = (char *) coco_allocate_memory(COCO_PATH_MAX);
  algorithm_info = (char *) coco_allocate_memory(5 * COCO_PATH_MAX);
  /* Read result_folder, algorithm_name and algorithm_info from the observer_options and use
   * them to initialize the observer */
  if (coco_options_read_string(observer_options, "result_folder", result_folder) == 0) {
    strcpy(result_folder, "results");
  }
  coco_create_unique_path(&result_folder);

  if (coco_options_read_string(observer_options, "algorithm_name", algorithm_name) == 0) {
    strcpy(algorithm_name, "ALG");
  }

  if (coco_options_read_string(observer_options, "algorithm_info", algorithm_info) == 0) {
    strcpy(algorithm_info, "");
  }

  precision_x = 8;
  if (coco_options_read_int(observer_options, "precision_x", &precision_x) != 0) {
    if ((precision_x < 1) || (precision_x > 32))
      precision_x = 8;
  }

  precision_f = 16;
  if (coco_options_read_int(observer_options, "precision_f", &precision_f) != 0) {
    if ((precision_f < 1) || (precision_f > 32))
      precision_f = 16;
  }

  observer = coco_observer_allocate(result_folder, algorithm_name, algorithm_info, precision_x, precision_f);

  coco_free_memory(result_folder);
  coco_free_memory(algorithm_name);
  coco_free_memory(algorithm_info);

  /* Update the coco_log_level */
  log_level = (char *) coco_allocate_memory(COCO_PATH_MAX);
  if (coco_options_read_string(observer_options, "log_level", log_level) > 0) {
    if (strcmp(log_level, "error") == 0)
      coco_log_level = COCO_ERROR;
    else if (strcmp(log_level, "warning") == 0)
      coco_log_level = COCO_WARNING;
    else if (strcmp(log_level, "info") == 0)
      coco_log_level = COCO_INFO;
    else if (strcmp(log_level, "debug") == 0)
      coco_log_level = COCO_DEBUG;
  }
  coco_free_memory(log_level);

  /* Here each observer must have an entry */
  if (0 == strcmp(observer_name, "observer_toy")) {
    observer_toy(observer, observer_options);
  } else if (0 == strcmp(observer_name, "observer_bbob")) {
    observer_bbob2009(observer, observer_options);
  } else if (0 == strcmp(observer_name, "observer_biobj")) {
    observer_biobj(observer, observer_options);
  } else {
    coco_warning("Unknown observer!");
    return NULL;
  }

  return observer;
}

/**
 * Adds the observer to the problem if the observer is not NULL (invokes initialization of the
 * corresponding logger).
 */
coco_problem_t *coco_problem_add_observer(coco_problem_t *problem, coco_observer_t *observer) {

  if ((observer == NULL) || (observer->is_active == 0)) {
    coco_warning("The problem is not being observed.");
    return problem;
  }

  return observer->logger_initialize_function(observer, problem);
}

