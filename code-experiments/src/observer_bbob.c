/**
 * @file observer_bbob.c
 * @brief Implementation of the bbob observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_bbob(coco_observer_t *observer, coco_problem_t *problem);
static void logger_bbob_free(void *logger);
static void logger_bbob_signal_restart(coco_problem_t *problem);
static void logger_bbob_data_nullify_observer(void *logger_data);

/**
 * @brief The bbob observer data type.
 *
 * There is a cyclic reference between the bbob_logger and the bbob_observer (through the observer's
 * observed_problem and the logger's data, which points to the observer). This is needed to be able
 * to free both objects without problems.
 */
typedef struct {
  coco_problem_t *observed_problem;  /**< @brief Pointer to the observed problem (NULL if none is observed) */
  char *prefix;                      /**< @brief Prefix in the name of the info and data files */

  /* Store the information on the last function, dimension and dat file used, to be able to tell when the 
   * logger needs to write in a new dat file.  */
  size_t last_function;             /**< @brief The function that was logged last */
  size_t last_dimension;            /**< @brief The dimension that was logged last */
  char *last_dat_file;              /**< @brief The name of the .dat file that was last used for logging */
} observer_bbob_data_t;

/**
 * @brief  Frees the memory of the given observer_bbob_data_t object.
 */
static void observer_bbob_data_free(void *stuff) {

  observer_bbob_data_t *data = (observer_bbob_data_t *) stuff;
  coco_problem_t *problem;

  coco_debug("Started observer_bbob_data_free()");

  if (data->prefix != NULL) {
    coco_free_memory(data->prefix);
    data->prefix = NULL;
  }

  if (data->last_dat_file != NULL) {
    coco_free_memory(data->last_dat_file);
    data->last_dat_file = NULL;
  }

  /* Make sure that the observed problem's pointer to the observer points to NULL */
  if (data->observed_problem != NULL) {
    problem = (coco_problem_t *) data->observed_problem;
    if (problem->data != NULL) {
      logger_bbob_data_nullify_observer(coco_problem_transformed_get_data(problem));
    }
    data->observed_problem = NULL;
  }

  coco_debug("Ended   observer_bbob_data_free()");
}

/**
 * @brief Initializes the bbob observer.
 *
 * Possible options:
 *
 * - "prefix: STRING" defines the prefix of the name of the info files. The default value is "bbobex".
 */
static void observer_bbob(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_bbob_data_t *observer_data;
  /* Sets the valid keys for bbob observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "prefix" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_bbob_data_t *) coco_allocate_memory(sizeof(*observer_data));
  observer_data->observed_problem = NULL;
  observer_data->prefix = coco_allocate_string(COCO_PATH_MAX + 1);

  observer_data->last_function = 0;      
  observer_data->last_dimension = 0; 
  observer_data->last_dat_file = coco_allocate_string(COCO_PATH_MAX + 1); 
  strncpy(observer_data->last_dat_file, "init", COCO_PATH_MAX);

  if (coco_options_read_string(options, "prefix", observer_data->prefix) == 0) {
    strcpy(observer_data->prefix, "bbobexp");
  }

  observer->logger_allocate_function = logger_bbob;
  observer->logger_free_function = logger_bbob_free;
  observer->restart_function = logger_bbob_signal_restart;
  observer->data_free_function = observer_bbob_data_free;
  observer->data = observer_data;

  (void) options; /* To silence the compiler */
}
