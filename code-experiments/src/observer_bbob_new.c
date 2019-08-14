/**
 * @file observer_bbob.c
 * @brief Implementation of the bbob observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_bbob_new(coco_observer_t *observer, coco_problem_t *problem);
static void logger_bbob_new_free(void *logger);

/**
 * @brief The bbob observer data type.
 */
typedef struct {
  int logger_is_used;            /**< @brief Whether the logger is already used on a problem */
  char *prefix;                  /**< @brief Prefix in the name of the info and data files */

  /* last_dimensions and functions_array have the same number of values - corresponding to the same function index
   * While functions_array can contain duplicate values, only the first occurrences count */
  size_t num_functions;          /**< @brief The number of all functions in the suite */
  size_t *last_dimensions;       /**< @brief The dimension that was last used for the function index */
  size_t *functions_array;       /**< @brief The function number that corresponds to the function index */
} observer_bbob_new_data_t;

/**
 * @brief  Frees the memory of the given observer_bbob_new_data_t object.
 */
static void observer_bbob_new_data_free(void *stuff) {

  observer_bbob_new_data_t *data = (observer_bbob_new_data_t *) stuff;

  coco_debug("Started observer_bbob_new_data_free()");

  if (data->prefix != NULL) {
    coco_free_memory(data->prefix);
    data->prefix = NULL;
  }

  if (data->last_dimensions != NULL) {
    coco_free_memory(data->last_dimensions);
    data->last_dimensions = NULL;
  }

  if (data->functions_array != NULL) {
    coco_free_memory(data->functions_array);
    data->functions_array = NULL;
  }

  coco_debug("Ended   observer_bbob_new_data_free()");
}

/**
 * @brief Initializes the bbob_new observer.
 *
 * Possible options:
 *
 * - "prefix: STRING" defines the prefix of the name of the info files. The default value is "bbobex".
 */
static void observer_bbob_new(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_bbob_new_data_t *observer_data;
  /* Sets the valid keys for bbob_new observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation TODO */
  const char *known_keys[] = { "prefix", "unif_target_trigger" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_bbob_new_data_t *) coco_allocate_memory(sizeof(*observer_data));
  observer_data->logger_is_used = 0;
  observer_data->prefix = coco_allocate_string(COCO_PATH_MAX + 1);

  observer_data->num_functions = 0;      /* Needs to be initialized when the suite is known */
  observer_data->last_dimensions = NULL; /* Needs to be allocated when the suite is known */
  observer_data->functions_array = NULL; /* Needs to be allocated when the suite is known */

  if (coco_options_read_string(options, "prefix", observer_data->prefix) == 0) {
    strcpy(observer_data->prefix, "bbobexp");
  }

  observer->logger_allocate_function = logger_bbob_new;
  observer->logger_free_function = logger_bbob_new_free;
  observer->data_free_function = observer_bbob_new_data_free;
  observer->data = observer_data;

  *option_keys = NULL;

  (void) options; /* To silence the compiler */
}
