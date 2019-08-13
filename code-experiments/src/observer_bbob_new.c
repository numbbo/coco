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
  size_t current_dim;                      /**< @brief Current dimension */
  size_t current_fun;                      /**< @brief Current function */
  char *prefix;                            /**< @brief Prefix in the info files */
  int logger_is_used;                      /**< @brief Whether the logger is already used on a problem */
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
  coco_debug("Ended   observer_bbob_new_data_free()");
}

/**
 * @brief Initializes the bbob observer.
 */
static void observer_bbob_new(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_bbob_new_data_t *observer_data;
  /* Sets the valid keys for bbob observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation TODO */
  const char *known_keys[] = { "prefix" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_bbob_new_data_t *) coco_allocate_memory(sizeof(*observer_data));
  observer_data->current_dim = 0;
  observer_data->current_fun = 0;
  observer_data->prefix = coco_allocate_string(COCO_PATH_MAX + 1);
  observer_data->logger_is_used = 0;

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
