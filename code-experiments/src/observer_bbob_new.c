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
  /* TODO: Can be used to store variables that need to be accessible during one run (i.e. for multiple
   * problems). For example, the following global variables from logger_bbob.c could be stored here: */
  size_t current_dim;
  size_t current_fun_id;
  size_t info_first_instance;
  char *info_first_instance_char;
  /* ... and others */
} observer_bbob_new_data_t;

/**
 * @brief  Frees the memory of the given observer_bbob_new_data_t object.
 */
static void observer_bbob_new_data_free(void *stuff) {

  observer_bbob_new_data_t *data;

  assert(stuff != NULL);
  data = (observer_bbob_new_data_t *) stuff;

  if (data->info_first_instance_char) {
    coco_free_memory(data->info_first_instance_char);
  }
  data->info_first_instance_char = NULL;
}

/**
 * @brief Initializes the bbob observer.
 */
static void observer_bbob_new(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_bbob_new_data_t *observer_data;
  observer_data = (observer_bbob_new_data_t *) coco_allocate_memory(sizeof(*observer_data));

  observer_data->current_dim = 0;
  observer_data->current_fun_id = 0;
  observer_data->info_first_instance = 0;
  observer_data->info_first_instance_char = NULL;

  observer->logger_allocate_function = logger_bbob_new;
  observer->logger_free_function = logger_bbob_new_free;
  observer->data_free_function = observer_bbob_new_data_free;
  observer->data = observer_data;

  *option_keys = NULL;

  (void) options; /* To silence the compiler */
}
