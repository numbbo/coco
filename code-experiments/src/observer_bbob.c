/**
 * @file observer_bbob.c
 * @brief Implementation of the bbob observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_bbob(coco_observer_t *observer, coco_problem_t *problem);
static void logger_bbob_free(void *logger);

#define MAX_NB_DIMS 10 /* TODO: remove in favor of a dynamically set value */

/**
 * @brief The bbob observer data type.
 */
typedef struct {
  /* TODO: Can be used to store variables that need to be accessible during one run (i.e. for multiple
   * problems). For example, the following global variables from logger_bbob.c could be stored here: */
  size_t current_dim;
  size_t current_fun_id;
  size_t info_file_first_instance;
  size_t number_of_dimensions;
  size_t dimensions_in_current_info_file[MAX_NB_DIMS];
  int logger_is_open;
} observer_bbob_data_t;

/**
 * @brief frees the data of bbob observer
 */
static void observer_bbob_data_free(void *data){
  /*coco_free_memory(((observer_bbob_data_t *)data)->dimensions_in_current_info_file);*/
  (void) data; /* to silence warning*/
}


/**
 * @brief Initializes the bbob observer.
 */
static void observer_bbob(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  size_t i;
  observer_bbob_data_t *observer_bbob;
  observer_bbob = (observer_bbob_data_t *) coco_allocate_memory(sizeof(*observer_bbob));
  observer_bbob->current_dim = 0;
  observer_bbob->current_fun_id = 0;
  observer_bbob->info_file_first_instance = 0;
  observer_bbob->number_of_dimensions = MAX_NB_DIMS;
  for (i = 0; i < observer_bbob->number_of_dimensions; i++) {
    observer_bbob->dimensions_in_current_info_file[i] = 0;
  }
  observer_bbob->logger_is_open = 0;

  observer->logger_allocate_function = logger_bbob;
  observer->logger_free_function = logger_bbob_free;
  observer->data_free_function = observer_bbob_data_free;
  observer->data = observer_bbob;


  *option_keys = NULL;

  (void) options; /* To silence the compiler */
}
