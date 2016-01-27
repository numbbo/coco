/**
 * @file observer_toy.c
 * @brief Implementation of the toy observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *problem);

/**
 * @brief The toy observer data type.
 */
typedef struct {
  FILE *log_file;            /**< @brief File used for logging. */
  size_t number_of_targets;  /**< @brief The number of targets. */
  double *targets;           /**< @brief The targets. */
} observer_toy_data_t;

/**
 * @brief Frees memory of the toy observer data structure.
 */
static void observer_toy_free(void *stuff) {

  observer_toy_data_t *data;

  assert(stuff != NULL);
  data = stuff;

  if (data->log_file != NULL) {
    fclose(data->log_file);
    data->log_file = NULL;
  }

  if (data->targets != NULL) {
    coco_free_memory(data->targets);
    data->targets = NULL;
  }

}

/**
 * @brief Initializes the toy observer.
 *
 * Possible options:
 * - file_name : name_of_the_output_file (name of the output file; default value is "first_hitting_times.txt")
 * - number_of_targets : 1-100 (number of targets; default value is 20)
 */
static void observer_toy(coco_observer_t *observer, const char *options) {

  observer_toy_data_t *observer_toy;
  char *string_value;
  char *file_name;
  size_t i;

  observer_toy = (observer_toy_data_t *) coco_allocate_memory(sizeof(*observer_toy));

  /* Read file_name and number_of_targets from the options and use them to initialize the observer */
  string_value = coco_allocate_string(COCO_PATH_MAX);
  if (coco_options_read_string(options, "file_name", string_value) == 0) {
    strcpy(string_value, "first_hitting_times.txt");
  }
  if ((coco_options_read_size_t(options, "number_of_targets", &observer_toy->number_of_targets) == 0)
      || (observer_toy->number_of_targets < 1) || (observer_toy->number_of_targets > 100)) {
    observer_toy->number_of_targets = 20;
  }

  /* Open log_file */
  file_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(file_name, observer->output_folder, strlen(observer->output_folder) + 1);
  coco_create_directory(file_name);
  coco_join_path(file_name, COCO_PATH_MAX, string_value, NULL);

  observer_toy->log_file = fopen(file_name, "a");
  if (observer_toy->log_file == NULL) {
    coco_error("observer_toy(): failed to open file %s.", file_name);
    return; /* Never reached */
  }

  /* Compute targets */
  observer_toy->targets = coco_allocate_vector(observer_toy->number_of_targets);
  for (i = observer_toy->number_of_targets; i > 0; --i) {
    observer_toy->targets[i - 1] = pow(10.0, (double) (long) (observer_toy->number_of_targets - i) - 9.0);
  }

  observer->logger_initialize_function = logger_toy;
  observer->data_free_function = observer_toy_free;
  observer->data = observer_toy;
}
