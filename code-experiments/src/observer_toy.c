/**
 * @file observer_toy.c
 * @brief Implementation of the toy observer.
 */

#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *problem);
static void logger_toy_free(void *logger);

/**
 * @brief The toy observer data type.
 */
typedef struct {
  FILE *log_file;            /**< @brief File used for logging. */
} observer_toy_data_t;

/**
 * @brief Frees memory of the toy observer data structure.
 */
static void observer_toy_free(void *stuff) {

  observer_toy_data_t *data;

  assert(stuff != NULL);
  data = (observer_toy_data_t *) stuff;

  if (data->log_file != NULL) {
    fclose(data->log_file);
    data->log_file = NULL;
  }

}

/**
 * @brief Initializes the toy observer.
 *
 * Possible options:
 * - file_name: string (name of the output file; default value is "first_hitting_times.dat")
 */
static void observer_toy(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_toy_data_t *observer_data;
  char *string_value;
  char *file_name;

  /* Sets the valid keys for toy observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "file_name" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_toy_data_t *) coco_allocate_memory(sizeof(*observer_data));

  /* Read file_name and number_of_targets from the options and use them to initialize the observer */
  string_value = coco_allocate_string(COCO_PATH_MAX + 1);
  if (coco_options_read_string(options, "file_name", string_value) == 0) {
    strcpy(string_value, "first_hitting_times.dat");
  }

  /* Open log_file */
  file_name = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(file_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_create_directory(file_name);
  coco_join_path(file_name, COCO_PATH_MAX, string_value, NULL);

  observer_data->log_file = fopen(file_name, "a");
  if (observer_data->log_file == NULL) {
    coco_error("observer_toy(): failed to open file %s.", file_name);
    return; /* Never reached */
  }

  coco_free_memory(string_value);
  coco_free_memory(file_name);

  observer->logger_allocate_function = logger_toy;
  observer->logger_free_function = logger_toy_free;
  observer->restart_function = NULL;
  observer->data_free_function = observer_toy_free;
  observer->data = observer_data;
}
