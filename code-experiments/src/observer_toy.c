#include "coco.h"
#include "coco_utilities.c"

static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *problem);

/* Data for the toy observer */
typedef struct {
  FILE *log_file;
  size_t number_of_targets;
  double *targets;
} observer_toy_t;

/**
 * Frees memory for the given coco_observer_t's data field observer_toy_t.
 */
static void observer_toy_free(void *stuff) {

  observer_toy_t *data;

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
 * Initializes the toy observer. Possible options:
 * - file_name : name_of_the_output_file (name of the output file; default value is "first_hitting_times.txt")
 * - number_of_targets : 1-100 (number of targets; default value is 20)
 */
static void observer_toy(coco_observer_t *self, const char *options) {

  observer_toy_t *data;
  char *string_value;
  char *file_name;
  size_t i;

  data = coco_allocate_memory(sizeof(*data));

  /* Read file_name and number_of_targets from the options and use them to initialize the observer */
  string_value = (char *) coco_allocate_memory(COCO_PATH_MAX);
  if (coco_options_read_string(options, "file_name", string_value) == 0) {
    strcpy(string_value, "first_hitting_times.txt");
  }
  if ((coco_options_read_size_t(options, "number_of_targets", &data->number_of_targets) == 0)
      || (data->number_of_targets < 1) || (data->number_of_targets > 100)) {
    data->number_of_targets = 20;
  }

  /* Open log_file */
  file_name = (char *) coco_allocate_memory(COCO_PATH_MAX);
  memcpy(file_name, self->output_folder, strlen(self->output_folder) + 1);
  coco_create_path(file_name);
  coco_join_path(file_name, COCO_PATH_MAX, string_value, NULL);

  data->log_file = fopen(file_name, "a");
  if (data->log_file == NULL) {
    coco_error("observer_toy(): failed to open file %s.", file_name);
    return; /* Never reached */
  }

  /* Compute targets */
  data->targets = coco_allocate_vector(data->number_of_targets);
  for (i = data->number_of_targets; i > 0; --i) {
    data->targets[i - 1] = pow(10.0, (double) (long) (data->number_of_targets - i) - 9.0);
  }

  self->logger_initialize_function = logger_toy;
  self->data_free_function = observer_toy_free;
  self->data = data;
}
