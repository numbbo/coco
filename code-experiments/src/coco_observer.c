#include "coco.h"
#include "coco_internal.h"

#include "observer_bbob2009.c"
#include "observer_mo_toy.c"
#include "logger_mo.c"
#include "observer_toy.c"

static coco_observer_t *coco_observer_allocate(char *output_folder, int verbosity) {

  coco_observer_t *observer;
  observer = (coco_observer_t *) coco_allocate_memory(sizeof(*observer));
  /* Initialize fields to sane/safe defaults */
  observer->output_folder = coco_strdup(output_folder);
  observer->verbosity = verbosity;
  observer->data = NULL;
  observer->observer_free_function = NULL;
  observer->logger_initialize_function = NULL;
  return observer;
}

void coco_observer_free(coco_observer_t *self) {
  assert(self != NULL);
  if (self->observer_free_function != NULL) {
    self->observer_free_function(self);
  } else {
    /* Best guess at freeing all relevant structures */
    if (self->output_folder != NULL)
      coco_free_memory(self->output_folder);
    if (self->data != NULL)
      coco_free_memory(self->data);
    self->data = NULL;
    coco_free_memory(self);
  }
}

coco_observer_t *coco_observer(const char *observer_name, const char *options) {

  coco_observer_t *observer;
  char *string_value;
  int verbosity;

  if (0 == strcmp(observer_name, "no_observer")) {
    return NULL;
  } else if (strlen(observer_name) == 0) {
    coco_warning("Empty observer_name has no effect. To prevent this warning use 'no_observer' instead");
    return NULL;
  }

  string_value = (char *) coco_allocate_memory(COCO_PATH_MAX);
  /* Read result_folder and verbosity from the options and use them to initialize the observer */
  if (coco_options_read_string(options, "result_folder", string_value) == 0) {
    strcpy(string_value, "results");
  }
  coco_create_unique_path(&string_value);
  if (coco_options_read_int(options, "verbosity", &verbosity) == 0)
    verbosity = 0;

  observer = coco_observer_allocate(string_value, verbosity);

  /* Here each observer must have an entry */
  if (0 == strcmp(observer_name, "observer_toy")) {
    observer_toy(observer, options);
  } else if (0 == strcmp(observer_name, "observer_bbob2009")) {
    observer_bbob2009(observer, options);
  } else if (0 == strcmp(observer_name, "observer_mo_toy")) {
    observer_mo_toy(observer, options);
  } else if (0 == strcmp(observer_name, "observer_mo")) {
    observer_mo(observer, options);
  } else {
    coco_warning("Unknown observer!");
    return NULL;
  }

  return observer;
}

coco_problem_t *coco_problem_add_observer(coco_problem_t *problem, coco_observer_t *observer) {

  if (observer == NULL) {
    coco_warning("The problem is not being observed.");
    return problem;
  }

  return observer->logger_initialize_function(observer, problem);
}

