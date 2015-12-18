#include "coco.h"
#include "coco_utilities.c"
/*#include "logger_bbob2009.c"*/

/*
static coco_problem_t *deprecated__observer_bbob2009(coco_problem_t *problem, const char *options) {
  if (problem == NULL)
    return problem;
  coco_create_path(options);
  problem = depreciated_logger_bbob2009(problem, options);
  return problem;
}
*/
static coco_problem_t *logger_bbob2009(coco_observer_t *observer, coco_problem_t *problem);

typedef struct {
  size_t verbosity;
  
} observer_bbob2009_t;



static void observer_bbob2009_free(void *stuff) {
  
  observer_bbob2009_t *data;
  
  assert(stuff != NULL);
  data = stuff;
  
}

/**
 * Initializes the bbob2009 observer. Minimalistic implementation with no options yet
 */
static void observer_bbob2009(coco_observer_t *self, const char *options) {
  
  observer_bbob2009_t *data;
  
  data = coco_allocate_memory(sizeof(*data));
  
  if ((coco_options_read_size_t(options, "verbosity", &(data->verbosity)) == 0)) {
    data->verbosity = 2;
  }

  self->logger_initialize_function = logger_bbob2009;
  self->data_free_function = observer_bbob2009_free;
  self->data = data;
}
