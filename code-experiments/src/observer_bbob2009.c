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
  size_t bbob2009_nbpts_nbevals;
  size_t bbob2009_nbpts_fval;
} observer_bbob2009_t;



static void observer_bbob2009_free(void *stuff) {
  
  observer_bbob2009_t *data;
  
  assert(stuff != NULL);
  data = stuff;
  
}

/**
 * Initializes the bbob2009 observer. Possible options:
 * - bbob2009_nbpts_nbevals: nb fun eval triggers are at 10**(i/bbob2009_nbpts_nbevals) (the default value in bbob2009 is 20 )
 * - bbob2009_nbpts_fval: f value difference to the optimal triggers are at 10**(i/bbob2009_nbpts_fval)(the default value in bbob2009 is 5 )
 */
static void observer_bbob2009(coco_observer_t *self, const char *options) {
  
  observer_bbob2009_t *data;
  
  data = coco_allocate_memory(sizeof(*data));  

  if ((coco_options_read_size_t(options, "nbpts_nbevals", &(data->bbob2009_nbpts_nbevals)) == 0)) {
    data->bbob2009_nbpts_nbevals = 20;
  }
  if ((coco_options_read_size_t(options, "nbpts_fval", &(data->bbob2009_nbpts_fval)) == 0)) {
    data->bbob2009_nbpts_fval = 5;
  }

  self->logger_initialize_function = logger_bbob2009;
  self->data_free_function = observer_bbob2009_free;
  self->data = data;
}
