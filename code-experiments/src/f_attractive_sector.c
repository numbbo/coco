#include <assert.h>
#include <math.h>

#include "coco.h"

#include "coco_problem.c"

typedef struct {
  double *xopt;
} f_attractive_sector_data_t;

static void f_attractive_sector_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  f_attractive_sector_data_t *data;

  assert(self->number_of_objectives == 1);
  data = self->data;
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    if (data->xopt[i] * x[i] > 0.0) {
      y[0] += 100.0 * 100.0 * x[i] * x[i];
    } else {
      y[0] += x[i] * x[i];
    }
  }
}

static void f_attractive_sector_free(coco_problem_t *self) {
  f_attractive_sector_data_t *data;
  data = self->data;
  coco_free_memory(data->xopt);
  self->free_problem = NULL;
  coco_problem_free(self);
}

static coco_problem_t *f_attractive_sector(const size_t number_of_variables, const double *xopt) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);
  f_attractive_sector_data_t *data;
  data = coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, number_of_variables);

  problem->problem_name = coco_strdup("attractive sector function");
  /* Construct a meaningful problem id */
  problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "attractive_sector", (long) number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "attractive_sector",
      (long) number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->data = data;
  problem->evaluate_function = f_attractive_sector_evaluate;
  problem->free_problem = f_attractive_sector_free;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  f_attractive_sector_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
