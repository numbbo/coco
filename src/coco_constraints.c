#include "coco.h"

static void linear_constraint(coco_problem_t *self, const double *x, double *y){
  size_t i;
  
  double *a;
  a = coco_allocate_vector(self->number_of_variables);
  
  for (i = 0; i < self->number_of_variables; ++i) {
    a[i] = 1.0;
  }
  
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    y[0] += a[i] * x[i];
  }
}

static coco_problem_t *coco_add_constraints(coco_problem_t *self, const char *constraint_type) {
  
  if (strcmp(constraint_type, "linear_constraint") == 0) {
      self->evaluate_constraint = linear_constraint;
  }
  return self;
}
