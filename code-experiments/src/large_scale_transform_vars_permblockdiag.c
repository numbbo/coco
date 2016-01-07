#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double *B, *x;
  size_t *P1, *P2, *block_sizes, nb_blocks;/*permutation matrices, P1 for the columns of M and P2 for its rows*/
} ls_transform_vars_permblockdiag_t;

static void transform_vars_affine_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i, j;
  ls_transform_vars_permblockdiag_t *data;
  coco_problem_t *inner_problem;
  
  data = coco_transformed_get_data(self);
  inner_problem = coco_transformed_get_inner_problem(self);
  
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    const double *current_row = data->B + data->P2[i] * self->number_of_variables;
    data->x[i] = 0;/*data->b[i];*/
    for (j = 0; j < self->number_of_variables; ++j) {
      data->x[i] += x[j] * current_row[data->P1[j]];
    }
  }
  coco_evaluate_function(inner_problem, data->x, y);
}

static void transform_vars_affine_free(void *thing) {
  transform_vars_affine_data_t *data = thing;
  coco_free_memory(data->M);
  coco_free_memory(data->P1);
  coco_free_memory(data->P1);
  coco_free_memory(data->block_sizes);
  coco_free_memory(data->x);
}

/*
 * Apply a double permuted orthogonal block-diagonal transfromation matrix to the search space
 *
 *
 * The matrix M is stored in row-major format.
 */
static coco_problem_t *f_ls_transform_vars_permblockdiag(coco_problem_t *inner_problem,
                                               const double *B,
                                               const double *P1,
                                               const double *P2,
                                               const size_t number_of_variables) {
  coco_problem_t *self;
  ls_transform_vars_permblockdiag_t *data;
  size_t entries_in_B;
  
  /*entries_in_B = inner_problem->number_of_variables * number_of_variables;*/
  data = coco_allocate_memory(sizeof(*data));
  data->B = coco_duplicate_vector(M, entries_in_M);
  data->P1 = coco_duplicate_vector(P1, inner_problem->number_of_variables);
  data->P2 = coco_duplicate_vector(P2, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  
  self = coco_transformed_allocate(inner_problem, data, transform_vars_affine_free);
  self->evaluate_function = transform_vars_affine_evaluate;
  return self;
}
