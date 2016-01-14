#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "large_scale_transformations.c"

typedef struct {
  double **B;
  double *x;
  size_t *P1; /*permutation matrices, P1 for the columns of B and P2 for its rows*/
  size_t *P2;
  size_t *block_sizes;
  size_t nb_blocks;
  size_t *block_map; /* maps rows to blocksizes, keep until better way is found */
} ls_transform_vars_permblockdiag_t;

static void ls_transform_vars_permblockdiag_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i, j;
  ls_transform_vars_permblockdiag_t *data;
  coco_problem_t *inner_problem;
  
  data = coco_transformed_get_data(self);
  inner_problem = coco_transformed_get_inner_problem(self);

  printf("eval started\n");
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    /*
    printf("in ls_transform\n");
    printf("P1: ");
    for (j = 0; j < self->number_of_variables; ++j) {
      printf("%d ", data->P1[i]);
    }
    printf("\n");
    
    printf("P2: ");
    for (j = 0; j < self->number_of_variables; ++j) {
      printf("%d ", data->P2[i]);
    }
    printf("\n");
    */
    /*const double *current_row = data->B;*/
    data->x[i] = 0;/*data->b[i];*/
    /*compute data->x[i] = < B[P2[i]] , x[P1[i]] >  */
    for (j = 0; j < data->block_map[i]; ++j) {/*blocksize[P2[i]]*/
      
      data->x[i] += data->B[data->P2[i]][j] * x[data->P1[j]];
    }
    /*printf("x_%d=%f\n", i, x[0]);*/

  }
  printf("eval finished\n\n");
  coco_evaluate_function(inner_problem, data->x, y);
}

static void ls_transform_vars_permblockdiag_free(void *thing) {
  ls_transform_vars_permblockdiag_t *data = thing;
  coco_free_memory(data->B);
  coco_free_memory(data->P1);
  coco_free_memory(data->P2);
  coco_free_memory(data->block_sizes);
  coco_free_memory(data->x);
  coco_free_memory(data->block_map);
}

/*
 * Apply a double permuted orthogonal block-diagonal transfromation matrix to the search space
 *
 *
 * The matrix M is stored in row-major format.
 */
static coco_problem_t *f_ls_transform_vars_permblockdiag(coco_problem_t *inner_problem,
                                               const double **B,
                                               const size_t *P1,
                                               const size_t *P2,
                                               const size_t number_of_variables,
                                               const size_t *block_sizes,
                                               const size_t nb_blocks) {
  coco_problem_t *self;
  ls_transform_vars_permblockdiag_t *data;
  size_t entries_in_M, idx_blocksize, next_bs_change, current_blocksize;
  int i;
  entries_in_M = 0;
  assert(number_of_variables > 0);/*tmp*/
  for (i = 0; i < nb_blocks; i++) {
    entries_in_M += block_sizes[i] * block_sizes[i];
  }
  data = coco_allocate_memory(sizeof(*data));
  data->B = ls_copy_block_matrix(B, number_of_variables, block_sizes, nb_blocks);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->P1 = coco_duplicate_size_t_vector(P1, inner_problem->number_of_variables);
  data->P2 = coco_duplicate_size_t_vector(P2, inner_problem->number_of_variables);
  data->block_sizes = coco_duplicate_size_t_vector(block_sizes, nb_blocks);
  data->nb_blocks = nb_blocks;
  data->block_map = (size_t *)coco_allocate_memory(number_of_variables * sizeof(size_t));
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];
  for (i = 0; i < number_of_variables; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    data->block_map[i] = current_blocksize;
  }
  
  self = coco_transformed_allocate(inner_problem, data, ls_transform_vars_permblockdiag_free);
  self->evaluate_function = ls_transform_vars_permblockdiag_evaluate;
  return self;
}
