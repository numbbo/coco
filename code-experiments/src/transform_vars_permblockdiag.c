/**
 * @file transform_vars_permblockdiag.c
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
/* #include "large_scale_transformations.c" */
#include "transform_vars_permutation_helpers.c"
#include "transform_vars_blockrotation_helpers.c"


/**
 * @brief Data type for transform_vars_permblockdiag.
 */
typedef struct {
  double **B;
  double *x;
  size_t *P1; /*permutation matrices, P1 for the columns of B and P2 for its rows*/
  size_t *P2;
  size_t *block_sizes;
  size_t nb_blocks;
  size_t *block_size_map; /* maps rows to blocksizes, keep until better way is found */
  size_t *first_non_zero_map; /* maps a row to the index of its first non zero element */
} transform_vars_permblockdiag_t;

static void transform_vars_permblockdiag_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j, current_blocksize, first_non_zero_ind;
  transform_vars_permblockdiag_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_permblockdiag_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    current_blocksize = data->block_size_map[data->P2[i]];/*the block_size is that of the permuted line*/
    first_non_zero_ind = data->first_non_zero_map[data->P2[i]];
    data->x[i] = 0;
    /*compute data->x[i] = < B[P2[i]] , x[P1] >  */
    for (j = first_non_zero_ind; j < first_non_zero_ind + current_blocksize; ++j) {/*blocksize[P2[i]]*/
      data->x[i] += data->B[data->P2[i]][j - first_non_zero_ind] * x[data->P1[j]];/*all B lines start at 0*/
    }
  }

  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

static void transform_vars_permblockdiag_free(void *thing) {
  transform_vars_permblockdiag_t *data = (transform_vars_permblockdiag_t *) thing;
  coco_free_memory(data->B);
  coco_free_memory(data->P1);
  coco_free_memory(data->P2);
  coco_free_memory(data->block_sizes);
  coco_free_memory(data->x);
  coco_free_memory(data->block_size_map);
}

/*
 * Apply a double permuted orthogonal block-diagonal transfromation matrix to the search space
 *
 *
 * The matrix M is stored in row-major format.
 */
static coco_problem_t *transform_vars_permblockdiag(coco_problem_t *inner_problem,
                                                    const double * const *B,
                                                    const size_t *P1,
                                                    const size_t *P2,
                                                    const size_t number_of_variables,
                                                    const size_t *block_sizes,
                                                    const size_t nb_blocks) {
  coco_problem_t *problem;
  transform_vars_permblockdiag_t *data;
  size_t entries_in_M, idx_blocksize, next_bs_change, current_blocksize;
  int i;
  entries_in_M = 0;
  assert(number_of_variables > 0);/*tmp*/
  for (i = 0; i < nb_blocks; i++) {
    entries_in_M += block_sizes[i] * block_sizes[i];
  }
  data = (transform_vars_permblockdiag_t *) coco_allocate_memory(sizeof(*data));
  data->B = ls_copy_block_matrix(B, number_of_variables, block_sizes, nb_blocks);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->P1 = coco_duplicate_size_t_vector(P1, inner_problem->number_of_variables);
  data->P2 = coco_duplicate_size_t_vector(P2, inner_problem->number_of_variables);
  data->block_sizes = coco_duplicate_size_t_vector(block_sizes, nb_blocks);
  data->nb_blocks = nb_blocks;
  data->block_size_map = coco_allocate_vector_size_t(number_of_variables);
  data->first_non_zero_map = coco_allocate_vector_size_t(number_of_variables);
  
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];
  for (i = 0; i < number_of_variables; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    data->block_size_map[i] = current_blocksize;
    data->first_non_zero_map[i] = next_bs_change - current_blocksize;/* next_bs_change serves also as a cumsum for blocksizes*/
  }
  
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_permblockdiag_free, "transform_vars_permblockdiag");
  problem->evaluate_function = transform_vars_permblockdiag_evaluate;
  return problem;
}


