/**
 * @file transform_vars_blockrotation.c
 * @brief Implementation of performing a block-rotation transformation on decision values.
 *
 * x |-> Bx
 * The matrix B is stored in a 2-D array. Only the content of the blocks are stored
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "transform_vars_blockrotation_helpers.c"

/**
 * @brief Data type for transform_vars_blockrotation.
 */
typedef struct {
  double **B; /**< @brief the block-diagonal matrices*/
  double *x;
  size_t *block_sizes; /**< @brief the list of block-sizes*/
  size_t nb_blocks; /**< @brief the number of blocks in the matrix */
  size_t nb_rows; /**< @brief number of rows, needed to free the matrix correctly */
  size_t *block_size_map; /**< @brief maps a row to the block-size of the block to which it belong, keep until better way is found */
  size_t *first_non_zero_map; /**< @brief maps a row to the index of its first non zero element */
} transform_vars_blockrotation_t;

static void transform_vars_blockrotation_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j, current_blocksize, first_non_zero_ind;
  transform_vars_blockrotation_t *data;
  coco_problem_t *inner_problem;
  
  data = (transform_vars_blockrotation_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    current_blocksize = data->block_size_map[i];
    first_non_zero_ind = data->first_non_zero_map[i];
    data->x[i] = 0;
    /*compute data->x[i] = < B[i,:] , x >  */
    for (j = first_non_zero_ind; j < first_non_zero_ind + current_blocksize; ++j) {
      data->x[i] += data->B[i][j - first_non_zero_ind] * x[j]; /*all B lines start at 0*/
    }
  }

  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

static void transform_vars_blockrotation_free(void *thing) {
  transform_vars_blockrotation_t *data = (transform_vars_blockrotation_t *) thing;
  /* coco_free_memory(data->B); */
  coco_free_block_matrix(data->B, data->nb_rows);
  coco_free_memory(data->x);
  coco_free_memory(data->block_sizes);
  coco_free_memory(data->block_size_map);
  coco_free_memory(data->first_non_zero_map);
}


static coco_problem_t *transform_vars_blockrotation(coco_problem_t *inner_problem,
                                                    const double * const *B,
                                                    const size_t number_of_variables,
                                                    const size_t *block_sizes,
                                                    const size_t nb_blocks) {
  coco_problem_t *problem;
  transform_vars_blockrotation_t *data;
  size_t entries_in_M, idx_blocksize, next_bs_change, current_blocksize;
  size_t i;
  entries_in_M = 0;
  assert(number_of_variables > 0);/*tmp*/
  for (i = 0; i < nb_blocks; i++) {
    entries_in_M += block_sizes[i] * block_sizes[i];
  }
  data = (transform_vars_blockrotation_t *) coco_allocate_memory(sizeof(*data));
  data->B = coco_copy_block_matrix(B, number_of_variables, block_sizes, nb_blocks);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->block_sizes = coco_duplicate_size_t_vector(block_sizes, nb_blocks);
  data->nb_blocks = nb_blocks;
  data->nb_rows = number_of_variables;
  data->block_size_map = coco_allocate_vector_size_t(number_of_variables);
  data->first_non_zero_map = coco_allocate_vector_size_t(number_of_variables);
  
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];
  for (i = 0; i < number_of_variables; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize = block_sizes[idx_blocksize];
    data->block_size_map[i] = current_blocksize;
    data->first_non_zero_map[i] = next_bs_change - current_blocksize;/* next_bs_change serves also as a cumsum for blocksizes*/
  }
  if (coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_debug("transform_vars_blockrotation(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_blockrotation_free, "transform_vars_blockrotation");
  problem->evaluate_function = transform_vars_blockrotation_evaluate;
  return problem;
}


