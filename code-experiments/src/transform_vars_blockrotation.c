/**
 * @file transform_vars_blockrotation.c
 * @brief Implementation of performing a block-rotation transformation on
 * decision values.
 *
 * x |-> Bx
 * The matrix B is stored in a 2-D array. Only the content of the blocks are
 * stored
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "transform_vars_blockrotation_helpers.c"
#include "transform_vars_permutation_helpers.c" /* for coco_duplicate_size_t_vector */

/**
 * @brief Data type for transform_vars_blockrotation.
 */
typedef struct {
  double **B; /**< @brief the block-diagonal matrices*/
  double *Bx;
  size_t dimension;
  size_t *block_sizes; /**< @brief the list of block-sizes*/
  size_t nb_blocks;    /**< @brief the number of blocks in the matrix */
  size_t
      *block_size_map; /**< @brief maps a row to the block-size of the block to
                          which it belong, keep until better way is found */
  size_t *first_non_zero_map; /**< @brief maps a row to the index of its first
                                 non zero element */
} transform_vars_blockrotation_t;

/*
 * @brief return i-th row of blockrotation problem->data->B in y.
 */
static void transform_vars_blockrotation_get_row(coco_problem_t *problem,
                                                 size_t i, double *y) {
  size_t j, current_blocksize, first_non_zero_ind; /* cp-paste from apply */
  transform_vars_blockrotation_t *data;

  data = (transform_vars_blockrotation_t *)coco_problem_transformed_get_data(
      problem);
  current_blocksize = data->block_size_map[i];
  first_non_zero_ind = data->first_non_zero_map[i];

  for (j = 0; j < data->dimension; ++j) {
    y[j] =
        (j < first_non_zero_ind || j >= first_non_zero_ind + current_blocksize)
            ? 0
            : data->B[i][j - first_non_zero_ind]; /*all B lines start at 0*/
  }
}

/*
 * @brief Computes y = Bx, where all the pertinent information about B is given
 * in the problem data.
 */
static void transform_vars_blockrotation_apply(coco_problem_t *problem,
                                               const double *x, double *y) {
  size_t i, j, current_blocksize, first_non_zero_ind;
  transform_vars_blockrotation_t *data;

  data = (transform_vars_blockrotation_t *)coco_problem_transformed_get_data(
      problem);
  assert(x != data->Bx);
  for (i = 0; i < data->dimension; ++i) {
    current_blocksize = data->block_size_map[i];
    first_non_zero_ind = data->first_non_zero_map[i];
    data->Bx[i] = 0;
    /*compute y[i] = < B[i,:] , x >  */
    for (j = first_non_zero_ind; j < first_non_zero_ind + current_blocksize;
         ++j) {
      data->Bx[i] +=
          data->B[i][j - first_non_zero_ind] * x[j]; /*all B lines start at 0*/
    }
  }
  if (y != data->Bx) {
    for (i = 0; i < data->dimension; ++i) {
      y[i] = data->Bx[i];
    }
  }
}

static void transform_vars_blockrotation_evaluate(coco_problem_t *problem,
                                                  const double *x, double *y) {
  coco_problem_t *inner_problem =
      coco_problem_transformed_get_inner_problem(problem);
  transform_vars_blockrotation_t *data;
  data = (transform_vars_blockrotation_t *)coco_problem_transformed_get_data(
      problem);

  transform_vars_blockrotation_apply(problem, x, data->Bx);

  coco_evaluate_function(inner_problem, data->Bx, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

static void transform_vars_blockrotation_free(void *stuff) {
  transform_vars_blockrotation_t *data =
      (transform_vars_blockrotation_t *)stuff;
  coco_free_block_matrix(data->B, data->dimension);
  coco_free_memory(data->block_sizes);
  coco_free_memory(data->Bx);
  coco_free_memory(data->block_size_map);
  coco_free_memory(data->first_non_zero_map);
}

/*
 * @brief test blockrotation on its own rows and raise coco_error in case
 */
static void transform_vars_blockrotation_test(coco_problem_t *problem,
                                              double precision) {
  size_t i, j;
  size_t number_of_variables = coco_problem_get_dimension(problem);
  double *y = coco_allocate_vector(number_of_variables);

  for (i = 0; i < number_of_variables; ++i) { /* for each row */
    transform_vars_blockrotation_get_row(problem, i, y);
    transform_vars_blockrotation_apply(problem, y, y);
    for (j = 0; j < number_of_variables; ++j) { /* check result */
      if (!coco_double_almost_equal(y[j], (i == j) ? 1.0 : 0.0, precision)) {
        coco_error("transform_vars_blockrotation_test() with precision %e "
                   "failed on row %i",
                   precision, i);
      }
      /* printf("%f ", y[j]); */
    }
    /* printf("\n"); */
  }
  coco_free_memory(y);
}

static coco_problem_t *transform_vars_blockrotation(
    coco_problem_t *inner_problem, const double *const *B,
    const size_t number_of_variables, const size_t *block_sizes,
    const size_t nb_blocks) {
  coco_problem_t *problem;
  transform_vars_blockrotation_t *data;
  size_t idx_blocksize, next_bs_change, current_blocksize;
  size_t i;
  assert(number_of_variables > 0); /*tmp*/
  data = (transform_vars_blockrotation_t *)coco_allocate_memory(sizeof(*data));
  data->dimension = number_of_variables;
  data->B =
      coco_copy_block_matrix(B, number_of_variables, block_sizes, nb_blocks);
  data->Bx = coco_allocate_vector(inner_problem->number_of_variables);
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
    current_blocksize = block_sizes[idx_blocksize];
    data->block_size_map[i] = current_blocksize;
    data->first_non_zero_map[i] =
        next_bs_change - current_blocksize; /* next_bs_change serves also as a
                                               cumsum for blocksizes*/
  }
  if (coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_debug("transform_vars_blockrotation(): 'best_parameter' not updated, "
               "set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter,
                           inner_problem->number_of_variables);
  }
  problem = coco_problem_transformed_allocate(inner_problem, data,
                                              transform_vars_blockrotation_free,
                                              "transform_vars_blockrotation");
  problem->evaluate_function = transform_vars_blockrotation_evaluate;

  if (number_of_variables < 100) {
    /* 1e-11 still passes and 1e-12 fails under macOS */
    transform_vars_blockrotation_test(problem, 1e-5);
  }
  return problem;
}
