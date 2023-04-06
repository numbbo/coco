/**
 * @file transform_vars_blockrotation_helpers.c
 * @brief implements functions needed by transform_vars_blockrotation.c
 */

#include <stdio.h>
#include <assert.h>
#include "coco.h"

#include "coco_random.c" /*tmp*/
#include "suite_bbob_legacy_code.c" /*tmp*/

#include <time.h> /*tmp*/

/* TODO: Document this file in doxygen style! */

/**
 * @brief Returns block size for block rotation matrices.
 *
 */
static size_t coco_rotation_matrix_block_size(size_t const dimension) {
  const double BLOCK_SIZE_RELATIVE = 1;   /* for block rotations, relative to dimension */
  const size_t MAX_BLOCK_SIZE_ABSOLUTE = 40;  /* for block rotations */

  return coco_double_to_size_t(coco_double_min(
                  BLOCK_SIZE_RELATIVE * (double)dimension,
                  (double)MAX_BLOCK_SIZE_ABSOLUTE));
}

/**
 * @brief
 * Allocate a ${n} by ${m} block matrix of nb_blocks block sizes block_sizes structured as an array of pointers
 * to double arrays.
 * each row contains only the block_sizes[i] possibly non-zero elements
 */
static double **coco_allocate_blockmatrix(const size_t n, const size_t* block_sizes, const size_t nb_blocks) {
  double **matrix = NULL;
  size_t current_blocksize;
  size_t next_bs_change;
  size_t idx_blocksize;
  size_t i;
  COCO_UNUSED size_t sum_block_sizes;

  sum_block_sizes = 0;
  for (i = 0; i < nb_blocks; i++){
    sum_block_sizes += block_sizes[i];
  }
  assert(sum_block_sizes == n);

  matrix = (double **) coco_allocate_memory(sizeof(double *) * n);
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];

  for (i = 0; i < n; ++i) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    matrix[i] = coco_allocate_vector(current_blocksize);

  }
  return matrix;
}


/**
 * @brief frees a block diagonal matrix (same as a matrix but in case of change, easier to update separately from free_matrix)
 */
static void coco_free_block_matrix(double **matrix, const size_t n) {
  size_t i;
  for (i = 0; i < n; ++i) {
    if (matrix[i] != NULL) {
      coco_free_memory(matrix[i]);
      matrix[i] = NULL;
    }
  }
  coco_free_memory(matrix);
}



/**
 * @brief Compute a ${DIM}x${DIM} block-diagonal matrix based on ${seed} and block_sizes and stores it in ${B}.
 * B is a 2D vector with DIM lines and each line has blocksize(line) elements (the zeros are not stored)
 */
static void coco_compute_blockrotation(double **B, long seed, COCO_UNUSED size_t n, size_t *block_sizes, size_t nb_blocks) {
  double **current_block;
  size_t i, j;
  size_t idx_block, current_blocksize, cumsum_prev_block_sizes;
  COCO_UNUSED size_t sum_block_sizes;
  sum_block_sizes = 0;
  for (i = 0; i < nb_blocks; i++){
    sum_block_sizes += block_sizes[i];
  }
  assert(sum_block_sizes == n);

  cumsum_prev_block_sizes = 0;/* shift in rows to account for the previous blocks */
  for (idx_block = 0; idx_block < nb_blocks; idx_block++) {
    current_blocksize = block_sizes[idx_block];
    current_block = bbob2009_allocate_matrix(current_blocksize, current_blocksize);
    assert(current_blocksize <= 44);
    bbob2009_compute_rotation(current_block, seed + (long) 1000000 * (long) idx_block, current_blocksize);

    /* now fill the block matrix*/
    for (i = 0 ; i < current_blocksize; i++) {
      for (j = 0; j < current_blocksize; j++) {
        B[i + cumsum_prev_block_sizes][j] = current_block[i][j];
      }
    }

    cumsum_prev_block_sizes+=current_blocksize;
    /*current_gvect_pos += current_blocksize * current_blocksize;*/
    coco_free_block_matrix(current_block, current_blocksize);
  }
}

/**
 * @brief makes a copy of a block_matrix
 */
static double **coco_copy_block_matrix(const double *const *B, const size_t dimension, const size_t *block_sizes, const size_t nb_blocks) {
  double **dest;
  size_t i, j, idx_blocksize, current_blocksize, next_bs_change;

  dest = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];
  assert(nb_blocks != 0); /*tmp*/ /*to silence warning*/
  for (i = 0; i < dimension; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    for (j = 0; j < current_blocksize; j++) {
      dest[i][j] = B[i][j];
    }
  }
  return dest;
}


/**
 * @brief returns the list of block_sizes and sets nb_blocks to its correct value
 */
static size_t *coco_get_block_sizes(size_t *nb_blocks, size_t dimension, const char *suite_name){
  size_t *block_sizes;
  size_t block_size;
  size_t i;

  if (strcmp(suite_name, "bbob-largescale") == 0) {
    /*block_size = coco_double_to_size_t(bbob2009_fmin((double)dimension / 4, 100));*/ /*old value*/
    /*block_size = coco_double_to_size_t(bbob2009_fmin((double)dimension, 40));*/
    block_size = coco_rotation_matrix_block_size(dimension);
    *nb_blocks = dimension / block_size + ((dimension % block_size) > 0);
    block_sizes = coco_allocate_vector_size_t(*nb_blocks);
    for (i = 0; i < *nb_blocks - 1; i++) {
      block_sizes[i] = block_size;
    }
    block_sizes[*nb_blocks - 1] = dimension - (*nb_blocks - 1) * block_size; /*add rest*/
    return block_sizes;
  } else {
    return NULL;
  }
}
