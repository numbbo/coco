#include <stdio.h>
#include <assert.h>
#include "coco.h"

/**
 * ls_allocate_blockmatrix(n, m, bs):
 *
 * Allocate a ${n} by ${m} block matrix of nb_blocks block sizes blocksizes structured as an array of pointers
 * to double arrays.
 * each row constains only the blocksizes[i] possibly non-zero elements
 */
static double **ls_allocate_blockmatrix(const size_t n, const size_t m, const size_t* blocksizes, const size_t nb_blocks) {
  double **matrix = NULL;
  size_t current_blocksize;
  size_t next_bs_change;
  size_t idx_blocksize;
  size_t i;
  size_t sum_blocksizes;
  assert(n == m);/* we only treat square matrices for now*/
  
  sum_blocksizes = 0;
  for (i = 0; i < nb_blocks; i++){
    sum_blocksizes += blocksizes[i];
  }
  assert(sum_blocksizes == n);
  
  matrix = (double **) coco_allocate_memory(sizeof(double *) * n);
  idx_blocksize = 0;
  next_bs_change = blocksizes[idx_blocksize];
  
  for (i = 0; i < n; ++i) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += blocksizes[idx_blocksize];
    }
    current_blocksize=blocksizes[idx_blocksize];
    matrix[i] = coco_allocate_vector(current_blocksize);
  }
  return matrix;
}

static void ls_free_blockmatrix(double **matrix, const size_t n) {
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
 * ls_compute_blockrotation(B, seed, DIM):
 *
 * Compute a ${DIM}x${DIM} block rotation matrix based on ${seed} and store
 * it in ${B}.
 */
static void ls_compute_blockrotation(double **B, long seed, size_t n, size_t *blocksizes, size_t nb_blocks) {
  double prod;
  double gvect[2000];
  double **current_block;
  long i, j, k; /* Loop over pairs of column vectors. */
  size_t idx_block, current_blocksize;
  
  cumsum_prev_blocksizes = 0;/* shift in rows to account for the previous blocks */
  for (idx_block = 0; idx_block < nb_blocks; idx_block++) {
    current_blocksize = blocksizes[idx_blocksize];
    current_block = bbob2009_allocate_matrix(current_blocksize, current_blocksize)
    bbob2009_gauss(gvect, current_blocksize * current_blocksize, seed);
    bbob2009_reshape(current_block, gvect, current_blocksize, current_blocksize);

    for (i = 0; i < current_blocksize; i++) {
      for (j = 0; j < i; j++) {
        prod = 0;
        for (k = 0; k < current_blocksize; k++)
          prod += current_block[k][i] * current_block[k][j];
        for (k = 0; k < current_blocksize; k++)
          current_block[k][i] -= prod * current_block[k][j];
      }
      prod = 0;
      for (k = 0; k < current_blocksize; k++)
        prod += B[k][i] * B[k][i];
      for (k = 0; k < current_blocksize; k++)
        B[k][i] /= sqrt(prod);
    }
    cum_prev_blocksizes+=current_blocksize;
    ls_free_blockmatrix(current_block, current_blocksize);
  }
    
}/*TODO: update with blocks*/

static void bbob2009_copy_rotation_matrix(double **rot, double *M, double *b, const size_t dimension) {
  size_t row, column;
  double *current_row;
  
  for (row = 0; row < dimension; ++row) {
    current_row = M + row * dimension;
    for (column = 0; column < dimension; ++column) {
      current_row[column] = rot[row][column];
    }
    b[row] = 0.0;
  }
}

