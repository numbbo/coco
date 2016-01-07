#include <stdio.h>
#include <assert.h>
#include "coco.h"

#include "coco_runtime_c.c" //
#include "suite_bbob2009_legacy_code.c"//
#include <time.h>//

static double *random_data;/* global variable used to generate the random permutations */

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
  size_t idx_block, current_blocksize,cumsum_prev_blocksizes;
  
  cumsum_prev_blocksizes = 0;/* shift in rows to account for the previous blocks */
  for (idx_block = 0; idx_block < nb_blocks; idx_block++) {
    current_blocksize = blocksizes[idx_block];
    current_block = bbob2009_allocate_matrix(current_blocksize, current_blocksize);
    bbob2009_gauss(gvect, current_blocksize * current_blocksize, seed);
    bbob2009_reshape(current_block, gvect, current_blocksize, current_blocksize);

    for (i = 0; i < current_blocksize; i++) {
      for (j = 0; j < i; j++) {
        prod = 0;
        for (k = 0; k < current_blocksize; k++){
          prod += current_block[k][i] * current_block[k][j];
        }
        for (k = 0; k < current_blocksize; k++){
          current_block[k][i] -= prod * current_block[k][j];
        }
      }
      prod = 0;
      for (k = 0; k < current_blocksize; k++){
        prod += current_block[k][i] * current_block[k][i];
      }
      for (k = 0; k < current_blocksize; k++){
        current_block[k][i] /= sqrt(prod);
      }
    }
    
    /* now fill the block matrix*/
    for (i = 0 ; i < current_blocksize; i++) {
      for (j = 0; j < current_blocksize; j++) {
        B[i+cumsum_prev_blocksizes][j]=current_block[i][j];
      }
    }
    
    cumsum_prev_blocksizes+=current_blocksize;
    ls_free_blockmatrix(current_block, current_blocksize);
  }
    
}

/*
 * copy the 2-D matrix into a 1-D vector
 */
static void ls_copy_block_matrix(double **rot, double *M, const size_t dimension, const size_t *blocksizes, const size_t nb_blocks) {
  size_t row, column, idx_blocksize, current_blocksize, current_idx, next_bs_change;
  double *current_row;
  
  current_idx = 0;
  idx_blocksize = 0;
  current_blocksize = blocksizes[idx_blocksize];
  next_bs_change = blocksizes[idx_blocksize];

  for (row = 0; row < dimension; ++row) {
    if (row >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += blocksizes[idx_blocksize];
    }
    current_blocksize=blocksizes[idx_blocksize];
    for (column = 0; column < current_blocksize; column++, current_idx++) {
      M[current_idx] = rot[row][column];
    }
  }
}

/**
 * Comparison function used for sorting.
 */
static int f_compare_doubles_for_random_permutation(const void *a, const void *b) {
  double temp = random_data[*(const size_t *) a] - random_data[*(const size_t *) b];
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}


static void ls_compute_random_permutation(size_t *P, long seed, size_t n) {
  //size_t *rperm;
  long i;
  random_data = coco_allocate_vector(n);
  bbob2009_unif(random_data, n, seed);
  //rperm = (size_t *) coco_allocate_memory(n * sizeof(size_t));
  for (i = 0; i < n; i++)
    P[i] = i;
  qsort(P, n, sizeof(size_t), f_compare_doubles_for_random_permutation);
  
  //coco_free_memory(rperm);
  
}

int main(){
  int testBlockMatrix = 0;
  int testPermutations = 1;

  size_t nb_blocks = 3;
  size_t blocksizes[3] = {2,3,1};
  size_t n, nb_entries, idx_blocksize, next_bs_change, current_blocksize, idx_block, cumsum_prev_blocksizes;
  long seed;
  srand(time(NULL));
  seed=rand();

  
  if (testPermutations) {
    size_t *P;
    int i, j, nbRuns = 19;
    n = 10;
    P = (size_t *) coco_allocate_memory(n * sizeof(size_t));
    for (j = 0; j < nbRuns; j++) {
      seed = rand();
      ls_compute_random_permutation(P, seed, n);
      for (i = 0; i < n; i++) {
        printf("%zu ",P[i]);
      }
      printf("\n");
      }
  }
  
  
  if (testBlockMatrix) {
    double **B, *R, prod, sum;
    int i, j, j1, j2;
    n=0;
    nb_entries = 0;
    for (i = 0; i < nb_blocks; i++) {
      n += blocksizes[i];
      nb_entries += blocksizes[i] * blocksizes[i];
    }
    printf("nb_entries:%zu\n",nb_entries);
    B = ls_allocate_blockmatrix(n,n,blocksizes,nb_blocks);
    ls_compute_blockrotation(B, seed, n, blocksizes, nb_blocks);
    R = coco_allocate_vector(nb_entries);
    ls_copy_block_matrix(B, R, n, blocksizes, nb_blocks);

    printf("1-D version: \n");
    for (i = 0; i < nb_entries; i++) {
      printf("%f ", R[i]);
    }
    printf("\n");
    
    // show matrix
    idx_blocksize = 0;
    next_bs_change = blocksizes[idx_blocksize];
    for (i = 0; i < n; ++i) {
      if (i >= next_bs_change) {
        idx_blocksize++;
        next_bs_change += blocksizes[idx_blocksize];
      }
      current_blocksize=blocksizes[idx_blocksize];
      for (j = 0; j < current_blocksize; j++) {
        printf("%f ",B[i][j]);
      }
      printf("\n");
    }
    
    // test orthonormality
    printf("\ntest orthonormality\n");
    cumsum_prev_blocksizes = 0;
    for (idx_block = 0; idx_block < nb_blocks; idx_block++) {
      current_blocksize = blocksizes[idx_block];
      for (j1 = 0; j1 < current_blocksize; j1++) {
        for (j2 = 0; j2 < j1; j2++) {
          prod = 0;
          for (i = 0; i < current_blocksize; i++){
            prod += B[i + cumsum_prev_blocksizes][j1] * B[i + cumsum_prev_blocksizes][j2];
          }
          printf("block%lu <%d,%d>: %f\n", idx_block + 1, j2 + 1, j1 + 1, prod);
        }
        
        sum = 0;
        for (i = 0; i < current_blocksize; i++) {
          sum += B[i + cumsum_prev_blocksizes][j1] * B[i + cumsum_prev_blocksizes][j1];
        }
        printf("block%lu |%d|: %f\n", idx_block + 1, j1 + 1, sum);
      }

      cumsum_prev_blocksizes+=current_blocksize;
    }
  }
  
  
  
  
  //ls_free_blockmatrix(B,n);
  
  
  return 0;
  
}

