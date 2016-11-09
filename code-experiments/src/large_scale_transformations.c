#include <stdio.h>
#include <assert.h>
#include "coco.h"

#include "coco_random.c" /*tmp*/
#include "suite_bbob_legacy_code.c" /*tmp*/

#include <time.h> /*tmp*/

/* TODO: Document this file in doxygen style! */

static double *ls_random_data;/* global variable used to generate the random permutations */

/**
 * ls_allocate_blockmatrix(n, m, bs):
 *
 * Allocate a ${n} by ${m} block matrix of nb_blocks block sizes block_sizes structured as an array of pointers
 * to double arrays.
 * each row constains only the block_sizes[i] possibly non-zero elements
 */
static double **ls_allocate_blockmatrix(const size_t n, const size_t* block_sizes, const size_t nb_blocks) {
  double **matrix = NULL;
  size_t current_blocksize;
  size_t next_bs_change;
  size_t idx_blocksize;
  size_t i;
  size_t sum_block_sizes;
  
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


/*
 * frees a block diagonal matrix (same as a matrix but in case of change, easier to update separatly from free_matrix)
 */
static void ls_free_block_matrix(double **matrix, const size_t n) {
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
 * Compute a ${DIM}x${DIM} block-diagonal matrix based on ${seed} and block_sizes and stores it in ${B}.
 * B is a 2D vector with DIM lines and each line has blocksize(line) elements (the zeros are not stored)
 */
static void ls_compute_blockrotation(double **B, long seed, size_t n, size_t *block_sizes, size_t nb_blocks) {
  double prod;
  /*double *gvect;*/
  double **current_block;
  size_t i, j, k; /* Loop over pairs of column vectors. */
  size_t idx_block, current_blocksize,cumsum_prev_block_sizes, sum_block_sizes;
  size_t nb_entries;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);
  
  nb_entries = 0;
  sum_block_sizes = 0;
  for (i = 0; i < nb_blocks; i++){
    sum_block_sizes += block_sizes[i];
    nb_entries += block_sizes[i] * block_sizes[i];
  }
  assert(sum_block_sizes == n);
  
  cumsum_prev_block_sizes = 0;/* shift in rows to account for the previous blocks */
  for (idx_block = 0; idx_block < nb_blocks; idx_block++) {
    current_blocksize = block_sizes[idx_block];
    current_block = bbob2009_allocate_matrix(current_blocksize, current_blocksize);
    for (i = 0; i < current_blocksize; i++) {
      for (j = 0; j < current_blocksize; j++) {
        current_block[i][j] = coco_random_normal(rng);
      }
    }
    
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
        B[i + cumsum_prev_block_sizes][j]=current_block[i][j];
      }
    }
    
    cumsum_prev_block_sizes+=current_blocksize;
    /*current_gvect_pos += current_blocksize * current_blocksize;*/
    ls_free_block_matrix(current_block, current_blocksize);
  }
  /*coco_free_memory(gvect);*/
  coco_random_free(rng);
}

/*
 * makes a copy of a block_matrix
 */
static double **ls_copy_block_matrix(const double *const *B, const size_t dimension, const size_t *block_sizes, const size_t nb_blocks) {
  double **dest;
  size_t i, j, idx_blocksize, current_blocksize, next_bs_change;
  
  dest = ls_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  idx_blocksize = 0;
  current_blocksize = block_sizes[idx_blocksize];
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
 * Comparison function used for sorting.
 * In our case, it serves as a random permutation generator
 */
static int f_compare_doubles_for_random_permutation(const void *a, const void *b) {
  double temp = ls_random_data[*(const size_t *) a] - ls_random_data[*(const size_t *) b];
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}

/*
 * generates a random, uniformly sampled, permutation and puts it in P
 */
static void ls_compute_random_permutation(size_t *P, long seed, size_t n) {
  long i;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);
  ls_random_data = coco_allocate_vector(n);
  for (i = 0; i < n; i++){
    P[i] = (size_t) i;
    ls_random_data[i] = coco_random_uniform(rng);
  }
  qsort(P, n, sizeof(size_t), f_compare_doubles_for_random_permutation);
  coco_random_free(rng);
}


/*
 * returns a uniformly distributed integer between lower_bound and upper_bound using seed.
 */
static long ls_rand_int(long lower_bound, long upper_bound, coco_random_state_t *rng){
  long range;
  range = upper_bound - lower_bound + 1;
  return ((long)(coco_random_uniform(rng) * (double) range)) + lower_bound;
}



/*
 * generates a random permutation resulting from nb_swaps truncated uniform swaps of range swap_range
 * missing paramteters: dynamic_not_static pool, seems empirically irrelevant
 * for now so dynamic is implemented (simple since no need for tracking indices
 * if swap_range is the largest possible size_t value ( (size_t) -1 ), a random uniform permutation is generated
 */
static void ls_compute_truncated_uniform_swap_permutation(size_t *P, long seed, size_t n, size_t nb_swaps, size_t swap_range) {
  long i, idx_swap;
  size_t lower_bound, upper_bound, first_swap_var, second_swap_var, tmp;
  size_t *idx_order;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);

  ls_random_data = coco_allocate_vector(n);
  idx_order = coco_allocate_vector_size_t(n);
  for (i = 0; i < n; i++){
    P[i] = (size_t) i;
    idx_order[i] = (size_t) i;
    ls_random_data[i] = coco_random_uniform(rng);
  }
  
  if (swap_range > 0) {
    /*sort the random data in random_data and arange idx_order accordingly*/
    /*did not use ls_compute_random_permutation to only use the seed once*/
    qsort(idx_order, n, sizeof(size_t), f_compare_doubles_for_random_permutation);
    for (idx_swap = 0; idx_swap < nb_swaps; idx_swap++) {
      first_swap_var = idx_order[idx_swap];
      if (first_swap_var < swap_range) {
        lower_bound = 0;
      }
      else{
        lower_bound = first_swap_var - swap_range;
      }
      if (first_swap_var + swap_range > n - 1) {
        upper_bound = n - 1;
      }
      else{
        upper_bound = first_swap_var + swap_range;
      }

      second_swap_var = (size_t) ls_rand_int((long) lower_bound, (long) upper_bound, rng);
      while (first_swap_var == second_swap_var) {
        second_swap_var = (size_t) ls_rand_int((long) lower_bound, (long) upper_bound, rng);
      }
      /* swap*/
      tmp = P[first_swap_var];
      P[first_swap_var] = P[second_swap_var];
      P[second_swap_var] = tmp;
    }
  } else {
    if ( swap_range == (size_t) -1) {
      /* generate random permutation instead */
      ls_compute_random_permutation(P, seed, n);
    }

  }
  coco_random_free(rng);
}



/*
 * duplicates a size_t vector
 */
static size_t *coco_duplicate_size_t_vector(const size_t *src, const size_t number_of_elements) {
  size_t i;
  size_t *dst;
  
  assert(src != NULL);
  assert(number_of_elements > 0);
  
  dst = coco_allocate_vector_size_t(number_of_elements);
  for (i = 0; i < number_of_elements; ++i) {
    dst[i] = src[i];
  }
  return dst;
}


/*
 * returns the list of block_sizes and sets nb_blocks to its correct value
 * TODO: update with chosen parameter setting
 */
static size_t *ls_get_block_sizes(size_t *nb_blocks, size_t dimension){
  size_t *block_sizes;
  size_t block_size;
  int i;

  block_size = coco_double_to_size_t(bbob2009_fmin((double)dimension / 4, 100));
  *nb_blocks = dimension / block_size + ((dimension % block_size) > 0);
  block_sizes = coco_allocate_vector_size_t(*nb_blocks);
  for (i = 0; i < *nb_blocks - 1; i++) {
    block_sizes[i] = block_size;
  }
  block_sizes[*nb_blocks - 1] = dimension - (*nb_blocks - 1) * block_size; /*add rest*/
  return block_sizes;
}


/*
 * return the swap_range corresponding to the problem
 */
static size_t ls_get_swap_range(size_t dimension){
  return dimension / 3;
}


/*
 * return the number of swaps corresponding to the problem
 */
size_t ls_get_nb_swaps(size_t dimension){
  return dimension;
}




