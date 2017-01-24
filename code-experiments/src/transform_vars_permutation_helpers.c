/**
 * @file transform_vars_permutation_helpers.c
 * @brief implements fonctions needed by transform_vars_permutation.c
 */

#include <stdio.h>
#include <assert.h>
#include "coco.h"

#include "coco_random.c"
#include "suite_bbob_legacy_code.c" /*tmp*/

#include <time.h> /*tmp*/

/* TODO: Document this file in doxygen style! */

static double *perm_random_data;/* global variable used to generate the random permutations */

/**
 * @brief Comparison function used for sorting. In our case, it serves as a random permutation generator
 */
static int f_compare_doubles_for_random_permutation(const void *a, const void *b) {
  double temp = perm_random_data[*(const size_t *) a] - perm_random_data[*(const size_t *) b];
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}

/**
 * @brief generates a random, uniformly sampled, permutation and puts it in P
 * Wassim: move to coco_utilities?
 */
static void coco_compute_random_permutation(size_t *P, long seed, size_t n) {
  long i;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);
  perm_random_data = coco_allocate_vector(n);
  for (i = 0; i < n; i++){
    P[i] = (size_t) i;
    perm_random_data[i] = coco_random_uniform(rng);
  }
  qsort(P, n, sizeof(size_t), f_compare_doubles_for_random_permutation);
  coco_random_free(rng);
}


/**
 * @brief returns a uniformly distributed integer between lower_bound and upper_bound using seed.
 * Wassim: move to coco_utilities?
 */
static long coco_random_unif_int(long lower_bound, long upper_bound, coco_random_state_t *rng){
  long range;
  range = upper_bound - lower_bound + 1;
  return ((long)(coco_random_uniform(rng) * (double) range)) + lower_bound;
}



/**
 * @brief generates a random permutation resulting from nb_swaps truncated uniform swaps of range swap_range
 * missing paramteters: dynamic_not_static pool, seems empirically irrelevant
 * for now so dynamic is implemented (simple since no need for tracking indices
 * if swap_range is the largest possible size_t value ( (size_t) -1 ), a random uniform permutation is generated
 */
static void coco_compute_truncated_uniform_swap_permutation(size_t *P, long seed, size_t dimension, size_t nb_swaps, size_t swap_range) {
  long i, idx_swap;
  size_t lower_bound, upper_bound, first_swap_var, second_swap_var, tmp;
  size_t *idx_order;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);
  
  perm_random_data = coco_allocate_vector(dimension);
  idx_order = coco_allocate_vector_size_t(dimension);
  for (i = 0; i < dimension; i++){
    P[i] = (size_t) i;
    idx_order[i] = (size_t) i;
    perm_random_data[i] = coco_random_uniform(rng);
  }
  if (dimension <= 40) {/* provisional solution to have the large-scale suite backward compatible with the bbob suite. Should probably take the value from the bbob suite itself.*/
    coco_random_free(rng);
    coco_free_memory(idx_order);
    return; /* return the identity permutation*/
  }
  
  if (swap_range > 0) {
    /*sort the random data in random_data and arange idx_order accordingly*/
    /*did not use coco_compute_random_permutation to only use the seed once*/
    qsort(idx_order, dimension, sizeof(size_t), f_compare_doubles_for_random_permutation);
    for (idx_swap = 0; idx_swap < nb_swaps; idx_swap++) {
      first_swap_var = idx_order[idx_swap];
      if (first_swap_var < swap_range) {
        lower_bound = 0;
      }
      else{
        lower_bound = first_swap_var - swap_range;
      }
      if (first_swap_var + swap_range > dimension - 1) {
        upper_bound = dimension - 1;
      }
      else{
        upper_bound = first_swap_var + swap_range;
      }
      
      second_swap_var = (size_t) coco_random_unif_int((long) lower_bound, (long) upper_bound, rng);
      while (first_swap_var == second_swap_var) {
        second_swap_var = (size_t) coco_random_unif_int((long) lower_bound, (long) upper_bound, rng);
      }
      /* swap*/
      tmp = P[first_swap_var];
      P[first_swap_var] = P[second_swap_var];
      P[second_swap_var] = tmp;
    }
  } else {
    if ( swap_range == (size_t) -1) {
      /* generate random permutation instead */
      coco_compute_random_permutation(P, seed, dimension);
    }
    
  }
  coco_free_memory(idx_order);
  coco_random_free(rng);
}



/**
 * @brief duplicates a size_t vector
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



/**
 * @brief return the swap_range corresponding to the problem in the given suite
 */
static size_t coco_get_swap_range(size_t dimension, const char *suite_name){
  if (strcmp(suite_name, "bbob-largescale") == 0) {
    return dimension / 3;
  } else {
    coco_error("coco_get_swap_range(): unknown problem suite");
    return (size_t) NULL;
  }
}


/**
 * @brief return the number of swaps corresponding to the problem in the given suite
 */
size_t coco_get_nb_swaps(size_t dimension, const char *suite_name){
  if (strcmp(suite_name, "bbob-largescale") == 0) {
    return dimension;
  } else {
    coco_error("coco_get_nb_swaps(): unknown problem suite");
    return (size_t) NULL;
  }
}




