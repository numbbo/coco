/**
 * @file suite_noisy_utilities.c 
 * @brief Implementation of some functions (mostly handling instances) used by the bi-objective suites. 
 * These are used throughout the COCO code base but should not be used by any external code.
 */

#include "coco.h"
#include "coco_internal.h"
#include "coco_random.c"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"

/***********************************************************************************************************/
/**
 * @name Methods and global variables needed to perform random sampling
 */
/**{@*/
static long _RANDNSEED = 30; /** < @brief Random seed for sampling Uniform noise*/
static long _RANDSEED = 30;  /** < @brief Random seed for sampling Gaussian noise*/

/**
 * @brief Increases the normal random seed by one unit
 * Needed to sample different values at each time
 */
void increase_random_n_seed(void){
  _RANDNSEED += 1;
  if (_RANDNSEED > (long) 1.0e9)
    _RANDNSEED = 1;
}

/**
 * @brief Increases the uniform random seed by one unit
 * Needed to sample different values at each time
 */
void increase_random_seed(void){
  _RANDSEED += 1;
  if (_RANDSEED > (long) 1.0e9)
    _RANDSEED = 1;
}

/**
 * @brief Resets both random seeds to the initial values
 */
void reset_seeds(void){
  _RANDSEED = 30;
  _RANDNSEED = 30;
}

/**
 * @brief Returns a sample from the gaussian distribution 
 * using the legacy code for random number generations
 * Returns a double
 */
double coco_sample_gaussian_noise(void){
  double gaussian_noise;
  double gaussian_noise_ptr[1] = {0.0};
  increase_random_n_seed();
  bbob2009_gauss(&gaussian_noise_ptr[0], 1, _RANDNSEED);
  gaussian_noise = gaussian_noise_ptr[0];
  return gaussian_noise;
}


/**
 * @brief Returns a sample from the uniform distribution 
 * using the legacy code for random number generations
 * Returns a double
 */
double coco_sample_uniform_noise(void){
  double uniform_noise_term;
  double noise_vector[1] = {0.0};
  increase_random_seed();
  bbob2009_unif(&noise_vector[0], 1, _RANDSEED);
  uniform_noise_term = noise_vector[0];
  return uniform_noise_term;
}
/**@}*/

/***********************************************************************************************************/
/**
 * @name Methods regarding boundary handling
 */

/**
 * @brief Applies a penalty to solution outside the feasible hypercube 
 */
/**@{*/
double coco_boundary_handling(coco_problem_t * problem, const double * x){ 
  double penalty, lower_bound, upper_bound;
  size_t dimension;
  penalty = 0.0;
  for(dimension = 0; dimension < problem->number_of_variables; dimension ++ ){
    lower_bound = problem->smallest_values_of_interest[dimension];
    upper_bound = problem->largest_values_of_interest[dimension];
    if(x[dimension] > upper_bound){
      penalty += x[dimension] - upper_bound;
    }
    else if(x[dimension] < lower_bound){
      penalty += fabs(x[dimension] - lower_bound);
    }
  }
  return 100.0 * penalty;
}
/**@}*/

/***********************************************************************************************************/
