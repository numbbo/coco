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
 * @brief Resets both random seeds to the initial values
 */
void coco_reset_seeds(void){
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
  _RANDNSEED += 1;
  if (_RANDNSEED > (long) 1.0e9)
    _RANDNSEED = 1;
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
  _RANDSEED += 1;
  if (_RANDSEED > (long) 1.0e9)
    _RANDSEED = 1;
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
  double penalty = 0.0;
  for(size_t dimension = 0; dimension < problem -> number_of_variables; dimension ++ ){
    penalty += fabs(x[dimension]) - 5 > 0 ? pow(fabs(x[dimension]) - 5, 2) : 0; 
  }
  return 100.0 * penalty;
}
/**@}*/

/***********************************************************************************************************/
