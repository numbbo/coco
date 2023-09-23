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

/***********************************************************************************************************/

/**
 * @name Methods evaluating noise models
 */

/**
 * @brief Evaluates bbob-noisy gaussian noise model 
 */
/**@{*/
void coco_problem_gaussian_noise_model(coco_problem_t * problem, double *y){
  double fvalue = *(y);
  assert(fvalue != NAN);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  double scale = *(distribution_theta);
  double gaussian_noise = coco_sample_gaussian_noise();
  gaussian_noise = exp(scale * gaussian_noise);
  problem -> last_noise_value = gaussian_noise;
  double tol = 1e-8;
  y[0] = y[0] * problem -> last_noise_value  + 1.01 * tol;
}

/**
 * @brief Evaluates bbob-noisy uniform noise model
 */
void coco_problem_uniform_noise_model(coco_problem_t * problem, double *y){
  double fvalue = *(y);
  assert(fvalue != NAN);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  double alpha = distribution_theta[0];
  double beta = distribution_theta[1];  
  double uniform_noise_term1, uniform_noise_term2;
  uniform_noise_term1 = coco_sample_uniform_noise();
  uniform_noise_term2 = coco_sample_uniform_noise();
  double uniform_noise_factor = pow(uniform_noise_term1, beta);
  double scaling_factor = 1e9/(fvalue + 1e-99);
  scaling_factor = pow(scaling_factor, alpha * uniform_noise_term2);
  scaling_factor = scaling_factor > 1 ? scaling_factor : 1;
  double uniform_noise = uniform_noise_factor * scaling_factor;
  problem -> last_noise_value = uniform_noise;
  double tol = 1e-8;
  y[0] = y[0] * problem -> last_noise_value + 1.01 * tol; 
}

/**
 * @brief Evaluates bbob-noisy cauchy noise model
 */
void coco_problem_cauchy_noise_model(coco_problem_t * problem, double *y){
  double fvalue = *(y);
  assert(fvalue != NAN);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  double alpha = distribution_theta[0];
  double p = distribution_theta[1];
  double uniform_indicator, numerator_normal_variate, denominator_normal_variate;
  uniform_indicator = coco_sample_uniform_noise();
  numerator_normal_variate = coco_sample_gaussian_noise();
  denominator_normal_variate = coco_sample_gaussian_noise();
  denominator_normal_variate = fabs(denominator_normal_variate  + 1e-199);
  double cauchy_noise = numerator_normal_variate / (denominator_normal_variate);
  cauchy_noise = uniform_indicator < p ?  1e3 + cauchy_noise : 1e3;
  cauchy_noise = alpha * cauchy_noise;
  problem -> last_noise_value = cauchy_noise > 0 ? cauchy_noise : 0.;
  double tol = 1e-8;
  y[0] = y[0] + problem -> last_noise_value + 1.01 * tol;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding the noisy COCO problem wrapper functions
 * These are the methods to allocate and construct a noisy instances 
 * of a coco_problem_t, mostly by means of wrapping functions around the 
 * original deterministic ones.
 */

/**
  * @brief Evaluates the coco problem noisy function
  * works as a wrapper around the function type coco_problem_f_evaluate
  * 1) Computes deterministic function value
  * 2) Samples the noise according to the given distribution
  * 3) Computes the final function value by applying the given noise model 
 */
void coco_problem_f_evaluate_wrap_noisy(
        coco_problem_t *problem, 
        const double *x, 
        double *y
    ){
    double fopt = *(problem -> best_value);
    assert(problem -> evaluate_function != NULL);
    assert(problem -> noise_model != NULL);
    assert(problem -> noise_model -> noise_sampler != NULL);
    problem -> placeholder_evaluate_function(problem, x, y);
    *(y) = *(y) - fopt;
    problem -> noise_model -> noise_sampler(problem, y);
    *(y) = *(y) + fopt;
}

/**
 * @brief Allocates the bbob problem
 * @param coco_problem_bbbob_allocator The function allocating the noiseless problem
 * @param noise_model The function evaluating the noise model
 * It can be one of the three functions defined in the "Methods evaluating noise models"
 * The additional arguments are the ones that need to be passed to the coco_problem_bbob_allocator function
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy(
    coco_problem_bbob_allocator_t coco_problem_bbob_allocator,
    coco_problem_evaluate_noise_model_t noise_model,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed, 
    const char *problem_id_template, 
    const char *problem_name_template,
    double *distribution_theta

  ){
  coco_problem_t *problem = NULL;
  problem = coco_problem_bbob_allocator(
    function,
    dimension, 
    instance,
    rseed, 
    problem_id_template, 
    problem_name_template 
  );
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}

/**
 * @brief Allocates the bbob problem
 * @param coco_problem_bbbob_allocator The function allocating the noiseless problem
 * @param noise_model The function evaluating the noise model
 * @param conditioning The conditioning value for the problem instance
 * It can be one of the three functions defined in the "Methods evaluating noise models"
 * The additional arguments are the ones that need to be passed to the coco_problem_bbob_allocator function.
 * The definition of this function is required because there are some objective functions (schaffers and ellipsoid)
 * that require an additional argument, a double, the conditioning indeed, to be passed to the coco_problem_allocator function
 * Initially, this was not required for the ellipsoid function, nor for the schaffers one. However, in the old code, these parameters
 * were different across the noiseless and noisy version of the same objective function. 
 * This difference required the definition of a different wrapper for these particular functions.
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_conditioned(
    coco_problem_bbob_conditioned_allocator_t coco_problem_bbob_allocator,
    coco_problem_evaluate_noise_model_t noise_model,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const double conditioning,
    const char *problem_id_template, 
    const char *problem_name_template, 
    double *distribution_theta

  ){
  coco_problem_t *problem = NULL;
  problem = coco_problem_bbob_allocator(
    function,
    dimension, 
    instance,
    rseed,
    conditioning, 
    problem_id_template, 
    problem_name_template 
  );
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}


/**
 * @brief Allocates the bbob problem
 * @param coco_problem_bbbob_allocator The function allocating the noiseless problem
 * @param noise_model The function evaluating the noise model
 * @param conditioning The conditioning value for the problem instance
 * It can be one of the three functions defined in the "Methods evaluating noise models"
 * The additional arguments are the ones that need to be passed to the coco_problem_bbob_allocator function.
 * The definition of this function is required because the gallagher objective function
 * requires an additional argument, a size_t, the number of peaks, to be passed to the coco_problem_allocator function.
 * Initially, this was not required for the ellipsoid function, nor for the schaffers one. However, in the old code, this parameter
 * was different across the noiseless and noisy version of the same objective function. 
 * This difference required the definition of a different wrapper for these particular functions.
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_gallagher(
    coco_problem_bbob_gallagher_allocator_t coco_problem_bbob_allocator,
    coco_problem_evaluate_noise_model_t noise_model,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const size_t n_peaks,
    const char *problem_id_template, 
    const char *problem_name_template, 
    double *distribution_theta

  ){
  coco_problem_t *problem = NULL;
  problem = coco_problem_bbob_allocator(
    function,
    dimension, 
    instance,
    rseed,
    n_peaks, 
    problem_id_template, 
    problem_name_template 
  );
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods allocating the parameter of the noise model distribution
 */

/**
 * @brief Allocates the parameters of the gaussian noise model given the level of severeness of the noise
 */
void coco_gaussian_noise_model_allocate_params(double *distribution_theta, const size_t severeness){
  assert(severeness == 0 || severeness == 1);
  double sigma;
  if(severeness == 0){
    sigma = 0.01;
  }
  if(severeness == 1){
    sigma = 1.0;
  }
  distribution_theta[0] = sigma;
}

/**
  * @brief Allocates the parameter of the uniform noise model given the level of severeness of the noise
  */
void coco_uniform_noise_model_allocate_params(double *distribution_theta, const size_t severeness, size_t dimension){
  assert(severeness == 0 || severeness == 1);
  double alpha, beta;
  if(severeness == 0){
    alpha = 0.01 * (0.49 + 1 / (double) dimension);
    beta = 0.01;
  }
  if(severeness == 1){
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
  }
  distribution_theta[0] = alpha;
  distribution_theta[1] = beta;
}

/**
  * @brief Allocates the parameter of the cauchy noise model given the level of severeness of the noise
  */
void coco_cauchy_noise_model_allocate_params(double *distribution_theta, const size_t severeness){
  assert(severeness == 0 || severeness == 1);
  double alpha, p;
  if(severeness == 0){
    alpha = 0.01;
    p = 0.05;
  }
  if(severeness == 1){
    alpha = 1.0; 
    p = 0.2; 
  }
  distribution_theta[0] = alpha;
  distribution_theta[1] = p;
}
