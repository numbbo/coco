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

static long _RANDNSEED = 30;
static long _RANDSEED = 30;

void increase_random_n_seed(void){
  _RANDNSEED += 1;
  if (_RANDNSEED > (long) 1.0e9)
    _RANDNSEED = 1;
}

void increase_random_seed(void){
  _RANDSEED += 1;
  if (_RANDSEED > (long) 1.0e9)
    _RANDSEED = 1;
}

void reset_seeds(void){
  _RANDSEED = 30;
  _RANDNSEED = 30;
}

double coco_sample_gaussian_noise(void){
  double gaussian_noise;
  double gaussian_noise_ptr[1] = {0.0};
  increase_random_n_seed();
  bbob2009_gauss(&gaussian_noise_ptr[0], 1, _RANDNSEED);
  gaussian_noise = gaussian_noise_ptr[0];
  return gaussian_noise;
}

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
 * @name Methods regarding the noisy COCO samplers
 */

/**
 * @brief Samples gaussian noise for a noisy problem
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
 * @brief Samples uniform noise for a noisy problem
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
 * @brief Samples cauchy noise for a noisy problem
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
 * These are the methods to allocate and construct a coco_noisy_problem_t instance 
 * from a coco_problem_t one, mostly by means of wrapping functions around the 
 * original ones of the deterministic version of those problems,
 * in order to implement some sort of "inheritance" relation between the two types
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
    int test_noise_free_values = 0;
    double fopt = *(problem -> best_value);
    assert(problem -> evaluate_function != NULL);
    assert(problem -> noise_model != NULL);
    assert(problem -> noise_model -> noise_sampler != NULL);
    problem -> placeholder_evaluate_function(problem, x, y);
    if (test_noise_free_values == 0){
      *(y) = *(y) - fopt;
      problem -> noise_model -> noise_sampler(problem, y);
      *(y) = *(y) + fopt;
    }
}

/**
 * @brief Allocates the gaussian noise model to the coco_noisy_problem instance
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
 * @brief Allocates the gaussian noise model to the coco_noisy_problem instance
  * This new function signature is needed for the objective functions
  * that need the conditioning variable as argument for allocating the 
  * coco_problem_t
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
 * @brief Allocates the gaussian noise model to the coco_noisy_problem instance
  * This new function signature is needed for the objective functions
  * that need the conditioning variable as argument for allocating the 
  * coco_problem_t
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
