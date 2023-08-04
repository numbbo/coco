/**
 * @file suite_noisy_utilities.c 
 * @brief Implementation of some functions (mostly handling instances) used by the bi-objective suites. 
 * These are used throughout the COCO code base but should not be used by any external code.
 */

#include "coco.h"
#include "coco_random.c"
#include "coco_problem.c"

/***********************************************************************************************************/

/**
 * @name Getter methods for the coco_noisy_problem_t template
 */

uint32_t coco_problem_get_random_seed(const coco_noisy_problem_t *problem){
  assert(problem != NULL);
  assert(problem -> random_seed != NAN); /**<@ warning: comparison between pointer and integer>*/
  return problem ->  noise_model -> random_seed;
}

double *coco_problem_get_distribution_theta(const coco_noisy_problem_t *problem){
  assert(problem != NULL);
  assert(problem -> distribution_theta != NULL);
  return problem -> noise_model -> distribution_theta;
} 

double coco_problem_get_last_noise_value(const coco_noisy_problem_t *problem){
  assert(problem != NULL);
  assert(problem -> last_noise_value != NAN);
  return problem -> noise_model -> last_noise_value;
}

/***********************************************************************************************************/

/**
 * @name Methods regarding the noisy COCO samplers
 */

/**
 * @brief Samples gaussian noise for a noisy problem
 */
/**@{*/
void coco_problem_sample_gaussian_noise(coco_noisy_problem_t * problem, const double fvalue){
  uint32_t random_seed = coco_problem_get_random_seed(problem);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  assert(random_seed != NAN);
  double scale = *(distribution_theta);
  coco_random_state_t * coco_seed = coco_random_new(random_seed);
  double gaussian_noise = coco_random_normal(coco_seed);
  gaussian_noise = exp(scale * gaussian_noise);
  problem -> last_noise_value = gaussian_noise;
}

/**
 * @brief Samples uniform noise for a noisy problem
 */
void coco_problem_sample_uniform_noise(coco_noisy_problem_t * problem, const double fvalue){
  uint32_t random_seed = coco_problem_get_random_seed(problem);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  assert(random_seed != NAN);
  double alpha = *(distribution_theta);
  double beta = *(distribution_theta++);
  coco_random_state_t * coco_seed1 = coco_random_new(random_seed);
  coco_random_state_t * coco_seed2 = coco_random_new(random_seed);
  double uniform_noise_term1 = coco_random_uniform(coco_seed1);
  double uniform_noise_term2 = coco_random_uniform(coco_seed2);
  double uniform_noise_factor = pow(uniform_noise_term1, beta);
  double scaling_factor = pow(10, 9)/(fvalue + 10e-99);
  scaling_factor = pow(scaling_factor, alpha * uniform_noise_term2);
  scaling_factor = scaling_factor > 1 ? scaling_factor : 1;
  double uniform_noise = uniform_noise_factor * scaling_factor;
  problem -> last_noise_value = uniform_noise;
}

/**
 * @brief Samples cauchy noise for a noisy problem
 */
void coco_problem_sample_cauchy_noise(coco_noisy_problem_t * problem, const double fvalue){
  uint32_t random_seed = coco_problem_get_random_seed(problem);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  assert(random_seed != NAN);
  double alpha = *(distribution_theta);
  double p = *(distribution_theta++);
  coco_random_state_t * coco_seed1 = coco_random_new(random_seed);
  coco_random_state_t * coco_seed2 = coco_random_new(random_seed); 
  coco_random_state_t * coco_seed3 = coco_random_new(random_seed);
  double uniform_indicator = coco_random_uniform(coco_seed1);
  double numerator_normal_variate = coco_random_normal(coco_seed2);
  double denominator_normal_variate = coco_random_normal(coco_seed3);
  denominator_normal_variate = fabs(denominator_normal_variate);
  double cauchy_noise = numerator_normal_variate / (denominator_normal_variate + 10e-99);
  cauchy_noise = uniform_indicator < p ?  1000 + cauchy_noise : 1000;
  cauchy_noise = alpha * cauchy_noise;
  problem -> last_noise_value = cauchy_noise;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Noise models
 */

/**{@*/
/**
  * @brief Applies additive noise to the function
  * Used to apply the additive noise model to the function value
  * Should be used as a wrapper around f_<function_name>_evaluate
 */
void coco_problem_evaluate_additive_noise_model(
        const coco_noisy_problem_t *problem,
        const double * x,  
        double * y
    ){
    double noise_value = coco_problem_get_last_noise_value(problem);
    y[0] = y[0] + noise_value;
}

/**
  * @brief Applies additive noise to the function
  * Used to apply the additive noise model to the function value
  * Should be used as a wrapper around f_<function_name>_evaluate
 */
void coco_problem_evaluate_multiplicative_noise_model(
        const coco_noisy_problem_t *problem, 
        const double * x,
        double * y
    ){
    double noise_value = coco_problem_get_last_noise_value(problem);
    y[0] = y[0] * noise_value;
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
static void coco_problem_f_evaluate_wrap_noisy(
        const coco_noisy_problem_t *problem, 
        const double *x, 
        double *y
    ){
    assert(inner_problem -> evaluate_function != NULL);
    assert(problem -> noise_sampler != NULL);
    assert(problem -> noise_model != NULL);
    problem -> inner_problem -> evaluate_function(inner_problem, x, y);
    problem -> noise_model -> noise_sampler(problem, y[0]);
    problem -> noise_model -> noise_model_evaluator(problem, y);
}

/**
  * @brief Allocates a coco_noisy_problem_t instance from a coco_problem_t one
  * @param inner_problem The inner problem as input from which to construct the returned coco_noisy_problem instance
  * @return problem, a coco_noisy_problem_t "instance" 
 */
static coco_noisy_problem_t *coco_problem_allocate_f_wrap_noisy(
        const coco_problem_t *inner_problem,
    ){
    coco_noisy_problem_t problem;
    problem -> inner_problem = inner_problem;
    problem -> evaluate_noisy_function = coco_problem_f_evaluate_wrap_noisy;
    coco_problem_get_noisy_problem_id_from_problem_id(inner_problem, problem);
    return problem;
}

/**
  * @brief Evaluates the coco problem noisy function
  * works as a wrapper around the function type coco_problem_f_evaluate
  * @param noise_model Noise model used to evaluate the function (additive or multiplicative)
  * @param noise_sampler Noise sampler used to compute the noise values
  * @param coco_f_evaluator Function evaluator used to compute the deterministic function value
 */
static coco_noisy_problem_t *coco_problem_allocate_bbob_wrap_noisy(
        const coco_problem_bbob_allocator_t *coco_problem_bbob_allocator_t,
        const size_t function, 
        const size_t dimension, 
        const size_t instance, 
        const long rseed, 
        const char *problem_id_template, 
        const char *problem_name_template, 
        const uint32_t random_seed, 
        const double *distribution_theta,
        const coco_problem_evaluate_noise_model_t *noise_model, 
        const coco_problem_noise_sampler_t *noise_sampler,
    ){
    coco_problem_t *inner_problem = coco_problem_bbob_allocator_t(
      function, 
      dimension, 
      instance, 
      rseed, 
      problem_id_template, 
      problem_name_template
    );
    coco_noisy_problem_t *problem = coco_problem_allocate_f_wrap_noisy(inner_problem);
    problem -> noise_model -> random_seed = random_seed;
    problem -> noise_model -> distribution_theta = distribution_theta;
    problem -> noise_model -> noise_sampler = noise_sampler;
    problem -> noise_model -> noise_model_evaluator = noise_model;
}

/**
  * @brief Sets the noisy problem id from its inner problem id, noise sampler and theta distribution
 */
void coco_problem_get_noisy_problem_id_from_problem_id(coco_problem_t inner_problem, coco_noisy_problem_t problem){
  ;
}
/**@}*/

/***********************************************************************************************************/
