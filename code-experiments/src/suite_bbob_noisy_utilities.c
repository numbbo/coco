/**
 * @file suite_noisy_utilities.c 
 * @brief Implementation of some functions (mostly handling instances) used by the bi-objective suites. 
 * These are used throughout the COCO code base but should not be used by any external code.
 */

#include "coco.h"
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

/***********************************************************************************************************/

/**
 * @name Methods regarding the noisy COCO samplers
 */

/**
 * @brief Samples gaussian noise for a noisy problem
 */
/**@{*/
void coco_problem_sample_gaussian_noise(coco_problem_t * problem, double *y){
  int test_against_bbob2009 = 1;
  double fvalue = *(y);
  assert(fvalue != NAN);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  double scale = *(distribution_theta);
  double gaussian_noise;
  if (test_against_bbob2009 != 1){
    uint32_t random_seed = (uint32_t) coco_problem_get_random_seed(problem);
    assert(random_seed != NAN);
    coco_random_state_t * coco_state = coco_random_new(random_seed);
    gaussian_noise = coco_random_normal(coco_state);
  }
  else{
    double gaussian_noise_ptr[1] = {0.0};
    increase_random_n_seed();
    bbob2009_gauss(&gaussian_noise_ptr[0], 1, _RANDNSEED);
    gaussian_noise = gaussian_noise_ptr[0];
  }
  gaussian_noise = exp(scale * gaussian_noise);
  problem -> last_noise_value = gaussian_noise;
}

/**
 * @brief Samples uniform noise for a noisy problem
 */
void coco_problem_sample_uniform_noise(coco_problem_t * problem, double *y){
  int test_against_bbob2009 = 1;
  double fvalue = *(y);
  assert(fvalue != NAN);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  double alpha = distribution_theta[0];
  double beta = distribution_theta[1];  
  double uniform_noise_term1, uniform_noise_term2;
  if (test_against_bbob2009 != 1){
    uint32_t random_seed = (uint32_t) coco_problem_get_random_seed(problem);
    assert(random_seed != NAN);
    coco_random_state_t * coco_state1 = coco_random_new(random_seed);
    coco_random_state_t * coco_state2 = coco_random_new(random_seed);
    uniform_noise_term1 = coco_random_uniform(coco_state1);
    uniform_noise_term2 = coco_random_uniform(coco_state2);
  }
  else{
    double noise_vector_1[1] = {0.0};
    double noise_vector_2[1] = {0.0};
    increase_random_seed();
    bbob2009_unif(&noise_vector_1[0], 1, _RANDSEED);
    uniform_noise_term1 = noise_vector_1[0];
    increase_random_seed();
    bbob2009_unif(&noise_vector_2[0], 1, _RANDSEED);
    uniform_noise_term2 = noise_vector_2[0];
  }
  double uniform_noise_factor = pow(uniform_noise_term1, beta);
  double scaling_factor = 1e9/(fvalue + 1e-99);
  scaling_factor = pow(scaling_factor, alpha * uniform_noise_term2);
  scaling_factor = scaling_factor > 1 ? scaling_factor : 1;
  double uniform_noise = uniform_noise_factor * scaling_factor;
  problem -> last_noise_value = uniform_noise;
}

/**
 * @brief Samples cauchy noise for a noisy problem
 */
void coco_problem_sample_cauchy_noise(coco_problem_t * problem, double *y){
  int test_against_bbob2009 = 1;
  double fvalue = *(y);
  assert(fvalue != NAN);
  double *distribution_theta = coco_problem_get_distribution_theta(problem);
  double alpha = distribution_theta[0];
  double p = distribution_theta[1];
  double uniform_indicator, numerator_normal_variate, denominator_normal_variate;
  if(test_against_bbob2009 != 1){
    uint32_t random_seed = (uint32_t) coco_problem_get_random_seed(problem);
    assert(random_seed != NAN);
    coco_random_state_t * coco_state1 = coco_random_new(random_seed);
    coco_random_state_t * coco_state2 = coco_random_new(random_seed); 
    coco_random_state_t * coco_state3 = coco_random_new(random_seed);
    uniform_indicator = coco_random_uniform(coco_state1);
    numerator_normal_variate = coco_random_normal(coco_state2);
    denominator_normal_variate = coco_random_normal(coco_state3);
  }
  else{
    double noise_vector_unif[1] = {0.0};
    double noise_vector_normal_1[1] = {0.0};
    double noise_vector_normal_2[1] = {0.0};
    increase_random_seed();
    bbob2009_unif(&noise_vector_unif[0], 1, _RANDSEED);
    increase_random_n_seed();
    bbob2009_gauss(&noise_vector_normal_1[0], 1, _RANDNSEED);
    increase_random_n_seed();
    bbob2009_gauss(&noise_vector_normal_2[0], 1, _RANDNSEED);
    uniform_indicator = noise_vector_unif[0];
    numerator_normal_variate = noise_vector_normal_1[0];
    denominator_normal_variate = noise_vector_normal_2[0];
  }
  denominator_normal_variate = fabs(denominator_normal_variate  + 1e-199);
  double cauchy_noise = numerator_normal_variate / (denominator_normal_variate);
  cauchy_noise = uniform_indicator < p ?  1e3 + cauchy_noise : 1e3;
  cauchy_noise = alpha * cauchy_noise;
  problem -> last_noise_value = cauchy_noise > 0 ? cauchy_noise : 0.;
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
        coco_problem_t *problem,
        double * y
    ){
    double tol = 1e-8;
    double noise_value = coco_problem_get_last_noise_value(problem);
    y[0] = y[0] + noise_value + 1.01 * tol;
}

/**
  * @brief Applies additive noise to the function
  * Used to apply the additive noise model to the function value
  * Should be used as a wrapper around f_<function_name>_evaluate
 */
void coco_problem_evaluate_multiplicative_noise_model(
        coco_problem_t *problem,
        double * y
    ){
    double tol = 1e-8;
    double noise_value = coco_problem_get_last_noise_value(problem);
    y[0] = y[0] * noise_value + 1.01 * tol;

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
    assert(problem -> noise_model -> noise_model_evaluator != NULL);
    problem -> placeholder_evaluate_function(problem, x, y);
    if (test_noise_free_values == 0){
      *(y) = *(y) - fopt;
      problem -> noise_model -> noise_sampler(problem, y);
      problem -> noise_model -> noise_model_evaluator(problem, y);
      *(y) = *(y) + fopt;
    }
}

/**
 * @brief Allocates the gaussian noise model to the coco_noisy_problem instance
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_gaussian(
    coco_problem_bbob_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed, 
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_gaussian_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_multiplicative_noise_model;
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
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_gaussian_conditioned(
    coco_problem_bbob_conditioned_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const double conditioning,
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_gaussian_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_multiplicative_noise_model;
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
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_gaussian_gallagher(
    coco_problem_bbob_gallagher_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const size_t n_peaks,
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_gaussian_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_multiplicative_noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}

/**
 * @brief Allocates the uniform noise model to the coco_noisy_problem instance
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_uniform(
    const coco_problem_bbob_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed, 
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_uniform_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_multiplicative_noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}

/**
 * @brief Allocates the uniform noise model to the coco_noisy_problem instance
  * This new function signature is needed for the objective functions
  * that need the conditioning variable as argument for allocating the 
  * coco_problem_t 
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_uniform_conditioned(
    const coco_problem_bbob_conditioned_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const double conditioning, 
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_uniform_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_multiplicative_noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}

/**
 * @brief Allocates the uniform noise model to the coco_noisy_problem instance
  * This new function signature is needed for the objective functions
  * that need the conditioning variable as argument for allocating the 
  * coco_problem_t 
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_uniform_gallagher(
    const coco_problem_bbob_gallagher_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const size_t n_peaks, 
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_uniform_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_multiplicative_noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}

/**
 * @brief Allocates the cauchy noise model to the coco_noisy_problem instance
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_cauchy(
    const coco_problem_bbob_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed, 
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_cauchy_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_additive_noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}

/**
 * @brief Allocates the cauchy noise model to the coco_noisy_problem instance
 * This new function signature is needed for the objective functions
  * that need the conditioning variable as argument for allocating the 
  * coco_problem_t 
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_cauchy_conditioned(
    const coco_problem_bbob_conditioned_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const double conditioning, 
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_cauchy_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_additive_noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}

/**
 * @brief Allocates the cauchy noise model to the coco_noisy_problem instance
 * This new function signature is needed for the objective functions
  * that need the conditioning variable as argument for allocating the 
  * coco_problem_t 
 */
coco_problem_t *coco_problem_allocate_bbob_wrap_noisy_cauchy_gallagher(
    const coco_problem_bbob_gallagher_allocator_t coco_problem_bbob_allocator,
    const size_t function, 
    const size_t dimension, 
    const size_t instance, 
    const long rseed,
    const size_t n_peaks, 
    const char *problem_id_template, 
    const char *problem_name_template, 
    const uint32_t random_seed, 
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
  problem -> noise_model -> random_seed = random_seed;
  problem -> noise_model -> distribution_theta = distribution_theta;
  problem -> noise_model -> noise_sampler = coco_problem_sample_cauchy_noise;
  problem -> noise_model -> noise_model_evaluator = coco_problem_evaluate_additive_noise_model;
  problem -> placeholder_evaluate_function = problem -> evaluate_function;
  problem -> evaluate_function = coco_problem_f_evaluate_wrap_noisy;
  return problem;
}
/**@}*/

/***********************************************************************************************************/