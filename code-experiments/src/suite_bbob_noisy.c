/**
 * @file suite_bbob_noisy.c
 * @brief Implementation of the bbob suite containing 24 noiseless single-objective functions in 6
 * dimensions.
 */

#include "suite_bbob_noisy_utilities.c"

#include "f_different_powers.c"
#include "f_ellipsoid.c"
#include "f_gallagher.c"
#include "f_griewank_rosenbrock.c"
#include "f_griewank_rosenbrock.c"
#include "f_rosenbrock.c"
#include "f_schaffers.c"
#include "f_sphere.c"
#include "f_step_ellipsoid.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob suite.
 */
static coco_suite_t *suite_bbob_noisy_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };
  const size_t num_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("noisy-bbob", 30, num_dimensions, dimensions, "year: 0");

  return suite;
}



/**
 * @brief Sets the instances associated with years for the bbob suite.
 */
static const char *suite_bbob_noisy_get_instances_by_year(const int year) {
  assert(year != NAN);
  return "1-5,1-5,1-5";
}


/**
 * @brief Creates and returns a BBOB problem without needing the actual bbob suite.
 *
 * Useful for other suites as well (see for example suite_biobj.c).
 */
static coco_problem_t *coco_get_bbob_noisy_problem(const size_t function,
                                             const size_t dimension,
                                             const size_t instance) {
  coco_problem_t *problem = NULL;

  const char *problem_id_template = "bbob_noisy_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "BBOB-NOISY suite problem f%lu instance %lu in %luD";

  const long rseed = (long) (function + 10000 * instance);
  const uint32_t coco_seed = (uint32_t) (31 + 1000 + instance);

  if (function == 1){
    double distribution_theta_arr[] = {0.1};
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_sphere_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 2){
    double alpha, beta;
    alpha = 0.01 * (0.49 + 1 / (double) dimension);
    beta = 0.01;
    double distribution_theta_arr[] = {alpha, beta};
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_sphere_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 3){
    double distribution_theta_arr[] = {0.01, 0.05}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_sphere_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 4){
    double distribution_theta_arr[] = {0.01}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 5){
    double alpha, beta;
    alpha = 0.01 * (0.49 + 1 / (double) dimension);
    beta = 0.01;
    double distribution_theta_arr[] = {alpha, beta};
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 6){
    double distribution_theta_arr[] = {0.01, 0.05}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 7){
    double distribution_theta_arr[] = {1.0}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_sphere_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 8){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta};
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_sphere_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 9){
    double distribution_theta_arr[] = {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_sphere_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 10){
    double distribution_theta_arr[]= {1.0}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 11){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta};
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 12){
    double distribution_theta_arr[]= {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 13){
    double distribution_theta_arr[] = {1.0}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_step_ellipsoid_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 14){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta};
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_step_ellipsoid_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 15){
    double distribution_theta_arr[]= {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_step_ellipsoid_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 16){
    double distribution_theta_arr[] = {1.0}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_ellipsoid_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 17){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_ellipsoid_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 18){
    double distribution_theta_arr[]= {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_ellipsoid_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 19){
    double distribution_theta_arr[] = {1.0}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_different_powers_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 20){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_different_powers_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 21){
    double distribution_theta_arr[]= {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_different_powers_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 22){
    double distribution_theta_arr[]= {1.0}; 
    double *distribution_theta = &distribution_theta_arr[0];
    double conditioning = 10; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian_schaffers(
        f_schaffers_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed,
        conditioning, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 23){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    double conditioning = 10;
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform_schaffers(
        f_schaffers_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed,
        conditioning, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 24){
    double distribution_theta_arr[] = {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0];
    double conditioning = 10; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy_schaffers(
        f_schaffers_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed,
        conditioning, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 25){
    double distribution_theta_arr[]= {1.0}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian(
        f_griewank_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 26){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform(
        f_griewank_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 27){
    double distribution_theta_arr[] = {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy(
        f_griewank_rosenbrock_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 28){
    double distribution_theta_arr[]= {1.0};
    double *distribution_theta = &distribution_theta_arr[0];
    size_t n_peaks = 101; 
    problem = coco_problem_allocate_bbob_wrap_noisy_gaussian_gallagher(
        f_gallagher_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed,
        n_peaks,
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 29){
    double alpha, beta;
    alpha = 0.49 + 1.0 / (double) dimension;
    beta = 1.0;
    double distribution_theta_arr[] = {alpha, beta}; 
    double *distribution_theta = &distribution_theta_arr[0]; 
    size_t n_peaks = 101;
    problem = coco_problem_allocate_bbob_wrap_noisy_uniform_gallagher(
        f_gallagher_bbob_problem_allocate, 
        function, 
        dimension, 
        instance, 
        rseed,
        n_peaks, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else if (function == 30){
    double distribution_theta_arr[] = {1.0, 0.2}; 
    double *distribution_theta = &distribution_theta_arr[0];
    size_t n_peaks = 101; 
    problem = coco_problem_allocate_bbob_wrap_noisy_cauchy_gallagher(
        f_gallagher_bbob_problem_allocate, 
        function,
        dimension, 
        instance, 
        rseed,
        n_peaks, 
        problem_id_template, 
        problem_name_template,
        coco_seed, 
        distribution_theta
    );
  } else {
    coco_error("coco_get_bbob_noisy_problem(): cannot retrieve problem f%lu instance %lu in %luD",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension);
    return NULL; /* Never reached */
  }    
  return problem;
}

/**
 * @brief Returns the problem from the bbob suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_bbob_noisy_get_problem( coco_suite_t *suite,
                                              const size_t function_idx,
                                              const size_t dimension_idx,
                                              const size_t instance_idx) {

  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = coco_get_bbob_noisy_problem(function, dimension, instance);

  problem -> suite_dep_function = function;
  problem -> suite_dep_instance = instance;
  problem -> suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);
  return problem;
}
