/**
 * @file transform_obj_cauchy_noise.c
 * @brief Implementation of the Cauchy noise model
 */
#include "coco.h"
#include "suite_bbob_noisy_utilities.c"
#include <stddef.h>


/**
 @brief Data type for transform_obj_cauchy_noise
 */
typedef struct{
    double alpha;
    double p;    
} transform_obj_cauchy_noise_data_t;


/**
 * @brief Evaluates the transformed objective function by applying cauchy additive noise.
 */
static void transform_obj_cauchy_noise_evaluate_function(
        coco_problem_t *problem,
        const double *x, 
        double *y 
    ){
    coco_problem_t *inner_problem = coco_problem_transformed_get_inner_problem(problem);    
    double fopt = *(inner_problem -> best_value);
    transform_obj_cauchy_noise_data_t *data;
    data = (transform_obj_cauchy_noise_data_t *) coco_problem_transformed_get_data(problem);
    double uniform_indicator, numerator_normal_variate, denominator_normal_variate;
    uniform_indicator = coco_sample_uniform_noise();
    numerator_normal_variate = coco_sample_gaussian_noise();
    denominator_normal_variate = coco_sample_gaussian_noise();
    denominator_normal_variate = fabs(denominator_normal_variate  + 1e-199);
    double cauchy_noise = numerator_normal_variate / (denominator_normal_variate);
    cauchy_noise = uniform_indicator < data -> p ?  1e3 + cauchy_noise : 1e3;
    cauchy_noise = data -> alpha * cauchy_noise;
    cauchy_noise = cauchy_noise > 0 ? cauchy_noise : 0.;
    double tol = 1e-8;
    inner_problem -> evaluate_function(inner_problem, x, y);
    for(size_t i = 0; i < problem -> number_of_objectives; i++){
        problem -> last_noise_free_values[i] = y[i];
    }
    *(y) = *(y) + cauchy_noise + 1.01 * tol + coco_boundary_handling(problem, x);
}

/**
 * @brief Allocates a noisy problem with cauchy noise.
 */
static coco_problem_t *transform_obj_cauchy_noise(
        coco_problem_t *inner_problem,
        const double alpha,
        const double p
    ){
    coco_problem_t *problem;
    transform_obj_cauchy_noise_data_t *data;
    data = (transform_obj_cauchy_noise_data_t *) coco_allocate_memory(sizeof(*data));
    data -> alpha = alpha;
    data -> p = p;
    problem = coco_problem_transformed_allocate(inner_problem, data, 
        NULL, "cauchy_noise_model");
    problem->evaluate_function = transform_obj_cauchy_noise_evaluate_function;
    problem->is_noisy = 1;
    return problem;
}
