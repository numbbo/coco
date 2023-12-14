/**
 * @file transform_obj_uniform_noise.c
 * @brief Implementation of the Uniform noise model
 */
#include "coco.h"
#include "suite_bbob_noisy_utilities.c"


/**
 @brief Data type for transform_obj_uniform_noise
 */
typedef struct{
    double alpha;
    double beta;    
} transform_obj_uniform_noise_data_t;


/**
 * @brief Evaluates the transformed objective function by applying uniform multiplicative noise.
 */
static void transform_obj_uniform_noise_evaluate_function(
        coco_problem_t *problem,
        const double *x, 
        double *y 
    ){    
    double uniform_noise_term1, uniform_noise_term2, fopt, uniform_noise_factor,
        scaling_factor, uniform_noise, tol;
    transform_obj_uniform_noise_data_t *data;
    size_t i;    
    coco_problem_t *inner_problem; 
    inner_problem = coco_problem_transformed_get_inner_problem(problem);
    fopt = *(inner_problem->best_value);
    data = (transform_obj_uniform_noise_data_t *) coco_problem_transformed_get_data(problem);
    uniform_noise_term1 = coco_sample_uniform_noise();
    uniform_noise_term2 = coco_sample_uniform_noise();
    uniform_noise_factor = pow(uniform_noise_term1, data->beta);
    inner_problem->evaluate_function(inner_problem, x, y);
    for(i = 0; i < problem->number_of_objectives; i++){
        problem->last_noise_free_values[i] = y[i];
    }
    *(y) = *(y) - fopt;
    scaling_factor = 1e9/(*(y) + 1e-99);
    scaling_factor = pow(scaling_factor, data -> alpha * uniform_noise_term2);
    scaling_factor = scaling_factor > 1 ? scaling_factor : 1;
    uniform_noise = uniform_noise_factor * scaling_factor;
    tol = 1e-8;
    *(y) = *(y) * uniform_noise + 1.01 * tol; 
    *(y) = *(y) + fopt + coco_boundary_handling(problem, x);
}

/**
 * @brief Allocates a noisy problem with uniform noise.
 */
static coco_problem_t *transform_obj_uniform_noise(
        coco_problem_t *inner_problem,
        const double alpha,
        const double beta
    ){
    coco_problem_t *problem;
    transform_obj_uniform_noise_data_t *data;
    data = (transform_obj_uniform_noise_data_t *) coco_allocate_memory(sizeof(*data));
    data->alpha = alpha;
    data->beta = beta;
    problem = coco_problem_transformed_allocate(inner_problem, data, 
        NULL, "uniform_noise_model");
    problem->evaluate_function = transform_obj_uniform_noise_evaluate_function;
    problem->is_noisy = 1;
    return problem;
}
