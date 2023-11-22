/**
 * @file transform_obj_uniform_noise.c
 * @brief Implementation of the Uniform noise model
 */
#include "coco.h"
#include "suite_bbob_noisy_utilities.c"

/**
 @brief Data type for transform_obj_gaussian_noise
 */
typedef struct{
    double beta;    
} transform_obj_gaussian_noise_data_t;


/**
 * @brief Evaluates the transformed objective function by applying gaussian multiplicative noise.
 */
static void transform_obj_gaussian_noise_evaluate_function(
        coco_problem_t *problem,
        const double *x, 
        double *y 
    ){
    coco_problem_t *inner_problem = coco_problem_transformed_get_inner_problem(problem);    
    double fopt = *(inner_problem -> best_value);
    transform_obj_gaussian_noise_data_t *data;
    data = (transform_obj_gaussian_noise_data_t *) coco_problem_transformed_get_data(problem);
    double gaussian_noise = coco_sample_gaussian_noise();
    gaussian_noise = exp(data -> beta * gaussian_noise);
    double tol = 1e-8;
    inner_problem -> evaluate_function(inner_problem, x, y);
    *(y) = *(y) - fopt;
    *(y) = *(y) * gaussian_noise  + 1.01 * tol;
    *(y) = *(y) + fopt + coco_boundary_handling(problem, x);
}


/**
 * @brief Allocates a noisy problem with gaussian noise.
 */
static coco_problem_t *transform_obj_gaussian_noise(
        coco_problem_t *inner_problem,
        const double beta
    ){
    coco_problem_t *problem;
    transform_obj_gaussian_noise_data_t *data;
    data = (transform_obj_gaussian_noise_data_t *) coco_allocate_memory(sizeof(*data));
    data -> beta = beta;
    problem = coco_problem_transformed_allocate(inner_problem, data, 
        NULL, "gaussian_noise_model");
    problem->evaluate_function = transform_obj_gaussian_noise_evaluate_function;
    return problem;

}
