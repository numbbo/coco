/**
    * @file f_sphere_gaussian.c
    * @brief Implementation of the gaussian noisy version of the sphere function in the bbob-noisy suite
*/

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "f_sphere.c"
#include "coco.h"
#include "coco_problem.c"
#include "coco_random.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_shift.c"
#include "transform_obj_norm_by_dim.c"


/**
    * @brief Implements the sphere function with gaussian noise, by calling the functions of the f_sphere.c and coco_random.c files
*/
static double f_sphere_gaussian_raw(
        const double * x,
        const double gaussian_noise,
        const size_t number_of_variables,
    ){       
        double result = f_sphere_raw(x, number_of_variables);
        assert(result != NULL);
        result = result * gaussian_noise;
        return result;
}


/**
    * @brief Uses the raw function to evaluate the COCO problem.
*/
static void f_sphere_gaussian_evaluate(
        coco_problem_t * problem, 
        const double * x,
        double * y
    ){
        assert(problem->number_of_objectives == 1);
        coco_problem_sample_gaussian_noise(problem);
        double gaussian_noise = coco_problem_get_last_noise_value(problem);
        y[0] = f_sphere_gaussian_raw(x, gaussian_noise, number_of_variables);    
        assert(y[0] + 1e-13 >= problem->best_value[0]);/**<How to handle the tolerance considering the noise??>*/
}

/**
    *@brief Evaluates gradient of the sphere function with gaussian noise 
*/
static void f_sphere_gaussian_evaluate_gradient(
        coco_problem_t * problem,
        const double * x, 
        double *y, 
    ){
    size_t i;
    double gaussian_noise = coco_problem_get_last_noise_value(problem);
    for (i = 0; i < problem -> number_of_variables; i++){
        y[i] = 2*x[i]*gaussian_noise;
    }
}

/**
    *@brief Allocates the sphere problem with gaussian noise
*/
static coco_problem_t *f_sphere_gaussian_allocate(
        const size_t number_of_variables,
        const uint32_t seed,
        const double scale,
    ){
    const double *distribution_theta = &scale;
    coco_problem_t *problem = coco_problem_allocate_from_scalars("sphere function with gaussian noise",
     f_sphere_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);    
    problem -> random_seed = random_seed;
    problem -> distribution_theta = distribution_theta;
    problem -> evaluate_gradient = f_sphere_gaussian_evaluate_gradient;
    coco_problem_set_id(problem, "%s_d%02lu", "sphere-gaussian", number_of_variables); 
    /* Compute best solution */
    f_sphere_evaluate(problem, problem->best_parameter, problem->best_value);
    return problem;
}

/**
 * @brief Creates the BBOB sphere gaussian problem.
*/
static coco_problem_t *f_sphere_gaussian_bbob_problem_allocate(
            const size_t function,
            const size_t dimension,
            const size_t instance,
            const long rseed,
            const char *problem_id_template,
            const char *problem_name_template,
            const uint32_t seed, 
            const double scale,
        ) {

        double *xopt, fopt;
        coco_problem_t *problem = NULL;

        xopt = coco_allocate_vector(dimension);
        bbob2009_compute_xopt(xopt, rseed, dimension);
        fopt = bbob2009_compute_fopt(function, instance);

        problem = f_sphere_gaussian_allocate(dimension, seed, scale);
        problem = transform_vars_shift(problem, xopt, 0);

        /*if large scale test-bed, normalize by dim*/
        if (coco_strfind(problem_name_template, "BBOB large-scale suite") >= 0){
        problem = transform_obj_norm_by_dim(problem);
        }
        problem = transform_obj_shift(problem, fopt);

        coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
        coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
        coco_problem_set_type(problem, "1-separable");

        coco_free_memory(xopt);
        return problem;
}