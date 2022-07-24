/**
 * @file transform_vars_asymmetric.c
 * @brief Implementation of performing an asymmetric transformation on decision values.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_asymmetric.
 */
typedef struct {
  double *x;
  double beta;
} transform_vars_asymmetric_data_t;

typedef struct {
  double beta;
  size_t i;
  size_t n;
} tasy_data;


static double tasy_uv(double xi, tasy_data *d) {
  double yi;
  double exponent;
  if (xi > 0.0) {
        exponent = 1.0
            + ((d->beta * (double) (long) d->i) / ((double) (long) d->n - 1.0)) * sqrt(xi);
        yi = pow(xi, exponent);
      } else {
        yi = xi;
      }
  return yi;
}


/**
 * @brief Multivariate, coordinate-wise, asymmetric non-linear transformation.
 */
static transform_vars_asymmetric_data_t *tasy(transform_vars_asymmetric_data_t *data,
                                              const double *x,
                                              size_t number_of_variables) {
  size_t i;
  tasy_data *d;

  d->beta = data->beta;
  d->n = number_of_variables;

  for (i = 0; i < number_of_variables; ++i) {
      d->i = i;
      data->x[i] = tasy_uv(x[i], d);
  }
  return data;
}


/**
 * @brief Evaluates the transformed function.
 */
static void transform_vars_asymmetric_evaluate_function(coco_problem_t *problem, 
                                                        const double *x, 
                                                        double *y) {

  double *cons_values;
  int is_feasible;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  data = tasy(data, x, problem->number_of_variables);
  
  coco_evaluate_function(inner_problem, data->x, y);
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed constraint.
 */
static void transform_vars_asymmetric_evaluate_constraint(coco_problem_t *problem, 
                                                          const double *x, 
                                                          double *y) {
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  data = tasy(data, x, problem->number_of_variables);

  coco_evaluate_constraint(inner_problem, data->x, y);
}

static void transform_vars_asymmetric_free(void *thing) {
  transform_vars_asymmetric_data_t *data = (transform_vars_asymmetric_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_asymmetric(coco_problem_t *inner_problem, const double beta) {
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *problem;
  
  data = (transform_vars_asymmetric_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->beta = beta;
  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_asymmetric_free, "transform_vars_asymmetric");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_asymmetric_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_vars_asymmetric_evaluate_constraint;
  
  if (inner_problem->number_of_objectives > 0 && coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_warning("transform_vars_asymmetric(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }
  return problem;
}
