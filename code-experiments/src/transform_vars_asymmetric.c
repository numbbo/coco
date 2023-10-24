/**
 * @file transform_vars_asymmetric.c
 * @brief Implementation of performing an asymmetric transformation on decision values.
 * @author ??
 * @author Paul Dufosse
 * @note Edited to fulfill needs from the constrained test bed.
 */

#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "brentq.c"

/**
 * @brief Data type for transform_vars_asymmetric.
 */
typedef struct {
  double *x;
  double beta;
} transform_vars_asymmetric_data_t;

/**
 * @brief Data type for univariate function tasy_uv.
 */
typedef struct {
  double beta;
  size_t i;
  size_t n;
} tasy_data;

/**
 * @brief Univariate asymmetric non-linear transformation.
 */
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
 * @brief Inverse of asymmetric non-linear transformation tasy_uv obtained with brentq.
 */
static double tasy_uv_inv(double yi, tasy_data *d) {
  double xi;
  xi = brentinv((callback_type) &tasy_uv, yi, d);
  return xi;
}


/**
 * @brief Multivariate, coordinate-wise, asymmetric non-linear transformation.
 */
static void tasy(transform_vars_asymmetric_data_t *data,
                                              const double *x,
                                              size_t number_of_variables) {
  size_t i;
  tasy_data *d;
  d = coco_allocate_memory(sizeof(*d));

  d->beta = data->beta;
  d->n = number_of_variables;

  for (i = 0; i < number_of_variables; ++i) {
      d->i = i;
      data->x[i] = tasy_uv(x[i], d);
  }
  coco_free_memory(d);
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

  tasy(data, x, problem->number_of_variables);
  
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
                                                          double *y,
                                                          int update_counter) {
  size_t i;
  double exponent;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* FIXME (OME): Old  pre-logger
  tasy(data, x, problem->number_of_variables);

  coco_evaluate_constraint(inner_problem, data->x, y);
  */
  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent = 1.0
          + ((data->beta * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0)) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  inner_problem->evaluate_constraint(inner_problem, data->x, y, update_counter);
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

/**
 * @brief Applies the inverse of the asymmetric transformation tasy to the initial solution.
 * 
 *        Takes xopt as input to check the solution remains in the bounds
 *        If not, a curve search is performed
 *        xopt is needed because transform_vars_shift is not yet called
 *        in f_{function}_rotated_c_linear_cons_bbob_problem_allocate
 */
static void transform_inv_initial_asymmetric(coco_problem_t *problem, const double *xopt) {
  size_t i;
  size_t j;
  int is_in_bounds;
  double di;
  double xi;
  double *sol = NULL;
  double halving_factor = .5;

  transform_vars_asymmetric_data_t *data;
  tasy_data *d;

  sol = coco_allocate_vector(problem->number_of_variables);
  d = coco_allocate_memory(sizeof(*d));

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);

  d->beta = data->beta;
  d->n = problem->number_of_variables;

  j = 0;
  while (1) {

    for (i = 0; i < problem->number_of_variables; ++i) {
      d->i = i;
      di = tasy_uv_inv(problem->initial_solution[i] * pow(halving_factor, (double) (long) j), d);
      xi = di + xopt[i];
      is_in_bounds = (int) (-5.0 < xi  && xi < 5.0);
      /* Line search for the inverse-transformed feasible initial solution
         to remain within the bounds
        */
      if (!is_in_bounds) {
        j = j + 1;
        break;
      }
      sol[i] = di;
    }
    if (!is_in_bounds && !coco_is_nan(di)){
      continue;
    }
    else {
      break;
    }   
  }
  if (!coco_vector_contains_nan(sol, problem->number_of_variables)) {
    for (i = 0; i < problem->number_of_variables; ++i) {
      problem->initial_solution[i] = sol[i];
    }
  }
  
  coco_free_memory(d);
  coco_free_memory(sol);
}
