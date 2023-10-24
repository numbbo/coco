/**
 * @file transform_vars_oscillate.c
 * @brief Implementation of oscillating the decision values.
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
 * @brief Data type for transform_vars_oscillate.
 */
typedef struct {
  double alpha;
  double *oscillated_x;
} transform_vars_oscillate_data_t;

/**
 * @brief Data type for univariate function tosz_uv
 */
typedef struct {
  double alpha;
} tosz_data;

/**
 * @brief Univariate oscillating non-linear transformation.
 */
static double tosz_uv(double xi, tosz_data *d) {
  double yi;
  double tmp, base;
  if (xi > 0.0) {
      tmp = log(xi) / d->alpha;
      base = exp(tmp + 0.49 * (sin(tmp) + sin(0.79 * tmp)));
      yi = pow(base, d->alpha);
    } else if (xi < 0.0) {
      tmp = log(-xi) / d->alpha;
      base = exp(tmp + 0.49 * (sin(0.55 * tmp) + sin(0.31 * tmp)));
      yi = -pow(base, d->alpha);
    } else {
      yi = 0.0;
    }
  return yi;
}

/**
 * @brief Inverse of oscillating non-linear transformation tosz_uv obtained with brentq.
 */
static double tosz_uv_inv(double yi, tosz_data *d) {
  double xi;
  xi = brentinv((callback_type) &tosz_uv, yi, d);
  return xi;
}

/**
 * @brief Multivariate, coordinate-wise, oscillating non-linear transformation.
 */
static void tosz(transform_vars_oscillate_data_t *data,
                                              const double *x,
                                              size_t number_of_variables) {
  size_t i;
  tosz_data *d;
  d = coco_allocate_memory(sizeof(*d));

  d->alpha = data->alpha;

  for (i = 0; i < number_of_variables; ++i) {
    data->oscillated_x[i] = tosz_uv(x[i], d);
  }
  coco_free_memory(d);
}


/**
 * @brief Evaluates the transformed objective functions.
 */
static void transform_vars_oscillate_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  double *cons_values;
  int is_feasible;
  transform_vars_oscillate_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_oscillate_data_t *) coco_problem_transformed_get_data(problem);

  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  tosz(data, x, problem->number_of_variables);

  coco_evaluate_function(inner_problem, data->oscillated_x, y);
  
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
 * @brief Evaluates the transformed constraints.
 */
static void transform_vars_oscillate_evaluate_constraint(coco_problem_t *problem,
                                                         const double *x,
                                                         double *y,
                                                         int update_counter) {
  static const double alpha = 0.1;
  double tmp, base, *oscillated_x;
  size_t i;
  transform_vars_oscillate_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

  data = (transform_vars_oscillate_data_t *) coco_problem_transformed_get_data(problem);
  oscillated_x = data->oscillated_x; /* short cut to make code more readable */
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      tmp = log(x[i]) / alpha;
      base = exp(tmp + 0.49 * (sin(tmp) + sin(0.79 * tmp)));
      oscillated_x[i] = pow(base, alpha);
    } else if (x[i] < 0.0) {
      tmp = log(-x[i]) / alpha;
      base = exp(tmp + 0.49 * (sin(0.55 * tmp) + sin(0.31 * tmp)));
      oscillated_x[i] = -pow(base, alpha);
    } else {
      oscillated_x[i] = 0.0;
    }
  }
  inner_problem->evaluate_constraint(inner_problem, oscillated_x, y, update_counter);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_oscillate_free(void *thing) {
  transform_vars_oscillate_data_t *data = (transform_vars_oscillate_data_t *) thing;
  coco_free_memory(data->oscillated_x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_oscillate(coco_problem_t *inner_problem) {
  transform_vars_oscillate_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_oscillate_data_t *) coco_allocate_memory(sizeof(*data));
  data->alpha = 0.1;
  data->oscillated_x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_oscillate_free, "transform_vars_oscillate");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_oscillate_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_vars_oscillate_evaluate_constraint;

  return problem;
}

/**
 * @brief Applies the inverse of the oscillating transformation tasy(tosz(.)) to the initial solution.
 * 
 *        Takes xopt as input to check the solution remains in the bounds
 *        If not, a curve search is performed
 *        xopt is needed because transform_vars_shift is not yet called
 *        in f_{function}_rotated_c_linear_cons_bbob_problem_allocate
 */
static void transform_inv_initial_oscillate(coco_problem_t *problem, const double *xopt) {
  size_t i;
  size_t j;
  int is_in_bounds;
  double di = 0.0;
  double xi;
  double *sol = NULL;
  double halving_factor = .5;

  transform_vars_oscillate_data_t *data;
  tosz_data *d;

  sol = coco_allocate_vector(problem->number_of_variables);
  d = coco_allocate_memory(sizeof(*d));

  data = (transform_vars_oscillate_data_t *) coco_problem_transformed_get_data(problem);
  d->alpha = data->alpha;

  j = 0;
  while (1) {

    for (i = 0; i < problem->number_of_variables; ++i) {
      di = tosz_uv_inv(problem->initial_solution[i] * pow(halving_factor, (double) (long) j), d);
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

