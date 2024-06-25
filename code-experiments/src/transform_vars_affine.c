/**
 * @file transform_vars_affine.c
 * @brief Implementation of performing an affine transformation on decision values.
 *
 * x |-> Mx + b <br>
 * The matrix M is stored in row-major format.
 *
 * Currently, the best parameter is transformed correctly only in the simple 
 * cases where M is orthogonal which is always the case for the `bbob`
 * functions. How to code this for general transformations of the above form,
 * see https://github.com/numbbo/coco/issues/814#issuecomment-303724400
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"

/**
 * @brief Data type for transform_vars_affine.
 */
typedef struct {
  double *M, *b, *x;
} transform_vars_affine_data_t;

/**
 * @brief Evaluates the transformed objective function.
 */
static void transform_vars_affine_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j;
  double *cons_values;
  int is_feasible;
  transform_vars_affine_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_affine_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    /* data->M has problem->number_of_variables columns and inner_problem->number_of_variables rows. */
    const double *current_row = data->M + i * problem->number_of_variables;
    data->x[i] = data->b[i];
    for (j = 0; j < problem->number_of_variables; ++j) {
      data->x[i] += x[j] * current_row[j];
    }
  }
  
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
static void transform_vars_affine_evaluate_constraint(coco_problem_t *problem, const double *x, double *y, int update_counter) {
  size_t i, j;  
  transform_vars_affine_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

  data = (transform_vars_affine_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    /* data->M has problem->number_of_variables columns and inner_problem->number_of_variables rows. */
    const double *current_row = data->M + i * problem->number_of_variables;
    data->x[i] = data->b[i];
    for (j = 0; j < problem->number_of_variables; ++j) {
      data->x[i] += x[j] * current_row[j];
    }
  }
  inner_problem->evaluate_constraint(inner_problem, data->x, y, update_counter);
}

/**
 * @brief Evaluates the gradient of the transformed function.
 */
static void transform_vars_affine_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j;
  transform_vars_affine_data_t *data;
  coco_problem_t *inner_problem;
  double *current_row;
  double *gradient;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }
  
  data = (transform_vars_affine_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  gradient = coco_allocate_vector(inner_problem->number_of_variables);
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
    gradient[i] = 0.0;

  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    /* data->M has problem->number_of_variables columns and inner_problem->number_of_variables rows. */
    current_row = data->M + i * problem->number_of_variables;
    data->x[i] = data->b[i];
    for (j = 0; j < problem->number_of_variables; ++j) {
      data->x[i] += x[j] * current_row[j];
    }
  }
  
  bbob_evaluate_gradient(inner_problem, data->x, y);
  
  /* grad_(f o g )(x), where g(x) = M * x + b, equals to
   * M^T * grad_f(M *x + b) 
   */
  for (j = 0; j < inner_problem->number_of_variables; ++j) {
    for (i = 0; i < inner_problem->number_of_variables; ++i) {
       current_row = data->M + i * problem->number_of_variables;
       gradient[j] += y[i] * current_row[j];
    }
  }
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
     y[i] = gradient[i];
  
  current_row = NULL;
  coco_free_memory(gradient);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_affine_free(void *thing) {
  transform_vars_affine_data_t *data = (transform_vars_affine_data_t *) thing;
  coco_free_memory(data->M);
  coco_free_memory(data->b);
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_affine(coco_problem_t *inner_problem,
                                             const double *M,
                                             const double *b,
                                             const size_t number_of_variables) {
  /*
   * TODOs:
   * - Calculate new smallest/largest values of interest?
   * - Resize bounds vectors if input and output dimensions do not match
   */

  size_t i, j;
  coco_problem_t *problem;
  transform_vars_affine_data_t *data;
  size_t entries_in_M;

  entries_in_M = inner_problem->number_of_variables * number_of_variables;
  data = (transform_vars_affine_data_t *) coco_allocate_memory(sizeof(*data));
  data->M = coco_duplicate_vector(M, entries_in_M);
  data->b = coco_duplicate_vector(b, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_affine_free, "transform_vars_affine");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_affine_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_vars_affine_evaluate_constraint;
    
  problem->evaluate_gradient = transform_vars_affine_evaluate_gradient;

  /* Update the best parameter by computing
     problem->best_parameter = M^T * (inner_problem->best_parameter - b).

     The update takes place only if the best parameter or b are different than zero
     and the transformation matrix M is orthogonal.
  */
  if (coco_problem_best_parameter_not_zero(inner_problem) || !coco_vector_is_zero(data->b, inner_problem->number_of_variables)) {
    if (!coco_is_orthogonal(data->M, problem->number_of_variables, inner_problem->number_of_variables))
        coco_warning("transform_vars_affine(): rotation matrix is not orthogonal. Best parameter not updated");
    else {
        for (i = 0; i < inner_problem->number_of_variables; ++i) {
            data->x[i] = inner_problem->best_parameter[i] - data->b[i];
        }
        for (i = 0; i < problem->number_of_variables; ++i) {
            problem->best_parameter[i] = 0;
            for (j = 0; j < inner_problem->number_of_variables; ++j) {
                problem->best_parameter[i] += data->M[j * problem->number_of_variables + i] * data->x[j];
            }
        }
    }
  }

  return problem;
}
