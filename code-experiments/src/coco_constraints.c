#include <stdio.h>
#include <assert.h>

#include "coco.h"

#include "coco_problem.c"

/* Evaluate the linear constraint defined in "self" at the point x and
 * store the resulting value at y[0]
 */ 
static void linear_constraint_evaluate(coco_problem_t *self, const double *x, double *y) {
  
  size_t i;
  
  y[0] = 0.0;
  
  for (i = 0; i < self->number_of_variables; ++i) { 
	  y[0] += self->data[i]*x[i]; 
  }
}

/* Guarantee that the negative gradient of the constraint in problem_c1
 * is feasible w.r.t. the constraint in problem_c2
 */
static coco_problem_t guarantee_feasible_point(coco_problem_t *problem_c2, coco_problem_t *problem_c1) {
  
  size_t i;
  double constraint_value = 0.0;
  
  assert(problem_c1->number_of_variables ==  problem_c2->number_of_variables);
  assert(problem_c1->number_of_constraints > 0);
  assert(problem_c2->number_of_constraints > 0);
  
  /* Define the feasible point as the negative gradient of the constraint
   * in problem_c1. Let p be the gradient of the constraint
   * in problem_c2 and q be the gradient of the constraint in problem_c1. 
   * 
   * We desire that p' * (-q) <= 0, or, equivalently, p' * q >= 0. 
   * To check whether the inequality bove holds, compute the dot product 
   * of the two vectors p and q, which are the gradients of the linear 
   * constraints.
   */
  
  for (i = 0; i < problem_c2->number_of_variables; ++i) {
	  constraint_value += problem_c2->data[i] * problem_c1->data[i];
  }
  
  /* Flip the constraint in problem_c2 if the chosen feasible point is
   * not feasible w.r.t the constraint in problem_c2, i.e. if p' * q < 0.
   */
  if (constraint_value < 0) {
      for (i = 0; i < problem_c2->number_of_variables; ++i) {
			problem_c2->data[i] *= -1.0;
      }
  }
  
  return problem_c2;
}

/* Defines a new problem containing one single linear constraint and no
 * objective function. The linear constraint is defined by its gradient,
 * which is stored in problem->data.
 */
static coco_problem_t *linear_constraint_problem(const size_t number_of_variables, 
                                                 const double *gradient) {
  
  size_t problem_id_length;
  double *cons_gradient;
  
  cons_gradient = coco_duplicate_vector(gradient, number_of_variables);
  
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 0, 1);
  problem->problem_name = coco_strdup("linear constraint");
  
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "linear constraint", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "linear constraint",
           (int)number_of_variables);
  
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 0;
  problem->number_of_constraints = 1;
  problem->evaluate_constraint = linear_constraint_evaluate;
  problem->data = cons_gradient;
  
  return problem;
}
