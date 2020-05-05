/**
 * @file  c_linear.c
 * @brief Implements the linear constraints for the suite of 
 *        constrained problems.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_problem.c"
/**
 * @brief Data type for the linear constraints.
 */
typedef struct {
  double *gradient;
  double *x;
} linear_constraint_data_t;	

static void c_sum_variables_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y,
                                     int update_counter);
                                     
static void c_linear_single_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y,
                                     int update_counter);
                                        
static coco_problem_t *c_guarantee_feasible_point(coco_problem_t *problem,
                                                  const double *feasible_point);
                                               
static void c_linear_gradient_free(void *thing);

static coco_problem_t *c_sum_variables_allocate(const size_t number_of_variables);

static coco_problem_t *c_linear_transform(coco_problem_t *inner_problem, 
                                          const double *gradient);
         
static coco_problem_t *c_linear_single_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t constraint_number,
                                                      const double factor1,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      double *gradient,
                                                      const double *feasible_direction);
                                                      
static coco_problem_t *c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      const double *feasible_direction);

/**
 * @brief Evaluates the linear constraint with all-ones gradient at
 *        the point 'x' and stores the result into 'y'.
 */
static void c_sum_variables_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y,
                                     int update_counter) {
	
  size_t i;

  assert(self->number_of_constraints == 1);
  
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i)
    y[0] += x[i];

  (void) update_counter; /* To silence the compiler */
}	

/**
 * @brief Evaluates the linear constraint at the point 'x' and stores
 *        the result in 'y'.
 */
static void c_linear_single_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y,
                                     int update_counter) {
	
  size_t i;
  
  linear_constraint_data_t *data;
  coco_problem_t *inner_problem;
  
  data = (linear_constraint_data_t *) coco_problem_transformed_get_data(self);
  inner_problem = coco_problem_transformed_get_inner_problem(self);
  
  assert(self->number_of_constraints == 1);
			
  for (i = 0; i < self->number_of_variables; ++i)
    data->x[i] = (data->gradient[i])*x[i];
  
  inner_problem->evaluate_constraint(inner_problem, data->x, y, update_counter);
  
  inner_problem = NULL;
  data = NULL;
}

/**
 * @brief Guarantees that "feasible_direction" is feasible w.r.t. 
 *        the constraint in "problem" and records it as the 
 *        initial feasible solution to this coco_problem.
 */
static coco_problem_t *c_guarantee_feasible_point(coco_problem_t *problem,
                                                  const double *feasible_direction) {
  
  size_t i;
  linear_constraint_data_t *data;
  double constraint_value = 0.0;
  
  data = (linear_constraint_data_t *) coco_problem_transformed_get_data(problem);
  
  assert(problem->number_of_constraints == 1);
  
  /* Let p be the gradient of the constraint in "problem".
   * Check whether p' * (feasible_direction) <= 0.
   */
  problem->evaluate_constraint(problem, feasible_direction, &constraint_value, 0);
  
  /* Flip the constraint in "problem" if feasible_direction
   * is not feasible w.r.t. the constraint in "problem".
   */
  if (constraint_value > 0)
    for (i = 0; i < problem->number_of_variables; ++i)
      data->gradient[i] *= -1.0;
          
  problem->initial_solution = coco_duplicate_vector(feasible_direction, 
      problem->number_of_variables);
 
  data = NULL;  
  return problem;
}

/**
 * @brief Frees the data object.
 */
static void c_linear_gradient_free(void *thing) {
	
  linear_constraint_data_t *data = (linear_constraint_data_t *) thing;
  coco_free_memory(data->gradient);
  coco_free_memory(data->x);
}

/**
 * @brief Allocates a linear constraint coco_problem_t with all-ones gradient.
 */
static coco_problem_t *c_sum_variables_allocate(const size_t number_of_variables) {

  size_t i;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 0, 1);

  problem->evaluate_constraint = c_sum_variables_evaluate;
  
  coco_problem_set_id(problem, "%s_d%02lu", "linearconstraint", number_of_variables);

  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
  }
  return problem;
}

/**
 * @brief Transforms a linear constraint with all-ones gradient
 *        into a linear constraint whose gradient is passed 
 *        as argument.
 */
static coco_problem_t *c_linear_transform(coco_problem_t *inner_problem, 
                                          const double *gradient) {
  
  linear_constraint_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->gradient = coco_duplicate_vector(gradient, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  self = coco_problem_transformed_allocate(inner_problem, data, 
      c_linear_gradient_free, "gradient_linear_constraint");
  self->evaluate_constraint = c_linear_single_evaluate;

  return self;
}

/**
 * @brief Builds a coco_problem_t containing one single linear constraint.
 * 
 * This function is called by c_linear_cons_bbob_problem_allocate(),
 * the central function that stacks all the constraints built by
 * c_linear_single_cons_bbob_problem_allocate() into one single
 * coco_problem_t object.
 */
static coco_problem_t *c_linear_single_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t constraint_number,
                                                      const double factor1,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      double *gradient,
                                                      const double *feasible_direction) {
																			
  size_t i;
  
  double *gradient_linear_constraint = NULL;
  coco_problem_t *problem = NULL;
  coco_random_state_t *random_generator;
  long seed_cons_i;
  double factor2;
  
  problem = c_sum_variables_allocate(dimension);
  
  seed_cons_i = (long)(function + 10000 * instance 
                                + 50000 * constraint_number);
  random_generator = coco_random_new((uint32_t) seed_cons_i);
  
  /* The constraints gradients are scaled with random numbers
   * 10**U[0,1] and 10**U_i[0,2], where U[a, b] is uniform in [a,b] 
   * and only U_i is drawn for each constraint individually. 
   * The random number 10**U[0,1] is given by the variable 'factor1' 
   * while the random number 10**U_i[0,2] is calculated below and 
   * stored as 'factor2'. (The exception is when the number of
   * constraints is n+1, in which case 'factor2' defines a random
   * number 10**U_i[0,1])
   */
     
  factor2 = pow(100.0, coco_random_uniform(random_generator));
    
  
  /* Set the gradient of the linear constraint if it is given.
   * This should be the case of the construction of the first 
   * linear constraint only.
   */
  if(gradient) {
	  
    coco_vector_scale(gradient, dimension,
                      factor1 * factor2,
                      coco_vector_norm(gradient, dimension));
    problem = c_linear_transform(problem, gradient);

  }
  else{ /* Randomly generate the gradient of the linear constraint */
	  
    gradient_linear_constraint = coco_allocate_vector(dimension);
     
    /* Generate a pseudorandom vector with distribution N_i(0, I)
     * and scale it with 'factor1' and 'factor2' (see comments above)
     */
    for (i = 0; i < dimension; ++i)
      gradient_linear_constraint[i] = factor1 *
                coco_random_normal(random_generator) * factor2 / sqrt((double)dimension);

    problem = c_linear_transform(problem, gradient_linear_constraint);
    coco_free_memory(gradient_linear_constraint);
  }
  
  /* Guarantee that the vector feasible_point is feasible w.r.t. to
   * this constraint and set it as the initial solution.
   * The initial solution will be copied later to the constrained function
   * coco_problem_t object once the objective function and the constraint(s) 
   * are stacked together in coco_problem_stacked_allocate().
   */
  if(feasible_direction)
    problem = c_guarantee_feasible_point(problem, feasible_direction);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "linear");
  coco_random_free(random_generator);
  return problem;
  
}

/**
 * @brief Builds a coco_problem_t containing all the linear constraints
 *        by stacking them all.
 * 
 * The constraints' gradients are randomly generated with distribution
 * 10**U[0,1] * N_i(0, I) * 10**U_i[0,2], where U[a, b] is uniform 
 * in [a,b] and only U_i is drawn for each constraint individually. 
 * The exception is the first constraint, whose gradient is given by
 * 10**U[0,1] * (-feasible_direction) * 10**U_i[0,2].
 * 
 * Each constraint is built by calling the function
 * c_linear_single_cons_bbob_problem_allocate(), which returns a
 * coco_problem_t object that defines the constraint. The resulting
 * coco_problem_t objects are then stacked together into one single
 * coco_problem_t object that is returned by the function.
 */
static coco_problem_t *c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      const double *feasible_direction) {
																																			
  const double global_scaling_factor = 100.;
  size_t i;
  
  coco_problem_t *problem_c = NULL;
  coco_problem_t *problem_c2 = NULL;
  coco_random_state_t *random_generator;
  double *gradient_c1 = NULL;
  double *gradient;
  long seed_cons;
  double exp1, factor1;
  
  gradient_c1 = coco_allocate_vector(dimension);
  																	
  for (i = 0; i < dimension; ++i)
    gradient_c1[i] = -feasible_direction[i];

  /* Build a coco_problem_t object for each constraint. 
   * The constraints' gradients are generated randomly with
   * distriution 10**U[0,1] * N_i(0, I) * 10**U_i[0,2]
   * where U[a, b] is uniform in [a,b] and only U_i is drawn 
   * for each constraint individually.
   */
  
  /* Calculate the first random factor 10**U[0,1]. */
  seed_cons = (long)(function + 10000 * instance);
  random_generator = coco_random_new((uint32_t) seed_cons);
  exp1 = coco_random_uniform(random_generator);
  factor1 = global_scaling_factor * pow(10.0, exp1);

  /* Build the first linear constraint using 'gradient_c1' to build
   * its gradient.
   */ 
  /* set gradient depending on instance number */
  gradient = instance % number_of_linear_constraints ? NULL : gradient_c1;
  problem_c = c_linear_single_cons_bbob_problem_allocate(function,
      dimension, instance, 1, factor1,
      problem_id_template, problem_name_template, gradient,
      feasible_direction);
	 
  /* Instantiate the other linear constraints (if any) and stack them 
   * all into problem_c
   */     
  for (i = 2; i <= number_of_linear_constraints; ++i) {
	 
    /* Instantiate a new problem containing one linear constraint only */
    /* set gradient depending on instance number */
    gradient = (i - 1 + instance) % number_of_linear_constraints ? NULL : gradient_c1;
    problem_c2 = c_linear_single_cons_bbob_problem_allocate(function,
        dimension, instance, i, factor1,
        problem_id_template, problem_name_template, gradient,
        feasible_direction);
		
    problem_c = coco_problem_stacked_allocate(problem_c, problem_c2,
        problem_c2->smallest_values_of_interest, problem_c2->largest_values_of_interest);
	 
    /* Use the standard stacked problem_id as problem_name and 
     * construct a new suite-specific problem_id 
     */
    coco_problem_set_name(problem_c, problem_c->problem_id);
    coco_problem_set_id(problem_c, "bbob-constrained_f%02lu_i%02lu_d%02lu", 
        (unsigned long)function, (unsigned long)instance, (unsigned long)dimension);

    /* Construct problem type */
    coco_problem_set_type(problem_c, "%s_%s", problem_c2->problem_type, 
        problem_c2->problem_type);
  }
  
  coco_free_memory(gradient_c1);
  coco_random_free(random_generator);
  
  return problem_c;
 
}

