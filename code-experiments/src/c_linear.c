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
  double x_shift_factor; /* shift solution by - factor * gradient */
  double gradient_norm;
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
                                          const double *gradient,
                                          const double x_shift_factor);
         
static coco_problem_t *c_linear_single_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t constraint_number,
                                                      const double factor1,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      double *gradient,
                                                      double x_shift_factor,
                                                      const double *feasible_direction);
                                                      
static coco_problem_t *c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_active_constraints,
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
    /* prevent that the optimal solution is infeasible due to loss of precision when shifted */
    if (x[i] > 1e-11 || x[i] < 0)
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
  double factor;
  
  linear_constraint_data_t *data;
  coco_problem_t *inner_problem;
  
  data = (linear_constraint_data_t *) coco_problem_transformed_get_data(self);
  inner_problem = coco_problem_transformed_get_inner_problem(self);
  
  assert(self->number_of_constraints == 1);

  for (i = 0, factor = data->x_shift_factor / data->gradient_norm;
       i < self->number_of_variables; ++i)
    data->x[i] = (data->gradient[i]) * (x[i] - factor * data->gradient[i]);
  
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
 *        as argument and with the feasible domain extended by
 *        shift_factor.
 */
static coco_problem_t *c_linear_transform(coco_problem_t *inner_problem, 
                                          const double *gradient,
                                          const double shift_factor) {
  
  linear_constraint_data_t *data;
  coco_problem_t *self;
  double gradient_norm = coco_vector_norm(gradient, inner_problem->number_of_variables);

  if (gradient_norm <= 0)
    coco_error("c_linear_transform(): gradient norm %f<=0 zero", gradient_norm);

  data = coco_allocate_memory(sizeof(*data));
  data->gradient = coco_duplicate_vector(gradient, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->x_shift_factor = shift_factor;
  data->gradient_norm = gradient_norm;
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
                                                      const double x_shift_factor,
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
    problem = c_linear_transform(problem, gradient, x_shift_factor);

  }
  else{ /* Randomly generate the gradient of the linear constraint */
	  
    gradient_linear_constraint = coco_allocate_vector(dimension);
     
    /* Generate a pseudorandom vector with distribution N_i(0, I)
     * and scale it with 'factor1' and 'factor2' (see comments above)
     */
    for (i = 0; i < dimension; ++i)
      gradient_linear_constraint[i] = factor1 *
                coco_random_normal(random_generator) * factor2 / sqrt((double)dimension);

    problem = c_linear_transform(problem, gradient_linear_constraint, x_shift_factor);
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
 * @brief helper function to successively compute a linear combination of
 *        constraint gradients, add current gradient if weight is != 0.
 *
*/
static void con_update_linear_combination(double *linear_combination,
                                          const coco_problem_t *problem,
                                          double weight) {
  size_t i;
  linear_constraint_data_t *data;

  data = (linear_constraint_data_t *) coco_problem_transformed_get_data(problem);
  if (data->gradient == NULL) {
    if (weight != 0) {
      coco_error("con_update_linear_combination(): gradient of constraint was zero");
    } else {
      coco_warning("con_update_linear_combination(): gradient of constraint was zero");
    }
  } else {
    if (data->x_shift_factor != 0)
      coco_warning("Inactive constraint passed to update_linear_combination, x_shift_factor=%f",
          data->x_shift_factor);
    if (weight == 0)
      return; /* nothing to add */
    if (weight < 0)
      coco_warning("con_update_linear_combination: weight=%f < 0, should be > 0", weight);
    for (i = 0; i < problem->number_of_variables; ++i)
      linear_combination[i] += weight * data->gradient[i];
  }
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
                                                      const size_t number_of_active_constraints,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      const double *feasible_direction) {
																																			
  const double global_scaling_factor = 100.;
  size_t i, j;
  
  coco_problem_t *problem_c = NULL;
  coco_problem_t *problem_c2 = NULL;
  coco_random_state_t *random_generator;
  coco_random_state_t *random_generator2;
  double *gradient_c1 = NULL;
  double *gradient;
  double *linear_combination;
  linear_constraint_data_t *first_constraint_data = NULL;
  double shift_factor;
  long seed_cons;
  double fac, norm, factor1;
  size_t number_of_linear_constraints;
  size_t inactive_constraints_left = number_of_active_constraints / 2;  /* TODO: decide whether this should go into the interface */
  int disguise_gradient = 1;

  if (strncmp(problem_id_template, "bbob-constrained-active-only", 28) == 0)
    inactive_constraints_left = 0;

  if (strncmp(problem_id_template, "bbob-constrained-no-disguise", 28) == 0)
    disguise_gradient = 0;

  number_of_linear_constraints = number_of_active_constraints + inactive_constraints_left;

  gradient_c1 = coco_allocate_vector(dimension);
  linear_combination = coco_allocate_vector_with_value(dimension, 0.0);
  																	
  for (i = 0; i < dimension; ++i)
    gradient_c1[i] = -feasible_direction[i];

  /* Build a coco_problem_t object for each constraint. 
   * The constraints' gradients are generated randomly with
   * distribution 10**U[0,1] * N_i(0, I/n) * 10**U_i[0,2]
   * where U[a, b] is uniform in [a,b] and only U_i is drawn 
   * for each constraint individually.
   */
  
  /* Calculate the first random factor 10**U[0,1]. */
  seed_cons = (long)(function + 10000 * instance);
  random_generator = coco_random_new((uint32_t) seed_cons);
  random_generator2 = coco_random_new((uint32_t) seed_cons);
  factor1 = global_scaling_factor * pow(10.0, coco_random_uniform(random_generator));

  /* Build the first linear constraint using 'gradient_c1' to build
   * its gradient.
   */ 
  /* set gradient depending on instance number */
  gradient = instance % number_of_active_constraints ? NULL : gradient_c1;
  problem_c = c_linear_single_cons_bbob_problem_allocate(function,
      dimension, instance, 1, factor1,
      problem_id_template, problem_name_template, gradient, 0.0,
      feasible_direction);
  if (gradient != NULL)  /* preserve the constraint based on function gradient */
    first_constraint_data = (linear_constraint_data_t *) coco_problem_transformed_get_data(problem_c);
  else
    con_update_linear_combination(linear_combination, problem_c,
                                  coco_random_uniform(random_generator2));
	 
  /* Instantiate the other linear constraints (if any) and stack them 
   * all into problem_c
   */
  for (j = 2, i = 1; j <= number_of_linear_constraints; ++j) {
	 
    /* Instantiate a new problem containing one linear constraint only */
    /* set gradient depending on instance number */

    if (i < number_of_active_constraints && coco_random_uniform(random_generator)
          + 1e-23 > (double) inactive_constraints_left / (double) number_of_linear_constraints
        ) {  /* create an active constraint */
      gradient = (++i - 1 + instance) % number_of_active_constraints ? NULL : gradient_c1;
      shift_factor = 0.0;
    } else {  /* create an inactive (shifted) constraint */
      if (inactive_constraints_left-- <= 0)
        coco_error("c_linear_cons_bbob_problem_allocate(): no inactive left i=%d j=%d nb_act=%ul nb_con=%ul",
                   i, j, number_of_active_constraints, number_of_linear_constraints);
      gradient = NULL;
      shift_factor = 0.01 + 2.0 * coco_random_uniform(random_generator);
    }

    problem_c2 = c_linear_single_cons_bbob_problem_allocate(function, dimension, instance,
        i <= number_of_active_constraints ? i : number_of_linear_constraints - inactive_constraints_left,
        factor1, problem_id_template, problem_name_template, gradient, shift_factor,
        feasible_direction);
    if (shift_factor == 0) {  /* active constraint */
      if (gradient != NULL) {  /* preserve the constraint based on function gradient */
        if (first_constraint_data)
          coco_warning("c_linear_cons_bbob_problem_allocate(): first_constraint_data already assigned, this is probably a bug");
        first_constraint_data = (linear_constraint_data_t *) coco_problem_transformed_get_data(problem_c2);
      } else
        con_update_linear_combination(linear_combination, problem_c2, coco_random_uniform(random_generator2));
    }

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

  /* Modify first constraint without changing the feasible solution,
   * thereby disguising the gradient of the function
   */
  if (number_of_active_constraints > 1 && disguise_gradient) {
    norm = first_constraint_data->gradient_norm;  /* for reading convenience only */
    gradient = first_constraint_data->gradient;   /* ditto */
    fac = coco_vector_scalar_product(linear_combination, gradient, dimension);
    if (fac < 0)
      coco_error("scalar product between first constraint and linear combination = %f < 0 should be > 0", fac);
    fac = fac <= 0 ? (double) dimension : coco_double_min((double) dimension, norm * norm / fac);
    fac *= (1 + coco_random_uniform(random_generator2)) / (2 + 1. / (double) dimension);  /* bound to <1 and >1/3 */
    for (i = 0; i < dimension; ++i)
      gradient[i] -= fac * linear_combination[i];
    fac = coco_vector_norm(gradient, dimension);  /* new gradient norm */
    if (fac == 0)  /* make sure that gradient did not become zero */
      coco_error("new vector norm = %f == 0, should be > 0", fac);
    coco_vector_scale(gradient, dimension, norm, fac); /* restore original norm */
  }
  
  coco_free_memory(linear_combination);
  coco_free_memory(gradient_c1);
  coco_random_free(random_generator);
  coco_random_free(random_generator2);

  return problem_c;
 
}

