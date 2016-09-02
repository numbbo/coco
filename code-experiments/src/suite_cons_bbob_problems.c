/**
 * @file  suite_cons_bbob_problems.c
 * @brief Implementation of the problems in the constrained BBOB suite.
 * 
 * This suite contains 48 constrained functions in continuous domain 
 * which are derived from combining 8 single-objective functions of the
 * noiseless bbob test suite with randomly-generated 
 * linear constraints perturbed by nonlinear transformations.
 * Each one of the 8 functions is combined with 6 different numbers of 
 * constraints: 1, 2, 10, n/2, n-1 and n+1.
 * 
 * The generic algorithm for defining the constrained functions is given below:
 * 
 * 1. Pick up a 'bbob' function f whose raw version is pseudoconvex 
 *    to be the objective function.
 * 2. Remove possible nonlinear transformations from f. (see note below)
 * 3. In order to make sure that the feasible set to be built is not
 *    empty, we must choose a direction p that should be kept feasible.
 *    Define p as the gradient of f at the origin.
 * 4. Define the first constraint function g_1(x) by setting its 
 *    gradient to a_1 = -p. By definition, g_1(p) < 0. Thus p is feasible
 *    for g_1.
 * 5. Generate the other constraints randomly while making sure that 
 *    p remains feasible for each one.
 * 6. Apply to the whole constrained function the nonlinear transformations 
 *    that were removed from the objective function in Step 2. (see note below)
 * 7. Choose a random point xopt and move the optimum away from the 
 *    origin to xopt by translating the constrained function by -xopt.
 * 
 * @note a_1 is set to -p, i.e. the gradient of f at the origin, in order to 
 *       have the KKT conditions easily satisfied.
 * 
 * @note The pseudoconvexity in Step 1 guarantees that the KKT conditions 
 *       are sufficient for optimality.
 * 
 * @note The removal of possible nonlinear transformations in Step 2
 *       and posterior application in Step 6 are necessary to make
 *       sure that the KKT conditions are satisfied in the optimum
 *       - until then the origin. As explained in the documentation, 
 *       the application of the nonlinear transformations in Step 6 
 *       does not affect the location of the optimum.
 * 
 * @note Steps 1 and 2 are done within the 'allocate' function of the
 *       objective function, e.g. f_ellipsoid_cons_bbob_problem_allocate().
 * 
 * @note Steps 4 and 5 are done within c_linear_cons_bbob_problem_allocate().
 * 
 * @note The constrained Rastrigin function's construction differs a bit
 *       from the steps above. Since it is a multimodal function with
 *       well distributed local optima, we choose one of its local optima
 *       to be the global constrained optimum by adding constraints 
 *       that pass through that point.
 * 
 * @note An initial solution is provided to the user by the testbed.
 *       With exception of the the constrained Rastrigin function, the
 *       initial solution is the feasible direction p scaled by
 *       a constant.
 * 
 * 
 */

#include <math.h>

#include "coco.h"
#include "coco_utilities.c"
#include "c_linear.c"
#include "f_different_powers.c"
#include "f_discus.c"
#include "f_ellipsoid.c"
#include "f_linear_slope.c"
#include "f_rastrigin.c"
#include "f_sphere.c"

/**
 * @brief Calculates the obj. function type based on the value
 *        of "function"
 */
static size_t obj_function_type(const size_t function) {
  
  size_t problems_per_obj_function_type = 6;
  return (size_t)ceil((double)function/problems_per_obj_function_type);
  
}

/**
 * @brief Returns the number of linear constraints associated to the
 *        value of "function"
 */
static size_t nb_of_linear_constraints(const size_t function,
                                       const size_t dimension) {
  
  int problems_per_obj_function_type = 6;
  int p;
  
  /* Map "function" value into {1, ..., problems_per_obj_function_type} */
  p = (((int)function - 1) % problems_per_obj_function_type) + 1;
  
  if (p == 1) return 1;
  else if (p == 2) return 2;
  else if (p == 3) return 10;
  else if (p == 4) return dimension/2;
  else if (p == 5) return dimension - 1;
  else return dimension + 1;
  
}

/**
 * @brief Objective function: sphere
 *        Constraint(s): linear
 */
static coco_problem_t *f_sphere_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
                                                         		
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;                
  
  all_zeros = coco_allocate_vector(dimension);             
  
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;    
  
  problem = f_sphere_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
	 
  coco_evaluate_gradient(problem, all_zeros, feasible_direction);	 
  coco_scale_vector(feasible_direction, dimension, 50.0);
	 
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
	    
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
	    
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;  
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);  
  problem->evaluations = 0;  
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
	
  return problem;
 
}

/**
 * @brief Objective function: ellipsoid
 *        Constraint(s): linear
 */
static coco_problem_t *f_ellipsoid_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
     
  problem = f_ellipsoid_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);

  coco_evaluate_gradient(problem, all_zeros, feasible_direction);
  coco_scale_vector(feasible_direction, dimension, 50.0);
  
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);
  problem->evaluations = 0;  
  
  problem = transform_vars_oscillate(problem);
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: rotated ellipsoid
 *        Constraint(s): linear
 */
static coco_problem_t *f_ellipsoid_rotated_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  problem = f_ellipsoid_rotated_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  coco_evaluate_gradient(problem, all_zeros, feasible_direction);
  coco_scale_vector(feasible_direction, dimension, 50.0);
  
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
     
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);
  problem->evaluations = 0;  
  
  problem = transform_vars_oscillate(problem);
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: linear slope
 *        Constraint(s): linear
 */
static coco_problem_t *f_linear_slope_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  problem = f_linear_slope_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  coco_evaluate_gradient(problem, all_zeros, feasible_direction);
  coco_scale_vector(feasible_direction, dimension, 50.0);
  
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);
  problem->evaluations = 0;  
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: discus
 *        Constraint(s): linear
 */
static coco_problem_t *f_discus_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;

  problem = f_discus_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  coco_evaluate_gradient(problem, all_zeros, feasible_direction);
  coco_scale_vector(feasible_direction, dimension, 50.0);

  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);
  problem->evaluations = 0;  
     
  problem = transform_vars_oscillate(problem);
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: bent cigar
 *        Constraint(s): linear
 */
static coco_problem_t *f_bent_cigar_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  problem = f_bent_cigar_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  coco_evaluate_gradient(problem, all_zeros, feasible_direction);
  coco_scale_vector(feasible_direction, dimension, 50.0);

  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);
  problem->evaluations = 0;  
     
  problem = transform_vars_asymmetric(problem, 0.5);
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: different powers
 *        Constraint(s): linear
 */
static coco_problem_t *f_different_powers_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  problem = f_different_powers_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  coco_evaluate_gradient(problem, all_zeros, feasible_direction);
  coco_scale_vector(feasible_direction, dimension, 50.0);
  
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);
  problem->evaluations = 0;  
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: rastrigin
 *        Constraint(s): linear
 */
static coco_problem_t *f_rastrigin_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  double norm_factor = 10.0;
  
  char *problem_type_temp = NULL;
	 
  /* The function below has global minimum at (-1,-1,...,-1) */
  problem = f_rastrigin_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
  
  /* Define the feasible_direction and, consequently, the initial
   * solution provided by the testbed as a point in (1,2)^n. This avoids
   * providing a local optimal solution (such as all-ones) as initial 
   * solution to the user. Otherwise, algorithms that look for local 
   * optima would stop at iteration 1 if they used such an initial
   * solution.
   */
  srand(rseed);
  for (i = 0; i < dimension; ++i)
    feasible_direction[i] = 1.0 + (double)rand()/((double)RAND_MAX+1);
     
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints, norm_factor,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  coco_evaluate_function(problem, problem->best_parameter, problem->best_value);
  problem->evaluations = 0;  
  
  problem = transform_vars_asymmetric(problem, 0.2);
  problem = transform_vars_oscillate(problem);
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  
  return problem;
 
}
