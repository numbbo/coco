/**
 * @file  suite_cons_bbob.c
 * @brief Implementation of the constrained bbob suite containing 
 *        48 constrained problems in 6 dimensions. See comments in
 *        "suite_cons_bbob_problems.c" for more details.
 */

#include "coco.h"
#include "suite_cons_bbob_problems.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob suite.
 */
static coco_suite_t *suite_cons_bbob_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("bbob-constrained", 48, 6, dimensions, "year: 2016");

  return suite;
}

/**
 * @brief Sets the instances associated with years for the constrained
 *        bbob suite.
 */
static const char *suite_cons_bbob_get_instances_by_year(const int year) {

  if ((year == 2016) || (year == 0)) {
    return "1-15";
  }
  else {
    coco_error("suite_cons_bbob_get_instances_by_year(): year %d not defined for suite_cons_bbob", year);
    return NULL;
  }
}

/**
 * @brief Creates and returns a constrained BBOB problem.
 */
static coco_problem_t *coco_get_cons_bbob_problem(const size_t function,
                                                  const size_t dimension,
                                                  const size_t instance) {
  
  size_t number_of_linear_constraints; 
  coco_problem_t *problem = NULL;
  
  double *feasible_direction = coco_allocate_vector(dimension);  
  double *xopt = coco_allocate_vector(dimension);  
  
  const char *problem_id_template = "bbob-constrained_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "bbob-constrained suite problem f%lu instance %lu in %luD";
  
  /* Seed value used for shifting the whole constrained problem */
  long rseed = (long) (function + 10000 * instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  
  /* Choose a different seed value for building the objective function */
  rseed = (long) (function + 20000 * instance);
  
  number_of_linear_constraints = nb_of_linear_constraints(function, dimension);
  
  if (obj_function_type(function) == 1) {
	  
    problem = f_sphere_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	 
  } else if (obj_function_type(function) == 2) {
	  
    problem = f_ellipsoid_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 3) {
	  
    problem = f_linear_slope_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 4) {
	  
    problem = f_ellipsoid_rotated_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 5) {
	  
    problem = f_discus_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 6) {
	  
    problem = f_bent_cigar_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 7) {
	  
    problem = f_different_powers_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 8) {
	  
    problem = f_rastrigin_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else {
    coco_error("get_cons_bbob_problem(): cannot retrieve problem f%lu instance %lu in %luD", 
        function, instance, dimension);
    coco_free_memory(xopt);
    coco_free_memory(feasible_direction);
    return NULL; /* Never reached */
  }
  
  coco_free_memory(xopt);
  coco_free_memory(feasible_direction);
  
  return problem;
}

/**
 * @brief Returns the problem from the constrained bbob suite that 
 *        corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_cons_bbob_get_problem(coco_suite_t *suite,
                                                   const size_t function_idx,
                                                   const size_t dimension_idx,
                                                   const size_t instance_idx) {
  
  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = coco_get_cons_bbob_problem(function, dimension, instance);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);
  
  /* Use the standard stacked problem_id as problem_name and 
   * construct a new suite-specific problem_id 
   */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "bbob-constrained_f%02lu_i%02lu_d%02lu", 
  (unsigned long)function, (unsigned long)instance, (unsigned long)dimension);
  
  return problem;
}
