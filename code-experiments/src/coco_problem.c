/**
 * @file coco_problem.c
 * @brief Definitions of functions regarding COCO problems.
 */

#include <float.h>
#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"

/***********************************************************************************************************/

/**
 * @name Methods regarding the basic COCO problem
 */
/**@{*/
/**
 * Evaluates the problem function, increases the number of evaluations and updates the best observed value
 * and the best observed evaluation number.
 *
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The objective vector that is the result of the evaluation (in single-objective problems only the
 * first vector item is being set).
 */
void coco_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of problem->evaluate(problem, x, y) */
  size_t i, j;
  int is_feasible;
  double *z;
  
  assert(problem != NULL);
  assert(problem->evaluate_function != NULL);

  /* Set objective vector to INFINITY if the decision vector contains any INFINITY values */
  for (i = 0; i < coco_problem_get_dimension(problem); i++) {
    if (coco_is_inf(x[i])) {
      for (j = 0; j < coco_problem_get_number_of_objectives(problem); j++) {
        y[j] = fabs(x[i]);
      }
      return;
    }
  }
  
  /* Set objective vector to NAN if the decision vector contains any NAN values */
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  problem->evaluate_function(problem, x, y);
  problem->evaluations++; /* each derived class has its own counter, only the most outer will be visible */

  /* A little bit of bookkeeping */
  if (y[0] < problem->best_observed_fvalue[0]) {
    is_feasible = 1;
    if (coco_problem_get_number_of_constraints(problem) > 0) {
      z = coco_allocate_vector(coco_problem_get_number_of_constraints(problem));
      is_feasible = coco_is_feasible(problem, x, z);
      coco_free_memory(z);
    }
    if (is_feasible) {
      problem->best_observed_fvalue[0] = y[0];
      problem->best_observed_evaluation[0] = problem->evaluations;
    }
  }
}

/**
 * Evaluates the problem constraint.
 * 
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The vector of constraints that is the result of the evaluation.
 */
void coco_evaluate_constraint_optional_update(coco_problem_t *problem,
                                              const double *x,
                                              double *y,
                                              int update_counter) {
  /* implements a safer version of problem->evaluate(problem, x, y) */
  size_t i, j;
  assert(problem != NULL);
  if (problem->evaluate_constraint == NULL) {
    coco_error("coco_evaluate_constraint_optional_update(): No constraint function implemented for problem %s",
        problem->problem_id);
  }
  
  /* Set constraints vector to INFINITY if the decision vector contains any INFINITY values */
  for (i = 0; i < coco_problem_get_dimension(problem); i++) {
    if (coco_is_inf(x[i])) {
      for (j = 0; j < coco_problem_get_number_of_constraints(problem); j++) {
        y[j] = fabs(x[i]);
      }
      return;
    }
  }
  
  /* Set constraints vector to NAN if the decision vector contains any NAN values */
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
    return;
  }
  
  problem->evaluate_constraint(problem, x, y, update_counter);
  if (update_counter)
    problem->evaluations_constraints++;

}

void coco_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  coco_evaluate_constraint_optional_update(problem, x, y, 1);
}

/**
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The gradient of the function evaluated at the point x.
 */
static void bbob_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of problem->evaluate_gradient(problem, x, y) */
  assert(problem != NULL);
  if (problem->evaluate_gradient == NULL) {
    coco_error("bbob_evaluate_gradient(): No gradient function implemented for problem %s",
        problem->problem_id);
  }
  problem->evaluate_gradient(problem, x, y);
}

/**
 * Evaluates and logs the given solution (as the coco_evaluate_function), but does not return the evaluated
 * value.
 *
 * @note x must point to a correctly sized allocated memory region.

 * @param problem The given COCO problem.
 * @param x The decision vector.
 */
void coco_recommend_solution(coco_problem_t *problem, const double *x) {
  assert(problem != NULL);
  if (problem->recommend_solution == NULL) {
    coco_error("coco_recommend_solutions(): No recommend solution function implemented for problem %s",
        problem->problem_id);
  }
  problem->recommend_solution(problem, x);
}

/***********************************************************************************************************/

/**
 * @brief Allocates a new coco_problem_t for the given number of variables, number of objectives and
 * number of constraints.
 */
static coco_problem_t *coco_problem_allocate(const size_t number_of_variables,
                                             const size_t number_of_objectives,
                                             const size_t number_of_constraints) {
  coco_problem_t *problem;
  problem = (coco_problem_t *) coco_allocate_memory(sizeof(*problem));
  
  /* Initialize fields to sane/safe defaults */
  problem->initial_solution = NULL;
  problem->evaluate_function = NULL;
  problem->evaluate_constraint = NULL;
  problem->evaluate_gradient = NULL;
  problem->recommend_solution = NULL;
  problem->problem_free_function = NULL;
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = number_of_objectives;
  problem->number_of_constraints = number_of_constraints;
  problem->smallest_values_of_interest = coco_allocate_vector(number_of_variables);
  problem->largest_values_of_interest = coco_allocate_vector(number_of_variables);
  problem->number_of_integer_variables = 0; /* No integer variables by default */

  if (number_of_objectives > 1) {
    problem->is_opt_known = 0;        /* Optimum of multi-objective problems is unknown by default */
    problem->best_parameter = NULL;
    problem->best_value = coco_allocate_vector(number_of_objectives);
    problem->nadir_value = coco_allocate_vector(number_of_objectives);
  }
  else {
    problem->is_opt_known = 1;        /* Optimum of single-objective problems is known by default */
    problem->best_parameter = coco_allocate_vector(number_of_variables);
    problem->best_value = coco_allocate_vector(1);
    problem->nadir_value = NULL;
  }
  problem->problem_name = NULL;
  problem->problem_id = NULL;
  problem->problem_type = NULL;
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
  problem->final_target_delta[0] = 1e-8; /* in case to be modified by the benchmark */
  problem->best_observed_fvalue[0] = DBL_MAX;
  problem->best_observed_evaluation[0] = 0;
  problem->suite = NULL; /* To be initialized in the coco_suite_get_problem_from_indices() function */
  problem->suite_dep_index = 0;
  problem->suite_dep_function = 0;
  problem->suite_dep_instance = 0;
  problem->data = NULL;
  problem->versatile_data = NULL; /* Wassim: added to be able to pass data from one transformation to another*/
  return problem;
}

/**
 * @brief Creates a duplicate of the 'other' problem for all fields except for data, which points to NULL.
 */
static coco_problem_t *coco_problem_duplicate(const coco_problem_t *other) {
  size_t i;
  coco_problem_t *problem;
  problem = coco_problem_allocate(other->number_of_variables, other->number_of_objectives,
      other->number_of_constraints);

  problem->evaluate_function = other->evaluate_function;
  problem->evaluate_constraint = other->evaluate_constraint;
  problem->recommend_solution = other->recommend_solution;
  problem->problem_free_function = other->problem_free_function;
  
  problem->versatile_data = other->versatile_data; /* Wassim: make the pointers the same*/

  for (i = 0; i < problem->number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = other->smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] = other->largest_values_of_interest[i];
    if (other->best_parameter)
      problem->best_parameter[i] = other->best_parameter[i];
  }
  problem->number_of_integer_variables = other->number_of_integer_variables;

  if (other->initial_solution)
    problem->initial_solution = coco_duplicate_vector(other->initial_solution, other->number_of_variables);

  problem->is_opt_known = other->is_opt_known;
  if (other->best_value)
    for (i = 0; i < problem->number_of_objectives; ++i) {
      problem->best_value[i] = other->best_value[i];
    }

  if (other->nadir_value)
    for (i = 0; i < problem->number_of_objectives; ++i) {
      problem->nadir_value[i] = other->nadir_value[i];
    }

  problem->problem_name = coco_strdup(other->problem_name);
  problem->problem_id = coco_strdup(other->problem_id);
  problem->problem_type = coco_strdup(other->problem_type);

  problem->evaluations = other->evaluations;
  problem->evaluations_constraints = other->evaluations_constraints;
  problem->final_target_delta[0] = other->final_target_delta[0];
  problem->best_observed_fvalue[0] = other->best_observed_fvalue[0];
  problem->best_observed_evaluation[0] = other->best_observed_evaluation[0];

  problem->suite = other->suite;
  problem->suite_dep_index = other->suite_dep_index;
  problem->suite_dep_function = other->suite_dep_function;
  problem->suite_dep_instance = other->suite_dep_instance;

  problem->data = NULL;

  return problem;
}

/**
 * @brief Allocates a problem using scalar values for smallest_value_of_interest, largest_value_of_interest
 * and best_parameter. Assumes all variables are continuous.
 */
static coco_problem_t *coco_problem_allocate_from_scalars(const char *problem_name,
                                                          coco_evaluate_function_t evaluate_function,
                                                          coco_problem_free_function_t problem_free_function,
                                                          const size_t number_of_variables,
                                                          const double smallest_value_of_interest,
                                                          const double largest_value_of_interest,
                                                          const double best_parameter) {
  size_t i;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);

  problem->problem_name = coco_strdup(problem_name);
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = evaluate_function;
  problem->problem_free_function = problem_free_function;

  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = smallest_value_of_interest;
    problem->largest_values_of_interest[i] = largest_value_of_interest;
    problem->best_parameter[i] = best_parameter;
  }
  problem->number_of_integer_variables = 0;
  return problem;
}

void coco_problem_free(coco_problem_t *problem) {
  assert(problem != NULL);
  if (problem->problem_free_function != NULL) {
    problem->problem_free_function(problem);
  } else {
    /* Best guess at freeing all relevant structures */
    if (problem->smallest_values_of_interest != NULL)
      coco_free_memory(problem->smallest_values_of_interest);
    if (problem->largest_values_of_interest != NULL)
      coco_free_memory(problem->largest_values_of_interest);
    if (problem->best_parameter != NULL)
      coco_free_memory(problem->best_parameter);
    if (problem->best_value != NULL)
      coco_free_memory(problem->best_value);
    if (problem->nadir_value != NULL)
      coco_free_memory(problem->nadir_value);
    if (problem->problem_name != NULL)
      coco_free_memory(problem->problem_name);
    if (problem->problem_id != NULL)
      coco_free_memory(problem->problem_id);
    if (problem->problem_type != NULL)
      coco_free_memory(problem->problem_type);
    if (problem->data != NULL)
      coco_free_memory(problem->data);
    if (problem->initial_solution != NULL)
      coco_free_memory(problem->initial_solution);
    problem->smallest_values_of_interest = NULL;
    problem->largest_values_of_interest = NULL;
    problem->best_parameter = NULL;
    problem->best_value = NULL;
    problem->nadir_value = NULL;
    problem->suite = NULL;
    problem->data = NULL;
    problem->initial_solution = NULL;
    coco_free_memory(problem);
  }
}

/***********************************************************************************************************/

/**
 * @brief Checks whether the given string is in the right format to be a problem_id.
 *
 * No non-alphanumeric characters besides '-', '_' and '.' are allowed.
 */
static int coco_problem_id_is_fine(const char *id, ...) {
  va_list args;
  const int reject = 0;
  const int accept = 1;
  const char *cp;
  char *s;
  int result = accept;

  va_start(args, id);
  s = coco_vstrdupf(id, args);
  va_end(args);
  for (cp = s; *cp != '\0'; ++cp) {
    if (('A' <= *cp) && (*cp <= 'Z'))
      continue;
    if (('a' <= *cp) && (*cp <= 'z'))
      continue;
    if ((*cp == '_') || (*cp == '-'))
      continue;
    if (('0' <= *cp) && (*cp <= '9'))
      continue;
    result = reject;
  }
  coco_free_memory(s);
  return result;
}

/**
 * @brief Sets the problem_id using formatted printing (as in printf).
 *
 * Takes care of memory (de-)allocation and verifies that the problem_id is in the correct format.
 */
static void coco_problem_set_id(coco_problem_t *problem, const char *id, ...) {
  va_list args;

  va_start(args, id);
  if (problem->problem_id != NULL)
    coco_free_memory(problem->problem_id);
  problem->problem_id = coco_vstrdupf(id, args);
  va_end(args);
  if (!coco_problem_id_is_fine(problem->problem_id)) {
    coco_error("Problem id should only contain standard chars, not like '%s'", problem->problem_id);
  }
}

/**
 * @brief Sets the problem_name using formatted printing (as in printf).
 *
 * Takes care of memory (de-)allocation.
 */
static void coco_problem_set_name(coco_problem_t *problem, const char *name, ...) {
  va_list args;

  va_start(args, name);
  if (problem->problem_name != NULL)
    coco_free_memory(problem->problem_name);
  problem->problem_name = coco_vstrdupf(name, args);
  va_end(args);
}

/**
 * @brief Sets the problem_type using formatted printing (as in printf).
 *
 * Takes care of memory (de-)allocation.
 */
static void coco_problem_set_type(coco_problem_t *problem, const char *type, ...) {
  va_list args;

  va_start(args, type);
  if (problem->problem_type != NULL)
    coco_free_memory(problem->problem_type);
  problem->problem_type = coco_vstrdupf(type, args);
  va_end(args);
}

size_t coco_problem_get_evaluations(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->evaluations;
}

size_t coco_problem_get_evaluations_constraints(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->evaluations_constraints;
}

/**
 * @brief Returns 1 if the best parameter is not (close to) zero and 0 otherwise.
 */
static int coco_problem_best_parameter_not_zero(const coco_problem_t *problem) {
  size_t i = 0;
  int best_is_zero = 1;

  if (coco_vector_contains_nan(problem->best_parameter, problem->number_of_variables))
    return 1;

  while (i < problem->number_of_variables && best_is_zero) {
    best_is_zero = coco_double_almost_equal(problem->best_parameter[i], 0, 1e-9);
    i++;
  }

  return !best_is_zero;
}

/**
 * @note Can be used to prevent unnecessary burning of CPU time.
 */
int coco_problem_final_target_hit(const coco_problem_t *problem) {
  assert(problem != NULL);
  if (coco_problem_get_number_of_objectives(problem) != 1 ||
      coco_problem_get_evaluations(problem) < 1) 
    return 0;
  if (problem->best_value == NULL)
    return 0;
  return problem->best_observed_fvalue[0] <= problem->best_value[0] + problem->final_target_delta[0] ?
    1 : 0;
}
/**
 * @note Tentative...
 */
double coco_problem_get_best_observed_fvalue1(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->best_observed_fvalue[0];
}

/**
 * @brief Returns the optimal function value of the problem
 */
double coco_problem_get_best_value(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->best_value != NULL);
  return problem->best_value[0];
}

/**
 * @note This function breaks the black-box property: the returned  value is not
 * meant to be used by the optimization algorithm.
 */
double coco_problem_get_final_target_fvalue1(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->best_value != NULL);
  assert(problem->final_target_delta != NULL);
  return problem->best_value[0] + problem->final_target_delta[0];
}

/**
 * @note Do not modify the returned string! If you free the problem, the returned pointer becomes invalid.
 * When in doubt, use coco_strdup() on the returned value.
 */
const char *coco_problem_get_name(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_name != NULL);
  return problem->problem_name;
}

/**
 * The ID is guaranteed to contain only characters in the set [a-z0-9_-]. It should therefore be safe to use
 * it to construct filenames or other identifiers.
 *
 * Each problem ID should be unique within each benchmark suite.
 *
 * @note Do not modify the returned string! If you free the problem, the returned pointer becomes invalid.
 * When in doubt, use coco_strdup() on the returned value.
 */
const char *coco_problem_get_id(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_id != NULL);
  return problem->problem_id;
}

const char *coco_problem_get_type(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_type != NULL);
  return problem->problem_type;
}

size_t coco_problem_get_dimension(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->number_of_variables > 0);
  return problem->number_of_variables;
}

size_t coco_problem_get_number_of_objectives(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->number_of_objectives;
}

size_t coco_problem_get_number_of_constraints(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->number_of_constraints;
}

const double *coco_problem_get_smallest_values_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->smallest_values_of_interest != NULL);
  return problem->smallest_values_of_interest;
}

const double *coco_problem_get_largest_values_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->largest_values_of_interest != NULL);
  return problem->largest_values_of_interest;
}

size_t coco_problem_get_number_of_integer_variables(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->number_of_integer_variables;
}

const double *coco_problem_get_largest_fvalues_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  if (problem->number_of_objectives == 1)
    coco_error("coco_problem_get_largest_fvalues_of_interest(): f-values of interest undefined for single-objective problems");
  if (problem->nadir_value == NULL)
    coco_error("coco_problem_get_largest_fvalues_of_interest(): f-values of interest undefined");
  return problem->nadir_value;
}

/**
 * Copies problem->initial_solution into initial_solution if not null, 
 * otherwise the center of the problem's region of interest is the 
 * initial solution. Takes care of rounding the solution in case of integer variables.
 * 
 * @param problem The given COCO problem.
 * @param initial_solution The pointer to the initial solution being set by this method.
 */
void coco_problem_get_initial_solution(const coco_problem_t *problem, double *initial_solution) {
  
  size_t i; 
   
  assert(problem != NULL);
  if (problem->initial_solution != NULL) {
    for (i = 0; i < problem->number_of_variables; ++i)
      initial_solution[i] = problem->initial_solution[i];
  } else {
    assert(problem->smallest_values_of_interest != NULL);
    assert(problem->largest_values_of_interest != NULL);
    for (i = 0; i < problem->number_of_variables; ++i)
      initial_solution[i] = problem->smallest_values_of_interest[i] + 0.5
          * (problem->largest_values_of_interest[i] - problem->smallest_values_of_interest[i]);
    if (problem->number_of_integer_variables > 0) {
      for (i = 0; i < problem->number_of_integer_variables; ++i) {
        initial_solution[i] = coco_double_round(initial_solution[i]);
      }
    }
  }
}

static coco_suite_t *coco_problem_get_suite(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->suite;
}

static void coco_problem_set_suite(coco_problem_t *problem, coco_suite_t *suite) {
  assert(problem != NULL);
  problem->suite = suite;
}

size_t coco_problem_get_suite_dep_index(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->suite_dep_index;
}

static size_t coco_problem_get_suite_dep_function(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_function > 0);
  return problem->suite_dep_function;
}

static size_t coco_problem_get_suite_dep_instance(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_instance > 0);
  return problem->suite_dep_instance;
}
/**@}*/

void bbob_problem_best_parameter_print(const coco_problem_t *problem) {
  size_t i;
  FILE *file;
  assert(problem != NULL);
  assert(problem->best_parameter != NULL);
  file = fopen("._bbob_problem_best_parameter.txt", "w");
  if (file != NULL) {
    for (i = 0; i < problem->number_of_variables; ++i)
      fprintf(file, " %.16f ", problem->best_parameter[i]);
    fclose(file);
  }
}

void bbob_biobj_problem_best_parameter_print(const coco_problem_t *problem) {
  size_t i;
  FILE *file;
  coco_problem_t *problem1, *problem2;
  assert(problem != NULL);
  assert(problem->data != NULL);
  problem1 = ((coco_problem_stacked_data_t *) problem->data)->problem1;
  problem2 = ((coco_problem_stacked_data_t *) problem->data)->problem2;
  assert(problem1 != NULL);
  assert(problem2 != NULL);
  assert(problem1->best_parameter != NULL);
  assert(problem2->best_parameter != NULL);
  file = fopen("._bbob_biobj_problem_best_parameter.txt", "w");
  if (file != NULL) {
    for (i = 0; i < problem->number_of_variables; ++i)
      fprintf(file, " %.16f ", problem1->best_parameter[i]);
    fprintf(file, "\n");
    for (i = 0; i < problem->number_of_variables; ++i)
      fprintf(file, " %.16f ", problem2->best_parameter[i]);
    fclose(file);
  }
}

/***********************************************************************************************************/

/**
 * @name Methods regarding the transformed COCO problem
 */
/**@{*/

/**
 * @brief Returns the data of the transformed problem.
 */
static void *coco_problem_transformed_get_data(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->data != NULL);
  assert(((coco_problem_transformed_data_t *) problem->data)->data != NULL);

  return ((coco_problem_transformed_data_t *) problem->data)->data;
}

/**
 * @brief Returns the inner problem of the transformed problem.
 */
static coco_problem_t *coco_problem_transformed_get_inner_problem(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->data != NULL);
  assert(((coco_problem_transformed_data_t *) problem->data)->inner_problem != NULL);

  return ((coco_problem_transformed_data_t *) problem->data)->inner_problem;
}

/**
 * @brief Calls the coco_evaluate_function function on the inner problem.
 */
static void coco_problem_transformed_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  coco_evaluate_function(data->inner_problem, x, y);
}

/**
 * @brief Calls the coco_evaluate_constraint_optional_update function on the inner problem.
 */
static void coco_problem_transformed_evaluate_constraint(coco_problem_t *problem, const double *x, double *y, int update_counter) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  coco_evaluate_constraint_optional_update(data->inner_problem, x, y, update_counter);
}

static void bbob_problem_transformed_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  bbob_evaluate_gradient(data->inner_problem, x, y);
}

/**
 * @brief Calls the coco_recommend_solution function on the inner problem.
 */
static void coco_problem_transformed_recommend_solution(coco_problem_t *problem, const double *x) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  coco_recommend_solution(data->inner_problem, x);
}

/**
 * @brief Frees only the data of the transformed problem leaving the inner problem intact.
 *
 * @note If there is no other pointer to the inner problem, access to it will be lost.
 */
static void coco_problem_transformed_free_data(coco_problem_t *problem) {
  coco_problem_transformed_data_t *data;

  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;

  if (data->data != NULL) {
    if (data->data_free_function != NULL) {
      data->data_free_function(data->data);
      data->data_free_function = NULL;
    }
    coco_free_memory(data->data);
    data->data = NULL;
  }
  /* Let the generic free problem code deal with the rest of the fields. For this we clear the free_problem
   * function pointer and recall the generic function. */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Frees the transformed problem.
 */
static void coco_problem_transformed_free(coco_problem_t *problem) {
  coco_problem_transformed_data_t *data;

  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);
  if (data->inner_problem != NULL) {
    coco_problem_free(data->inner_problem);
    data->inner_problem = NULL;
  }
  coco_problem_transformed_free_data(problem);
}

/**
 * @brief Allocates a transformed problem that wraps the inner_problem.
 *
 * By default all methods will dispatch to the inner_problem. A prefix is prepended to the problem name
 * in order to reflect the transformation somewhere.
 */
static coco_problem_t *coco_problem_transformed_allocate(coco_problem_t *inner_problem,
                                                         void *user_data,
                                                         coco_data_free_function_t data_free_function,
                                                         const char *name_prefix) {
  coco_problem_transformed_data_t *problem;
  coco_problem_t *inner_copy;
  char *old_name = coco_strdup(inner_problem->problem_name);

  problem = (coco_problem_transformed_data_t *) coco_allocate_memory(sizeof(*problem));
  problem->inner_problem = inner_problem;
  problem->data = user_data;
  problem->data_free_function = data_free_function;

  inner_copy = coco_problem_duplicate(inner_problem);
  inner_copy->evaluate_function = coco_problem_transformed_evaluate_function;
  inner_copy->evaluate_constraint = coco_problem_transformed_evaluate_constraint;
  inner_copy->evaluate_gradient = bbob_problem_transformed_evaluate_gradient;
  inner_copy->recommend_solution = coco_problem_transformed_recommend_solution;
  inner_copy->problem_free_function = coco_problem_transformed_free;
  inner_copy->data = problem;

  coco_problem_set_name(inner_copy, "%s(%s)", name_prefix, old_name);
  coco_free_memory(old_name);

  return inner_copy;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding the stacked COCO problem
 */
/**@{*/

/**
 * @brief Calls the coco_evaluate_function function on the underlying problems.
 */
static void coco_problem_stacked_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_stacked_data_t* data = (coco_problem_stacked_data_t *) problem->data;

  const size_t number_of_objectives_problem1 = coco_problem_get_number_of_objectives(data->problem1);
  const size_t number_of_objectives_problem2 = coco_problem_get_number_of_objectives(data->problem2);
  double *cons_values = NULL;
  int is_feasible;
    
  assert(coco_problem_get_number_of_objectives(problem)
      == number_of_objectives_problem1 + number_of_objectives_problem2);
  
  if (number_of_objectives_problem1 > 0)
     coco_evaluate_function(data->problem1, x, &y[0]);
  if (number_of_objectives_problem2 > 0)
     coco_evaluate_function(data->problem2, x, &y[number_of_objectives_problem1]);

  /* Make sure that no feasible point has a function value lower
   * than the minimum's.
   */
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);   
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
}

/**
 * @brief Calls the coco_evaluate_constraint_optional_update function on the underlying problems.
 */
static void coco_problem_stacked_evaluate_constraint(coco_problem_t *problem, const double *x, double *y, int update_counter) {
  coco_problem_stacked_data_t* data = (coco_problem_stacked_data_t*) problem->data;

  const size_t number_of_constraints_problem1 = coco_problem_get_number_of_constraints(data->problem1);
  const size_t number_of_constraints_problem2 = coco_problem_get_number_of_constraints(data->problem2);
  assert(coco_problem_get_number_of_constraints(problem)
      == number_of_constraints_problem1 + number_of_constraints_problem2);

  if (number_of_constraints_problem1 > 0)
    coco_evaluate_constraint_optional_update(data->problem1, x, y, update_counter);
  if (number_of_constraints_problem2 > 0)
    coco_evaluate_constraint_optional_update(data->problem2, x, &y[number_of_constraints_problem1], update_counter);
  
}

/* TODO: Missing coco_problem_stacked_recommend_solution function! */

/**
 * @brief Frees the stacked problem.
 */
static void coco_problem_stacked_free(coco_problem_t *problem) {
  coco_problem_stacked_data_t *data;

  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_stacked_data_t*) problem->data;

  if (data->problem1 != NULL) {
    coco_problem_free(data->problem1);
    data->problem1 = NULL;
  }
  if (data->problem2 != NULL) {
    coco_problem_free(data->problem2);
    data->problem2 = NULL;
  }
  /* Let the generic free problem code deal with the rest of the fields. For this we clear the free_problem
   * function pointer and recall the generic function. */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Allocates a problem constructed by stacking two COCO problems.
 * 
 * This is particularly useful for generating multi-objective problems, e.g. a bi-objective problem from two
 * single-objective problems. The stacked problem must behave like a normal COCO problem accepting the same
 * input.
 *
 * @note Regions of interest in the decision space must either agree or at least one of them must be NULL.
 * @note Best parameter becomes somewhat meaningless, but the nadir value make sense now.
 */
static coco_problem_t *coco_problem_stacked_allocate(coco_problem_t *problem1, 
                                                     coco_problem_t *problem2,
                                                     const double *smallest_values_of_interest,
                                                     const double *largest_values_of_interest) {

  size_t number_of_variables, number_of_objectives, number_of_constraints;
  size_t i;
  char *s;
  coco_problem_stacked_data_t *data;
  coco_problem_t *problem; /* the new coco problem */

  assert(problem1);
  assert(problem2);
  assert(coco_problem_get_dimension(problem1) == coco_problem_get_dimension(problem2));

  number_of_variables = coco_problem_get_dimension(problem1);
  number_of_objectives = coco_problem_get_number_of_objectives(problem1)
      + coco_problem_get_number_of_objectives(problem2);
  number_of_constraints = coco_problem_get_number_of_constraints(problem1)
      + coco_problem_get_number_of_constraints(problem2);

  problem = coco_problem_allocate(number_of_variables, number_of_objectives, number_of_constraints);
  
  s = coco_strconcat(coco_problem_get_id(problem1), "__");
  problem->problem_id = coco_strconcat(s, coco_problem_get_id(problem2));
  coco_free_memory(s);
  s = coco_strconcat(coco_problem_get_name(problem1), " + ");
  problem->problem_name = coco_strconcat(s, coco_problem_get_name(problem2));
  coco_free_memory(s);

  problem->evaluate_function = coco_problem_stacked_evaluate_function;
  if (number_of_constraints > 0)
    problem->evaluate_constraint = coco_problem_stacked_evaluate_constraint;

  assert(smallest_values_of_interest);
  assert(largest_values_of_interest);
  
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] = largest_values_of_interest[i];
  }
  assert(problem1->number_of_integer_variables == problem2->number_of_integer_variables);
  problem->number_of_integer_variables = problem1->number_of_integer_variables;

  assert(problem->best_value);
    
  if (number_of_constraints > 0) {
     
    /* The best_value must be set up afterwards in suite_cons_bbob_problems.c */
    problem->best_value[0] = -FLT_MAX;
    
    /* Define problem->initial_solution as problem2->initial_solution */
    if (coco_problem_get_number_of_constraints(problem2) > 0 && problem2->initial_solution)
      problem->initial_solution = coco_duplicate_vector(problem2->initial_solution, number_of_variables);
      
  }
  else {
     
    /* Compute the ideal and nadir values */
    assert(problem->nadir_value);
    
    problem->best_value[0] = problem1->best_value[0];
    problem->best_value[1] = problem2->best_value[0];
    coco_evaluate_function(problem1, problem2->best_parameter, &problem->nadir_value[0]);
    coco_evaluate_function(problem2, problem1->best_parameter, &problem->nadir_value[1]);
    
  }

  /* setup data holder */
  data = (coco_problem_stacked_data_t *) coco_allocate_memory(sizeof(*data));
  data->problem1 = problem1;
  data->problem2 = problem2;

  problem->data = data;
  problem->problem_free_function = coco_problem_stacked_free;

  return problem;
}
/**@}*/

/***********************************************************************************************************/
