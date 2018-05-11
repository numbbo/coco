#include "coco.h"
#include "minunit_c89.h"

/**
 * Tests whether the coco_evaluate_function returns a vector of NANs when given a vector with one or more
 * NAN values.
 */
MU_TEST(test_coco_evaluate_function) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *x;
  double *y;

  suite = coco_suite("bbob", NULL, "dimensions: 2 instance_indices: 1");
  x = coco_allocate_vector(2);
  y = coco_allocate_vector(1);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    x[0] = 0;
    x[1] = NAN;
    coco_evaluate_function(problem, x, y);
    mu_check(coco_vector_contains_nan(y, 1));
  }
  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(y);

  suite = coco_suite("bbob-biobj", NULL, "dimensions: 2 instance_indices: 1");
  x = coco_allocate_vector(2);
  y = coco_allocate_vector(2);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
  	x[0] = 0;
  	x[1] = NAN;
    coco_evaluate_function(problem, x, y);
    mu_check(coco_vector_contains_nan(y, 2));
  }
  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(y);

  suite = coco_suite("bbob-constrained", NULL, "dimensions: 2 instance_indices: 1");
  x = coco_allocate_vector(2);
  y = coco_allocate_vector(1);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
  	x[0] = 0;
  	x[1] = NAN;
    coco_evaluate_function(problem, x, y);
    mu_check(coco_vector_contains_nan(y, 1));
  }
  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(y);
}

/**
 * Tests whether coco_evaluate_constraint returns a vector of NANs 
 * when given a vector with one or more NAN values.
 */
MU_TEST(test_coco_evaluate_constraint) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *x;
  double *y;
  size_t number_of_constraints;

  suite = coco_suite("bbob-constrained", NULL, "dimensions: 2 instance_indices: 1");
  x = coco_allocate_vector(2);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    x[0] = 0;
    x[1] = NAN;
  	number_of_constraints = coco_problem_get_number_of_constraints(problem);
  	y = coco_allocate_vector(number_of_constraints);
    coco_evaluate_constraint(problem, x, y);
    mu_check(coco_vector_contains_nan(y, number_of_constraints));
    coco_free_memory(y);
  }
  coco_suite_free(suite);
  coco_free_memory(x);
}

/**
 * Tests whether the coco_is_feasible returns 0 when given a vector 
 * with one or more NAN values and 1 for the initial solution given
 * by COCO.
 */
MU_TEST(test_coco_is_feasible) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *x, *initial_solution;
  double *y;
  size_t number_of_constraints;

  suite = coco_suite("bbob-constrained", NULL, "dimensions: 2 instance_indices: 1");
  x = coco_allocate_vector(2);
  initial_solution = coco_allocate_vector(2);
  
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    x[0] = 0;
    x[1] = NAN;
  	number_of_constraints = coco_problem_get_number_of_constraints(problem);
  	y = coco_allocate_vector(number_of_constraints);
    mu_check(coco_is_feasible(problem, x, y) == 0);
    coco_problem_get_initial_solution(problem, initial_solution);
    mu_check(coco_is_feasible(problem, initial_solution, y) == 1);
    coco_free_memory(y);
  }
  coco_suite_free(suite);
  coco_free_memory(x);
}

/**
 * Tests whether coco_problem_get_largest_fvalues_of_interest returns non-NULL values
 * on problems from the "bbob-biobj" test suite.
 */
MU_TEST(test_coco_problem_get_largest_fvalues_of_interest_bbob_biobj) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  const double *result;

  suite = coco_suite("bbob-biobj", NULL, NULL);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    result = coco_problem_get_largest_fvalues_of_interest(problem);
    mu_check(result);
  }
  coco_suite_free(suite);
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_coco_problem) {
  MU_RUN_TEST(test_coco_evaluate_function);
  MU_RUN_TEST(test_coco_evaluate_constraint);
  MU_RUN_TEST(test_coco_is_feasible);
  MU_RUN_TEST(test_coco_problem_get_largest_fvalues_of_interest_bbob_biobj);
}

