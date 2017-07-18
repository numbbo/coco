#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

/**
 * Tests whether the coco_evaluate_function returns a vector of NANs when given a vector with one or more
 * NAN values.
 */
static void test_coco_evaluate_function(void **state) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *x;
  double *y;

  suite = coco_suite("bbob", NULL, NULL);
  x = coco_allocate_vector(2);
  y = coco_allocate_vector(1);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    if (coco_problem_get_dimension(problem) > 2)
  		continue;
    x[0] = 0;
    x[1] = NAN;
    coco_evaluate_function(problem, x, y);
    assert(coco_vector_contains_nan(y, 1));
  }
  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(y);

  suite = coco_suite("bbob-biobj", NULL, NULL);
  x = coco_allocate_vector(2);
  y = coco_allocate_vector(2);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
  	 if (coco_problem_get_dimension(problem) > 2)
  		continue;
  	 x[0] = 0;
  	 x[1] = NAN;
    coco_evaluate_function(problem, x, y);
    assert(coco_vector_contains_nan(y, 2));
  }
  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(y);
  
  suite = coco_suite("bbob-constrained", NULL, NULL);
  x = coco_allocate_vector(2);
  y = coco_allocate_vector(1);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
  	 if (coco_problem_get_dimension(problem) > 2)
  		continue;
  	 x[0] = 0;
  	 x[1] = NAN;
    coco_evaluate_function(problem, x, y);
    assert(coco_vector_contains_nan(y, 1));
  }
  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(y);

  (void)state; /* unused */
}

/**
 * Tests whether coco_evaluate_constraint returns a vector of NANs 
 * when given a vector with one or more NAN values.
 */
static void test_coco_evaluate_constraint(void **state) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *x;
  double *y;
  size_t number_of_constraints;

  suite = coco_suite("bbob-constrained", NULL, NULL);
  x = coco_allocate_vector(2);
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
  	 if (coco_problem_get_dimension(problem) > 2)
      continue;
    x[0] = 0;
    x[1] = NAN;
  	
  	 number_of_constraints = coco_problem_get_number_of_constraints(problem);
  	 y = coco_allocate_vector(number_of_constraints);
    coco_evaluate_constraint(problem, x, y);
    assert(coco_vector_contains_nan(y, number_of_constraints));
    coco_free_memory(y);
  }
  coco_suite_free(suite);
  coco_free_memory(x);

  (void)state; /* unused */
}

/**
 * Tests whether the coco_is_feasible returns 0 when given a vector 
 * with one or more NAN values and 1 for the initial solution given
 * by COCO.
 */
static void test_coco_is_feasible(void **state) {

  coco_suite_t *suite;
  coco_problem_t *problem;
  double *x, *initial_solution;
  double *y;
  size_t number_of_constraints;

  suite = coco_suite("bbob-constrained", NULL, NULL);
  x = coco_allocate_vector(2);
  initial_solution = coco_allocate_vector(2);
  
  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
  	 if (coco_problem_get_dimension(problem) > 2)
      continue;
    x[0] = 0;
    x[1] = NAN;
  	
  	 number_of_constraints = coco_problem_get_number_of_constraints(problem);
  	 y = coco_allocate_vector(number_of_constraints);
    
    assert(coco_is_feasible(problem, x, y) == 0);
    
    coco_problem_get_initial_solution(problem, initial_solution);
    assert(coco_is_feasible(problem, initial_solution, y) == 1);
    
    coco_free_memory(y);
  }
  coco_suite_free(suite);
  coco_free_memory(x);

  (void)state; /* unused */
}

static int test_all_coco_problem(void) {

  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_coco_evaluate_function),
    cmocka_unit_test(test_coco_evaluate_constraint),
    cmocka_unit_test(test_coco_is_feasible),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
