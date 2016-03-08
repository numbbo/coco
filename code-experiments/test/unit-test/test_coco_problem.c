#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

/**
 * Tests that the coco_evaluate_function returns a vector of NANs when given a vector with one or more
 * NAN values.
 */
static void test_coco_evaluate_function(void **state) {

	coco_suite_t *suite = coco_suite("bbob", NULL, NULL);
	coco_problem_t *problem;
	double *x = coco_allocate_vector(2);
	double *y = coco_allocate_vector(2);

  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {

  	if (coco_problem_get_dimension(problem) > 2)
  		continue;

  	x[0] = 0;
  	x[1] = NAN;
    coco_evaluate_function(problem, x, y);

    if (!coco_vector_contains_nan(y, coco_problem_get_number_of_objectives(problem))) {
    	coco_warning("nan = %f, y0 = %f, y1 = %f", NAN, y[0], y[1]);
    }

    assert(coco_vector_contains_nan(y, coco_problem_get_number_of_objectives(problem)));

  }

  coco_suite_free(suite);
  coco_free_memory(x);
  coco_free_memory(y);

  (void)state; /* unused */
}

static int test_all_coco_problem(void) {

  const struct CMUnitTest tests[] = {
  cmocka_unit_test(test_coco_evaluate_function) };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
