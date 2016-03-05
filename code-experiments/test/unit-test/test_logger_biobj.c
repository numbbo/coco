#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

static int about_equal_vector(const double *a, const double *b, const size_t dimension);
static int about_equal_2d(const double *a, const double b1, const double b2);

/**
 * Tests several things in the computation of the modified hypervolume indicator for a specific
 * biobjective problem (problem data is contained in test_logger_biobj.txt).
 */
static void test_logger_biobj_evaluate(void **state) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  size_t number_of_evaluations;
  double *x = coco_allocate_vector(2);
  double *y = coco_allocate_vector(2);
  double *y_eval = coco_allocate_vector(2);
  double hypervolume;

  logger_biobj_data_t *logger;
  logger_biobj_indicator_t *indicator;

  int scan_return;
  char file_name[] = "test_hypervolume.txt";
  FILE *f_results = fopen(file_name, "r");
  if (f_results == NULL) {
    coco_error("test_logger_biobj_evaluate() failed to open file '%s'.", file_name);
  }

  suite = coco_suite("bbob-biobj", "year: 2016", "dimensions: 2 function_indices: 23 instance_indices: 5");
  observer = coco_observer("bbob-biobj", "");

/* TODO: Enable this test again once you figure out the problem with the mingw compiler */
#if 0

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {

    /* Tests the ideal and nadir points */
    assert(about_equal_2d(problem->best_value, 3.914e+01, -4.328e+01));
    assert(about_equal_2d(problem->nadir_value, 1.080278978306634e+02, -3.795129793854297e+01));

    while (f_results) {

      /* Reads the values from the file */
      scan_return = fscanf(f_results, "%lu\t%lf\t%lf\t%lf\t%lf\t%lf\n", &number_of_evaluations, &x[0], &x[1],
          &y[0], &y[1], &hypervolume);

      if (scan_return != 6)
        break;

      coco_evaluate_function(problem, x, y_eval);

      /* Checks the function values are right */
      assert(about_equal_vector(y, y_eval, 2));

      logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
      indicator = (logger_biobj_indicator_t *) (logger->indicators[0]);

      /* Checks the hypervolume */
      assert(about_equal_value(hypervolume, indicator->current_value));
    }
#endif

  }

  fclose(f_results);

  coco_free_memory(x);
  coco_free_memory(y);
  coco_free_memory(y_eval);

  coco_observer_free(observer);
  coco_suite_free(suite);

  (void)state; /* unused */
}


static int test_all_logger_biobj(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_logger_biobj_evaluate)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
