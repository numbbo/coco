#include <math.h>

#include "minunit.h"
#include "coco.c"
#include "about_equal.h"

/**
 * Tests several things in the computation of the modified hypervolume indicator for a specific
 * biobjective problem (problem data is contained in test_logger_biobj.txt).
 */
MU_TEST(test_logger_biobj_evaluate) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  unsigned long number_of_evaluations;
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

  suite = coco_suite("bbob-biobj", "instances: 1-10", "dimensions: 2 function_indices: 23 instance_indices: 5");
  observer = coco_observer("bbob-biobj", "");

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {

    /* Tests the ideal and nadir points */
    mu_check(about_equal_2d(problem->best_value, 3.914e+01, -4.328e+01));
    mu_check(about_equal_2d(problem->nadir_value, 1.080278978306634e+02, -3.795129793854297e+01));

    while (f_results) {

      /* Reads the values from the file */
      scan_return = fscanf(f_results, "%lu\t%lf\t%lf\t%lf\t%lf\t%lf\n", &number_of_evaluations, &x[0], &x[1],
          &y[0], &y[1], &hypervolume);

      if (scan_return != 6)
        break;

      coco_evaluate_function(problem, x, y_eval);

      /* Checks the function values are right */
      mu_check(about_equal_vector(y, y_eval, 2));

      logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
      indicator = (logger_biobj_indicator_t *) (logger->indicators[0]);

      /* Checks the hypervolume */
      mu_check(about_equal_value(hypervolume, indicator->current_value));
    }

  }

  fclose(f_results);

  coco_free_memory(x);
  coco_free_memory(y);
  coco_free_memory(y_eval);

  coco_observer_free(observer);
  coco_suite_free(suite);
}


MU_TEST(test_logger_biobj_evaluate2) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  double *x = coco_allocate_vector(2);
  double *y = coco_allocate_vector(2);

  logger_biobj_data_t *logger;
  logger_biobj_indicator_t *indicator;

  suite = coco_suite("bbob-biobj", "instances: 1-10", "dimensions: 2 function_indices: 12 instance_indices: 7");
  observer = coco_observer("bbob-biobj", "");

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {

    logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
    indicator = (logger_biobj_indicator_t *) (logger->indicators[0]);

    /* Tests the ideal and nadir points */
    mu_check(about_equal_2d(problem->best_value, 2.872000000000000e+001, 6.587000000000001e+001));
    mu_check(about_equal_2d(problem->nadir_value, 8.182999486471047e+006, 2.973909225537448e+005));

    indicator->best_value = 9.999761801899431e-01;

    x[0] = 3.70126935e+00;
    x[1] = 4.86336949e+00;
    coco_evaluate_function(problem, x, y);
    mu_check(about_equal_2d(y, 6.526894600733597e+07, 9.773829237996117e+04));

    x[0] = -4.07708054e+00;
    x[1] = -4.23301504e+00;
    coco_evaluate_function(problem, x, y);
    mu_check(about_equal_2d(y, 1.469166639901164e+06, 1.038943690282359e+06));

    x[0] = 3.62985262e+00;
    x[1] = 3.27345862e-01;
    coco_evaluate_function(problem, x, y);
    mu_check(about_equal_2d(y, 1.102461181155135e+07, 1.984282346358726e+03));

    x[0] = -6.83296521e-01;
    x[1] = -2.01717321e+00;
    coco_evaluate_function(problem, x, y);
    mu_check(about_equal_2d(y, 1.269962944284010e+06, 2.877987262850722e+05));

    x[0] = 1.93618303e+00;
    x[1] = -6.32608857e-01;
    coco_evaluate_function(problem, x, y);
    mu_check(about_equal_2d(y, 6.736465672667241e+06, 3.426169774397720e+04));

  }

  coco_free_memory(x);
  coco_free_memory(y);

  coco_observer_free(observer);
  coco_suite_free(suite);
}

/**
 * Tests the coco_logger_biobj_feed_solution function.
 */
MU_TEST(test_coco_logger_biobj_feed_solution) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  size_t number_of_evaluations;
  double *y = coco_allocate_vector(2);

  logger_biobj_data_t *logger;
  logger_biobj_indicator_t *indicator;

  suite = coco_suite("bbob-biobj", "instances: 7", "dimensions: 10 function_indices: 12");
  observer = coco_observer("bbob-biobj", "log_nondominated: read");
  problem = coco_suite_get_next_problem(suite, observer);

  mu_check(problem != NULL);

  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
  indicator = (logger_biobj_indicator_t *) (logger->indicators[0]);

  indicator->best_value = 9.999545292785900e-01;

  number_of_evaluations = 1;
  y[0] = 3.998010498047934e+06;
  y[1] = 3.032832203646995e+05;
  coco_logger_biobj_feed_solution(problem, number_of_evaluations, y);
  mu_check(about_equal_value(indicator->overall_value, 5.627463907182547e-01));

  number_of_evaluations = 3;
  y[0] = 2.455928115758100e+06;
  y[1] = 4.832453316303584e+05;
  coco_logger_biobj_feed_solution(problem, number_of_evaluations, y);
  mu_check(about_equal_value(indicator->overall_value, 5.517002077559761e-01));

  number_of_evaluations = 7;
  y[0] = 1.905500967744320e+06;
  y[1] = 7.225364079041419e+05;
  coco_logger_biobj_feed_solution(problem, number_of_evaluations, y);
  mu_check(about_equal_value(indicator->overall_value, 5.517002077559761e-01));

  number_of_evaluations = 15;
  y[0] = 5.134664589788270e+06;
  y[1] = 2.859068205979772e+05;
  coco_logger_biobj_feed_solution(problem, number_of_evaluations, y);
  mu_check(about_equal_value(indicator->overall_value, 5.282781576255882e-01));

  coco_free_memory(y);
  coco_observer_free(observer);
  coco_suite_free(suite);
}

/**
 * Run all tests in this file.
 */
int main(void) {
  MU_RUN_TEST(test_logger_biobj_evaluate);
  MU_RUN_TEST(test_logger_biobj_evaluate2);
  MU_RUN_TEST(test_coco_logger_biobj_feed_solution);

  MU_REPORT();

  return minunit_status;
}
