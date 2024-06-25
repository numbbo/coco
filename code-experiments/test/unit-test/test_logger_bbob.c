#include "minunit.h"

#include "coco.h"
#include "about_equal.h"

/**
 * Tests the logarithmic and linear triggers via the bbob logger.
 */
MU_TEST(test_logger_bbob_triggers) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  double *x = coco_allocate_vector(2);
  double *y = coco_allocate_vector(1);
  double target;

  logger_bbob_data_t *logger;

  /* Using only the logarithmic performance targets */
  suite = coco_suite("bbob", "", "dimensions: 2 function_indices: 1 instance_indices: 1");
  observer = coco_observer("bbob", "number_target_triggers: 1");
  /* Use the 2-D sphere function */
  problem = coco_suite_get_next_problem(suite, observer);
  logger = (logger_bbob_data_t *) coco_problem_transformed_get_data(problem);

  x[0] = 0; x[1] = 0;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 80.88209408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 1.40209408));
  target = coco_observer_targets_get_last_target((coco_observer_targets_t *)logger->targets);
  mu_check(about_equal_value(target, 10));

  x[0] = 0; x[1] = -0.5;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.97529408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.49529408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 1));

  x[0] = 0; x[1] = -1;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.56849408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.08849408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 0.1));

  x[0] = 0; x[1] = -1.1;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.54713408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.06713408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 0.1));

  x[0] = 0.2; x[1] = -1.1;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.48601408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.00601408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 0.01));

  coco_observer_free(observer);
  coco_suite_free(suite);

  /* Using the linear performance targets */
  suite = coco_suite("bbob", "", "dimensions: 2 function_indices: 1 instance_indices: 1");
  suite->known_optima = 0;
  observer = coco_observer("bbob", "number_target_triggers: 1 lin_target_precision: 1e-3 log_target_precision: 1e-5");
  /* Use the 2-D sphere function */
  problem = coco_suite_get_next_problem(suite, observer);
  logger = (logger_bbob_data_t *) coco_problem_transformed_get_data(problem);

  x[0] = 0; x[1] = 0;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 80.88209408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 1.40209408));
  target = coco_observer_targets_get_last_target((coco_observer_targets_t *)logger->targets);
  mu_check(about_equal_value(target, 1.403));

  x[0] = 0; x[1] = -0.5;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.97529408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.49529408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 0.496));

  x[0] = 0; x[1] = -1;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.56849408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.08849408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 0.089));

  x[0] = 0; x[1] = -1.1;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.54713408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.06713408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 0.068));

  x[0] = 0.2; x[1] = -1.1;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.48601408));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 0.00601408));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 0.007));

  /* Testing the targets that fall below log_target_precision */

  x[0] = 0.252; x[1] = -1.156;
  coco_evaluate_function(problem, x, y);
  mu_check(about_equal_value(y[0], 79.48000128));
  coco_observer_targets_trigger(logger->targets, logger->best_found_value - logger->optimal_value);
  mu_check(about_equal_value(logger->best_found_value - logger->optimal_value, 1.28 * 1e-6));
  target = coco_observer_targets_get_last_target(logger->targets);
  mu_check(about_equal_value(target, 1e-5));

  coco_observer_free(observer);
  coco_suite_free(suite);

  coco_free_memory(x);
  coco_free_memory(y);
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_logger_bbob) {
  MU_RUN_TEST(test_logger_bbob_triggers);
}
