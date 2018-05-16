#include "coco.h"
#include "minunit_c89.h"

/**
 * Tests the function coco_observer_targets_trigger.
 */
MU_TEST(test_coco_observer_targets_trigger) {

  coco_observer_targets_t *targets = coco_observer_targets(10, 1e-8);
  int update;

  update = coco_observer_targets_trigger(targets, 12);
  mu_check(update);
  mu_check(targets->value >= 12);

  update = coco_observer_targets_trigger(targets, 10);
  mu_check(update);
  mu_check(targets->value >= 10);

  update = coco_observer_targets_trigger(targets, 2);
  mu_check(update);
  mu_check(targets->value >= 2);

  update = coco_observer_targets_trigger(targets, 1.2);
  mu_check(update);
  mu_check(targets->value >= 1.2);

  update = coco_observer_targets_trigger(targets, 0.12);
  mu_check(update);
  mu_check(targets->value >= 0.12);

  update = coco_observer_targets_trigger(targets, 10);
  mu_check(!update);

  update = coco_observer_targets_trigger(targets, 2);
  mu_check(!update);

  update = coco_observer_targets_trigger(targets, 0.000012);
  mu_check(update);
  mu_check(targets->value >= 0.000012);

  update = coco_observer_targets_trigger(targets, 12);
  mu_check(!update);

  update = coco_observer_targets_trigger(targets, 1e-8);
  mu_check(update);
  mu_check(targets->value >= 1e-8);

  update = coco_observer_targets_trigger(targets, 1e-9);
  mu_check(!update);

  update = coco_observer_targets_trigger(targets, -1.2e-8);
  mu_check(update);
  mu_check(targets->value >= -1.2e-8);

  update = coco_observer_targets_trigger(targets, -1.2e-7);
  mu_check(update);
  mu_check(targets->value >= -1.2e-7);

  update = coco_observer_targets_trigger(targets, 2);
  mu_check(!update);

  update = coco_observer_targets_trigger(targets, -1200);
  mu_check(update);
  mu_check(targets->value >= -1200);

  coco_free_memory(targets);

  targets = coco_observer_targets(10, 1e-8);
  update = coco_observer_targets_trigger(targets, 1e-9);
  mu_check(update);
  update = coco_observer_targets_trigger(targets, 0);
  mu_check(update);
  update = coco_observer_targets_trigger(targets, -1.2e-8);
  mu_check(update);
  update = coco_observer_targets_trigger(targets, -1.2e-7);
  mu_check(update);
  coco_free_memory(targets);

  targets = coco_observer_targets(10, 1e-8);
  update = coco_observer_targets_trigger(targets, -1.2e-7);
  mu_check(update);
  coco_free_memory(targets);
}

/**
 * Tests the function coco_observer_evaluations_trigger.
 */
MU_TEST(test_coco_observer_evaluations_trigger) {

  size_t evals[53] = { 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 17, 19, 20, 22, 25, 28, 31, 35, 39, 40, 44,
      50, 56, 63, 70, 79, 89, 100, 112, 125, 141, 158, 177, 199, 200, 223, 251, 281, 316, 354, 398, 400, 446,
      501, 562, 630, 707, 794, 891, 1000};

  size_t i, j;
  int update, found;

  coco_observer_evaluations_t *evaluations;
  evaluations = coco_observer_evaluations("1,2,5", 2);

  for (i = 1; i <= 1000; i++) {
    update = coco_observer_evaluations_trigger(evaluations, i);
    found = 0;
    for (j = 0; j < 53; j++) {
      if (i == evals[j]) {
        found = 1;
        break;
      }
    }
    if (update != found) {
      coco_warning("test_coco_observer_evaluations_trigger(): mu_check fails for evaluation number = %lu",
      		(unsigned long) i);
    }
    mu_check(update == found);
  }

  coco_observer_evaluations_free(evaluations);
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_coco_observer) {
  MU_RUN_TEST(test_coco_observer_targets_trigger);
  MU_RUN_TEST(test_coco_observer_evaluations_trigger);
}
