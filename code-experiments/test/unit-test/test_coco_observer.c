#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

/**
 * Tests the function coco_observer_evaluation_to_log.
 */
static void test_coco_observer_evaluation_to_log(void **state) {

  size_t dimensions[6] = { 2, 3, 5, 10, 20, 40 };
  size_t evals, i, dim;
  int result;

  for (i = 0; i < 6; i++) {
    dim = dimensions[i];
    for (evals = 1; evals < 1500; evals++) {
      result = deprecated__coco_observer_evaluation_to_log(evals, dim);
      if ((evals == 1) || (evals == dim) || (evals == 2 * dim) || (evals == 5 * dim))
        assert(result);
      else if ((evals == 10 * dim) || (evals == 20 * dim) || (evals == 50 * dim))
        assert(result);
      else if ((evals == 100 * dim) || (evals == 200 * dim) || (evals == 500 * dim))
        assert(result);
      else
        assert(!result);
    }
  }

  (void)state; /* unused */
}

/**
 * Tests the function coco_observer_targets_trigger.
 */
static void test_coco_observer_targets_trigger(void **state) {

  coco_observer_targets_t *targets = coco_observer_targets(10, 1e-8);
  int update;

  update = coco_observer_targets_trigger(targets, 12);
  assert(update);
  assert(targets->value >= 12);

  update = coco_observer_targets_trigger(targets, 10);
  assert(update);
  assert(targets->value >= 10);

  update = coco_observer_targets_trigger(targets, 2);
  assert(update);
  assert(targets->value >= 2);

  update = coco_observer_targets_trigger(targets, 1.2);
  assert(update);
  assert(targets->value >= 1.2);

  update = coco_observer_targets_trigger(targets, 0.12);
  assert(update);
  assert(targets->value >= 0.12);

  update = coco_observer_targets_trigger(targets, 10);
  assert(!update);

  update = coco_observer_targets_trigger(targets, 2);
  assert(!update);

  update = coco_observer_targets_trigger(targets, 0.000012);
  assert(update);
  assert(targets->value >= 0.000012);

  update = coco_observer_targets_trigger(targets, 12);
  assert(!update);

  update = coco_observer_targets_trigger(targets, 1e-8);
  assert(update);
  assert(targets->value >= 1e-8);

  update = coco_observer_targets_trigger(targets, 1e-9);
  assert(!update);

  update = coco_observer_targets_trigger(targets, -1.2e-8);
  assert(update);
  assert(targets->value >= -1.2e-8);

  update = coco_observer_targets_trigger(targets, -1.2e-7);
  assert(update);
  assert(targets->value >= -1.2e-7);

  update = coco_observer_targets_trigger(targets, 2);
  assert(!update);

  update = coco_observer_targets_trigger(targets, -1200);
  assert(update);
  assert(targets->value >= -1200);

  coco_free_memory(targets);

  targets = coco_observer_targets(10, 1e-8);
  update = coco_observer_targets_trigger(targets, 1e-9);
  assert(update);
  update = coco_observer_targets_trigger(targets, -1.2e-8);
  assert(update);
  update = coco_observer_targets_trigger(targets, -1.2e-7);
  assert(update);
  coco_free_memory(targets);

  targets = coco_observer_targets(10, 1e-8);
  update = coco_observer_targets_trigger(targets, -1.2e-7);
  assert(update);
  coco_free_memory(targets);

  (void)state; /* unused */
}

/**
 * Tests the function coco_observer_evaluations_trigger.
 */
static void test_coco_observer_evaluations_trigger(void **state) {

  size_t dimensions[6] = { 2, 3, 5, 10, 20, 40 };
  coco_observer_evaluations_t *evaluations;
  size_t evals, i, dim;
  int update;

  for (i = 0; i < 6; i++) {
    dim = dimensions[i];
    evaluations = coco_observer_evaluations("1,2,5", dim);
    for (evals = 1; evals < 1500; evals++) {
      update = coco_observer_evaluations_trigger(evaluations, evals);
      if ((evals == 1) || (evals == dim) || (evals == 2 * dim) || (evals == 5 * dim))
        assert(update);
      else if ((evals == 10 * dim) || (evals == 20 * dim) || (evals == 50 * dim))
        assert(update);
      else if ((evals == 100 * dim) || (evals == 200 * dim) || (evals == 500 * dim))
        assert(update);
      else
        assert(!update);
    }
    coco_observer_evaluations_free(evaluations);
  }

  (void)state; /* unused */
}

static int test_all_coco_observer(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_observer_evaluation_to_log),
      cmocka_unit_test(test_coco_observer_targets_trigger),
      cmocka_unit_test(test_coco_observer_evaluations_trigger)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
