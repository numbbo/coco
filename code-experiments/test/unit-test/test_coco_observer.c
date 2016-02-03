#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

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

  size_t evals[89] = { 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 17, 19, 20, 22, 25, 28, 31, 35, 39, 40, 44,
      50, 56, 63, 70, 79, 89, 100, 112, 125, 141, 158, 177, 199, 200, 223, 251, 281, 316, 354, 398, 400, 446,
      501, 562, 630, 707, 794, 891, 1000, 1122, 1258, 1412, 1584, 1778, 1995, 2000, 2238, 2511, 2818, 3162,
      3548, 3981, 4000, 4466, 5011, 5623, 6309, 7079, 7943, 8912, 10000, 11220, 12589, 14125, 15848, 17782,
      19952, 20000, 22387, 25118, 28183, 31622, 35481, 39810, 40000 };

  size_t i, j;
  int update, found;

  coco_observer_evaluations_t *evaluations;
  evaluations = coco_observer_evaluations("1,2,5", 2);

  for (i = 0; i <= 40000; i++) {
    update = coco_observer_evaluations_trigger(evaluations, i);
    found = 0;
    for (j = 0; j < 89; j++) {
      if (i == evals[j]) {
        found = 1;
        break;
      }
    }
    if (update != found) {
      coco_warning("test_coco_observer_evaluations_trigger(): Assert fails for evaluation number = %lu", i);
    }
    assert(update == found);
  }

  coco_observer_evaluations_free(evaluations);

  (void)state; /* unused */
}

static int test_all_coco_observer(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_observer_targets_trigger),
      cmocka_unit_test(test_coco_observer_evaluations_trigger)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
