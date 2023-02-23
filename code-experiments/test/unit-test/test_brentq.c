#include "minunit.h"

#include "coco.c"

/**
 * Tests the function coco_observer_targets_trigger.
 */
double my_sqrt(double x, COCO_UNUSED void *other) {
  return sqrt(x);
}

MU_TEST(test_brentq) {
  double y;

  y = brentq((callback_type) &my_sqrt, 4, 1, 4, 1E-14, 1E-10, 200, NULL);
  mu_assert_double_eq(2.0, y);
}

MU_TEST(test_brentinv) {
  double y;

  y = brentinv((callback_type) &my_sqrt, 4, NULL);
  mu_assert_double_eq(16.0, y);
}

/**
 * Run all tests in this file.
 */
int main(void) {
  MU_RUN_TEST(test_brentq);
  MU_RUN_TEST(test_brentinv);

  MU_REPORT();

  return minunit_status;
}
