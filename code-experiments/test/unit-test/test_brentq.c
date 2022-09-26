#include "coco.h"

#include "minunit_c89.h"

/**
 * Tests the function coco_observer_targets_trigger.
 */

MU_TEST(test_brentq) {
    double y;

    y = brentq((callback_type) &sqrt, 4, 1, 4, 1E-14, 1E-10, 200, NULL);
    mu_check(y - 2 < 1E-14);
    mu_check(2 - y < 1E-14);
}

MU_TEST(test_brentinv) {
    double y;

    y = brentinv((callback_type) &sqrt, 4, NULL);
    mu_check(y - 2 < 1E-14);
    mu_check(2 - y < 1E-14);
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_brent) {
  MU_RUN_TEST(test_brentq);
  MU_RUN_TEST(test_brentinv);
}