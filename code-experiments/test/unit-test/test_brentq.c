#include "coco.h"
#include "minunit_c89.h"

/**
 * Tests the function coco_observer_targets_trigger.
 */
MU_TEST(test_brentq) {

    //brentq(double sq2(double x) {sqrt(x) - 4}, 1, 4, 1E-14, 1E-10, 1, NULL)

    y = brentq(double sq2(double x) {sqrt(x) - 4}, 1, 4, 1E-14, 1E-10, 200, NULL)
    mu_check(y - 2 < 1E-14)
}

MU_TEST(test_brentinv) {
    mu_check(true)
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_coco_observer) {
  MU_RUN_TEST(test_brentq);
  MU_RUN_TEST(test_brentinv);
}