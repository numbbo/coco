#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

/**
 * Tests the function mo_get_norm.
 */
static void test_mo_get_norm(void **state) {

  double norm = 0;

  double first[40] = { 0.51, 0.51, 0.53, 0.54, 0.63, 0.83, 0.25, 0.05, 0.60, 0.30, 0.01, 0.97, 0.55, 0.39,
      0.85, 0.49, 0.86, 0.63, 0.85, 0.63, 0.73, 0.49, 0.09, 0.40, 0.66, 0.45, 0.99, 0.83, 0.92, 0.42, 0.29,
      0.18, 0.75, 0.81, 0.57, 0.11, 0.89, 0.61, 0.03, 0.40 };
  double second[40] = { 0.46, 0.11, 0.47, 0.51, 0.05, 0.18, 0.41, 0.03, 0.62, 0.54, 0.30, 0.21, 0.13, 0.47,
      0.23, 0.39, 0.93, 0.52, 0.21, 0.38, 0.14, 0.54, 0.67, 0.02, 0.73, 0.89, 0.32, 0.77, 0.99, 0.76, 0.18,
      0.53, 0.84, 0.94, 0.78, 0.38, 0.78, 0.58, 0.27, 0.57 };

  norm = mo_get_norm(first, second, 1);
  assert(norm >= 0.04999);  assert(norm <= 0.05001);

  norm = mo_get_norm(first, second, 2);
  assert(norm >= 0.40310);  assert(norm <= 0.40312);

  norm = mo_get_norm(first, second, 3);
  assert(norm >= 0.40754);  assert(norm <= 0.40756);

  norm = mo_get_norm(first, second, 4);
  assert(norm >= 0.40865);  assert(norm <= 0.40867);

  norm = mo_get_norm(first, second, 5);
  assert(norm >= 0.70950);  assert(norm <= 0.70952);

  norm = mo_get_norm(first, second, 10);
  assert(norm >= 1.00493);  assert(norm <= 1.00495);

  norm = mo_get_norm(first, second, 20);
  assert(norm >= 1.65465);  assert(norm <= 1.65467);

  norm = mo_get_norm(first, second, 40);
  assert(norm >= 2.17183);  assert(norm <= 2.17185);

  (void)state; /* unused */
}

/**
 * Tests the function mo_normalize.
 */
static void test_mo_normalize(void **state) {

  double *y = coco_allocate_vector(2);
  double *ideal = coco_allocate_vector(2);
  double *nadir = coco_allocate_vector(2);
  double *result;

  ideal[0] = 100;
  ideal[1] = 0.2;
  nadir[0] = 200;
  nadir[1] = 0.22;

  result = mo_normalize(ideal, ideal, nadir, 2);
  /* Note that the ideal point gets adjusted to be equal to an extreme point! */
  assert(about_equal_2d(result, 0, 1));
  coco_free_memory(result);

  result = mo_normalize(nadir, ideal, nadir, 2);
  assert(about_equal_2d(result, 1, 1));
  coco_free_memory(result);

  y[0] = 50;
  y[1] = 0.1;
  result = mo_normalize(y, ideal, nadir, 2);
  /* Note that a point better than the ideal point gets adjusted to be equal to an extreme point! */
  assert(about_equal_2d(result, 0, 1));
  coco_free_memory(result);

  y[0] = 180;
  y[1] = 0.21;
  result = mo_normalize(y, ideal, nadir, 2);
  assert(about_equal_2d(result, 0.8, 0.5));
  coco_free_memory(result);

  coco_free_memory(y);
  coco_free_memory(ideal);
  coco_free_memory(nadir);

  (void)state; /* unused */
}

/**
 * Tests the function mo_get_dominance.
 */
static void test_mo_get_dominance(void **state) {

  double *a = coco_allocate_vector(2);
  double *b = coco_allocate_vector(2);
  double *c = coco_allocate_vector(2);
  double *d = coco_allocate_vector(2);

  a[0] = 0.8;  a[1] = 0.2;
  b[0] = 0.5;  b[1] = 0.3;
  c[0] = 0.6;  c[1] = 0.4;
  d[0] = 0.6;  d[1] = 0.4;

  assert(mo_get_dominance(a, b, 2) == 0);
  assert(mo_get_dominance(c, b, 2) == -1);
  assert(mo_get_dominance(b, d, 2) == 1);
  assert(mo_get_dominance(c, d, 2) == -2);
  assert(mo_get_dominance(a, a, 2) == -2);

  coco_free_memory(a);
  coco_free_memory(b);
  coco_free_memory(c);
  coco_free_memory(d);

  (void)state; /* unused */
}

/**
 * Tests the function mo_is_within_ROI.
 */
static void test_mo_is_within_ROI(void **state) {

  double *y = coco_allocate_vector(2);

  y[0] = 0.5; y[1] = 0.2;
  assert(mo_is_within_ROI(y, 2) == 1);

  y[0] = 0; y[1] = 0;
  assert(mo_is_within_ROI(y, 2) == 1);

  y[0] = 1; y[1] = 1;
  assert(mo_is_within_ROI(y, 2) == 1);

  y[0] = -0.00001; y[1] = 1;
  assert(mo_is_within_ROI(y, 2) == 0);

  y[0] = 1.2; y[1] = 0.5;
  assert(mo_is_within_ROI(y, 2) == 0);

  coco_free_memory(y);

  (void)state; /* unused */
}

/**
 * Tests the function mo_get_distance_to_ROI.
 */
static void test_mo_get_distance_to_ROI(void **state) {

  double *y = coco_allocate_vector(2);
  double result;

  y[0] = 0.5; y[1] = 0.2;
  assert(about_equal_value(mo_get_distance_to_ROI(y, 2), 0));

  y[0] = 0; y[1] = 0;
  assert(about_equal_value(mo_get_distance_to_ROI(y, 2), 0));

  y[0] = 1; y[1] = 1;
  assert(about_equal_value(mo_get_distance_to_ROI(y, 2), 0));

  y[0] = 1.00001; y[1] = 1;
  result = mo_get_distance_to_ROI(y, 2);
  assert(about_equal_value(result, 0.00001));

  y[0] = 1.2; y[1] = 1.5;
  result = mo_get_distance_to_ROI(y, 2);
  assert(about_equal_value(mo_get_distance_to_ROI(y, 2), 0.53851648071345037));

  coco_free_memory(y);


  (void)state; /* unused */
}

static int test_all_mo_utilities(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_mo_get_norm),
      cmocka_unit_test(test_mo_normalize),
      cmocka_unit_test(test_mo_get_dominance),
      cmocka_unit_test(test_mo_is_within_ROI),
      cmocka_unit_test(test_mo_get_distance_to_ROI)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
