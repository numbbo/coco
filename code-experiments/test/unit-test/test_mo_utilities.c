#include "minunit.h"

#include "coco.c"
#include "about_equal.h"

/**
 * Tests the function mo_get_norm.
 */
MU_TEST(test_mo_get_norm) {

  double norm = 0;

  double first[40] = { 0.51, 0.51, 0.53, 0.54, 0.63, 0.83, 0.25, 0.05, 0.60, 0.30, 0.01, 0.97, 0.55, 0.39,
      0.85, 0.49, 0.86, 0.63, 0.85, 0.63, 0.73, 0.49, 0.09, 0.40, 0.66, 0.45, 0.99, 0.83, 0.92, 0.42, 0.29,
      0.18, 0.75, 0.81, 0.57, 0.11, 0.89, 0.61, 0.03, 0.40 };
  double second[40] = { 0.46, 0.11, 0.47, 0.51, 0.05, 0.18, 0.41, 0.03, 0.62, 0.54, 0.30, 0.21, 0.13, 0.47,
      0.23, 0.39, 0.93, 0.52, 0.21, 0.38, 0.14, 0.54, 0.67, 0.02, 0.73, 0.89, 0.32, 0.77, 0.99, 0.76, 0.18,
      0.53, 0.84, 0.94, 0.78, 0.38, 0.78, 0.58, 0.27, 0.57 };

  norm = mo_get_norm(first, second, 1);
  mu_check(norm >= 0.04999);  mu_check(norm <= 0.05001);

  norm = mo_get_norm(first, second, 2);
  mu_check(norm >= 0.40310);  mu_check(norm <= 0.40312);

  norm = mo_get_norm(first, second, 3);
  mu_check(norm >= 0.40754);  mu_check(norm <= 0.40756);

  norm = mo_get_norm(first, second, 4);
  mu_check(norm >= 0.40865);  mu_check(norm <= 0.40867);

  norm = mo_get_norm(first, second, 5);
  mu_check(norm >= 0.70950);  mu_check(norm <= 0.70952);

  norm = mo_get_norm(first, second, 10);
  mu_check(norm >= 1.00493);  mu_check(norm <= 1.00495);

  norm = mo_get_norm(first, second, 20);
  mu_check(norm >= 1.65465);  mu_check(norm <= 1.65467);

  norm = mo_get_norm(first, second, 40);
  mu_check(norm >= 2.17183);  mu_check(norm <= 2.17185);
}

/**
 * Tests the function mo_normalize.
 */
MU_TEST(test_mo_normalize) {

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
  mu_check(about_equal_2d(result, 0, 1));
  coco_free_memory(result);

  result = mo_normalize(nadir, ideal, nadir, 2);
  mu_check(about_equal_2d(result, 1, 1));
  coco_free_memory(result);

  y[0] = 50;
  y[1] = 0.1;
  result = mo_normalize(y, ideal, nadir, 2);
  /* Note that a point better than the ideal point gets adjusted to be equal to an extreme point! */
  mu_check(about_equal_2d(result, 0, 1));
  coco_free_memory(result);

  y[0] = 180;
  y[1] = 0.21;
  result = mo_normalize(y, ideal, nadir, 2);
  mu_check(about_equal_2d(result, 0.8, 0.5));
  coco_free_memory(result);

  coco_free_memory(y);
  coco_free_memory(ideal);
  coco_free_memory(nadir);
}

/**
 * Tests the function mo_get_dominance.
 */
MU_TEST(test_mo_get_dominance) {

  double *a = coco_allocate_vector(2);
  double *b = coco_allocate_vector(2);
  double *c = coco_allocate_vector(2);
  double *d = coco_allocate_vector(2);

  a[0] = 0.8;  a[1] = 0.2;
  b[0] = 0.5;  b[1] = 0.3;
  c[0] = 0.6;  c[1] = 0.4;
  d[0] = 0.6;  d[1] = 0.4;

  mu_check(mo_get_dominance(a, b, 2) == 0);
  mu_check(mo_get_dominance(c, b, 2) == -1);
  mu_check(mo_get_dominance(b, d, 2) == 1);
  mu_check(mo_get_dominance(c, d, 2) == -2);
  mu_check(mo_get_dominance(a, a, 2) == -2);

  coco_free_memory(a);
  coco_free_memory(b);
  coco_free_memory(c);
  coco_free_memory(d);
}

/**
 * Tests the function mo_is_within_ROI.
 */
MU_TEST(test_mo_is_within_ROI) {

  double *y = coco_allocate_vector(2);

  y[0] = 0.5; y[1] = 0.2;
  mu_check(mo_is_within_ROI(y, 2) == 1);

  y[0] = 0; y[1] = 0;
  mu_check(mo_is_within_ROI(y, 2) == 1);

  y[0] = 1; y[1] = 1;
  mu_check(mo_is_within_ROI(y, 2) == 1);

  y[0] = -0.00001; y[1] = 1;
  mu_check(mo_is_within_ROI(y, 2) == 0);

  y[0] = 1.2; y[1] = 0.5;
  mu_check(mo_is_within_ROI(y, 2) == 0);

  coco_free_memory(y);
}

/**
 * Tests the function mo_get_distance_to_ROI.
 */
MU_TEST(test_mo_get_distance_to_ROI) {

  double *y = coco_allocate_vector(2);
  double result;

  y[0] = 0.5; y[1] = 0.2;
  mu_check(about_equal_value(mo_get_distance_to_ROI(y, 2), 0));

  y[0] = 0; y[1] = 0;
  mu_check(about_equal_value(mo_get_distance_to_ROI(y, 2), 0));

  y[0] = 1; y[1] = 1;
  mu_check(about_equal_value(mo_get_distance_to_ROI(y, 2), 0));

  y[0] = 1.00001; y[1] = 1;
  result = mo_get_distance_to_ROI(y, 2);
  mu_check(about_equal_value(result, 0.00001));

  y[0] = 1.2; y[1] = 1.5;
  result = mo_get_distance_to_ROI(y, 2);
  mu_check(about_equal_value(mo_get_distance_to_ROI(y, 2), 0.53851648071345037));

  coco_free_memory(y);
}

/**
 * Run all tests in this file.
 */
int main(void) {
  MU_RUN_TEST(test_mo_get_norm);
  MU_RUN_TEST(test_mo_normalize);
  MU_RUN_TEST(test_mo_get_dominance);
  MU_RUN_TEST(test_mo_is_within_ROI);
  MU_RUN_TEST(test_mo_get_distance_to_ROI);

  MU_REPORT();

  int minunit_status;
}
