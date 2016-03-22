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

static int test_all_mo_generics(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_mo_get_norm)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
