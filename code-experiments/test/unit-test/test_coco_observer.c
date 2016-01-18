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
      result = coco_observer_evaluation_to_log(evals, dim);
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

static int test_all_coco_observer(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_observer_evaluation_to_log)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
