#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

/**
 * Tests the function coco_set_log_level.
 */
static void test_coco_set_log_level(void **state) {

  char *previous_log_level;

  /* Check whether the default set to COCO_INFO */
  assert(strcmp(coco_set_log_level(""), "info") == 0);

  /* Check whether the method works */
  previous_log_level = coco_strdup(coco_set_log_level("error"));
  assert(strcmp(previous_log_level, "info") == 0);
  assert(strcmp(coco_set_log_level(""), "error") == 0);
  coco_free_memory(previous_log_level);

  previous_log_level = coco_strdup(coco_set_log_level("warning"));
  assert(strcmp(previous_log_level, "error") == 0);
  assert(strcmp(coco_set_log_level(""), "warning") == 0);
  coco_free_memory(previous_log_level);

  previous_log_level = coco_strdup(coco_set_log_level("debug"));
  assert(strcmp(previous_log_level, "warning") == 0);
  assert(strcmp(coco_set_log_level(""), "debug") == 0);
  coco_free_memory(previous_log_level);

  previous_log_level = coco_strdup(coco_set_log_level("info"));
  assert(strcmp(previous_log_level, "debug") == 0);
  assert(strcmp(coco_set_log_level(""), "info") == 0);
  coco_free_memory(previous_log_level);

  /* An invalid argument shouldn't change the current value */
  previous_log_level = coco_strdup(coco_set_log_level("bla"));
  assert(strcmp(previous_log_level, "info") == 0);
  assert(strcmp(coco_set_log_level(""), "info") == 0);
  coco_free_memory(previous_log_level);

  (void)state; /* unused */
}

static int test_all_coco_runtime_c(void) {

  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_coco_set_log_level)
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
