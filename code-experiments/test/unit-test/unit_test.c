#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.c"
#include "coco.h"
#include <time.h>

#include "unit_test_utilities.c"
#include "test_coco_string.c"
#include "test_coco_archive.c"
#include "test_coco_observer.c"
#include "test_coco_problem.c"
#include "test_coco_suite.c"
#include "test_coco_utilities.c"
#include "test_logger_biobj.c"
#include "test_mo_utilities.c"

static int run_all_tests(void)
{
  int result = 0;
  result += test_all_logger_biobj();
  result += test_all_coco_archive();
  result += test_all_coco_utilities();
  result += test_all_coco_string();
  result += test_all_mo_utilities();
  result += test_all_coco_observer();
  result += test_all_coco_problem();
  result += test_all_coco_suite();

  return result;
}

int main(void) {
  return run_all_tests();
}

