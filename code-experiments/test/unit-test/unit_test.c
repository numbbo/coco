/*
 ============================================================================
 Name        : test.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Unit tests for numbbo.
 ============================================================================
 */

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.c"
#include "coco.h"
#include <time.h>

#include "test_coco_suite.c"
#include "test_coco_utilities.c"
#include "test_logger_biobj.c"
#include "test_mo_generics.c"

static int run_all_tests(void)
{
  int result = test_all_coco_utilities();
  result += test_all_mo_generics();
  result += test_all_logger_biobj();
  /* result += test_all_coco_suite(); */

  return result;
}

int main(void) {
  return run_all_tests();
}

