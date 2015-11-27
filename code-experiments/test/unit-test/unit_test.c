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

#include "test_coco_suites.c"
#include "test_coco_utilities.c"

static int run_all_tests(void)
{
  int result = test_all_coco_utilities();
  /* result += test_all_coco_suites(); */

  return result;
}

int main(void) {
  return run_all_tests();
}

