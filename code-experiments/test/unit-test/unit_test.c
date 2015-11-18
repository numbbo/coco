/*
 ============================================================================
 Name        : test.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Unit tests for numbbo.
 ============================================================================
 */

#include "test_coco_utilities.c"

static int run_all_tests(void)
{
  return test_all_coco_utilities();
}

int main(void) {
  return run_all_tests();
}

