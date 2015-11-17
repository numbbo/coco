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

static void test(void **state)
{
	double firstValue = 5.0;
	double secondValue = 6.0;

	(void)state; /* unused */

	int fd;

	double maxValue = coco_max_double(firstValue, secondValue);
	assert_true(maxValue == secondValue);
	assert_return_code(fd, errno);
}

int main(void) {

    const struct CMUnitTest tests[] =
    {
		cmocka_unit_test(test),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}

