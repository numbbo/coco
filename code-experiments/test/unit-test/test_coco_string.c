#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

/**
 * Tests coco_archive-related functions.
 */
static void test_coco_string_trim(void **state) {

	char *sample_strings[] = {
			"nothing to trim",
			"    trim the front",
			"trim the back     ",
			" trim one char front and back ",
			" trim one char front",
			"trim one char back ",
			"                   ",
			" ",
			"a",
			"",
			NULL };
	char *result_strings[] = {
			"nothing to trim",
			"trim the front",
			"trim the back",
			"trim one char front and back",
			"trim one char front",
			"trim one char back",
			"",
			"",
			"a",
			"",
			NULL };
	int index;
	char str[64];

	for (index = 0; sample_strings[index] != NULL; ++index) {
		strcpy(str, sample_strings[index] );
		assert(strcmp(coco_string_trim(str), result_strings[index]) == 0);
	}

  (void)state; /* unused */
}

static int test_all_coco_string(void) {

  const struct CMUnitTest tests[] = {
  cmocka_unit_test(test_coco_string_trim) };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
