#include "minunit.h"
#include "coco.h"

/**
 * Tests coco_archive-related functions.
 */
static void test_coco_string_trim(void) {

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
		mu_check(strcmp(coco_string_trim(str), result_strings[index]) == 0);
	}
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_coco_string) {
  MU_RUN_TEST(test_coco_string_trim);
}
