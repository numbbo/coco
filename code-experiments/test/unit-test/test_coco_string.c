#include "coco.h"
#include "minunit_c89.h"

/**
 * Tests coco_string-related functions.
 */
MU_TEST(test_coco_string_trim) {

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

MU_TEST(test_coco_string_replace) {
	char *result_string;

	result_string = coco_string_replace("replace <A> <B> <A> <AB> <A", "<A>", "42");
	mu_check(strcmp(result_string, "replace 42 <B> 42 <AB> <A") == 0);
	coco_free_memory(result_string);

	result_string = coco_string_replace("replace <A> <B> <A> <AB> <A", "<A>>", "42");
	mu_check(strcmp(result_string, "replace <A> <B> <A> <AB> <A") == 0);
	coco_free_memory(result_string);

	result_string = coco_string_replace("replace <A> <B> <A> <AB> <A", "<A>", NULL);
	mu_check(strcmp(result_string, "replace  <B>  <AB> <A") == 0);
	coco_free_memory(result_string);

	result_string = coco_string_replace("replace <A> <B> <A> <AB> <A", NULL, "42");
	mu_check(result_string == NULL);
	coco_free_memory(result_string);
}

/**
 * Run all tests in this file.
 */
MU_TEST_SUITE(test_all_coco_string) {
  MU_RUN_TEST(test_coco_string_trim);
  MU_RUN_TEST(test_coco_string_replace);
}
