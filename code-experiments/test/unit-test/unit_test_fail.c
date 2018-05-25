#include "minunit_c89.h"

static int foo = 5;
static const double dbar = 0.1;

/**
 * A collection of tests that fail for different reasons.
 */

void test_setup(void) {
	foo = 7;
}

void test_teardown(void) {
	/* Nothing */
}

MU_TEST(test_check_fail) {
	mu_check(foo != 7);
}

MU_TEST(test_assert_fail) {
	mu_assert(foo != 7, "foo should be <> 7");
}

MU_TEST(test_assert_int_eq_fail) {
	mu_assert_int_eq(foo, 5);
}

MU_TEST(test_assert_double_eq_fail) {
	mu_assert_double_eq(0.2, dbar);
}

MU_TEST(test_fail) {
	mu_fail("Fail now!");
}

MU_TEST(test_string_eq_fail){
	mu_assert_string_eq("That string", "This string");
}


MU_TEST_SUITE(test_suite_fail) {

	MU_SUITE_CONFIGURE(&test_setup, &test_teardown);

	MU_RUN_TEST(test_check_fail);
	MU_RUN_TEST(test_assert_fail);
	MU_RUN_TEST(test_assert_int_eq_fail);
	MU_RUN_TEST(test_assert_double_eq_fail);
	MU_RUN_TEST(test_string_eq_fail);
	MU_RUN_TEST(test_fail);

}
