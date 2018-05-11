#include "coco.c"
#include "coco_minunit.h"
#include "unit_test_utilities.c"
#include "test_coco_archive.c"
#include "test_coco_observer.c"
#include "test_coco_problem.c"
#include "test_coco_string.c"
#include "test_coco_suite.c"
#include "test_coco_utilities.c"
#include "test_logger_biobj.c"
#include "test_mo_utilities.c"
#include "unit_test_fail.c"

int main(void) {

	MU_RUN_SUITE(test_all_coco_archive);
	MU_RUN_SUITE(test_all_coco_observer);
	MU_RUN_SUITE(test_all_coco_problem);
	MU_RUN_SUITE(test_all_coco_string);
	MU_RUN_SUITE(test_all_coco_suite);
	MU_RUN_SUITE(test_all_coco_utilities);
	MU_RUN_SUITE(test_all_logger_biobj);
	MU_RUN_SUITE(test_all_mo_utilities);

	/* Run this if you want to see some tests fail
	MU_RUN_SUITE(test_suite_fail); */

	MU_REPORT();

  coco_remove_directory("exdata");

  return minunit_status;
}
