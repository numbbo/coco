#include "coco.h"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

static coco_suite_t *suite_biobj_allocate(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  suite = coco_suite_allocate("suite_biobj", 24, 6, dimensions, "instances:1-5");

  return suite;
}

static char *suite_biobj_get_instances_by_year(int year) {

  if (year == 2016) {
    return "1-5";
  }
  else {
    coco_error("suite_biobj_get_instances_by_year(): year %d not defined for suite_biobj", year);
    return NULL;
  }
}

static coco_problem_t *suite_biobj_get_problem(size_t function_id, size_t dimension, size_t instance_id) {

  if (function_id + dimension + instance_id > 0)
    return NULL;

  return NULL;
}
