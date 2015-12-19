#include "coco.h"
#include "suite_bbob.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

static coco_suite_t *suite_biobj_allocate(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  suite = coco_suite_allocate("suite_biobj", 55, 6, dimensions, "instances:1-5");

  return suite;
}

static char *suite_biobj_get_instances_by_year(const int year) {

  if (year == 2016) {
    return "1-5";
  }
  else {
    coco_error("suite_biobj_get_instances_by_year(): year %d not defined for suite_biobj", year);
    return NULL;
  }
}

static coco_problem_t *suite_biobj_get_problem(coco_suite_t *suite,
                                               const size_t function_idx,
                                               const size_t dimension_idx,
                                               const size_t instance_idx) {

  const size_t num_bbob_functions = 10;
  const size_t bbob_functions[] = { 1, 2, 6, 8, 13, 14, 15, 17, 20, 21 };
  const size_t num_instances = 5;
  const size_t instance_mapping[5][2] = { { 2, 4 }, { 3, 5 }, { 7, 8 }, { 9, 10 }, { 11, 12 } };

  coco_problem_t *problem1, *problem2, *problem;
  size_t function1_idx, function2_idx;
  size_t instance1, instance2;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  if (instance_idx > num_instances) {
    coco_error("suite_biobj_get_problem(): the number of instances is limited to %lu", num_instances);
    return NULL; /* Never reached */
  }

  instance1 = instance_mapping[instance_idx][0];
  instance2 = instance_mapping[instance_idx][1];

  /* A "magic" formula to compute the BBOB function index from the bi-objective function index */
  function1_idx = num_bbob_functions -
      (size_t) (-0.5 + sqrt(0.25 + 2.0 * (double) (suite->number_of_functions - function_idx - 1))) - 1;
  function2_idx = function_idx - (function1_idx * num_bbob_functions) +
      (function1_idx * (function1_idx + 1)) / 2;

  problem1 = get_bbob_problem(bbob_functions[function1_idx], dimension, instance1);
  problem2 = get_bbob_problem(bbob_functions[function2_idx], dimension, instance2);

  problem = coco_stacked_problem_allocate(problem1, problem2);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  /* Use the standard stacked problem_id as problem_name and construct a new suite-specific problem_id */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "biobj_f%03d_i%02ld_d%02d", function, instance, dimension);

  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem1->problem_type, problem2->problem_type);

  return problem;
}

