#include "coco_generics.c"
#include "f_bueche_rastrigin.c"
#include "f_ellipsoid.c"
#include "f_linear_slope.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_sphere.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

static coco_suite_t *suite_toy_allocate(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20 };

  suite = coco_suite_allocate("toy", 6, 3, dimensions, "instances:1");

  return suite;
}

static coco_problem_t *suite_toy_get_problem(coco_suite_t *suite,
                                             const size_t function_idx,
                                             const size_t dimension_idx,
                                             const size_t instance_idx) {


  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  if (function == 1) {
    problem = f_sphere_allocate(dimension);
  } else if (function == 2) {
    problem = f_ellipsoid_allocate(dimension);
  } else if (function == 3) {
    problem = f_rastrigin_allocate(dimension);
  } else if (function == 4) {
    problem = f_bueche_rastrigin_allocate(dimension);
  } else if (function == 5) {
    double xopt[40] = { 5.0 };
    problem = f_linear_slope_allocate(dimension, xopt);
  } else if (function == 6) {
    problem = f_rosenbrock_allocate(dimension);
  } else {
    coco_error("suite_toy_get_problem(): function %lu does not exist in this suite", function);
    return NULL; /* Never reached */
  }

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}

/**
 * Initializes the toy suite composed from 6 functions.
 * Returns the problem corresponding to the given function_index.
 */
static coco_problem_t *deprecated__suite_toy(const long function_index) {
  static const size_t dims[] = { 2, 3, 5, 10, 20 };
  const size_t fid = (size_t) function_index % 6;
  const size_t did = (size_t) function_index / 6;
  coco_problem_t *problem;
  if (did >= 1)
    return NULL;

  if (fid == 0) {
    problem = f_sphere_allocate(dims[did]);
  } else if (fid == 1) {
    problem = f_ellipsoid_allocate(dims[did]);
  } else if (fid == 2) {
    problem = f_rastrigin_allocate(dims[did]);
  } else if (fid == 3) {
    problem = f_bueche_rastrigin_allocate(dims[did]);
  } else if (fid == 4) {
    double xopt[20] = { 5.0 };
    problem = f_linear_slope_allocate(dims[did], xopt);
  } else if (fid == 5) {
    problem = f_rosenbrock_allocate(dims[did]);
  } else {
    return NULL;
  }
  problem->suite_dep_index = fid;
  problem->suite_dep_function = fid;
  problem->suite_dep_instance = 0;
  return problem;
}
