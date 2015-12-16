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

  suite = coco_suite_allocate("suite_toy", 6, 5, dimensions, "1");

  return suite;
}

static coco_problem_t *suite_toy_get_problem(size_t function_id, size_t dimension, size_t instance_id) {

  if (function_id + dimension + instance_id > 0)
    return NULL;

  return NULL;
}

/**
 * Initializes the toy suite composed from 6 functions.
 * Returns the problem corresponding to the given function_index.
 */
static coco_problem_t *suite_toy(const long function_index) {
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
  problem->suite_dep_function_id = fid;
  problem->suite_dep_instance_id = 0;
  return problem;
}
