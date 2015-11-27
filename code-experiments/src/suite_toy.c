#include "coco_generics.c"
#include "f_bueche_rastrigin.c"
#include "f_ellipsoid.c"
#include "f_linear_slope.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_sphere.c"

/**
 * Initializes the toy suite composed from 6 functions.
 * Returns the problem corresponding to the given function_index.
 */
static coco_problem_t *suite_toy(const long function_index) {
  static const size_t dims[] = { 2, 3, 5, 10, 20 };
  const long fid = function_index % 6;
  const long did = function_index / 6;
  coco_problem_t *problem;
  if (did >= 1)
    return NULL;

  if (fid == 0) {
    problem = f_sphere(dims[did]);
  } else if (fid == 1) {
    problem = f_ellipsoid(dims[did]);
  } else if (fid == 2) {
    problem = f_rastrigin(dims[did]);
  } else if (fid == 3) {
    problem = f_bueche_rastrigin(dims[did]);
  } else if (fid == 4) {
    double xopt[20] = { 5.0 };
    problem = f_linear_slope(dims[did], xopt);
  } else if (fid == 5) {
    problem = f_rosenbrock(dims[did]);
  } else {
    return NULL;
  }
  problem->suite_dep_index = fid;
  problem->suite_dep_function_id = (int) fid;
  problem->suite_dep_instance_id = 0;
  return problem;
}
