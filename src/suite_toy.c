#include "coco_generics.c"
#include "f_1u_bueche_rastrigin.c"
#include "f_1u_ellipsoid.c"
#include "f_1u_linear_slope.c"
#include "f_1u_rastrigin.c"
#include "f_1u_rosenbrock.c"
#include "f_1u_sphere.c"

#include "logger_target_hits.c"

/**
 * suite_toy(function_index):
 *
 * Return the ${function_index}-th benchmark problem in the toy
 * benchmark suite. If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *suite_toy(const long function_index) {
  static const size_t dims[] = { 2, 3, 5, 10, 20 };
  const long fid = function_index % 6;
  const long did = function_index / 6;
  coco_problem_t *problem;
  if (did >= 1)
    return NULL;

  if (fid == 0) {
    problem = f_1u_sphere(dims[did]);
  } else if (fid == 1) {
    problem = f_1u_ellipsoid(dims[did]);
  } else if (fid == 2) {
    problem = f_1u_rastrigin(dims[did]);
  } else if (fid == 3) {
    problem = f_1u_bueche_rastrigin(dims[did]);
  } else if (fid == 4) {
    double xopt[20] = { 5.0 };
    problem = f_1u_linear_slope(dims[did], xopt);
  } else if (fid == 5) {
    problem = f_1u_rosenbrock(dims[did]);
  } else {
    return NULL;
  }
  return problem;
}
