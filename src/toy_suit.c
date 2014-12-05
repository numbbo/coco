#include "coco_generics.c"

#include "log_hitting_times.c"

#include "f_sphere.c"
#include "f_ellipsoid.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_bueche-rastrigin.c"
#include "f_linear_slope.c"

/**
 * toy_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem in the toy
 * benchmark suit. If the function index is out of bounds, return
 * NULL.
 */
coco_problem_t *toy_suit(const int function_index) {
  static const int dims[] = {2, 3, 5, 10, 20};
  const int fid = function_index % 6;
  const int did = function_index / 6;
  coco_problem_t *problem;
  if (did >= 1)
    return NULL;

  if (fid == 0) {
    problem = sphere_problem(dims[did]);
  } else if (fid == 1) {
    problem = ellipsoid_problem(dims[did]);
  } else if (fid == 2) {
    problem = rastrigin_problem(dims[did]);
  } else if (fid == 3) {
    problem = bueche_rastrigin_problem(dims[did]);
  } else if (fid == 4) {
    double xopt[20] = {5.0};
    problem = linear_slope_problem(dims[did], xopt);
  } else if (fid == 5) {
    problem = rosenbrock_problem(dims[did]);
  } else {
    return NULL;
  }
  return problem;
}
