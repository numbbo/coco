#include "numbbo_generics.c"

#include "log_hitting_times.c"

#include "f_sphere.c"
#include "f_ellipsoid.c"
#include "f_rastrigin.c"

/**
 * toy_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem in the toy
 * benchmark suit. If the function index is out of bounds, return
 * NULL.
 */
numbbo_problem_t *toy_suit(const int function_index) {
    static const int dims[] = {2, 3, 5, 10, 20};
    const int fid = function_index % 3;
    const int did = function_index / 3;
    numbbo_problem_t *problem;
    if (did >= 5)
        return NULL;

    if (fid == 0) {
        problem = sphere_problem(dims[did]);
    } else if (fid == 1) {
        problem = ellipsoid_problem(dims[did]);
    } else if (fid == 2) {
        problem = rastrigin_problem(dims[did]);
    } else {
        return NULL;
    }
    return problem;
}
