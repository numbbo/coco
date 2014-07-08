#include "numbbo_generics.c"

#include "f_sphere.c"
#include "f_ellipsoid.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_skewRastriginBueche.c"
#include "f_linearSlope.c"

/**
 * bbob2009_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem from the BBOB2009
 * benchmark suit. If the function index is out of bounds, return *
 * NULL.
 */
numbbo_problem_t *bbob2009_suit(const int function_index) {
    static const int dims[] = {2, 3, 5, 10, 20, 40};
    /* Decode the new function_index into the old convention of
     * function, instance and dimension. We have 24 functions in 6
     * different dimensions so a total of 144 functions and any number
     * of instances so we order them so that the function varies
     * faster than the dimension which is still faster than the
     * instance. This gives us:
     *
     * function_index | function_id | instance_id | dimension
     * ---------------+-------------+-------------+-----------
     *              0 |           1 |           1 |         2
     *              1 |           2 |           1 |         2
     *              2 |           3 |           1 |         2
     *                           ...
     *             23 |          24 |           1 |         2
     *             24 |           1 |           1 |         3
     *             25 |           2 |           1 |         3
     *                           ...
     *            143 |          24 |           1 |        40
     *            144 |           1 |           2 |         2
     *            145 |           2 |           2 |         2
     *                           ...
     *           2157 |          22 |           15|        40
     *           2158 |          23 |           15|        40
     *           2159 |          24 |           15|        40
     *
     * The quickest way to decode this is using integer division and
     * remainders. Every block of 144 consecutive function indexes is
     * a single instance. Within that block every consecutive block of
     * 24 function indexes is a fixed dimension and the 24 unique
     * function indexes in that block are the function_id.
     */
    const int instance_id = function_index / 144 + 1;
    int rest = function_index % 144;
    const int dimension = dims[rest / 24];
    const int function_id = rest % 24 + 1;

    if (instance_id > 15) return NULL;

    if (fid == 0) {
        problem = sphere_problem(dimension);
    } else if (fid == 1) {
        problem = ellipsoid_problem(dimension);
    } else if (fid == 2) {
        problem = rastrigin_problem(dimension);
    } else if (fid == 3) {
        problem = skewRastriginBueche_problem(dimension);
    } else if (fid == 4) {
        problem = linearSlope_problem(dimension);
    } else if (fid == 5) {
        problem = rosenbrock_problem(dimension);
    } else {
        return NULL;
    }

    /* FIXME: Apply instance specific transformations! */
    if (instance_id == 1) {
        problem = shift_objective(problem, 1.0);
    } else if (instance_id == 2) {
        problem = shift_objective(problem, 2.0);
    }
    return problem;
}
