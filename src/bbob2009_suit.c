#include <assert.h>

#include "coco_generics.c"

#include "bbob2009_legacy_code.c"

#include "f_sphere.c"
#include "f_ellipsoid.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_bueche-rastrigin.c"
#include "f_linear_slope.c"

#include "shift_objective.c"

#include "asymmetric_variable_transform.c"
#include "brs_transform.c"
#include "condition_variables.c"
#include "oscillate_variables.c"
#include "shift_variables.c"

/**
 * bbob2009_decode_function_index(function_index, function_id, instance_id, dimension):
 *
 * Decode the new function_index into the old convention of function,
 * instance and dimension. We have 24 functions in 6 different
 * dimensions so a total of 144 functions and any number of
 * instances. A natural thing would be to order them so that the
 * function varies faster than the dimension which is still faster
 * than the instance. For analysis reasons we want something
 * different. Our goal is to quickly produce 5 repetitions of a single
 * function in one dimension, then vary the function, then the
 * dimension.
 *
 * This gives us:
 *
 * function_index | function_id | instance_id | dimension
 * ---------------+-------------+-------------+-----------
 *              0 |           1 |           1 |         2
 *              1 |           1 |           2 |         2
 *              2 |           1 |           3 |         2
 *              3 |           1 |           4 |         2
 *              4 |           1 |           5 |         2
 *              5 |           2 |           1 |         2
 *              6 |           2 |           2 |         2
 *             ...           ...           ...         ...
 *            119 |          24 |           5 |         2
 *            120 |           1 |           1 |         3
 *            121 |           1 |           2 |         3
 *             ...           ...           ...         ...
 *           2157 |          24 |           13|        40
 *           2158 |          24 |           14|        40
 *           2159 |          24 |           15|        40
 *
 * The quickest way to decode this is using integer division and
 * remainders.
 */
void bbob2009_decode_function_index(const int function_index,
                                    int *function_id,
                                    int *instance_id,
                                    int *dimension) {
    static const int dims[] = {2, 3, 5, 10, 20, 40};
    static const int number_of_consecutive_instances = 5;
    static const int number_of_functions = 24;
    static const int number_of_dimensions = 6;
    const int high_instance_id = function_index /
        (number_of_consecutive_instances * number_of_functions * number_of_dimensions);
    int low_instance_id;
    int rest = function_index %
        (number_of_consecutive_instances * number_of_functions * number_of_dimensions);
    *dimension = dims[rest / (number_of_consecutive_instances * number_of_functions)];
    rest = rest % (number_of_consecutive_instances * number_of_functions);
    *function_id = rest / number_of_consecutive_instances + 1;
    rest = rest % number_of_consecutive_instances;
    low_instance_id = rest + 1;
    *instance_id = low_instance_id + 5 * high_instance_id;
}

/**
 * bbob2009_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem from the BBOB2009
 * benchmark suit. If the function index is out of bounds, return *
 * NULL.
 */
coco_problem_t *bbob2009_suit(const int function_index) {
    int i, instance_id, function_id, dimension, rseed;
    coco_problem_t *problem = NULL;
    bbob2009_decode_function_index(function_index, &function_id, &instance_id,
                                   &dimension);
    rseed = function_id + 10000 * instance_id;

    /* Break if we are past our 15 instances. */
    if (instance_id > 15) return NULL;

    if (function_id == 1) {
        double xopt[40], fopt;
        bbob2009_compute_xopt(xopt, rseed, dimension);
        fopt = bbob2009_compute_fopt(function_id, instance_id);

        problem = sphere_problem(dimension);
        problem = shift_variables(problem, xopt, 0);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 2) {
        double xopt[40], fopt;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        problem = ellipsoid_problem(dimension);
        problem = oscillate_variables(problem);
        problem = shift_variables(problem, xopt, 0);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 3) {
        double xopt[40], fopt;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        problem = rastrigin_problem(dimension);
        problem = condition_variables(problem, 10.0);
        problem = asymmetric_variable_transform(problem, 0.2);
        problem = oscillate_variables(problem);
        problem = shift_variables(problem, xopt, 0);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 4) {
        double xopt[40], fopt;
        rseed = 3 + 10000 * instance_id;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);
        /* 
         * OME: This step is in the legacy C code but _not_ in the
         * function description.
         */
        for (i = 0; i < dimension; i += 2) {
            xopt[i] = fabs(xopt[i]);
        }

        problem = bueche_rastrigin_problem(dimension);
        problem = brs_transform(problem);
        problem = oscillate_variables(problem);
        problem = shift_variables(problem, xopt, 0);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 5) {
        double xopt[40], fopt;
        bbob2009_compute_xopt(xopt, rseed, dimension);
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        problem = linear_slope_problem(dimension, xopt);
        problem = shift_objective(problem, fopt);
    } else {
        return NULL;
    }
    return problem;
}
