#include "numbbo_generics.c"

#include "f_sphere.c"
#include "f_ellipsoid.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_skewRastriginBueche.c"
#include "f_linearSlope.c"

#include "shift_objective.c"
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
 *This gives us:
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
    int rest = function_index / 
        (number_of_consecutive_instances * number_of_functions * number_of_dimensions);
    *dimension = dims[rest / (number_of_consecutive_instances * number_of_functions)];
    rest = rest % (number_of_consecutive_instances * number_of_functions);
    *function_id = rest / number_of_consecutive_instances + 1;
    rest = rest % number_of_consecutive_instances;
    const int low_instance_id = rest + 1;
    *instance_id = low_instance_id + high_instance_id;
}

/**
 * bbob2009_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem from the BBOB2009
 * benchmark suit. If the function index is out of bounds, return *
 * NULL.
 */
numbbo_problem_t *bbob2009_suit(const int function_index) {
    int instance_id, function_id, dimension;
    numbbo_problem_t *problem = NULL;
    bbob2009_decode_function_index(function_index, &function_id, &instance_id, 
                                   &dimension);
    
    /* Break if we are past our 15 instances. */
    if (instance_id > 15) return NULL;

    if (function_id == 0) {
        problem = sphere_problem(dimension);
    } else if (function_id == 1) {
        problem = ellipsoid_problem(dimension);
    } else if (function_id == 2) {
        problem = rastrigin_problem(dimension);
    } else if (function_id == 3) {
        problem = skewRastriginBueche_problem(dimension);
    } else if (function_id == 4) {
        problem = linearSlope_problem(dimension);
    } else if (function_id == 5) {
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
