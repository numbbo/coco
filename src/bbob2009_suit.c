#include <assert.h>
#include <math.h>

#include "coco_generics.c"

#include "bbob2009_legacy_code.c"

#include "f_bbob_step_ellipsoid.c"
#include "f_attractive_sector.c"
#include "f_bent_cigar.c"
#include "f_bueche-rastrigin.c"
#include "f_different_powers.c"
#include "f_discus.c"
#include "f_ellipsoid.c"
#include "f_linear_slope.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_schaffers.c"
#include "f_sharp_ridge.c"
#include "f_sphere.c"
#include "f_weierstrass.c"

#include "shift_objective.c"
#include "oscillate_objective.c"
#include "power_objective.c"

#include "affine_transform_variables.c"
#include "asymmetric_variable_transform.c"
#include "brs_transform.c"
#include "condition_variables.c"
#include "oscillate_variables.c"
#include "scale_variables.c"
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

static void bbob2009_copy_rotation_matrix(double **rot,
                                          double *M, double *b,
                                          const int dimension) {
    int row, column;
    double *current_row;

    for (row = 0; row < dimension; ++row) {
        current_row = M + row * dimension;
        for (column = 0; column < dimension; ++column) {
            current_row[column] = rot[row][column];
        }
        b[row] = 0.0;
    }
}

/**
 * bbob2009_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem from the BBOB2009
 * benchmark suit. If the function index is out of bounds, return *
 * NULL.
 */
coco_problem_t *bbob2009_suit(const int function_index) {
    size_t len;
    int i, instance_id, function_id, dimension, rseed;
    coco_problem_t *problem = NULL;
    bbob2009_decode_function_index(function_index, &function_id, &instance_id,
                                   &dimension);
    /* This assert is a hint for the static analyzer. */
    assert(dimension > 1);

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
    } else if (function_id == 6) {
        int i, j, k;
        double M[40 * 40], b[40], xopt[40], fopt, *current_row;
        double **rot1, **rot2;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);
        
        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        rot2 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_compute_rotation(rot2, rseed, dimension);
        for (i = 0; i < dimension; ++i) {
            b[i] = 0.0;
            current_row = M + i * dimension;
            for (j = 0; j < dimension; ++j) {
                current_row[j] = 0.0;
                for (k = 0; k < dimension; ++k) {
                    double exponent = k * 1.0 / (dimension - 1.0);
                    current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
                }
            }
        }
        bbob2009_free_matrix(rot1, dimension);
        bbob2009_free_matrix(rot2, dimension);

        problem = attractive_sector_problem(dimension, xopt);
        problem = shift_objective(problem, fopt);
        problem = oscillate_objective(problem);
        problem = power_objective(problem, 0.9);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);
    } else if (function_id == 7) {
        problem = bbob_step_ellipsoid_problem(dimension, instance_id);
    } else if (function_id == 8) {
        double xopt[40], minus_one[40], fopt, factor;
        bbob2009_compute_xopt(xopt, rseed, dimension);
        for (i = 0; i < dimension; ++i) {
            minus_one[i] = -1.0;
            xopt[i] *= 0.75;
        }
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        /* C89 version of
         *   fmax(1.0, sqrt(dimension) / 8.0);
         * follows
         */
        factor = sqrt(dimension) / 8.0;
        if (factor < 1.0)
            factor = 1.0;

        problem = rosenbrock_problem(dimension);
        problem = shift_variables(problem, minus_one, 0);
        problem = scale_variables(problem, factor);
        problem = shift_variables(problem, xopt, 0);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 9) {
        int row, column;
        double M[40*40], b[40], fopt, factor, *current_row;
        double **rot1;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed, dimension);
        /* C89 version of
         *   fmax(1.0, sqrt(dimension) / 8.0);
         * follows
         */
        factor = sqrt(dimension) / 8.0;
        if (factor < 1.0)
            factor = 1.0;
        /* Compute affine transformation */
        for (row = 0; row < dimension; ++row) {
            current_row = M + row * dimension;
            for (column = 0; column < dimension; ++column) {
                current_row[column] = rot1[row][column];
                if (row == column)
                    current_row[column] *= factor;
            }
            b[row] = 0.5;
        }
        bbob2009_free_matrix(rot1, dimension);

        problem = rosenbrock_problem(dimension);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 10) {
        double M[40*40], b[40], xopt[40], fopt;
        double **rot1;
        bbob2009_compute_xopt(xopt, rseed, dimension);
        fopt = bbob2009_compute_fopt(function_id, instance_id);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
        bbob2009_free_matrix(rot1, dimension);

        problem = ellipsoid_problem(dimension);
        problem = oscillate_variables(problem);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 11) {
        double M[40*40], b[40], xopt[40], fopt;
        double **rot1;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
        bbob2009_free_matrix(rot1, dimension);

        problem = discus_problem(dimension);
        problem = oscillate_variables(problem);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);
        problem = shift_objective(problem, fopt);
    } else if (function_id == 12) {
        double M[40*40], b[40], xopt[40], fopt;
        double **rot1;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed + 1000000, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
        bbob2009_free_matrix(rot1, dimension);

        problem = bent_cigar_problem(dimension);
        problem = shift_objective(problem, fopt);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = asymmetric_variable_transform(problem, 0.5);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);
    } else if (function_id == 13) {
        int i, j, k;
        double M[40*40], b[40], xopt[40], fopt, *current_row;
        double **rot1, **rot2;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        rot2 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_compute_rotation(rot2, rseed, dimension);
        for (i = 0; i < dimension; ++i) {
            b[i] = 0.0;
            current_row = M + i * dimension;
            for (j = 0; j < dimension; ++j) {
                current_row[j] = 0.0;
                for (k = 0; k < dimension; ++k) {
                    double exponent = k * 1.0 / (dimension - 1.0);
                    current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
                }
            }
        }
        bbob2009_free_matrix(rot1, dimension);
        bbob2009_free_matrix(rot2, dimension);
        problem = sharp_ridge_problem(dimension);
        problem = shift_objective(problem, fopt);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);
    } else if (function_id == 14) {
        double M[40*40], b[40], xopt[40], fopt;
        double **rot1;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
        bbob2009_free_matrix(rot1, dimension);

        problem = different_powers_problem(dimension);
        problem = shift_objective(problem, fopt);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);
    } else if (function_id == 15) {
        int i, j, k;
        double M[40*40], b[40], xopt[40], fopt, *current_row;
        double **rot1, **rot2;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        rot2 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_compute_rotation(rot2, rseed, dimension);
        for (i = 0; i < dimension; ++i) {
            b[i] = 0.0;
            current_row = M + i * dimension;
            for (j = 0; j < dimension; ++j) {
                current_row[j] = 0.0;
                for (k = 0; k < dimension; ++k) {
                    double exponent = k * 1.0 / (dimension - 1.0);
                    current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
                }
            }
        }


        problem = rastrigin_problem(dimension);
        problem = shift_objective(problem, fopt);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = asymmetric_variable_transform(problem, 0.2);
        problem = oscillate_variables(problem);
        bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);

        bbob2009_free_matrix(rot1, dimension);
        bbob2009_free_matrix(rot2, dimension);
    }  else if (function_id == 16) {
        int i, j, k;
        static double condition = 100.;
        double M[40*40], b[40], xopt[40], fopt, *current_row;
        double **rot1, **rot2;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        rot2 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_compute_rotation(rot2, rseed, dimension);
        for (i = 0; i < dimension; ++i) {
            b[i] = 0.0;
            current_row = M + i * dimension;
            for (j = 0; j < dimension; ++j) {
                current_row[j] = 0.0;
                for (k = 0; k < dimension; ++k) {
                    double exponent = k * 1.0 / (dimension - 1.0);
                    current_row[j] += rot1[i][k] * pow(1.0 / sqrt(condition), exponent) * rot2[k][j];
                }
            }
        }

        problem = weierstrass_problem(dimension);
        problem = shift_objective(problem, fopt);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = oscillate_variables(problem);
        bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);

        bbob2009_free_matrix(rot1, dimension);
        bbob2009_free_matrix(rot2, dimension);
    } else if (function_id == 17) {
        int i, j;
        double M[40*40], b[40], xopt[40], fopt, *current_row;
        double **rot1, **rot2;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        rot2 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_compute_rotation(rot2, rseed, dimension);
        for (i = 0; i < dimension; ++i) {
            b[i] = 0.0;
            current_row = M + i * dimension;
            for (j = 0; j < dimension; ++j) {
                    double exponent = j * 1.0 / (dimension - 1.0);
                    current_row[j] = rot1[i][j] * pow(sqrt(10), exponent);
                }
        }

        problem = schaffers_problem(dimension);
        problem = shift_objective(problem, fopt);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = asymmetric_variable_transform(problem, 0.5);
        bbob2009_copy_rotation_matrix(rot2, M, b, dimension);
        problem = affine_transform_variables(problem, M, b, dimension); 
        problem = shift_variables(problem, xopt, 0);

        bbob2009_free_matrix(rot1, dimension);
        bbob2009_free_matrix(rot2, dimension);
    } else if (function_id == 18) {
        int i, j;
        double M[40*40], b[40], xopt[40], fopt, *current_row;
        double **rot1, **rot2;
        fopt = bbob2009_compute_fopt(function_id, instance_id);
        bbob2009_compute_xopt(xopt, rseed, dimension);

        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        rot2 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
        bbob2009_compute_rotation(rot2, rseed, dimension);
        for (i = 0; i < dimension; ++i) {
            b[i] = 0.0;
            current_row = M + i * dimension;
            for (j = 0; j < dimension; ++j) {
                    double exponent = j * 1.0 / (dimension - 1.0);
                    current_row[j] = rot1[i][j] * pow(sqrt(1000), exponent);
                }
        }

        problem = schaffers_problem(dimension);
        problem = shift_objective(problem, fopt);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = asymmetric_variable_transform(problem, 0.5);
        bbob2009_copy_rotation_matrix(rot2, M, b, dimension);
        problem = affine_transform_variables(problem, M, b, dimension);
        problem = shift_variables(problem, xopt, 0);

        bbob2009_free_matrix(rot1, dimension);
        bbob2009_free_matrix(rot2, dimension);
    } else {
        return NULL;
    }
    
    /* Now set the problem name and problem id of the final problem */
    coco_free_memory(problem->problem_name);
    coco_free_memory(problem->problem_id);
    
    /* Construct a meaningful problem id */
    len = snprintf(NULL, 0, "bbbob2009_f%02i_i%02i_d%02i", 
                   function_id, instance_id, dimension);
    problem->problem_id = coco_allocate_memory(len + 1);
    snprintf(problem->problem_id, len + 1, "bbbob2009_f%02i_i%02i_d%02i",
                   function_id, instance_id, dimension);

    len = snprintf(NULL, 0, "BBOB2009 f%02i instance %i in %iD",
                   function_id, instance_id, dimension);
    problem->problem_name = coco_allocate_memory(len + 1);
    snprintf(problem->problem_name, len + 1, "BBOB2009 f%02i instance %i in %iD",
                   function_id, instance_id, dimension);
    return problem;
}
