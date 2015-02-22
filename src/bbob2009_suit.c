#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco_generics.c"

#include "bbob2009_legacy_code.c"

#include "f_bbob_step_ellipsoid.c"
#include "f_attractive_sector.c"
#include "f_bent_cigar.c"
#include "f_bueche-rastrigin.c"
#include "f_different_powers.c"
#include "f_discus.c"
#include "f_ellipsoid.c"
#include "f_griewankRosenbrock.c"
#include "f_linear_slope.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_schaffers.c"
#include "f_sharp_ridge.c"
#include "f_sphere.c"
#include "f_weierstrass.c"
#include "f_griewankRosenbrock.c"
#include "f_katsuura.c"
#include "f_schwefel.c"
#include "f_lunacek_bi_rastrigin.c"
#include "f_gallagher.c"

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
#include "x_hat_schwefel.c"
#include "z_hat_schwefel.c"
#include "penalize_uninteresting_values.c"

#define MAX_DIM 40
#define BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES 5
#define BBOB2009_NUMBER_OF_FUNCITONS 24
#define BBOB2009_NUMBER_OF_DIMENSIONS 6
const int BBOB2009_DIMS[] = {2, 3, 5, 10, 20, 40};/*might end up useful outside of bbob2009_decode_problem_index*/


/**
 * bbob2009_decode_problem_index(problem_index, function_id, instance_id,
 *dimension):
 *
 * Decode the new problem_index into the old convention of function,
 * instance and dimension. We have 24 functions in 6 different
 * dimensions so a total of 144 functions and any number of
 * instances. A natural thing would be to order them so that the
 * function varies faster than the dimension which is still faster
 * than the instance. For analysis reasons we want something
 * different. Our goal is to quickly produce 5 repetitions of a single
 * function in one dimension, then vary the function, then the
 * dimension.
 *
 * TODO: this is the default prescription for 2009. This is typically
 *       not what we want _now_. Maybe the benchmark initializer should
 *       take the desired instances as input? 
 * 
 * This gives us:
 *
 * problem_index | function_id | instance_id | dimension
 * ---------------+-------------+-------------+-----------
 *              0 |           1 |           1 |         2
 *              1 |           1 |           2 |         2
 *              2 |           1 |           3 |         2
 *              3 |           1 |           4 |         2
 *              4 |           1 |           5 |         2
 *              5 |           2 |           1 |         2
 *              6 |           2 |           2 |         2
 *             ...           ...           ...        ...
 *            119 |          24 |           5 |         2
 *            120 |           1 |           1 |         3
 *            121 |           1 |           2 |         3
 *             ...           ...           ...        ...
 *           2157 |          24 |           13|        40
 *           2158 |          24 |           14|        40
 *           2159 |          24 |           15|        40
 *
 * The quickest way to decode this is using integer division and
 * remainders.
 */
void bbob2009_decode_problem_index(const int problem_index, int *function_id,
                                    int *instance_id, int *dimension) {
  const int high_instance_id =
      problem_index / (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCITONS *
                        BBOB2009_NUMBER_OF_DIMENSIONS);
  int low_instance_id;
  int rest = problem_index % (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES *
                               BBOB2009_NUMBER_OF_FUNCITONS * BBOB2009_NUMBER_OF_DIMENSIONS);
  *dimension =
      BBOB2009_DIMS[rest / (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCITONS)];
  rest = rest % (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCITONS);
  *function_id = rest / BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES + 1;
  rest = rest % BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
  low_instance_id = rest + 1;
  *instance_id = low_instance_id + 5 * high_instance_id;
}

/* Encodes a triplet of (function_id, instance_id, dimension_idx) into a problem_index
 * The problem index can, then, be used to directly generate a problem
 * It helps allow easier control on instances, functions and dimensions one wants to run
 * all indices start from 0 TODO: start at 1 instead?
 */
int bbob2009_encode_problem_index(int function_id, int instance_id, int dimension_idx){
    int cycleLength = BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCITONS * BBOB2009_NUMBER_OF_DIMENSIONS;
    int tmp1 = instance_id % BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
    int tmp2 = function_id * BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
    int tmp3 = dimension_idx * (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCITONS);
    int tmp4 = ((int)(instance_id / BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES) ) * cycleLength; /* just for safety */
    
    return tmp1 + tmp2 + tmp3 + tmp4;
}

static void bbob2009_copy_rotation_matrix(double **rot, double *M, double *b,
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
 * bbob2009_suit(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from the BBOB2009
 * benchmark suit. If the function index is out of bounds, return *
 * NULL.
 */
coco_problem_t *bbob2009_suit(const int problem_index) {
  size_t len;
  int i, instance_id, function_id, dimension, rseed;
  int dimension_idx;
  coco_problem_t *problem = NULL;
  bbob2009_decode_problem_index(problem_index, &function_id, &instance_id,
                                 &dimension);
  /* This assert is a hint for the static analyzer. */
  assert(dimension > 1);
    switch (dimension) {/*TODO: make this more dynamic*/
            case 2:
            dimension_idx = 0;
            break;
            case 3:
            dimension_idx = 1;
            break;
            case 5:
            dimension_idx = 2;
            break;
            case 10:
            dimension_idx = 3;
            break;
            case 20:
            dimension_idx = 4;
            break;
            case 40:
            dimension_idx = 5;
            break;
        default:
            break;
    }
  assert(problem_index == bbob2009_encode_problem_index(function_id - 1, instance_id - 1 , dimension_idx));

  rseed = function_id + 10000 * instance_id;

  /* Break if we are past our 15 instances. */
  if (instance_id > 15)
    return NULL;

  if (function_id == 1) {
    double xopt[MAX_DIM], fopt;
    bbob2009_compute_xopt(xopt, rseed, dimension);
    fopt = bbob2009_compute_fopt(function_id, instance_id);

    problem = sphere_problem(dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 2) {
    double xopt[MAX_DIM], fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension);

    problem = ellipsoid_problem(dimension);
    problem = oscillate_variables(problem);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 3) {
    double xopt[MAX_DIM], fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension);

    problem = rastrigin_problem(dimension);
    problem = condition_variables(problem, 10.0);
    problem = asymmetric_variable_transform(problem, 0.2);
    problem = oscillate_variables(problem);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 4) {
    double xopt[MAX_DIM], fopt, penalty_factor = 100.0;
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
    problem = penalize_uninteresting_values(problem, penalty_factor);
  } else if (function_id == 5) {
    double xopt[MAX_DIM], fopt;
    bbob2009_compute_xopt(xopt, rseed, dimension);
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = linear_slope_problem(dimension, xopt);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 6) {
    int i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
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
    problem = oscillate_objective(problem);
    problem = power_objective(problem, 0.9);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
  } else if (function_id == 7) {
    problem = bbob_step_ellipsoid_problem(dimension, instance_id);
  } else if (function_id == 8) {
    double xopt[MAX_DIM], minus_one[MAX_DIM], fopt, factor;
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
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], fopt, factor, *current_row;
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
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
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
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
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
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
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
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
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
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
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
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
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
  } else if (function_id == 16) {
    int i, j, k;
    static double condition = 100.;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0/(double)dimension;
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
          const double base = 1.0 / sqrt(condition);
          const double exponent = k * 1.0 / (dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(base, exponent) * rot2[k][j];
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
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 17) {
    int i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0;
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
        double exponent = i / (dimension - 1.0);
        current_row[j] = rot2[i][j] * pow(sqrt(10), exponent);
      }
    }

    problem = schaffers_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = asymmetric_variable_transform(problem, 0.5);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 18) {
    int i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0;
    double **rot1, **rot2;
    /* Reuse rseed from f17. */
    rseed = 17 + 10000 * instance_id;

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
        double exponent = i * 1.0 / (dimension - 1.0);
        current_row[j] = rot2[i][j] * pow(sqrt(1000), exponent);
      }
    }

    problem = schaffers_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = asymmetric_variable_transform(problem, 0.5);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 19) {
    int i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], shift[MAX_DIM], fopt;
    double scales, **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    for (i = 0; i < dimension; ++i) {
      shift[i] = -0.5;
    }

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed, dimension);
    scales = fmax(1., sqrt((double)dimension) / 8.);
    for (i = 0; i < dimension; ++i) {
      for (j = 0; j < dimension; ++j) {
        rot1[i][j] *= scales;
      }
    }

    problem = griewankRosenbrock_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = shift_variables(problem, shift, 0);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);

    bbob2009_free_matrix(rot1, dimension);

  } else if (function_id == 20) {
    int i, j;
    static double condition = 10.;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row,
        *tmp1 = coco_allocate_vector(dimension),
        *tmp2 = coco_allocate_vector(dimension);
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_unif(tmp1, dimension, rseed);
    for (i = 0; i < dimension; ++i) {
      xopt[i] = 0.5 * 4.2096874633;
      if (tmp1[i] - 0.5 < 0) {
        xopt[i] *= -1;
      }
    }

    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        if (i == j) {
          double exponent = (double)i / (dimension - 1);
          current_row[j] = pow(sqrt(condition), exponent);
        }
      }
    }
    for (i = 0; i < dimension; ++i) {
      tmp1[i] = -2 * fabs(xopt[i]);
      tmp2[i] = 2 * fabs(xopt[i]);
    }
    problem = schwefel_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = scale_variables(problem, 100);
    problem = shift_variables(problem, tmp1, 0);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, tmp2, 0);
    problem = z_hat(problem, xopt);
    problem = scale_variables(problem, 2);
    problem = x_hat(problem, rseed);
    coco_free_memory(tmp1);
    coco_free_memory(tmp2);
  } else if (function_id == 21) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = bbob_gallagher_problem(dimension, instance_id, 101);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 22) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = bbob_gallagher_problem(dimension, instance_id, 21);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 23) {
    int i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], *current_row, fopt, penalty_factor = 1.0;
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
          current_row[j] += rot1[i][k] * pow(sqrt(100), exponent) * rot2[k][j];
        }
      }
    }
    problem = katsuura_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 24) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = bbob_lunacek_bi_rastrigin_problem(dimension, instance_id);
    problem = shift_objective(problem, fopt);
  } else {
    return NULL;
  }

  /* Now set the problem name and problem id of the final problem */
  coco_free_memory(problem->problem_name);
  coco_free_memory(problem->problem_id);

  /* Construct a meaningful problem id */
  len = snprintf(NULL, 0, "bbob2009_f%02i_i%02i_d%02i", function_id,
                 instance_id, dimension);
  problem->problem_id = coco_allocate_memory(len + 1);
  snprintf(problem->problem_id, len + 1, "bbob2009_f%02i_i%02i_d%02i",
           function_id, instance_id, dimension);

  len = snprintf(NULL, 0, "BBOB2009 f%02i instance %i in %iD", function_id,
                 instance_id, dimension);
  problem->problem_name = coco_allocate_memory(len + 1);
  snprintf(problem->problem_name, len + 1, "BBOB2009 f%02i instance %i in %iD",
           function_id, instance_id, dimension);
  return problem;
}

/* Return the bbob2009 function id of the problem or if it is not a bbob2009
 * problem -1. */
int bbob2009_get_function_id(const coco_problem_t *problem) {
  static const char *bbob_prefix = "bbob2009_";
  const char *problem_id = coco_get_problem_id(problem);
  assert(strlen(problem_id) >= 20);

  if (strncmp(bbob_prefix, problem_id, strlen(bbob_prefix)) != 0) {
    return -1;
  }

  /* OME: Ugly hardcoded extraction. In a perfect world, we would
   * parse the problem id by splitting on _ and then finding the 'f'
   * field. Instead, we cound out the position of the function id in
   * the string
   *
   *   01234567890123456789
   *   bbob2009_fXX_iYY_dZZ
   */
  return (problem_id[10] - '0') * 10 + (problem_id[11] - '0');
}

/* Return the bbob2009 instance id of the problem or if it is not a bbob2009
 * problem -1. */
int bbob2009_get_instance_id(const coco_problem_t *problem) {
  static const char *bbob_prefix = "bbob2009_";
  const char *problem_id = coco_get_problem_id(problem);
  assert(strlen(problem_id) >= 20);

  if (strncmp(bbob_prefix, problem_id, strlen(bbob_prefix)) != 0) {
    return -1;
  }

  /* OME: Ugly hardcoded extraction. In a perfect world, we would
   * parse the problem id by splitting on _ and then finding the 'i'
   * field. Instead, we cound out the position of the instance id in
   * the string
   *
   *   01234567890123456789
   *   bbob2009_fXX_iYY_dZZ
   */
  return (problem_id[14] - '0') * 10 + (problem_id[15] - '0');
}

/* Undefine constants */
#undef MAX_DIM
