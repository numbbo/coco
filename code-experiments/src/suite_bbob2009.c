#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco_generics.c"
#include "coco_utilities.c"

#include "f_attractive_sector.c"
#include "f_bbob_step_ellipsoid.c"
#include "f_bent_cigar.c"
#include "f_bueche_rastrigin.c"
#include "f_different_powers.c"
#include "f_discus.c"
#include "f_ellipsoid.c"
#include "f_gallagher.c"
#include "f_griewank_rosenbrock.c"
#include "f_griewank_rosenbrock.c"
#include "f_katsuura.c"
#include "f_linear_slope.c"
#include "f_lunacek_bi_rastrigin.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_schaffers.c"
#include "f_schwefel.c"
#include "f_sharp_ridge.c"
#include "f_sphere.c"

#include "f_weierstrass.c"
#include "suite_bbob2009_legacy_code.c"
#include "transform_obj_oscillate.c"
#include "transform_obj_penalize.c"
#include "transform_obj_power.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_asymmetric.c"
#include "transform_vars_brs.c"
#include "transform_vars_conditioning.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_scale.c"
#include "transform_vars_shift.c"
#include "transform_vars_x_hat.c"
#include "transform_vars_z_hat.c"

#define MAX_DIM SUITE_BBOB2009_MAX_DIM
#define SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES 5
#define SUITE_BBOB2009_NUMBER_OF_FUNCTIONS 24
#define SUITE_BBOB2009_NUMBER_OF_DIMENSIONS 6
static const int BBOB2009_DIMS[] = { 2, 3, 5, 10, 20, 40 };/*might end up useful outside of bbob2009_decode_problem_index*/

/**
 * bbob2009_decode_problem_index(problem_index, function_id, instance_id,
 * dimension):
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
 *       not what we want _now_, as the instances change in each
 *       workshop. We should have provide-problem-instance-indices
 *       methods to be able to run useful subsets of instances.
 * 
 * This gives us:
 *
 * problem_index | function_id | instance_id | dimension
 * --------------+-------------+-------------+-----------
 *             0 |           1 |           1 |         2
 *             1 |           1 |           2 |         2
 *             2 |           1 |           3 |         2
 *             3 |           1 |           4 |         2
 *             4 |           1 |           5 |         2
 *             5 |           2 |           1 |         2
 *             6 |           2 |           2 |         2
 *            ...           ...           ...        ...
 *           119 |          24 |           5 |         2
 *           120 |           1 |           1 |         3
 *           121 |           1 |           2 |         3
 *            ...           ...           ...        ...
 *          2157 |          24 |           13|        40
 *          2158 |          24 |           14|        40
 *          2159 |          24 |           15|        40
 *
 * The quickest way to decode this is using integer division and
 * remainders.
 */

static void suite_bbob2009_decode_problem_index(const long problem_index,
                                                int *function_id,
                                                long *instance_id,
                                                int *dimension) {
  const long high_instance_id = problem_index
      / (SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * SUITE_BBOB2009_NUMBER_OF_FUNCTIONS *
      SUITE_BBOB2009_NUMBER_OF_DIMENSIONS);
  long low_instance_id;
  long rest = problem_index % (SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES *
  SUITE_BBOB2009_NUMBER_OF_FUNCTIONS * SUITE_BBOB2009_NUMBER_OF_DIMENSIONS);
  *dimension = BBOB2009_DIMS[rest
      / (SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * SUITE_BBOB2009_NUMBER_OF_FUNCTIONS)];
  rest = rest % (SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * SUITE_BBOB2009_NUMBER_OF_FUNCTIONS);
  *function_id = (int) (rest / SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES + 1);
  rest = rest % SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
  low_instance_id = rest + 1;
  *instance_id = low_instance_id + 5 * high_instance_id;
}

/* Encodes a triplet of (function_id, instance_id, dimension_idx) into a problem_index
 * The problem index can, then, be used to directly generate a problem
 * It helps allow easier control on instances, functions and dimensions one wants to run
 * all indices start from 0 TODO: start at 1 instead?
 */
/* Commented to silence the compiler
static long suite_bbob2009_encode_problem_index(int function_id, long instance_id, int dimension_idx) {
  long cycleLength = SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * SUITE_BBOB2009_NUMBER_OF_FUNCTIONS
      * SUITE_BBOB2009_NUMBER_OF_DIMENSIONS;
  long tmp1 = instance_id % SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
  long tmp2 = function_id * SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
  long tmp3 = dimension_idx
      * (SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * SUITE_BBOB2009_NUMBER_OF_FUNCTIONS);
  long tmp4 = ((long) (instance_id / SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES)) * cycleLength;

  return tmp1 + tmp2 + tmp3 + tmp4;
} */

static coco_problem_t *suite_bbob2009_problem(int function_id, int dimension_, long instance_id) {
  size_t len;
  long rseed;
  coco_problem_t *problem = NULL;
  const size_t dimension = (size_t) dimension_;

  /* This assert is a hint for the static analyzer. */
  assert(dimension > 1);
  if (dimension > MAX_DIM)
    coco_error("bbob2009_suite currently supports dimension up to %lu (%lu given)",
    MAX_DIM, dimension);

#if 0
  { /* to be removed */
    int dimension_idx;
    switch (dimension) {/*TODO: make this more dynamic*//* This*/
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
      dimension_idx = -1;
      break;
    }
    assert(problem_index == suite_bbob2009_encode_problem_index(function_id - 1, instance_id - 1 , dimension_idx));
  }
#endif 
  rseed = function_id + 10000 * instance_id;

  /* Break if we are past our 15 instances. */
  if (instance_id > 15)
    return NULL;

  if (function_id == 1) {
    double xopt[MAX_DIM], fopt;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    fopt = bbob2009_compute_fopt(function_id, instance_id);

    problem = f_sphere(dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 2) {
    double xopt[MAX_DIM], fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    problem = f_ellipsoid(dimension);
    problem = f_transform_vars_oscillate(problem);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 3) {
    double xopt[MAX_DIM], fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    problem = f_rastrigin(dimension);
    problem = f_transform_vars_conditioning(problem, 10.0);
    problem = f_transform_vars_asymmetric(problem, 0.2);
    problem = f_transform_vars_oscillate(problem);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 4) {
    unsigned i; /*to prevent warnings, changed for all i,j and k variables used to iterate over coordinates*/
    double xopt[MAX_DIM], fopt, penalty_factor = 100.0;
    rseed = 3 + 10000 * instance_id;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    /*
     * OME: This step is in the legacy C code but _not_ in the
     * function description.
     */
    for (i = 0; i < dimension; i += 2) {
      xopt[i] = fabs(xopt[i]);
    }

    problem = f_bueche_rastrigin(dimension);
    problem = f_transform_vars_brs(problem);
    problem = f_transform_vars_oscillate(problem);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_obj_penalize(problem, penalty_factor);
  } else if (function_id == 5) {
    double xopt[MAX_DIM], fopt;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = f_linear_slope(dimension, xopt);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 6) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
        }
      }
    }
    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);

    problem = f_attractive_sector(dimension, xopt);
    problem = f_transform_obj_oscillate(problem);
    problem = f_transform_obj_power(problem, 0.9);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
  } else if (function_id == 7) {
    problem = f_bbob_step_ellipsoid(dimension, instance_id);
  } else if (function_id == 8) {
    unsigned i;
    double xopt[MAX_DIM], minus_one[MAX_DIM], fopt, factor;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      minus_one[i] = -1.0;
      xopt[i] *= 0.75;
    }
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    /* C89 version of
     *   fmax(1.0, sqrt(dimension) / 8.0);
     * follows
     */
    factor = coco_max_double(1.0, sqrt((double) dimension) / 8.0);

    problem = f_rosenbrock(dimension);
    problem = f_transform_vars_shift(problem, minus_one, 0);
    problem = f_transform_vars_scale(problem, factor);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 9) {
    unsigned row, column;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], fopt, factor, *current_row;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed, dimension_);
    /* C89 version of
     *   fmax(1.0, sqrt(dimension) / 8.0);
     * follows
     */
    factor = sqrt((double) (long) dimension) / 8.0;
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

    problem = f_rosenbrock(dimension);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 10) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    fopt = bbob2009_compute_fopt(function_id, instance_id);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = f_ellipsoid(dimension);
    problem = f_transform_vars_oscillate(problem);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 11) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = f_discus(dimension);
    problem = f_transform_vars_oscillate(problem);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 12) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed + 1000000, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = f_bent_cigar(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_asymmetric(problem, 0.5);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
  } else if (function_id == 13) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
        }
      }
    }
    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
    problem = f_sharp_ridge(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
  } else if (function_id == 14) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = f_different_powers(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
  } else if (function_id == 15) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
        }
      }
    }

    problem = f_rastrigin(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_asymmetric(problem, 0.2);
    problem = f_transform_vars_oscillate(problem);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 16) {
    unsigned i, j, k;
    static double condition = 100.;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0
        / (double) dimension;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          const double base = 1.0 / sqrt(condition);
          const double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(base, exponent) * rot2[k][j];
        }
      }
    }

    problem = f_weierstrass(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_oscillate(problem);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_penalize(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 17) {
    unsigned i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        double exponent = 1.0 * (int) i / ((double) (long) dimension - 1.0);
        current_row[j] = rot2[i][j] * pow(sqrt(10), exponent);
      }
    }

    problem = f_schaffers(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_asymmetric(problem, 0.5);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_penalize(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 18) {
    unsigned i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0;
    double **rot1, **rot2;
    /* Reuse rseed from f17. */
    rseed = 17 + 10000 * instance_id;

    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        double exponent = 1.0 * (int) i / ((double) (long) dimension - 1.0);
        current_row[j] = rot2[i][j] * pow(sqrt(1000), exponent);
      }
    }

    problem = f_schaffers(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_asymmetric(problem, 0.5);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_penalize(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 19) {
    unsigned i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], shift[MAX_DIM], fopt;
    double scales, **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    for (i = 0; i < dimension; ++i) {
      shift[i] = -0.5;
    }

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed, dimension_);
    scales = coco_max_double(1., sqrt((double) dimension) / 8.);
    for (i = 0; i < dimension; ++i) {
      for (j = 0; j < dimension; ++j) {
        rot1[i][j] *= scales;
      }
    }

    problem = f_griewank_rosenbrock(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_shift(problem, shift, 0);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = f_transform_vars_affine(problem, M, b, dimension);

    bbob2009_free_matrix(rot1, dimension);

  } else if (function_id == 20) {
    unsigned i, j;
    static double condition = 10.;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, *tmp1 = coco_allocate_vector(
        dimension), *tmp2 = coco_allocate_vector(dimension);
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_unif(tmp1, dimension_, rseed);
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
          double exponent = 1.0 * (int) i / ((double) (long) dimension - 1);
          current_row[j] = pow(sqrt(condition), exponent);
        }
      }
    }
    for (i = 0; i < dimension; ++i) {
      tmp1[i] = -2 * fabs(xopt[i]);
      tmp2[i] = 2 * fabs(xopt[i]);
    }
    problem = f_schwefel(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_scale(problem, 100);
    problem = f_transform_vars_shift(problem, tmp1, 0);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, tmp2, 0);
    problem = f_transform_vars_z_hat(problem, xopt);
    problem = f_transform_vars_scale(problem, 2);
    problem = f_transform_vars_x_hat(problem, rseed);
    coco_free_memory(tmp1);
    coco_free_memory(tmp2);
  } else if (function_id == 21) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = f_gallagher(dimension, instance_id, 101);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 22) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = f_gallagher(dimension, instance_id, 21);
    problem = f_transform_obj_shift(problem, fopt);
  } else if (function_id == 23) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], *current_row, fopt, penalty_factor = 1.0;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(100), exponent) * rot2[k][j];
        }
      }
    }
    problem = f_katsuura(dimension);
    problem = f_transform_obj_shift(problem, fopt);
    problem = f_transform_vars_affine(problem, M, b, dimension);
    problem = f_transform_vars_shift(problem, xopt, 0);
    problem = f_transform_obj_penalize(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 24) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = f_lunacek_bi_rastrigin(dimension, instance_id);
    problem = f_transform_obj_shift(problem, fopt);
  } else {
    return NULL;
  }

  /* Now set the problem name and problem id of the final problem */
  coco_free_memory(problem->problem_name);
  coco_free_memory(problem->problem_id);

  /* Construct a meaningful problem id */
  len = (size_t) snprintf(NULL, 0, "bbob2009_f%02i_i%02li_d%02lu", function_id, instance_id, dimension);
  problem->problem_id = coco_allocate_memory(len + 1);
  snprintf(problem->problem_id, len + 1, "bbob2009_f%02i_i%02li_d%02lu", function_id, instance_id, dimension);

  len = (size_t) snprintf(NULL, 0, "BBOB2009 f%02i instance %li in %luD", function_id, instance_id,
      dimension);
  problem->problem_name = coco_allocate_memory(len + 1);
  snprintf(problem->problem_name, len + 1, "BBOB2009 f%02i instance %li in %luD", function_id, instance_id,
      dimension);
  return problem;
}

/* TODO: specify selection_descriptor and implement
 *
 * Possible example for a descriptor: "instance:1-5, dimension:-20",
 * where instances are relative numbers (w.r.t. to the instances in
 * test bed), dimensions are absolute.
 *
 * Return successor of problem_index or first index if problem_index < 0 or -1 otherwise.
 *
 * Details: this function is not necessary unless selection is implemented.
 */
static long suite_bbob2009_get_next_problem_index(long problem_index, const char *selection_descriptor) {
  const long first_index = 0;
  const long last_index = 2159;

  if (problem_index < 0)
    problem_index = first_index - 1;

  if (strlen(selection_descriptor) == 0) {
    if (problem_index < last_index)
      return problem_index + 1;
    return -1;
  }

  /* TODO:
   o parse the selection_descriptor -> value bounds on funID, dimension, instance
   o increment problem_index until funID, dimension, instance match the restrictions
   or max problem_index is succeeded.
   */

  coco_error("next_problem_index is yet to be implemented for specific selections");
  return -1;
}

/**
 * suite_bbob2009(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from the BBOB2009
 * benchmark suite. If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *suite_bbob2009(long problem_index) {
  coco_problem_t *problem;
  int dimension, function_id;
  long instance_id;

  if (problem_index < 0)
    return NULL;
  suite_bbob2009_decode_problem_index(problem_index, &function_id, &instance_id, &dimension);
  problem = suite_bbob2009_problem(function_id, dimension, instance_id);
  problem->suite_dep_index = problem_index;
  problem->suite_dep_function_id = function_id;
  problem->suite_dep_instance_id = instance_id;
  return problem;
}

/* Undefine constants */
#undef MAX_DIM
#undef SUITE_BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES 
#undef SUITE_BBOB2009_NUMBER_OF_FUNCTIONS 
#undef SUITE_BBOB2009_NUMBER_OF_DIMENSIONS 
