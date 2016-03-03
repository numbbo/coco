/**
 * @file f_gallagher.c
 * @brief Implementation of the Gallagher function and problem.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_obj_norm_by_dim.c"
#include "transform_vars_permutation_helpers.c"

/**
 * @brief A random permutation type for the Gallagher problem.
 *
 * Needed to create a random permutation that is compatible with the old code.
 */
typedef struct {
  double value;
  size_t index;
} f_gallagher_permutation_t;

/**
 * @brief Data type for the Gallagher problem.
 */
typedef struct {
  long rseed;
  double *xopt;
  double **rotation, **x_local, **arr_scales;
  size_t number_of_peaks;
  double *peak_values;
  coco_problem_free_function_t old_free_problem;
} f_gallagher_data_t;

/**
 * Comparison function used for sorting.
 */
static int f_gallagher_compare_doubles(const void *a, const void *b) {
  double temp = (*(f_gallagher_permutation_t *) a).value - (*(f_gallagher_permutation_t *) b).value;
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}

/**
 * @brief Implements the Gallagher function without connections to any COCO structures.
 */
static double f_gallagher_raw(const double *x, const size_t number_of_variables, f_gallagher_data_t *data) {
  size_t i, j; /* Loop over dim */
  double *tmx;
  double a = 0.1;
  double tmp2, f = 0., f_add, tmp, f_pen = 0., f_true = 0.;
  double fac;
  double result;

  fac = -0.5 / (double) number_of_variables;

  /* Boundary handling */
  for (i = 0; i < number_of_variables; ++i) {
    tmp = fabs(x[i]) - 5.;
    if (tmp > 0.) {
      f_pen += tmp * tmp;
    }
  }
  f_add = f_pen;
  /* Transformation in search space */
  /* TODO: this should rather be done in f_gallagher */
  tmx = coco_allocate_vector(number_of_variables);
  for (i = 0; i < number_of_variables; i++) {
    tmx[i] = 0;
    for (j = 0; j < number_of_variables; ++j) {
      tmx[i] += data->rotation[i][j] * x[j];
    }
  }
  /* Computation core*/
  for (i = 0; i < data->number_of_peaks; ++i) {
    tmp2 = 0.;
    for (j = 0; j < number_of_variables; ++j) {
      tmp = (tmx[j] - data->x_local[j][i]);
      tmp2 += data->arr_scales[i][j] * tmp * tmp;
    }
    tmp2 = data->peak_values[i] * exp(fac * tmp2);
    f = coco_double_max(f, tmp2);
  }

  f = 10. - f;
  if (f > 0) {
    f_true = log(f) / a;
    f_true = pow(exp(f_true + 0.49 * (sin(f_true) + sin(0.79 * f_true))), a);
  } else if (f < 0) {
    f_true = log(-f) / a;
    f_true = -pow(exp(f_true + 0.49 * (sin(0.55 * f_true) + sin(0.31 * f_true))), a);
  } else
    f_true = f;

  f_true *= f_true;
  f_true += f_add;
  result = f_true;
  coco_free_memory(tmx);
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_gallagher_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_gallagher_raw(x, problem->number_of_variables, (f_gallagher_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the Gallagher data object.
 */
static void f_gallagher_free(coco_problem_t *problem) {
  f_gallagher_data_t *data;
  data = (f_gallagher_data_t *) problem->data;
  coco_free_memory(data->xopt);
  coco_free_memory(data->peak_values);
  bbob2009_free_matrix(data->rotation, problem->number_of_variables);
  bbob2009_free_matrix(data->x_local, problem->number_of_variables);
  bbob2009_free_matrix(data->arr_scales, data->number_of_peaks);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB Gallagher problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_gallagher_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const size_t number_of_peaks,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {

  f_gallagher_data_t *data;
  /* problem_name and best_parameter will be overwritten below */
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Gallagher function",
      f_gallagher_evaluate, f_gallagher_free, dimension, -5.0, 5.0, 0.0);

  const size_t peaks_21 = 21;
  const size_t peaks_101 = 101;

  double fopt;
  size_t i, j, k;
  double maxcondition = 1000.;
  /* maxcondition1 satisfies the old code and the doc but seems wrong in that it is, with very high
   * probability, not the largest condition level!!! */
  double maxcondition1 = 1000.;
  double *arrCondition;
  double fitvalues[2] = { 1.1, 9.1 };
  /* Parameters for generating local optima. In the old code, they are different in f21 and f22 */
  double b, c;
  /* Random permutation */
  f_gallagher_permutation_t *rperm;
  double *random_numbers;

  data = (f_gallagher_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->number_of_peaks = number_of_peaks;
  data->xopt = coco_allocate_vector(dimension);
  data->rotation = bbob2009_allocate_matrix(dimension, dimension);
  data->x_local = bbob2009_allocate_matrix(dimension, number_of_peaks);
  data->arr_scales = bbob2009_allocate_matrix(number_of_peaks, dimension);

  if (number_of_peaks == peaks_101) {
    maxcondition1 = sqrt(maxcondition1);
    b = 10.;
    c = 5.;
  } else if (number_of_peaks == peaks_21) {
    b = 9.8;
    c = 4.9;
  } else {
    coco_error("f_gallagher_bbob_problem_allocate(): '%lu' is a non-supported number of peaks",
        number_of_peaks);
  }
  data->rseed = rseed;
  bbob2009_compute_rotation(data->rotation, rseed, dimension);

  /* Initialize all the data of the inner problem */
  random_numbers = coco_allocate_vector(number_of_peaks * dimension); /* This is large enough for all cases below */
  bbob2009_unif(random_numbers, number_of_peaks - 1, data->rseed);
  rperm = (f_gallagher_permutation_t *) coco_allocate_memory(sizeof(*rperm) * (number_of_peaks - 1));
  for (i = 0; i < number_of_peaks - 1; ++i) {
    rperm[i].value = random_numbers[i];
    rperm[i].index = i;
  }
  qsort(rperm, number_of_peaks - 1, sizeof(*rperm), f_gallagher_compare_doubles);

  /* Random permutation */
  arrCondition = coco_allocate_vector(number_of_peaks);
  arrCondition[0] = maxcondition1;
  data->peak_values = coco_allocate_vector(number_of_peaks);
  data->peak_values[0] = 10;
  for (i = 1; i < number_of_peaks; ++i) {
    arrCondition[i] = pow(maxcondition, (double) (rperm[i - 1].index) / ((double) (number_of_peaks - 2)));
    data->peak_values[i] = (double) (i - 1) / (double) (number_of_peaks - 2) * (fitvalues[1] - fitvalues[0])
        + fitvalues[0];
  }
  coco_free_memory(rperm);

  rperm = (f_gallagher_permutation_t *) coco_allocate_memory(sizeof(*rperm) * dimension);
  for (i = 0; i < number_of_peaks; ++i) {
    bbob2009_unif(random_numbers, dimension, data->rseed + (long) (1000 * i));
    for (j = 0; j < dimension; ++j) {
      rperm[j].value = random_numbers[j];
      rperm[j].index = j;
    }
    qsort(rperm, dimension, sizeof(*rperm), f_gallagher_compare_doubles);
    for (j = 0; j < dimension; ++j) {
      data->arr_scales[i][j] = pow(arrCondition[i],
          ((double) rperm[j].index) / ((double) (dimension - 1)) - 0.5);
    }
  }
  coco_free_memory(rperm);

  bbob2009_unif(random_numbers, dimension * number_of_peaks, data->rseed);
  for (i = 0; i < dimension; ++i) {
    data->xopt[i] = 0.8 * (b * random_numbers[i] - c);
    problem->best_parameter[i] = 0.8 * (b * random_numbers[i] - c);
    for (j = 0; j < number_of_peaks; ++j) {
      data->x_local[i][j] = 0.;
      for (k = 0; k < dimension; ++k) {
        data->x_local[i][j] += data->rotation[i][k] * (b * random_numbers[j * dimension + k] - c);
      }
      if (j == 0) {
        data->x_local[i][j] *= 0.8;
      }
    }
  }
  coco_free_memory(arrCondition);
  coco_free_memory(random_numbers);

  problem->data = data;

  /* Compute best solution */
  f_gallagher_evaluate(problem, problem->best_parameter, problem->best_value);

  fopt = bbob2009_compute_fopt(function, instance);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  return problem;
}




/**
 * @brief Data type for the versatile_data_t
 */
typedef struct {
  size_t number_of_peaks;
  coco_problem_t **local_problems;
} f_gallagher_versatile_data_t;

/**
 * @brief allows to free the versatile_data part of the problem.
 */
static void f_gallagher_versatile_data_free(coco_problem_t *problem) {
  coco_free_memory((f_gallagher_versatile_data_t *) problem->versatile_data);
  problem->versatile_data = NULL;
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}


/**
 * @brief Implements the gallagher function without connections to any COCO structures.
 * Wassim: core to differentiate it from raw for now
 */
static double f_gallagher_core(const double *x, const size_t number_of_variables, f_gallagher_versatile_data_t *f_gallagher_versatile_data) {

  coco_problem_t *problem_i;
  double result = 0;
  double y;
  size_t i;
  double maxf;

  for (i = 0; i < f_gallagher_versatile_data->number_of_peaks; i++) {
    problem_i = f_gallagher_versatile_data->local_problems[i];
    problem_i->evaluate_function(problem_i, x, &y);
    if ( i == 1 || maxf < y ) {
      maxf = y;
    }
  }
  result = 10 - maxf;
  
  return result;
}

/**
 * @brief Uses the core function to evaluate the COCO problem.
 */
static void f_gallagher_evaluate_core(coco_problem_t *problem, const double *x, double *y) {

  assert(problem->number_of_objectives == 1);
  y[0] = f_gallagher_core(x, problem->number_of_variables, ((f_gallagher_versatile_data_t *) problem->versatile_data));
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic gallagher problem.
 */
static coco_problem_t *f_gallagher_problem_allocate(const size_t number_of_variables, size_t number_of_peaks) {
  
  coco_problem_t *problem = coco_problem_allocate_from_scalars("gallagher function",
                                                               f_gallagher_evaluate_core, NULL, number_of_variables, -5.0, 5.0, 0.0);

  coco_problem_set_id(problem, "%s_d%04lu", "gallagher", number_of_variables);
  
  /* Compute best solution */
  f_gallagher_evaluate_core(problem, problem->best_parameter, problem->best_value);
  return problem;
}

static coco_problem_t *f_gallagher_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                       const size_t dimension,
                                                                       const size_t instance,
                                                                       const long rseed,
                                                                       const size_t number_of_peaks,
                                                                       const char *problem_id_template,
                                                                       const char *problem_name_template) {
  
  size_t i;
  double *xopt, fopt;
  double penalty_factor = 1.0;
  coco_problem_t *problem = NULL, *problem_i;
  double **B;
  const double *const *B_copy;
  size_t *P1, *P2;
  size_t *block_sizes;
  size_t nb_blocks;
  size_t swap_range;
  size_t nb_swaps;
  
  xopt = coco_allocate_vector(dimension);/*xopt is y1 */
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  
  block_sizes = coco_get_block_sizes(&nb_blocks, dimension, "bbob-largescale");
  swap_range = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");

  B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;/*TODO: silences the warning, not sure if it prevents the modification of B at all levels*/
  coco_compute_blockrotation(B, rseed + 1000000, dimension, block_sizes, nb_blocks);

  P1 = coco_allocate_vector_size_t(dimension);
  P2 = coco_allocate_vector_size_t(dimension);
  coco_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  coco_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);

  problem = f_gallagher_problem_allocate(dimension, number_of_peaks);
  problem->versatile_data = (f_gallagher_versatile_data_t *) coco_allocate_memory(sizeof(f_gallagher_versatile_data_t));
  ((f_gallagher_versatile_data_t *) problem->versatile_data)->number_of_peaks = number_of_peaks;
  ((f_gallagher_versatile_data_t *) problem->versatile_data)->local_problems = (coco_problem_t **) coco_allocate_memory(number_of_peaks * sizeof(coco_problem_t *));
  for (i = 0; i < number_of_peaks; i++) {
    ((f_gallagher_versatile_data_t *) problem->versatile_data)->local_problems[i] = (coco_problem_t *) coco_allocate_memory(sizeof(coco_problem_t));
    problem_i = ((f_gallagher_versatile_data_t *) problem->versatile_data)->local_problems[i];
    problem_i = transform_vars_permutation(problem, P2, dimension);
    problem_i = transform_vars_blockrotation(problem, B_copy, dimension, block_sizes, nb_blocks);
    problem_i = transform_vars_permutation(problem, P1, dimension);
    problem_i = transform_vars_conditioning(problem, sqrt(19));
    problem_i = transform_vars_shift(problem, &xopt[19], 0);

    
  }/* TODO: free all this in the end */

  /*problem = transform_obj_norm_by_dim(problem);*/
  problem = transform_obj_oscillate(problem);
  problem = transform_obj_power(problem, 2.0);
  problem = transform_obj_penalize(problem, penalty_factor);
  problem = transform_obj_shift(problem, fopt);
  
/*
  problem = transform_vars_permutation(problem, P2, dimension);
  problem = transform_vars_blockrotation(problem, B_copy, dimension, block_sizes, nb_blocks);
  problem = transform_vars_permutation(problem, P1, dimension);
  problem = transform_vars_conditioning(problem, 10.0);
  problem = transform_vars_shift(problem, xopt, 0);
*/
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "large_scale_block_rotated");
  
  coco_free_block_matrix(B, dimension);
  coco_free_memory(P1);
  coco_free_memory(P2);
  coco_free_memory(block_sizes);
  coco_free_memory(xopt);
  return problem;
}



