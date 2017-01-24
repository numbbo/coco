/**
 * @file f_gallagher.c
 * @brief Implementation of the Gallagher function and problem.
 */

#include <assert.h>
#include <math.h>

/*Wassim: timing TODO: remove*/

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"

#include "transform_vars_permutation.c"
#include "transform_vars_blockrotation.c"
#include "transform_vars_permutation_helpers.c"
#include "transform_vars_scale.c"
#include "transform_obj_norm_by_dim.c"

#include "f_sphere.c"
#include "transform_vars_gallagher_blockrotation.c"
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
    		(unsigned long) number_of_peaks);
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




/* TODO: Functions used in/related to the large scale suite are below. Eventually either merge with above after the standard version is updated with the new approach or put what's below in a separate file */



/**
 * @brief Uses the core function to evaluate the sub problem.
 */
static void f_gallagher_sub_evaluate_core(coco_problem_t *problem_i, const double *x, double *y) {

  assert(problem_i->number_of_objectives == 1);
  y[0] = f_sphere_raw(x, problem_i->number_of_variables);
  /*assert(y[0] + 1e-13 >= problem->best_value[0]);*/
}

/**
 * @brief Allocates the basic gallagher sub problem.
 */
static coco_problem_t *f_gallagher_sub_problem_allocate(const size_t number_of_variables) {
  
  coco_problem_t *problem_i = coco_problem_allocate_from_scalars("gallagher_sub function",
                                                               f_gallagher_sub_evaluate_core, f_gallagher_versatile_data_free, number_of_variables, -5.0, 5.0, 0.0);
  f_gallagher_versatile_data_t *versatile_data_tmp;
  problem_i->versatile_data = (f_gallagher_versatile_data_t *) coco_allocate_memory(sizeof(f_gallagher_versatile_data_t));
  versatile_data_tmp = ((f_gallagher_versatile_data_t *) problem_i->versatile_data);
  /* the following are not needed in the sub-problems */
  versatile_data_tmp->number_of_peaks = 0;
  versatile_data_tmp->sub_problems = NULL;
  versatile_data_tmp->rotated_x = NULL;
  versatile_data_tmp->block_size_map = NULL;
  versatile_data_tmp->first_non_zero_map = NULL;
  versatile_data_tmp->block_sizes = NULL;
  versatile_data_tmp->B = NULL;

  coco_problem_set_id(problem_i, "%s_d%04lu", "gallagher_sub", number_of_variables);

  /* Compute best solution */
  /*f_gallagher_sub_evaluate_core(problem, problem->best_parameter, problem->best_value);*/
  return problem_i;
}



/**
 * @brief Implements the gallagher function without connections to any COCO structures.
 * Wassim: core to not conflict with raw for now
 */
static double f_gallagher_core(const double *x, size_t number_of_variables, f_gallagher_versatile_data_t *versatile_data) {

  coco_problem_t *problem_i;
  double result = 0;
  double y, w_i;
  size_t i,j;
  double maxf;
  double *x_local;
  x_local = coco_allocate_vector(number_of_variables);


  for (i = 0; i < versatile_data->number_of_peaks; i++) {
    for (j = 0; j < number_of_variables; j++) {
      x_local[j] = x[j];/*versatile_data->rotated_x[j];*/
    }
    problem_i = versatile_data->sub_problems[i];
    problem_i->evaluate_function(problem_i, x_local, &y);
    if (i == 0) {
      w_i = 10;
    } else {
      w_i = 1.1 + 8.0 * (((double) i + 1) - 2.0) / (((double)versatile_data->number_of_peaks) - 2.0);
    }
    y = w_i * exp(- 1.0 / (2.0 * ((double)number_of_variables)) * y);/* Wassim: problem_i->evaluate_function is the sphere on a transformed coordiante system with conditioning */
    if ( i == 0 || maxf < y ) {
      maxf = y;
    }
  }
  result = 10.0 - maxf;
  coco_free_memory(x_local);
  return result;
}

/**
 * @brief Uses the core function to evaluate the COCO problem.
 */
static void f_gallagher_evaluate_core(coco_problem_t *problem, const double *x, double *y) {

  assert(problem->number_of_objectives == 1);
  y[0] = f_gallagher_core(x, problem->number_of_variables, ((f_gallagher_versatile_data_t *) problem->versatile_data));
  if (! (y[0] + 1e-13 >= problem->best_value[0])) {
    printf("\n x[0]= %f: %f < %f\n", x[0], y[0] + 1e-13, problem->best_value[0]);
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic gallagher problem.
 */
static coco_problem_t *f_gallagher_problem_allocate(const size_t number_of_variables, size_t number_of_peaks) {

  size_t peak_index;
  f_gallagher_versatile_data_t *versatile_data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("gallagher function",
                                                               f_gallagher_evaluate_core, f_gallagher_versatile_data_free, number_of_variables, -5.0, 5.0, 0.0);
  problem->versatile_data = (f_gallagher_versatile_data_t *) coco_allocate_memory(sizeof(f_gallagher_versatile_data_t));
  versatile_data = (f_gallagher_versatile_data_t *)problem->versatile_data;/* shortcut */
  versatile_data->number_of_peaks = number_of_peaks;
  versatile_data->sub_problems = (coco_problem_t **) coco_allocate_memory(number_of_peaks * sizeof(coco_problem_t *));
  for (peak_index = 0; peak_index < number_of_peaks; peak_index++) {
    versatile_data->sub_problems[peak_index] = f_gallagher_sub_problem_allocate(number_of_variables);
  }
  versatile_data->rotated_x = coco_allocate_vector(number_of_variables);
  versatile_data->block_size_map = coco_allocate_vector_size_t(number_of_variables);
  versatile_data->first_non_zero_map = coco_allocate_vector_size_t(number_of_variables);
  versatile_data->block_sizes = coco_allocate_vector_size_t(number_of_variables);

  coco_problem_set_id(problem, "%s_d%04lu", "gallagher", number_of_variables);
  problem->best_value[0] = 0;
  
  /* Compute best solution */
  /*f_gallagher_evaluate_core(problem, problem->best_parameter, problem->best_value);*/
  return problem;
}


static coco_problem_t *f_gallagher_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                       const size_t dimension,
                                                                       const size_t instance,
                                                                       const long rseed,
                                                                       size_t number_of_peaks,
                                                                       const char *problem_id_template,
                                                                       const char *problem_name_template) {

  size_t i, j;
  int peak_index;
  double fopt, *y_i, *y_i_tmp;
  double penalty_factor = 1.0;
  coco_problem_t *problem = NULL, **problem_i;
  f_gallagher_versatile_data_t *versatile_data;
  double **B;
  const double *const *B_copy;
  /*size_t *P1, *P2;*/
  size_t *block_sizes;
  size_t nb_blocks;
  size_t idx_blocksize, current_blocksize, next_bs_change; /* needed for the rotated y_i*/
  size_t swap_range;
  size_t nb_swaps;
  coco_random_state_t *rng = coco_random_new((uint32_t) rseed);
  const size_t peaks_21 = 21;
  const size_t peaks_101 = 101;
  double a = 0.8, b, c;
  double maxcondition = 1000.0;
  double alpha_i, *alpha_i_vals;
  size_t *P_alpha_i, *P_Lambda;
  /* maxcondition satisfies the old code and the doc but seems wrong in that it is, with very high
   * probability, not the largest condition level!!! */

  fopt = bbob2009_compute_fopt(function, instance);
  if (number_of_peaks == peaks_101) {
    maxcondition = sqrt(maxcondition);
    b = 10.0;
    c = 5.0;
  } else if (number_of_peaks == peaks_21) {
    b = 9.8;
    c = 4.9;
  } else {
    coco_error("f_gallagher_permblockdiag_bbob_problem_allocate(): '%lu' is a non-supported number of peaks",
               number_of_peaks);
  }

  block_sizes = coco_get_block_sizes(&nb_blocks, dimension, "bbob-largescale");

  swap_range = coco_get_swap_range(dimension, "bbob-largescale");
  nb_swaps = coco_get_nb_swaps(dimension, "bbob-largescale");

  B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;
  coco_compute_blockrotation(B, rseed + 1000000, dimension, block_sizes, nb_blocks);

  /*P1 = coco_allocate_vector_size_t(dimension);
  P2 = coco_allocate_vector_size_t(dimension);
  coco_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  coco_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);*/

  problem = f_gallagher_problem_allocate(dimension, number_of_peaks);
  versatile_data = (f_gallagher_versatile_data_t *) problem->versatile_data;/* shortcut */
  /* set versatile_data fields needed later in transform_vars_gallagher_blockrotation.c */
  /* TODO: block-rotation related fields should be set in a separate function, not the definition of the problem*/
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];
  for (i = 0; i < dimension; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    versatile_data->block_size_map[i] = current_blocksize;
    versatile_data->first_non_zero_map[i] = next_bs_change - current_blocksize;
  }
  versatile_data->B = coco_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  versatile_data->B = coco_copy_block_matrix(B_copy, dimension, block_sizes, nb_blocks);
  
  alpha_i_vals = coco_allocate_vector(number_of_peaks - 1);
  for (i = 0; i < number_of_peaks - 1; i++) {
    alpha_i_vals[i] = pow(maxcondition, (double) (i) / ((double) (number_of_peaks - 2)));
  }
  P_alpha_i = coco_allocate_vector_size_t(number_of_peaks - 1);/* random permutation of the alpha_i's to allow sampling without replacement*/
  coco_compute_random_permutation(P_alpha_i, rseed + 4000000, number_of_peaks - 1);

  for (peak_index = 0; peak_index < number_of_peaks; peak_index++) {
    problem_i = &(versatile_data->sub_problems[peak_index]);
    /* compute transformation parameters: */
    /* y_i and block-rotate it once and for all */
    y_i = coco_allocate_vector(dimension);
    for (i = 0; i < dimension; i++) {
      y_i[i] = b * coco_random_uniform(rng) - c; /* Wassim: not sure the random numbers are used in the same order as for the standard case*/
    }
    y_i_tmp = coco_allocate_vector(dimension);
    /* block rotate y_i */
    /* TODO: there should be an independent code to apply the rotations and block-rotations, a helper_function maybe*/
    for (i = 0; i < dimension; ++i) {
      y_i_tmp[i] = 0;
      for (j = versatile_data->first_non_zero_map[i]; j < versatile_data->first_non_zero_map[i] + versatile_data->block_size_map[i]; ++j) {
        y_i_tmp[i] += B_copy[i][j - versatile_data->first_non_zero_map[i]] * y_i[j];
      }
      y_i[i] = y_i_tmp[i];
    }

    if (peak_index == 0) {
      for (i = 0; i < dimension; i++){
        y_i[i] *= a;
        problem->best_parameter[i] = y_i[i];
      }
      alpha_i = maxcondition;
    } else {
      alpha_i = alpha_i_vals[P_alpha_i[peak_index - 1]];/*already square-rooted */
    }

    (*problem_i)->best_value[0] = 0;/* to prevent raising the assert */
    /* P_Lambda the permutation */
    P_Lambda = coco_allocate_vector_size_t(dimension);/* random permutation of the values in C_i */
    coco_compute_random_permutation(P_Lambda, rseed + 5000000 + peak_index, dimension);

    /* apply var transformations to sub problem*/
    *problem_i = transform_vars_scale(*problem_i, 1. / sqrt(sqrt(sqrt(alpha_i))));/* sqrt( alpha^1/4) */
    *problem_i = transform_vars_conditioning(*problem_i, alpha_i);
    /**problem_i = transform_vars_blockrotation(*problem_i, B_copy, dimension, block_sizes, nb_blocks);*/
    *problem_i = transform_vars_permutation(*problem_i, P_Lambda, dimension);
    *problem_i = transform_vars_shift(*problem_i, y_i, 0);


    coco_free_memory(P_Lambda);
    coco_free_memory(y_i);
    coco_free_memory(y_i_tmp);
    y_i = NULL;
  }
  f_gallagher_evaluate_core(problem, problem->best_parameter, problem->best_value);

  /*transform global objective function*/
  problem = transform_obj_oscillate(problem);
  problem = transform_obj_power(problem, 2.0);
  problem = transform_obj_penalize(problem, penalty_factor);
  problem = transform_obj_shift(problem, fopt);
  /* Wassim: TODO: re-activate*/
  /*problem = transform_vars_gallagher_blockrotation(problem);*//* block-matrix in versatile_data*/

  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "large_scale_block_rotated");

  coco_random_free(rng);
  coco_free_block_matrix(B, dimension);
  /*coco_free_memory(P1);
  coco_free_memory(P2);*/
  coco_free_memory(block_sizes);
  coco_free_memory(alpha_i_vals);
  coco_free_memory(P_alpha_i);
  return problem;
}



/* 
 New, old style, Gallagher function for the large-scale suite with significantly faster CPU time.
 However, it does not separate the raw function implementation from the transformed problem (rotations are applied inside the raw function).
 */


/**
 * @brief Data type for the Gallagher problem.
 */
typedef struct {
  long rseed;
  double *xopt;
  double **B, **x_local, **arr_scales;
  size_t *block_sizes;
  size_t nb_blocks;
  size_t nb_rows;
  size_t *block_size_map;
  size_t *first_non_zero_map;
  size_t number_of_peaks;
  double *peak_values;
  coco_problem_free_function_t old_free_problem;
} f_largescale_gallagher_data_t;

/**
 * @brief Implements the raw Gallagher function.
 */
static double f_gallagher_largescale_raw(const double *x, const size_t number_of_variables, f_largescale_gallagher_data_t *data) {
  size_t i, j; /* Loop over dim */
  double *tmx;
  double a = 0.1;
  double tmp2, f = 0., f_add, tmp, f_pen = 0., f_true = 0.;
  double fac;
  double result;
  size_t current_blocksize, first_non_zero_ind;
  
  if (coco_vector_contains_nan(x, number_of_variables))
  return NAN;
  
  fac = -0.5 / (double) number_of_variables;
  
  /* Boundary handling */
  for (i = 0; i < number_of_variables; ++i) {
    tmp = fabs(x[i]) - 5.;
    if (tmp > 0.) {
      f_pen += tmp * tmp;
    }
  }
  f_add = f_pen;
  /* Transformation in search space, apply the block-rotation */
  /* TODO: this should rather be done in f_gallagher */
  tmx = coco_allocate_vector(number_of_variables);
  for (i = 0; i < number_of_variables; i++) {
    tmx[i] = 0;
    current_blocksize = data->block_size_map[i];
    first_non_zero_ind = data->first_non_zero_map[i];
    for (j = first_non_zero_ind; j < first_non_zero_ind + current_blocksize; ++j) {
      tmx[i] += data->B[i][j - first_non_zero_ind] * x[j];;
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
static void f_gallagher_largescale_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);

  y[0] = f_gallagher_largescale_raw(x, problem->number_of_variables, (f_largescale_gallagher_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the Gallagher data object.
 */
static void f_gallagher_largescale_free(coco_problem_t *problem) {
  f_largescale_gallagher_data_t *data;
  data = (f_largescale_gallagher_data_t *) problem->data;
  coco_free_memory(data->xopt);
  coco_free_memory(data->peak_values);
  coco_free_block_matrix(data->B, data->nb_rows);
  coco_free_memory(data->block_sizes);
  coco_free_memory(data->block_size_map);
  coco_free_memory(data->first_non_zero_map);
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
static coco_problem_t *f_gallagher_largescale_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const size_t number_of_peaks,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {
  f_largescale_gallagher_data_t *data;
  /* problem_name and best_parameter will be overwritten below */
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Gallagher function",
                                                               f_gallagher_largescale_evaluate, f_gallagher_largescale_free, dimension, -5.0, 5.0, 0.0);
  
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
  size_t current_blocksize, first_non_zero_ind, idx_blocksize, next_bs_change;
  data = (f_largescale_gallagher_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->number_of_peaks = number_of_peaks;
  data->xopt = coco_allocate_vector(dimension);
  /* set the block-rotation related fields*/
  data->nb_rows = dimension;
  data->block_sizes = coco_get_block_sizes(&(data->nb_blocks), dimension, "bbob-largescale");
  data->block_size_map = coco_allocate_vector_size_t(dimension);
  data->first_non_zero_map = coco_allocate_vector_size_t(dimension);
  idx_blocksize = 0;
  next_bs_change = data->block_sizes[idx_blocksize];
  for (i = 0; i < dimension; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += data->block_sizes[idx_blocksize];
    }
    current_blocksize = data->block_sizes[idx_blocksize];
    data->block_size_map[i] = current_blocksize;
    data->first_non_zero_map[i] = next_bs_change - current_blocksize;
  }
  data->B = coco_allocate_blockmatrix(dimension, data->block_sizes, data->nb_blocks);
  /*B_copy = (const double *const *)B;*/
  coco_compute_blockrotation(data->B, rseed + 1000000, dimension, data->block_sizes, data->nb_blocks);
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
               (unsigned long) number_of_peaks);
  }
  data->rseed = rseed;

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
  /* apply the block-diagonal rotation to the local optima*/
  for (i = 0; i < dimension; ++i) {
    data->xopt[i] = 0.8 * (b * random_numbers[i] - c);
    problem->best_parameter[i] = 0.8 * (b * random_numbers[i] - c);
    current_blocksize = data->block_size_map[i];
    first_non_zero_ind = data->first_non_zero_map[i];
    for (j = 0; j < number_of_peaks; ++j) {
      data->x_local[i][j] = 0.;
      for (k = first_non_zero_ind; k < first_non_zero_ind + current_blocksize; ++k) {
        data->x_local[i][j] += data->B[i][k - first_non_zero_ind] * (b * random_numbers[j * dimension + k] - c);
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
  f_gallagher_largescale_evaluate(problem, problem->best_parameter, problem->best_value);

  fopt = bbob2009_compute_fopt(function, instance);
  problem = transform_obj_shift(problem, fopt);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "large_scale_block_rotated");

  return problem;
}



