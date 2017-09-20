/**
 * @file f_lunacek_bi_rastrigin.c
 * @brief Implementation of the Lunacek bi-Rastrigin function and problem.
 */

#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "suite_bbob_legacy_code.c"
#include "transform_obj_shift.c"

/**
 * @brief Data type for the Lunacek bi-Rastrigin problem.
 */
typedef struct {
  double *x_hat, *z;
  double *xopt, fopt;
  double **rot1, **rot2;
  long rseed;
  coco_problem_free_function_t old_free_problem;
} f_lunacek_bi_rastrigin_data_t;

/**
 * @brief Implements the Lunacek bi-Rastrigin function without connections to any COCO structures.
 */
static double f_lunacek_bi_rastrigin_raw(const double *x,
                                         const size_t number_of_variables,
                                         f_lunacek_bi_rastrigin_data_t *data) {
  double result;
  static const double condition = 100.;
  size_t i, j;
  double penalty = 0.0;
  static const double mu0 = 2.5;
  static const double d = 1.;
  const double s = 1. - 0.5 / (sqrt((double) (number_of_variables + 20)) - 4.1);
  const double mu1 = -sqrt((mu0 * mu0 - d) / s);
  double *tmpvect, sum1 = 0., sum2 = 0., sum3 = 0.;

  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  for (i = 0; i < number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* x_hat */
  for (i = 0; i < number_of_variables; ++i) {
    data->x_hat[i] = 2. * x[i];
    if (data->xopt[i] < 0.) {
      data->x_hat[i] *= -1.;
    }
  }

  tmpvect = coco_allocate_vector(number_of_variables);
  /* affine transformation */
  for (i = 0; i < number_of_variables; ++i) {
    double c1;
    tmpvect[i] = 0.0;
    c1 = pow(sqrt(condition), ((double) i) / (double) (number_of_variables - 1));
    for (j = 0; j < number_of_variables; ++j) {
      tmpvect[i] += c1 * data->rot2[i][j] * (data->x_hat[j] - mu0);
    }
  }
  for (i = 0; i < number_of_variables; ++i) {
    data->z[i] = 0;
    for (j = 0; j < number_of_variables; ++j) {
      data->z[i] += data->rot1[i][j] * tmpvect[j];
    }
  }
  /* Computation core */
  for (i = 0; i < number_of_variables; ++i) {
    sum1 += (data->x_hat[i] - mu0) * (data->x_hat[i] - mu0);
    sum2 += (data->x_hat[i] - mu1) * (data->x_hat[i] - mu1);
    sum3 += cos(2 * coco_pi * data->z[i]);
  }
  result = coco_double_min(sum1, d * (double) number_of_variables + s * sum2)
      + 10. * ((double) number_of_variables - sum3) + 1e4 * penalty;
  coco_free_memory(tmpvect);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_lunacek_bi_rastrigin_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_lunacek_bi_rastrigin_raw(x, problem->number_of_variables, (f_lunacek_bi_rastrigin_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the Lunacek bi-Rastrigin data object.
 */
static void f_lunacek_bi_rastrigin_free(coco_problem_t *problem) {
  f_lunacek_bi_rastrigin_data_t *data;
  data = (f_lunacek_bi_rastrigin_data_t *) problem->data;
  coco_free_memory(data->x_hat);
  coco_free_memory(data->z);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, problem->number_of_variables);
  bbob2009_free_matrix(data->rot2, problem->number_of_variables);

  /* Let the generic free problem code deal with all of the
   * coco_problem_t fields.
   */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB Lunacek bi-Rastrigin problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_lunacek_bi_rastrigin_bbob_problem_allocate(const size_t function,
                                                                    const size_t dimension,
                                                                    const size_t instance,
                                                                    const long rseed,
                                                                    const char *problem_id_template,
                                                                    const char *problem_name_template) {

  f_lunacek_bi_rastrigin_data_t *data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Lunacek's bi-Rastrigin function",
      f_lunacek_bi_rastrigin_evaluate, f_lunacek_bi_rastrigin_free, dimension, -5.0, 5.0, 0.0);

  const double mu0 = 2.5;

  double fopt, *tmpvect;
  size_t i;

  data = (f_lunacek_bi_rastrigin_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x_hat = coco_allocate_vector(dimension);
  data->z = coco_allocate_vector(dimension);
  data->xopt = coco_allocate_vector(dimension);
  data->rot1 = bbob2009_allocate_matrix(dimension, dimension);
  data->rot2 = bbob2009_allocate_matrix(dimension, dimension);
  data->rseed = rseed;

  data->fopt = bbob2009_compute_fopt(24, instance);
  bbob2009_compute_xopt(data->xopt, rseed, dimension);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(data->rot2, rseed, dimension);

  problem->data = data;

  /* Compute best solution */
  tmpvect = coco_allocate_vector(dimension);
  bbob2009_gauss(tmpvect, dimension, rseed);
  for (i = 0; i < dimension; ++i) {
    data->xopt[i] = 0.5 * mu0;
    if (tmpvect[i] < 0.0) {
      data->xopt[i] *= -1.0;
    }
    problem->best_parameter[i] = data->xopt[i];
  }
  coco_free_memory(tmpvect);
  f_lunacek_bi_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);

  fopt = bbob2009_compute_fopt(function, instance);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  return problem;
}
