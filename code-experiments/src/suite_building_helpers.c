#include <assert.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_suites.c"
#include "coco_problem.c"
#include "f_attractive_sector.c" /* Added to get rid of compiler errors. */
#include "suite_bbob2009_legacy_code.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"

/**
 * A collection of "random" future code (snippets)
 */

#if 0  /* doesn't compile, not compliant with old code */
typedef void (*coco_optimizer_t)(coco_problem_t *problem, long budget);
#endif

/**
 * Construct a meaningful problem id for a bbob2009 problem,
 * the allocated memory must be free()ed by the caller. 
 * */
static char *bbob2009_problem_id(const char *name, size_t number_of_variables) {
  char * problem_id;

  problem_id = coco_strdupf("%s_%04lu", name, number_of_variables);
  if (coco_suite_problem_id_is_fine(problem_id))
    return problem_id;
  coco_warning("Problem ID '%s' is not valid", problem_id);
  coco_free_memory(problem_id);
  coco_error("Invalid problem ID");
  return NULL; /* never reached, but prevents pedantic warning */
}

/**
 * coco_problem_t *coco_allocate_so_problem_sss(
 *                            const char * problem_id,  // "unique" ID, only benign characters
 *                            const char * problem_name,
 *                            coco_evaluate_function_t fct,
 *                            size_t number_of_variables,
 *                            double smallest_value_of_interest,
 *                            double largest_value_of_interest,
 *                            double best_parameter):
 *
 * Return a single-objective coco problem constructed from a function and
 * scalar values for bounds and best parameter.
 *
 * Example:
 *
 *    coco_problem_t *problem = coco_allocate_so_problem_from_sss(
 *                          "sphere_20", "sphere", f_sphere_evaluate,
 *                          20, -5, 5, 0);
 *
 * Details: this is yet a tentative name/interface.
 * FIXME: find a better interface for best_parameter?
 */
coco_problem_t *coco_allocate_so_problem_from_sss(const char * problem_id,
                                                  const char * problem_name,
                                                  coco_evaluate_function_t fct,
                                                  size_t number_of_variables,
                                                  double smallest_value_of_interest,
                                                  double largest_value_of_interest,
                                                  double best_parameter) {
  size_t i;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);

  problem->problem_id = coco_strdup(problem_id);
  problem->problem_name = coco_strdup(problem_name);
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = fct;

  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = smallest_value_of_interest;
    problem->largest_values_of_interest[i] = largest_value_of_interest;
    problem->best_parameter[i] = best_parameter;
  }
  /* Calculate best parameter value */
  fct(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/***************** EXAMPLE: BENT CIGAR PROBLEM ****************/
/*** generic definition without any dependencies on COCO ***/
static double b2bob2009_raw_bent_cigar(size_t dimension, const double *x) {
  static const double condition = 1.0e6;
  size_t i;
  double res;

  res = x[0] * x[0];
  for (i = 1; i < dimension; ++i) {
    res += condition * x[i] * x[i];
  }
  return res;
}

/*** transform into coco_evaluate_function_t type ***/
static void b2bob2009_raw_bent_cigar_evaluate(coco_problem_t *self, const double *x, double *y) {
  assert(coco_problem_get_number_of_objectives(self) == 1);
  y[0] = b2bob2009_raw_bent_cigar(coco_problem_get_dimension(self), x);
}

/*** define as coco_problem_t ***/
static coco_problem_t *b2bob2009_raw_bent_cigar_problem(const size_t number_of_variables) {
  char *problem_id = bbob2009_problem_id("bent_cigar", number_of_variables);
  coco_problem_t *problem = coco_allocate_so_problem_from_sss(problem_id, "bent cigar function",
      b2bob2009_raw_bent_cigar_evaluate, number_of_variables, -5, 5, 0);
  coco_free_memory(problem_id);
  return problem;
}

/*** transform into final bbob2009 problem in coco format ***/
static coco_problem_t *b2bob2009_bent_cigar_problem(long dimension_, long instance_id) {
  const int function_id = 12;
  const size_t dimension = (size_t) dimension_; /* prevent subtle warnings */
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *xopt = coco_allocate_vector(dimension);
  double fopt;
  double **rot1;
  long rseed = function_id + 10000 * instance_id;

  coco_problem_t *problem;

  fopt = bbob2009_compute_fopt(function_id, instance_id);
  bbob2009_compute_xopt(xopt, rseed + 1000000, dimension_);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = b2bob2009_raw_bent_cigar_problem(dimension);
  problem = f_transform_obj_shift(problem, fopt);
  problem = f_transform_vars_affine(problem, M, b, dimension);
  problem = f_transform_vars_asymmetric(problem, 0.5);
  problem = f_transform_vars_affine(problem, M, b, dimension);
  problem = f_transform_vars_shift(problem, xopt, 0);
  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}

/***************** EXAMPLE: ATTRACTIVE SECTOR PROBLEM ****************/
#if 0
typedef struct {double *xopt;} f_attractive_sector_data_t;
#endif

/*** define computation of raw function as coco_evaluate_function_t type ***/
static void b2bob2009_raw_attractive_sector_evaluate(coco_problem_t *self, const double *x, double *y) {
  const size_t dimension = coco_problem_get_dimension(self);
  size_t i;
  f_attractive_sector_data_t *data;

  assert(coco_problem_get_number_of_objectives(self) == 1);
  data = self->data;
  y[0] = 0.0;
  for (i = 0; i < dimension; ++i) {
    if (data->xopt[i] * x[i] > 0.0) {
      y[0] += 100.0 * 100.0 * x[i] * x[i];
    } else {
      y[0] += x[i] * x[i];
    }
  }
}
/*** define raw function as coco_problem_t ***/
static coco_problem_t *b2bob2009_raw_attractive_sector_problem(const size_t number_of_variables,
                                                               const double *xopt) {
  f_attractive_sector_data_t *data;
  char *problem_id = bbob2009_problem_id("attractive_sector", number_of_variables);
  coco_problem_t *problem = coco_allocate_so_problem_from_sss(problem_id, "attractive sector function",
      f_attractive_sector_evaluate, number_of_variables, -5, 5, 0);
  coco_free_memory(problem_id);
  data = coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, number_of_variables);
  problem->data = data;
  problem->free_problem = f_attractive_sector_free;
  return problem;
}

/*** transform into final bbob2009 problem in coco format ***/
static coco_problem_t *b2bob2009_attractive_sector_problem(long dimension_, long instance_id) {
  const int function_id = 6;
  const size_t dimension = (size_t) dimension_; /* prevent subtle warnings */
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *xopt = coco_allocate_vector(dimension);
  double *current_row, fopt;
  size_t i, j, k;
  double **rot1, **rot2;

  long rseed = function_id + 10000 * instance_id;

  coco_problem_t *problem;

  fopt = bbob2009_compute_fopt(function_id, instance_id);
  bbob2009_compute_xopt(xopt, rseed, dimension_);

  /* compute affine transformation M from two rotation matrices */
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
        double exponent = (double) k / (double) (dimension - 1);
        current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
      }
    }
  }
  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  problem = b2bob2009_raw_attractive_sector_problem(dimension, xopt);
  problem = f_transform_obj_oscillate(problem);
  problem = f_transform_obj_power(problem, 0.9);
  problem = f_transform_obj_shift(problem, fopt);
  problem = f_transform_vars_affine(problem, M, b, dimension);
  problem = f_transform_vars_shift(problem, xopt, 0);
  return problem;
}
