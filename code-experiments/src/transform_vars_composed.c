/**
 * @file transform_vars_composed.c
 * @brief Implementation of inverse transformation composed by tasy and tosz.
 * @author Paul Dufosse
 */

#include "transform_vars_asymmetric.c"
#include "transform_vars_oscillate.c"

/**
 * @brief Inverse of composed non-linear transformation tcomp_uv obtained with brentq.
 */
static double tcomp_uv_inv(double yi, tasy_data *dasy, tosz_data *dosz) {
  double xi;
  xi = brentinv((callback_type) &tasy_uv, yi, dasy);
  xi = brentinv((callback_type) &tosz_uv, xi, dosz);
  return xi;
}

static void transform_inv_feas_dir_composed(coco_problem_t *problem, const double *xopt) {
  size_t i;
  size_t j;
  int is_in_bounds;
  double di;
  double xi;
  double *sol = NULL;
  double halving_factor = .5;

  transform_vars_asymmetric_data_t *data_asy;
  tasy_data *dasy;

  transform_vars_oscillate_data_t *data_osz;
  tosz_data *dosz;

  coco_problem_t *inner_problem;
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  sol = coco_allocate_vector(problem->number_of_variables);
  dasy = coco_allocate_memory(sizeof(*dasy));
  dosz = coco_allocate_memory(sizeof(*dosz));

  data_asy = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(inner_problem);
  data_osz = (transform_vars_oscillate_data_t *) coco_problem_transformed_get_data(problem);

  dasy->beta = data_asy->beta;
  dasy->n = problem->number_of_variables;
  dosz->alpha = data_osz->alpha;

  j = 0;
  while (1) {

    for (i = 0; i < problem->number_of_variables; ++i) {
      dasy->i = i;
      di = tcomp_uv_inv(problem->initial_solution[i] * pow(halving_factor, (double) (long) j), dasy, dosz);
      xi = di + xopt[i];
      is_in_bounds = (int) (-5.0 < xi  && xi < 5.0);
      /* Line search for the inverse-transformed feasible initial solution
         to remain within the bounds
        */
      if (!is_in_bounds) {
        j = j + 1;
        break;
      }
      sol[i] = di;
    }
    if (!is_in_bounds){
      continue;
    }
    else {
      break;
    }   
  }
  for (i = 0; i < problem->number_of_variables; ++i) {
    problem->initial_solution[i] = sol[i];
  }
  coco_free_memory(dasy);
  coco_free_memory(dosz);

  coco_free_memory(sol);
}
