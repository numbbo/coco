#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco_problem.c"

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


/**
 * mo_suit...(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from...
 * If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *mo_suite_first_attempt(const int problem_index) {
  coco_problem_t *problem, *problem1, *problem2;

  if (problem_index < 0)
    return NULL; 

  problem1 = coco_get_problem("bbob2009", 0);
  problem2 = coco_get_problem("bbob2009", problem_index);
  problem = coco_stacked_problem_allocate("0102-d02", "sphere-sepelli", problem1, problem2);

  return problem;
  
}

