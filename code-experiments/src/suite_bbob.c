#include "coco.h"

#include "f_attractive_sector.c"
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
#include "f_step_ellipsoid.c"
#include "f_weierstrass.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

static coco_suite_t *suite_bbob_allocate(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  suite = coco_suite_allocate("suite_bbob", 24, 6, dimensions, "1-15");

  return suite;
}

static char *suite_bbob_get_instances_by_year(int year) {
/* TODO: Fill in for other years! */

  if (year == 2009) {
    return "1-5,1-5,1-5";
  }
  else {
    coco_error("suite_bbob_get_instances_by_year(): year %d not defined for suite_bbob", year);
    return NULL;
  }
}

static coco_problem_t *suite_bbob_get_problem(size_t function_id, size_t dimension, size_t instance_id) {

  coco_problem_t *problem = NULL;

  const char *problem_id_template = "bbob_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "BBOB f%lu instance %lu in %luD";

  const long rseed = (long) (function_id + 10000 * instance_id);
  const long rseed_3 = (long) (3 + 10000 * instance_id);
  const long rseed_17 = (long) (17 + 10000 * instance_id);

  if (function_id == 1) {
    problem = f_sphere_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 2) {
    problem = f_ellipsoid_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 3) {
    problem = f_rastrigin_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 4) {
    problem = f_bueche_rastrigin_bbob_problem_allocate(function_id, dimension, instance_id, rseed_3,
        problem_id_template, problem_name_template);
  } else if (function_id == 5) {
    problem = f_linear_slope_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 6) {
    problem = f_attractive_sector_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 7) {
    problem = f_step_ellipsoid_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 8) {
    problem = f_rosenbrock_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 9) {
    problem = f_rosenbrock_rotated_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 10) {
    problem = f_ellipsoid_rotated_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 11) {
    problem = f_discus_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 12) {
    problem = f_bent_cigar_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 13) {
    problem = f_sharp_ridge_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 14) {
    problem = f_different_powers_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 15) {
    problem = f_rastrigin_rotated_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 16) {
    problem = f_weierstrass_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 17) {
    problem = f_schaffers_bbob_problem_allocate(function_id, dimension, instance_id, rseed, 10,
        problem_id_template, problem_name_template);
  } else if (function_id == 18) {
    problem = f_schaffers_bbob_problem_allocate(function_id, dimension, instance_id, rseed_17, 1000,
        problem_id_template, problem_name_template);
  } else if (function_id == 19) {
    problem = f_griewank_rosenbrock_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 20) {
    problem = f_schwefel_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 21) {
    problem = f_gallagher_bbob_problem_allocate(function_id, dimension, instance_id, rseed, 101,
        problem_id_template, problem_name_template);
  } else if (function_id == 22) {
    problem = f_gallagher_bbob_problem_allocate(function_id, dimension, instance_id, rseed, 21,
        problem_id_template, problem_name_template);
  } else if (function_id == 23) {
    problem = f_katsuura_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 24) {
    problem = f_lunacek_bi_rastrigin_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  }

  problem->suite_dep_function_id = function_id;
  problem->suite_dep_instance_id = instance_id;

  return problem;
}
