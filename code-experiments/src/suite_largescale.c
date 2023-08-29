/**
 * @file suite_largescale.c
 * @brief Implementation of the bbob large-scale suite containing 24 functions in 6 large dimensions.
 */

#include "coco.h"

#include "f_attractive_sector.c"
#include "f_bent_cigar_generalized.c"
#include "f_bueche_rastrigin.c"
#include "f_different_powers.c"
#include "f_discus_generalized.c"
#include "f_ellipsoid.c"
#include "f_griewank_rosenbrock.c"
#include "f_linear_slope.c"
#include "f_rastrigin.c"
#include "f_schaffers.c"
#include "f_schwefel_generalized.c"
#include "f_sharp_ridge_generalized.c"
#include "f_sphere.c"
#include "f_step_ellipsoid.c"
#include "f_weierstrass.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances,
                                         const int known_optima);

/**
 * @brief Sets the dimensions and default instances for the bbob large-scale suite.
 */
static coco_suite_t *suite_largescale_initialize(void) {
  
  coco_suite_t *suite;
  const size_t dimensions[] = { 20, 40, 80, 160, 320, 640};
  const size_t num_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);

  suite = coco_suite_allocate("bbob-largescale", 24, num_dimensions, dimensions, "instances: 1-15", 1);
  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob large-scale suite.
 */
static const char *suite_largescale_get_instances_by_year(const int year) {
  if (year >= 2016 || year == 0) {
    return "1-15";
  } else if (year >= 2009) {
    return "1-15";
  }
  else {
    coco_error("suite_largescale_get_instances_by_year(): year %d not defined for suite_largescale", year);
    return NULL;
  }
}



/**
 * @brief Creates and returns a large-scale problem without needing the actual large-scale suite.
 */
static coco_problem_t *coco_get_largescale_problem(const size_t function,
                                                   const size_t dimension,
                                                   const size_t instance) {
  coco_problem_t *problem = NULL;

  const char *problem_id_template = "bbob_f%03lu_i%02lu_d%04lu";
  const char *problem_name_template = "BBOB large-scale suite problem f%lu instance %lu in %luD";

  const long rseed = (long) (function + 10000 * instance);
  const long rseed_3 = (long) (3 + 10000 * instance);
  const long rseed_17 = (long) (17 + 10000 * instance);

  if (function == 1) {
    problem = f_sphere_bbob_problem_allocate(function, dimension, instance, rseed,
                                             problem_id_template, problem_name_template);
  } else if (function == 2) {
    double condition = 1.0e6;
    problem = f_ellipsoid_bbob_problem_allocate(function, dimension, instance, rseed,
                                                condition, problem_id_template, problem_name_template);
  } else if (function == 3) {
    problem = f_rastrigin_bbob_problem_allocate(function, dimension, instance, rseed,
                                                problem_id_template, problem_name_template);
  } else if (function == 4) {
    problem = f_bueche_rastrigin_bbob_problem_allocate(function, dimension, instance, rseed_3,
                                                       problem_id_template, problem_name_template);
  } else if (function == 5) {
    problem = f_linear_slope_bbob_problem_allocate(function, dimension, instance, rseed,
                                                   problem_id_template, problem_name_template);
  } else if (function == 6) {
    problem = f_attractive_sector_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                      problem_id_template, problem_name_template);
  } else if (function == 7) {
    problem = f_step_ellipsoid_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                   problem_id_template, problem_name_template);
  } else if (function == 8) {
    problem = f_rosenbrock_bbob_problem_allocate(function, dimension, instance, rseed,
                                                 problem_id_template, problem_name_template);
  } else if (function == 9) {
    problem = f_rosenbrock_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                               problem_id_template, problem_name_template);
  } else if (function == 10) {
    double condition = 1.0e6;
    problem = f_ellipsoid_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                              condition, problem_id_template, problem_name_template);
  } else if (function == 11) {
    problem = f_discus_generalized_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                       problem_id_template, problem_name_template);
  } else if (function == 12) {
    problem = f_bent_cigar_generalized_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                           problem_id_template, problem_name_template);
  } else if (function == 13) {
    problem = f_sharp_ridge_generalized_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                            problem_id_template, problem_name_template);
  } else if (function == 14) {
    problem = f_different_powers_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                     problem_id_template, problem_name_template);
  } else if (function == 15) {
    problem = f_rastrigin_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                              problem_id_template, problem_name_template);
  } else if (function == 16) {
    problem = f_weierstrass_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                problem_id_template, problem_name_template);
  } else if (function == 17) {
    problem = f_schaffers_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed, 10,
                                                              problem_id_template, problem_name_template);
  } else if (function == 18) {
    problem = f_schaffers_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed_17, 1000,
                                                              problem_id_template, problem_name_template);
  } else if (function == 19) {
    double facftrue = 10.0;
    problem = f_griewank_rosenbrock_permblockdiag_bbob_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                             facftrue, problem_id_template, problem_name_template);
  } else if (function == 20) {
    problem = f_schwefel_generalized_bbob_problem_allocate(function, dimension, instance, rseed,
                                                           problem_id_template, problem_name_template);
  } else if (function == 21) {
    problem = f_gallagher_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed, 101,
                                                              problem_id_template, problem_name_template);
  } else if (function == 22) {
    problem = f_gallagher_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed, 21,
                                                              problem_id_template, problem_name_template);
  } else if (function == 23) {
    problem = f_katsuura_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                             problem_id_template, problem_name_template);
  } else if (function == 24) {
    problem = f_lunacek_bi_rastrigin_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
                                                                         problem_id_template, problem_name_template);
  } else {
    coco_error("coco_get_largescale_problem(): cannot retrieve problem f%lu instance %lu in %luD",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension);
    return NULL; /* Never reached */
  }
  
  if (problem == NULL) {
    coco_error("coco_get_largescale_problem(): function f%lu not yet implemented  ", function);
  }
  
  return problem;
}

/**
 * @brief Returns the problem from the bbob large-scale suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_largescale_get_problem(coco_suite_t *suite,
                                                    const size_t function_idx,
                                                    const size_t dimension_idx,
                                                    const size_t instance_idx) {
  
  coco_problem_t *problem = NULL;
  
  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];
  
  problem = coco_get_largescale_problem(function, dimension, instance);
  
  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);
  
  return problem;
}
