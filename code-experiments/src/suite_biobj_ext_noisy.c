/**
 * @file suite_biobj_ext.c
 * @brief Implementation of the extended biobjective bbob-biobj-ext suite containing 92 functions and 6 dimensions.
 *
 * The bbob-biobj-ext suite is created by combining two single-objective problems from the bbob suite.
 * The first 55 functions are the same as in the original bbob-biobj test suite to which 37 functions are added.
 * Those additional functions are constructed by combining all not yet contained in-group combinations (i,j) of
 * single-objective bbob functions i and j such that i<j (i.e. in particular not all combinations (i,i) are
 * included in this bbob-biobj-ext suite), with the exception of the Weierstrass function (f16) for which
 * the optimum is not unique and thus a nadir point is difficult to compute, see
 * http://numbbo.github.io/coco-doc/bbob-biobj/functions/ for details.
 *
 * @note Because some bi-objective problems constructed from two single-objective ones have a single optimal
 * value, some care must be taken when selecting the instances. The already verified instances are stored in
 * suite_biobj_ext_instances. If a new instance of the problem is called, a check ensures that the two underlying
 * single-objective instances create a true bi-objective problem. However, these new instances need to be
 * manually added to suite_biobj_ext_instances, otherwise they will be computed each time the suite constructor
 * is invoked with these instances.
 *
 * @note This file is based on the original suite_bbob_biobj.c and extends it by 37 functions in 6 dimensions.
 */

#include "suite_biobj_ext.c"

#include "transform_obj_uniform_noise.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances,
                                         const int known_optima);
static void suite_biobj_ext_free(void *suite);

static size_t suite_biobj_ext_noisy_get_new_instance(coco_suite_t *suite,
                                           const size_t instance,
                                           const size_t instance1,
                                           const size_t num_bbob_functions,
                                           const size_t *bbob_functions);
/**
 * @brief Sets the dimensions and default instances for the bbob-biobj-ext suite.
 */
static coco_suite_t *suite_biobj_ext_noisy_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };
  const size_t num_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("bbob-biobj-ext-noisy", 55+37, num_dimensions, dimensions, "year: 2018", 1);

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj-ext suite.
 */
static const char *suite_biobj_ext_noisy_get_instances_by_year(const int year) {

  if (year == 0000) { /* default/test case */
    return "1-10";
  }
  else {
    coco_error("suite_biobj_ext_noisy_get_instances_by_year(): year %d not defined for suite_biobj_ext_noisy", year);
    return NULL;
  }
}

/**
 * @brief Returns the problem from the bbob-biobj-ext suite that corresponds to the given parameters.
 *
 * Creates the bi-objective problem by constructing it from two single-objective problems from the bbob
 * suite. If the invoked instance number is not in suite_biobj_ext_instances, the function uses the following
 * formula to construct a new appropriate instance:
 *
 *   problem1_instance = 2 * biobj_instance + 1
 *
 *   problem2_instance = problem1_instance + 1
 *
 * If needed, problem2_instance is increased (see also the explanation of suite_biobj_ext_get_new_instance).
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 * @note Copied from suite_bbob_biobj.c and extended.
 */
static coco_problem_t *suite_biobj_ext_noisy_get_problem(coco_suite_t *suite,
                                               const size_t function_idx,
                                               const size_t dimension_idx,
                                               const size_t instance_idx) {
    
    coco_problem_t *problem, *problem1, *problem2;

    coco_problem_stacked_data_t *inner_data, *transformed_data;
    problem = suite_biobj_ext_get_problem(
        suite,
        function_idx,
        dimension_idx,
        instance_idx
    );
    inner_data = (coco_problem_stacked_data_t *) problem -> data;

    double alpha = 0.01 * (0.49 + 1 / (double) problem -> number_of_variables);
    double beta = 0.01;
    problem1 = transform_obj_uniform_noise(inner_data -> problem1, alpha, beta);
    problem2 = transform_obj_uniform_noise(inner_data -> problem2, alpha, beta);

    transformed_data = (coco_problem_stacked_data_t *) coco_allocate_memory(sizeof(*transformed_data));
    transformed_data -> problem1 = problem1;
    transformed_data -> problem2 = problem2;
    problem -> data = transformed_data;
    problem -> problem_free_function = coco_problem_stacked_free;
    return problem;
}

