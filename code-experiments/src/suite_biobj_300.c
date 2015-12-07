#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco.h"
#include "coco_strdup.c"
#include "coco_problem.c"
#include "suite_bbob2009.c"
#include "mo_generics.c"

/**
 * The biobjective suite of 300 function combinations generated using the suite_bbob2009. For each
 * function, there are 11 preset instances in each of the five dimensions {2, 3, 5, 10, 20}.
 *
 * Ref: Benchmarking Numerical Multiobjective Optimizers Revisited @ GECCO'15
 */

#define SUITE_BIOBJ_NUMBER_OF_FUNCTIONS 300
#define SUITE_BIOBJ_NUMBER_OF_INSTANCES 5
#define SUITE_BIOBJ_NUMBER_OF_DIMENSIONS 5

/* Mapping to the suite_bbob2009 function instances */
static const size_t SUITE_BIOBJ_300_INSTANCE_LIST[5][2] = { { 2, 4 }, { 3, 5 }, { 7, 8 }, { 9, 10 },
    { 11, 12 } };
/* List of 300 functions (300 is the total number of bi-objective function combinations) */
static int SUITE_BIOBJ_300_FUNCTION_LIST[300][2];
/* Whether the suite is defined */
static int SUITE_BIOBJ_300_DEFINED = 0;
/* A list of problem types */
static const char SUITE_BIOBJ_300_PROBLEM_TYPE[24][30] = {
    "1-separable", "1-separable", "1-separable", "1-separable", "1-separable",
    "2-moderate", "2-moderate", "2-moderate", "2-moderate",
    "3-ill-conditioned", "3-ill-conditioned", "3-ill-conditioned", "3-ill-conditioned", "3-ill-conditioned",
    "4-multi-modal", "4-multi-modal", "4-multi-modal", "4-multi-modal", "4-multi-modal",
    "5-weakly-structured", "5-weakly-structured", "5-weakly-structured", "5-weakly-structured", "5-weakly-structured" };

/**
 * How: instance varies faster than function which is still faster than dimension
 * (instance_id and function_id actually start from 0, not 1, as is done in the underlying bbob2009 suite)
 * 
 *  problem_index | instance | function | dimension
 * ---------------+----------+----------+-----------
 *              0 |        1 |        1 |         2
 *              1 |        2 |        1 |         2
 *              2 |        3 |        1 |         2
 *              3 |        4 |        1 |         2
 *              4 |        5 |        1 |         2
 *              5 |        1 |        2 |         2
 *              6 |        2 |        2 |         2
 *             ...        ...        ...        ...
 *           1499 |        5 |      300 |         2
 *           1500 |        1 |        1 |         3
 *           1501 |        2 |        1 |         3
 *             ...        ...        ...        ...
 *           7497 |        3 |      300 |        20
 *           7498 |        4 |      300 |        20
 *           7499 |        5 |      300 |        20
 */

/**
 * Computes the function_id, instance_id and dimension_idx from the given problem_index.
 */
static void suite_biobj_300_decode_problem_index(const long problem_index,
                                                 int *function_id,
                                                 long *instance_id,
                                                 int *dimension_idx) {
  long rest;
  *dimension_idx = (int) problem_index / (SUITE_BIOBJ_NUMBER_OF_INSTANCES * SUITE_BIOBJ_NUMBER_OF_FUNCTIONS);
  rest = problem_index % (SUITE_BIOBJ_NUMBER_OF_INSTANCES * SUITE_BIOBJ_NUMBER_OF_FUNCTIONS);
  *function_id = (int) (rest / SUITE_BIOBJ_NUMBER_OF_INSTANCES);
  *instance_id = rest % SUITE_BIOBJ_NUMBER_OF_INSTANCES;
}
/**
 * Initializes the biobjective suite created from 300 combinations of the bbob2009 functions.
 * Returns the problem corresponding to the given problem_index.
 */
static coco_problem_t *suite_biobj_300(const long problem_index) {
  int function_id, function1_id, function2_id, dimension_idx;
  long instance_id;
  coco_problem_t *problem1, *problem2, *problem;
  mo_problem_data_t *data;
  double *x, *nadir;

  if (problem_index < 0)
    return NULL;

  if (SUITE_BIOBJ_300_DEFINED == 0) {
    int k = 0;
    int i, j;
    for (i = 0; i < 24; ++i) {
      for (j = i; j < 24; ++j) {
        SUITE_BIOBJ_300_FUNCTION_LIST[k][0] = i;
        SUITE_BIOBJ_300_FUNCTION_LIST[k][1] = j;
        k++;
      }
    }
    SUITE_BIOBJ_300_DEFINED = 1;
  }

  suite_biobj_300_decode_problem_index(problem_index, &function_id, &instance_id, &dimension_idx);

  function1_id = SUITE_BIOBJ_300_FUNCTION_LIST[function_id][0];
  function2_id = SUITE_BIOBJ_300_FUNCTION_LIST[function_id][1];

  problem1 = suite_bbob2009_problem(function1_id + 1, BBOB2009_DIMS[dimension_idx],
      (long) SUITE_BIOBJ_300_INSTANCE_LIST[instance_id][0]);
  problem2 = suite_bbob2009_problem(function2_id + 1, BBOB2009_DIMS[dimension_idx],
      (long) SUITE_BIOBJ_300_INSTANCE_LIST[instance_id][1]);

  /* Set the ideal and reference points*/
  data = mo_problem_data_allocate(2);
  data->ideal_point[0] = problem1->best_value[0];
  data->ideal_point[1] = problem2->best_value[0];
  nadir = coco_allocate_vector(2);
  x = problem2->best_parameter;
  coco_evaluate_function(problem1, x, &nadir[0]);
  x = problem1->best_parameter;
  coco_evaluate_function(problem2, x, &nadir[1]);
  data->reference_point[0] = nadir[0];
  data->reference_point[1] = nadir[1];

  /* ATTENTION! Changing the reference point affects the best values as well!
  data->reference_point[0] = data->ideal_point[0] + 2 * (nadir[0] - data->ideal_point[0]);
  data->reference_point[1] = data->ideal_point[1] + 2 * (nadir[1] - data->ideal_point[1]); */

  /* Construct the type of the problem */
  if (function1_id < function2_id)
    data->problem_type = coco_strdupf("%s_%s", SUITE_BIOBJ_300_PROBLEM_TYPE[function1_id],
        SUITE_BIOBJ_300_PROBLEM_TYPE[function2_id]);
  else
    data->problem_type = coco_strdupf("%s_%s", SUITE_BIOBJ_300_PROBLEM_TYPE[function2_id],
        SUITE_BIOBJ_300_PROBLEM_TYPE[function1_id]);

  mo_problem_data_compute_normalization_factor(data, 2);
  coco_free_memory(nadir);

  problem = coco_stacked_problem_allocate(problem1, problem2, data, mo_problem_data_free);
  problem->suite_dep_index = problem_index;
  problem->suite_dep_function_id = function_id;
  problem->suite_dep_instance_id = instance_id;

  coco_free_memory(problem->problem_name);
  problem->problem_name = coco_strdup(problem->problem_id);

  /* Construct the id for the suite_biobj_300 in the form "biobj_300_fxxx_DIMy" */
  coco_free_memory(problem->problem_id);
  problem->problem_id = coco_strdupf("biobj_300_f%03d_i%02ld_d%02d", function_id + 1, instance_id + 1,
      problem->number_of_variables);


  return problem;
}

/**
 * Return successor of problem_index or first index if problem_index < 0 or -1 otherwise.
 * Currently skips problems with bbob209 functions f07 and f20, because they don't define a
 * best value.
 */
static long suite_biobj_300_get_next_problem_index(long problem_index, const char *selection_descriptor) {

  const long first_index = 0;
  const long last_index = 7499;

  int function_id, function1_id, function2_id, dimension_idx;
  long instance_id;

  const int banned_functions_count = 2;
  const int banned_functions[2] = {6, 19};
  int i, is_banned;

  if (problem_index < 0)
    problem_index = first_index - 1;

  if (strlen(selection_descriptor) == 0) {
    if (problem_index < last_index) {
      do {
        is_banned = 0;
        suite_biobj_300_decode_problem_index(++problem_index, &function_id, &instance_id, &dimension_idx);

        function1_id = SUITE_BIOBJ_300_FUNCTION_LIST[function_id][0];
        function2_id = SUITE_BIOBJ_300_FUNCTION_LIST[function_id][1];

        for (i = 0; i < banned_functions_count; i++)
          if ((function1_id == banned_functions[i]) || (function2_id == banned_functions[i])) {
            is_banned = 1;
            break;
          }

      } while (is_banned);

      return problem_index;
    }
    return -1;
  }

  /* TODO:
   o parse the selection_descriptor -> value bounds on funID, dimension, instance
   o increment problem_index until funID, dimension, instance match the restrictions
   or max problem_index is succeeded.
   */

  coco_error("suite_biobj_300_get_next_problem_index(): specific selections not yet implemented");
  return -1; /* Never reached */
}

/* Undefine constants */
#undef SUITE_BIOBJ_NUMBER_OF_FUNCTIONS
#undef SUITE_BIOBJ_NUMBER_OF_INSTANCES
#undef SUITE_BIOBJ_NUMBER_OF_DIMENSIONS
