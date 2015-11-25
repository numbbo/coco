#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco.h"
#include "coco_strdup.c"
#include "coco_problem.c"
#include "suite_bbob2009.c"

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
static const size_t SUITE_BIOBJ_300_INSTANCE_LIST[5][2] = { { 2, 4 }, { 3, 5 }, { 7, 8 }, { 9, 10 }, { 11, 12 } };
/* List of 300 functions (300 is the total number of bi-objective function combinations) */
static int SUITE_BIOBJ_300_FUNCTION_LIST[300][2];
/* Whether the suite is defined */
static int SUITE_BIOBJ_300_DEFINED = 0;
/* A list of problem groups (TODO: to be replaced by a problem_info field) */
static const char SUITE_BIOBJ_300_PROBLEM_GROUP[24][30] = {
    "separable", "separable", "separable", "separable", "separable",
    "moderate", "moderate", "moderate", "moderate",
    "ill-conditioned", "ill-conditioned", "ill-conditioned", "ill-conditioned", "ill-conditioned",
    "multi-modal", "multi-modal", "multi-modal", "multi-modal", "multi-modal",
    "weakly-structured", "weakly-structured", "weakly-structured", "weakly-structured", "weakly-structured" };

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

static coco_problem_t *suite_biobj_300(const long problem_index) {
  int function_id, function1_id, function2_id, dimension_idx;
  long instance_id;
  coco_problem_t *problem1, *problem2, *problem;

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
  problem = coco_stacked_problem_allocate(problem1, problem2);
  problem->suite_dep_index = problem_index;
  problem->suite_dep_function_id = function_id;
  problem->suite_dep_instance_id = instance_id;

  /* Construct the id for the suite_biobj_300 in the form "biobj_300_fxxx_DIMy" */
  coco_free_memory(problem->problem_id);
  problem->problem_id = coco_strdupf("biobj_300_f%03d_i%02ld_d%02d", function_id + 1, instance_id + 1, problem->number_of_variables);

  /* Construct the information about the problem - its "type" */
  /* TODO: Use a new field (for example problem_type) instead of problem_name to store this information */
  coco_free_memory(problem->problem_name);
  if (function1_id < function2_id)
    problem->problem_name = coco_strdupf("%s_%s", SUITE_BIOBJ_300_PROBLEM_GROUP[function1_id], SUITE_BIOBJ_300_PROBLEM_GROUP[function2_id]);
  else
    problem->problem_name = coco_strdupf("%s_%s", SUITE_BIOBJ_300_PROBLEM_GROUP[function2_id], SUITE_BIOBJ_300_PROBLEM_GROUP[function1_id]);
  if (strstr(problem->problem_name, "biobj") != NULL) {
    printf("%s %d %d", problem->problem_name, function1_id, function2_id);
  }

  return problem;
}

/* Undefine constants */
#undef SUITE_BIOBJ_NUMBER_OF_FUNCTIONS
#undef SUITE_BIOBJ_NUMBER_OF_INSTANCES
#undef SUITE_BIOBJ_NUMBER_OF_DIMENSIONS
