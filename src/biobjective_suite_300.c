#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco.h"
#include "coco_strdup.c"
#include "coco_problem.c"
#include "bbob2009_suite.c"

#define BIOBJECTIVE_NUMBER_OF_COMBINATIONS 300
#define BIOBJECTIVE_NUMBER_OF_INSTANCES 5
#define BIOBJECTIVE_NUMBER_OF_DIMENSIONS 5


/**
 * The biobjective suite of 300 function combinations generated
 * using the bbob2009_suite. For each function combination, there are
 * 11 preset instances in each of the five dimensions {2, 3, 5, 10, 20}.
 * Ref: Benchmarking Numerical Multiobjective Optimizers Revisited @ GECCO'15
 */

/* const size_t DIMENSIONS[6] = {2, 3, 5, 10, 20, 40};*/
static const size_t instance_list[5][2] = { {2, 4},
                                            {3, 5},
                                            {7, 8},
                                            {9, 10},
                                            {11, 12} };
/* --> we must map this number to the two corresponding BBOB functions */
static int biobjective_list[300][2]; /* 300 is the total number of 2-obj combinations (< 24*24)*/
static int defined = 0;

/**
 * How: instance varies faster than combination which is still faster than dimension
 * 
 *  problem_index | instance | combination | dimension
 * ---------------+----------+-------------+-----------
 *              0 |        1 |           1 |         2
 *              1 |        2 |           1 |         2
 *              2 |        3 |           1 |         2
 *              3 |        4 |           1 |         2
 *              4 |        5 |           1 |         2
 *              5 |        1 |           2 |         2
 *              6 |        2 |           2 |         2
 *             ...        ...           ...        ...
 *           1499 |        5 |         300 |         2
 *           1500 |        1 |           1 |         3
 *           1501 |        2 |           1 |         3
 *             ...        ...           ...        ...
 *           7497 |        3 |         300 |        20
 *           7498 |        4 |         300 |        20
 *           7499 |        5 |         300 |        20
 */
static long biobjective_encode_problem_index(int combination_idx, long instance_idx, int dimension_idx) {
    long problem_index;
    problem_index = instance_idx + 
                    combination_idx * BIOBJECTIVE_NUMBER_OF_INSTANCES + 
                    dimension_idx * (BIOBJECTIVE_NUMBER_OF_INSTANCES * BIOBJECTIVE_NUMBER_OF_COMBINATIONS);
    return problem_index;
}

static void biobjective_decode_problem_index(const long problem_index, int *combination_idx,
                                             long *instance_idx, long *dimension_idx) {
    long rest;
    *dimension_idx = problem_index / (BIOBJECTIVE_NUMBER_OF_INSTANCES * BIOBJECTIVE_NUMBER_OF_COMBINATIONS);
    rest = problem_index % (BIOBJECTIVE_NUMBER_OF_INSTANCES * BIOBJECTIVE_NUMBER_OF_COMBINATIONS);
    *combination_idx = (int)(rest / BIOBJECTIVE_NUMBER_OF_INSTANCES);
    *instance_idx = rest % BIOBJECTIVE_NUMBER_OF_INSTANCES;
}

static coco_problem_t *biobjective_suite_300(const long problem_index) {  
    int combination_idx;
    long instance_idx, dimension_idx;
    coco_problem_t *problem1, *problem2, *problem;
      
  
    if (problem_index < 0) 
        return NULL;
    
    if (defined == 0) {
        int k = 0;
        int i, j;
        for (i = 1; i <= 24; ++i) {
            for (j = i; j <= 24; ++j) {
                biobjective_list[k][0] = i;
                biobjective_list[k][1] = j;
                k++;
            }
        }
        defined = 1;
    }
    
    biobjective_decode_problem_index(problem_index, &combination_idx, &instance_idx, &dimension_idx);
    
    problem1 = bbob2009_problem(biobjective_list[combination_idx][0],
                                BBOB2009_DIMS[dimension_idx],
                                (long)instance_list[instance_idx][0]);
    problem2 = bbob2009_problem(biobjective_list[combination_idx][1],
                                BBOB2009_DIMS[dimension_idx],
                                (long)instance_list[instance_idx][1]);
    problem = coco_stacked_problem_allocate(problem1, problem2);
    problem->index = problem_index;
    
    return problem; 
}


/* Undefine constants */
#undef BIOBJECTIVE_NUMBER_OF_COMBINATIONS
#undef BIOBJECTIVE_NUMBER_OF_INSTANCES
#undef BIOBJECTIVE_NUMBER_OF_DIMENSIONS
