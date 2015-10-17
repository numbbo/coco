#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco_generics.c"

#include "coco_constraints.c"

#include "bbob2009_legacy_code.c"

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

#define MAX_DIM CONSPROB2015_MAX_DIM
#define CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES 5
#define CONSPROB2015_NUMBER_OF_FUNCTIONS 1
#define CONSPROB2015_NUMBER_OF_DIMENSIONS 6
static const unsigned CONSPROB2015_DIMS[] = {2, 3, 5, 10, 20, 40};/*might end up useful outside of consprob2015_decode_problem_index*/

/**
 * consprob2015_decode_problem_index(problem_index, function_id, instance_id,
 *dimension):
 *
 * Decode the new problem_index into the old convention of function,
 * instance and dimension. We have 24 functions in 6 different
 * dimensions so a total of 144 functions and any number of
 * instances. A natural thing would be to order them so that the
 * function varies faster than the dimension which is still faster
 * than the instance. For analysis reasons we want something
 * different. Our goal is to quickly produce 5 repetitions of a single
 * function in one dimension, then vary the function, then the
 * dimension.
 *
 * TODO: this is the default prescription for 2009. This is typically
 *       not what we want _now_, as the instances change in each
 *       workshop. We should have provide-problem-instance-indices
 *       methods to be able to run useful subsets of instances.
 * 
 * This gives us:
 *
 * problem_index | function_id | instance_id | dimension
 * ---------------+-------------+-------------+-----------
 *              0 |           1 |           1 |         2
 *              1 |           1 |           2 |         2
 *              2 |           1 |           3 |         2
 *              3 |           1 |           4 |         2
 *              4 |           1 |           5 |         2
 *              5 |           2 |           1 |         2
 *              6 |           2 |           2 |         2
 *             ...           ...           ...        ...
 *            119 |          24 |           5 |         2
 *            120 |           1 |           1 |         3
 *            121 |           1 |           2 |         3
 *             ...           ...           ...        ...
 *           2157 |          24 |           13|        40
 *           2158 |          24 |           14|        40
 *           2159 |          24 |           15|        40
 *
 * The quickest way to decode this is using integer division and
 * remainders.
 */

static void consprob2015_decode_problem_index(const long problem_index, int *function_id,
                                    long *instance_id, long *dimension) {
  const long high_instance_id =
      problem_index / (CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES * CONSPROB2015_NUMBER_OF_FUNCTIONS *
                        CONSPROB2015_NUMBER_OF_DIMENSIONS);
  long low_instance_id;
  long rest = problem_index % (CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES *
                               CONSPROB2015_NUMBER_OF_FUNCTIONS * CONSPROB2015_NUMBER_OF_DIMENSIONS);
  *dimension =
      CONSPROB2015_DIMS[rest / (CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES * CONSPROB2015_NUMBER_OF_FUNCTIONS)];
  rest = rest % (CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES * CONSPROB2015_NUMBER_OF_FUNCTIONS);
  *function_id = (int)(rest / CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES + 1);
  rest = rest % CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES;
  low_instance_id = rest + 1;
  *instance_id = low_instance_id + 5 * high_instance_id;
}

/* Encodes a triplet of (function_id, instance_id, dimension_idx) into a problem_index
 * The problem index can, then, be used to directly generate a problem
 * It helps allow easier control on instances, functions and dimensions one wants to run
 * all indices start from 0 TODO: start at 1 instead?
 */
static long consprob2015_encode_problem_index(int function_id, long instance_id, int dimension_idx){
    long cycleLength = CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES * CONSPROB2015_NUMBER_OF_FUNCTIONS * CONSPROB2015_NUMBER_OF_DIMENSIONS;
    long tmp1 = instance_id % CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES;
    long tmp2 = function_id * CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES;
    long tmp3 = dimension_idx * (CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES * CONSPROB2015_NUMBER_OF_FUNCTIONS);
    long tmp4 = ((long)(instance_id / CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES) ) * cycleLength; /* just for safety */
    
    return tmp1 + tmp2 + tmp3 + tmp4;
}

static void consprob2015_copy_rotation_matrix(double **rot, double *M, double *b,
                                          const size_t dimension) {
  size_t row, column;
  double *current_row;

  for (row = 0; row < dimension; ++row) {
    current_row = M + row * dimension;
    for (column = 0; column < dimension; ++column) {
      current_row[column] = rot[row][column];
    }
    b[row] = 0.0;
  }
}

static coco_problem_t *consprob2015_problem(int function_id, long dimension_, long instance_id) {
  size_t len;
  long rseed;
  coco_problem_t *problem = NULL;
  const size_t dimension = (unsigned long) dimension_;
  
  /* This assert is a hint for the static analyzer. */
  assert(dimension > 1);
  if (dimension > MAX_DIM)
    coco_error("consprob2015_suite currently supports dimension up to %ld (%ld given)", 
        MAX_DIM, dimension);

#if 0
  {  /* to be removed */
    int dimension_idx;
    switch (dimension) {/*TODO: make this more dynamic*//* This*/
            case 2:
            dimension_idx = 0;
            break;
            case 3:
            dimension_idx = 1;
            break;
            case 5:
            dimension_idx = 2;
            break;
            case 10:
            dimension_idx = 3;
            break;
            case 20:
            dimension_idx = 4;
            break;
            case 40:
            dimension_idx = 5;
            break;
        default:
            dimension_idx = -1;
            break;
    }
    assert(problem_index == consprob2015_encode_problem_index(function_id - 1, instance_id - 1 , dimension_idx));
  }
#endif 
  rseed = function_id + 10000 * instance_id;

  /* Break if we are past our 15 instances. */
  if (instance_id > 15)
    return NULL;

  if (function_id == 1) {
    double xopt[MAX_DIM], fopt;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    fopt = bbob2009_compute_fopt(function_id, instance_id);

    problem = sphere_problem(dimension);
    problem = coco_add_constraints(problem, "linear_constraint");
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else {
    return NULL;
  }

  /* Now set the problem name and problem id of the final problem */
  coco_free_memory(problem->problem_name);
  coco_free_memory(problem->problem_id);

  /* Construct a meaningful problem id */
  len = snprintf(NULL, 0, "consprob2015_f%02i_i%02li_d%02lu", function_id,
                 instance_id, dimension);
  problem->problem_id = coco_allocate_memory(len + 1);
  snprintf(problem->problem_id, len + 1, "consprob2015_f%02i_i%02li_d%02lu",
           function_id, instance_id, dimension);

  len = snprintf(NULL, 0, "CONSPROB2015 f%02i instance %li in %luD", function_id,
                 instance_id, dimension);
  problem->problem_name = coco_allocate_memory(len + 1);
  snprintf(problem->problem_name, len + 1, "CONSPROB2015 f%02i instance %li in %luD",
           function_id, instance_id, dimension);
  return problem;
}

/* Return the consprob2015 function id of the problem or -1 if it is not a consprob2015
 * problem. */
static int consprob2015_get_function_id(const coco_problem_t *problem) {
  static const char *bbob_prefix = "consprob2015_";
  const char *problem_id = coco_get_problem_id(problem);
  assert(strlen(problem_id) >= 20);

  if (strncmp(bbob_prefix, problem_id, strlen(bbob_prefix)) != 0) {
    return -1;
  }

  /* OME: Ugly hardcoded extraction. In a perfect world, we would
   * parse the problem id by splitting on _ and then finding the 'f'
   * field. Instead, we cound out the position of the function id in
   * the string
   *
   *   01234567890123456789
   *   consprob2015_fXX_iYY_dZZ
   */
  return (problem_id[10] - '0') * 10 + (problem_id[11] - '0');
}

/* Return the consprob2015 instance id of the problem or -1 if it is not a consprob2015
 * problem. */
static int consprob2015_get_instance_id(const coco_problem_t *problem) {
  static const char *bbob_prefix = "consprob2015_";
  const char *problem_id = coco_get_problem_id(problem);
  assert(strlen(problem_id) >= 20);

  if (strncmp(bbob_prefix, problem_id, strlen(bbob_prefix)) != 0) {
    return -1;
  }

  /* OME: Ugly hardcoded extraction. In a perfect world, we would
   * parse the problem id by splitting on _ and then finding the 'i'
   * field. Instead, we cound out the position of the instance id in
   * the string
   *
   *   01234567890123456789
   *   consprob2015_fXX_iYY_dZZ
   */
  return (problem_id[14] - '0') * 10 + (problem_id[15] - '0');
}

/* TODO: specify selection_descriptor and implement
 *
 * Possible example for a descriptor: "instance:1-5, dimension:-20",
 * where instances are relative numbers (w.r.t. to the instances in
 * test bed), dimensions are absolute.
 *
 * Return successor of problem_index or first index if problem_index < 0 or -1 otherwise.
 *
 * Details: this function is not necessary unless selection is implemented. 
*/
static long consprob2015_next_problem_index(long problem_index, const char *selection_descriptor) {
  const long first_index = 0;
  const long last_index = 2159;
  
  if (problem_index < 0)
    problem_index = first_index - 1;
    
  if (strlen(selection_descriptor) == 0) {
    if (problem_index < last_index)
      return problem_index + 1;
    return -1;
  }
  
  /* TODO:
     o parse the selection_descriptor -> value bounds on funID, dimension, instance
     o inrement problem_index until funID, dimension, instance match the restrictions
       or max problem_index is succeeded. 
    */
  
  coco_error("next_problem_index is yet to be implemented for specific selections");
  return -1;
}

/**
 * consprob2015_suite(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from the CONSPROB2015
 * benchmark suit. If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *consprob2015_suite(long problem_index) {
  coco_problem_t *problem;
  int function_id;
  long dimension, instance_id;
  
  if (problem_index < 0)
    return NULL; 
  
  consprob2015_decode_problem_index(problem_index, &function_id, &instance_id,
                                 &dimension);
  
  problem = consprob2015_problem(function_id, dimension, instance_id);
  
  problem->index = problem_index;
  
  return problem;
}

/* Undefine constants */
#undef MAX_DIM
#undef CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES 
#undef CONSPROB2015_NUMBER_OF_FUNCTIONS 
#undef CONSPROB2015_NUMBER_OF_DIMENSIONS 
