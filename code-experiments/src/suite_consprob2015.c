#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco_generics.c"
#include "coco_constraints.c"

#include "f_attractive_sector.c"
#include "f_bbob_step_ellipsoid.c"
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
#include "f_weierstrass.c"

#include "suite_bbob2009_legacy_code.c"
#include "transform_obj_oscillate.c"
#include "transform_obj_penalize.c"
#include "transform_obj_power.c"
#include "transform_obj_shift.c"
#include "transform_vars_affine.c"
#include "transform_vars_asymmetric.c"
#include "transform_vars_brs.c"
#include "transform_vars_conditioning.c"
#include "transform_vars_oscillate.c"
#include "transform_vars_scale.c"
#include "transform_vars_shift.c"
#include "transform_vars_x_hat.c"
#include "transform_vars_z_hat.c"

#define MAX_DIM CONSPROB2015_MAX_DIM
#define CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES 5
#define CONSPROB2015_NUMBER_OF_FUNCTIONS 1
#define CONSPROB2015_NUMBER_OF_DIMENSIONS 6
static const unsigned CONSPROB2015_DIMS[] = {2, 3, 5, 10, 20, 40};

/**
 * consprob2015_decode_problem_index(problem_index, function_id, instance_id,
 *dimension):
 */

static void suite_consprob2015_decode_problem_index(const long problem_index, int *function_id,
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
static long suite_consprob2015_encode_problem_index(int function_id, long instance_id, int dimension_idx){
    long cycleLength = CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES * CONSPROB2015_NUMBER_OF_FUNCTIONS * CONSPROB2015_NUMBER_OF_DIMENSIONS;
    long tmp1 = instance_id % CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES;
    long tmp2 = function_id * CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES;
    long tmp3 = dimension_idx * (CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES * CONSPROB2015_NUMBER_OF_FUNCTIONS);
    long tmp4 = ((long)(instance_id / CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES) ) * cycleLength; /* just for safety */
    
    return tmp1 + tmp2 + tmp3 + tmp4;
}

static coco_problem_t *suite_consprob2015_problem(int function_id, long dimension_, long instance_id) {
	
  size_t len, i, j, k;
  long rseed_xopt, *rseed_cons;
  coco_problem_t *problem = NULL;
  const size_t dimension = (unsigned long) dimension_;
  
  /* This assert is a hint for the static analyzer. */
  assert(dimension > 1);
  if (dimension > MAX_DIM)
    coco_error("consprob2015_suite currently supports dimension up to %ld (%ld given)", 
        MAX_DIM, dimension);
        
  rseed_xopt = function_id + 10000 * instance_id;

  /* Break if we are past our 15 instances. */
  if (instance_id > 15)
    return NULL;

  if (function_id == 1) {
    
    size_t number_of_constraints = 5;
    double xopt[MAX_DIM];
    double **rotation_matrix;
    double *gradient, *shift, *origin, *e1;
    coco_problem_t *problem_c1, *problem_c2;
    
    rseed_cons = coco_allocate_vector(number_of_constraints);
    
    /* Define a seed for each linear constraint */
    for (i = 0; i < number_of_constraints; i++) { 
		 rseed_cons[i] =  function_id + 10000 * instance_id + 10000 * i;
	 }
    
    /* Define the shift vector as -xopt */
    bbob2009_compute_xopt(xopt, rseed_xopt, dimension);
    shift = coco_duplicate_vector(xopt, dimension);
    for (i = 0; i < dimension; i++) { shift[i] *= -1.0; }
    
    /* Define the origin vector */
    origin = coco_allocate_vector(dimension);
    for (i = 0; i < dimension; i++) { origin[i] = 0.0; } 
    
    /* Define the objective function of the problem */
    problem = sphere_problem(dimension);
    problem = shift_variables(problem, shift, 0);
    
    /* Compute the gradient of the shifted objective function
     * at the origin, or, equivalently, the gradient of the original 
     * function at -xopt, and store it into the vector "gradient"
     */
    gradient = coco_allocate_vector(dimension);
    coco_evaluate_gradient(problem, origin, gradient);
    
    /* Define the gradient of the first linear constraint as the
     * negative gradient of the shifted objective function at the 
     * origin 
     */
    for (i = 0; i < dimension; i++) { gradient[i] *= -1.0; }
    problem_c1 = linear_constraint_problem(dimension, gradient);
    
    /* Define the vector e1 = [1, 0, ... 0] */
    e1 = coco_allocate_vector(dimension);
    e1[0] = 1.0;
    for (i = 1; i < dimension; i++) { e1[i] = 0.0; }
    
    rotation_matrix = bbob2009_allocate_matrix(dimension, dimension);
    
    /* Use a rotation matrix to randomly generate the constraints
     * of the problem */
    for (i = 0; i < number_of_constraints; i++) {
      
      bbob2009_compute_rotation(rotation_matrix, rseed_cons[i], dimension);
		
		/* Define the gradient of the new linear constraint as the result
		 * of the multiplication of the rotation matrix by the vector e1
		 */
		for (j = 0; j < dimension; j++) {
				gradient[i] = 0.0;
				for (k = 0; k < dimension; k++) {
						gradient[j] += rotation_matrix[j][k] * e1[k];
				}
		}
		problem_c2 = linear_constraint_problem(dimension, gradient);
		problem_c2 = guarantee_feasible_point(problem_c2, problem_c1);
		
		/* Stack "problem_c2" and "problem_c1" */
		problem_c1 = coco_stacked_problem_allocate(problem_c1, problem_c2);
    }
    
    /* Create the linearly-constrained problem by stacking "problem" and 
     * "problem_c1"
     */
    problem = coco_stacked_problem_allocate(problem, problem_c1);
    
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
static int suite_consprob2015_get_function_id(const coco_problem_t *problem) {
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
static int suite_consprob2015_get_instance_id(const coco_problem_t *problem) {
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
static long suite_consprob2015_next_problem_index(long problem_index, const char *selection_descriptor) {
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
static coco_problem_t *suite_consprob2015(long problem_index) {
  coco_problem_t *problem;
  int function_id;
  long dimension, instance_id;
  
  if (problem_index < 0)
    return NULL; 
  
  suite_consprob2015_decode_problem_index(problem_index, &function_id, &instance_id,
                                 &dimension);
  
  problem = suite_consprob2015_problem(function_id, dimension, instance_id);
  
  problem->index = problem_index;
  
  return problem;
}

/* Undefine constants */
#undef MAX_DIM
#undef CONSPROB2015_NUMBER_OF_CONSECUTIVE_INSTANCES 
#undef CONSPROB2015_NUMBER_OF_FUNCTIONS 
#undef CONSPROB2015_NUMBER_OF_DIMENSIONS 
