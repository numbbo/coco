/**
 * @file transform_vars_gallagher_blockrotation.c
 * @brief Implementation of performing a block-rotation transformation on decision values for the gallagher function.
 * The block-rotation is applied only once per call and the result is stored such that the sub-problems can access it
 *
 */

#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "transform_vars_blockrotation_helpers.c"
/* #include "coco_utilities.c" */


/**
 * @brief Data type for transform_vars_gallagher_blockrotation.
 */
typedef struct {
  double *x;
} transform_vars_gallagher_blockrotation_t;

/**
 * @brief Frees the data object.
 */
static void transform_vars_gallagher_blockrotation_free(void *thing) {
  transform_vars_gallagher_blockrotation_t *data = (transform_vars_gallagher_blockrotation_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Data type in problem->versatile_data of f_gallagher.c
 */
typedef struct {
  double *rotated_x;
  size_t number_of_peaks;
  coco_problem_t **sub_problems;
  size_t nb_blocks, *block_sizes, *block_size_map, *first_non_zero_map;
  double **B;
} f_gallagher_versatile_data_t;

/**
 * @brief allows to free the gallagher_versatile_data part of the problem.
 */
static void f_gallagher_versatile_data_free(coco_problem_t *problem) {
  size_t i;
  f_gallagher_versatile_data_t *versatile_data = (f_gallagher_versatile_data_t *) problem->versatile_data;
  if (versatile_data->rotated_x != NULL) {
    coco_free_memory(versatile_data->rotated_x);
  }
  if (versatile_data->sub_problems != NULL) {
    for (i = 0; i < versatile_data->number_of_peaks; i++) {
      coco_problem_free(versatile_data->sub_problems[i]);
    }
    coco_free_memory(versatile_data->sub_problems);
  }
  if (versatile_data->block_sizes != NULL) {
    coco_free_memory(versatile_data->block_sizes);
  }
  if (versatile_data->block_size_map != NULL) {
    coco_free_memory(versatile_data->block_size_map);
  }
  if (versatile_data->first_non_zero_map != NULL) {
    coco_free_memory(versatile_data->first_non_zero_map);
  }
  if (versatile_data->B != NULL) {
    coco_free_block_matrix(versatile_data->B, problem->number_of_variables);
  }
  coco_free_memory(versatile_data);
  problem->versatile_data = NULL;
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}


/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_gallagher_blockrotation_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j;
  transform_vars_gallagher_blockrotation_t *data;
  coco_problem_t *inner_problem;
  f_gallagher_versatile_data_t *versatile_data;

  data = (transform_vars_gallagher_blockrotation_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  versatile_data = (f_gallagher_versatile_data_t *) problem->versatile_data;
  
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    versatile_data->rotated_x[i] = 0;
    for (j = versatile_data->first_non_zero_map[i]; j < versatile_data->first_non_zero_map[i] + versatile_data->block_size_map[i]; ++j) {
      versatile_data->rotated_x[i] += versatile_data->B[i][j - versatile_data->first_non_zero_map[i]] * x[j];
    }
    /*((f_gallagher_versatile_data_t *) problem->versatile_data)->rotated_x[i] = x[i];*/
    data->x[i] = x[i];/* to avoid pointer problems*/
  }

  coco_evaluate_function(inner_problem, data->x, y);/* does not modify the argument of the call since rotated_x will be used later in the sub_problems*/
  /* this function serves only to compute rotated_x on the problem level, not for each sub-problem
   */
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}


static coco_problem_t *transform_vars_gallagher_blockrotation(coco_problem_t *inner_problem) {
  coco_problem_t *problem;
  transform_vars_gallagher_blockrotation_t *data;

  data = (transform_vars_gallagher_blockrotation_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  if (coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_debug("transform_vars_gallagher_blockrotation(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_gallagher_blockrotation_free, "transform_vars_gallagher_blockrotation");
  problem->evaluate_function = transform_vars_gallagher_blockrotation_evaluate;
  return problem;
}


