/**
 * @file rw_gan.c
 *
 * @brief Implementation of the real-world problems of unsupervised learning of a Generative
 * Adversarial Network (GAN) that understands the structure of Super Mario Bros. levels.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"

/**
 * @brief Data type for the rw-gan problem.
 */
typedef struct {
  char *command;
  char *var_fname;
  char *obj_fname;
} rw_gan_data_t;


/**
 * @brief Calls the external function to evaluate the problem.
 */
static void rw_gan_evaluate(coco_problem_t *problem, const double *x, double *y) {

  rw_gan_data_t *data;
  data = (rw_gan_data_t *) problem->data;

  assert(problem->number_of_objectives == 1);
  if (coco_vector_contains_nan(x, problem->number_of_variables))
    y[0] = NAN;

  coco_external_evaluate(x, problem->number_of_variables, data->command, data->var_fname,
      data->obj_fname, y, problem->number_of_objectives);
}

/**
 * @brief Frees the rw_gan_data object.
 */
static void rw_gan_free(coco_problem_t *problem) {
  rw_gan_data_t *data;
  data = (rw_gan_data_t *) problem->data;
  coco_free_memory(data->var_fname);
  coco_free_memory(data->obj_fname);
  coco_free_memory(data->command);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates a rw-gan problem.
 */
static coco_problem_t *rw_gan_problem_allocate(const size_t function,
                                               const size_t dimension,
                                               const size_t instance) {

  coco_problem_t *problem = coco_problem_allocate(dimension, 1, 0);
  rw_gan_data_t *data;
  char dir1[] = "..";
  char dir2[] = "rw-problems";
  char dir3[] = "rw-gan";
  char command_template[COCO_PATH_MAX] = "";
  char *exe_fname, *tmp, *replace;
  FILE *exe_file;
  size_t i;

  for (i = 0; i < dimension; ++i) {
    problem->smallest_values_of_interest[i] = 0; /* TODO */
    problem->largest_values_of_interest[i] = 1;  /* TODO */
  }
  problem->are_variables_integer = NULL;
  problem->evaluate_function = rw_gan_evaluate;
  problem->problem_free_function = rw_gan_free;

  coco_problem_set_id(problem, "rw-gan_f%03lu_i%02lu_d%02lu", function, instance, dimension);
  coco_problem_set_name(problem, "real-world GAN problem f%lu instance %lu in %luD",
      function, instance, dimension);
  coco_problem_set_type(problem, "rw-gan-mario-single");

  data = (rw_gan_data_t *) coco_allocate_memory(sizeof(*data));
  data->var_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(data->var_fname, "", 1);
  coco_join_path(data->var_fname, COCO_PATH_MAX, dir1, dir1, dir2, dir3, "variables.txt", NULL);
  data->obj_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(data->obj_fname, "", 1);
  coco_join_path(data->obj_fname, COCO_PATH_MAX, dir1, dir1, dir2, dir3, "objectives.txt", NULL);

  /* Load the command template from exe_fname, replace <dim> with problem dimension and <fun> with
   * problem function and store the resulting command to the data structure */
  exe_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(exe_fname, "", 1);
  coco_join_path(exe_fname, COCO_PATH_MAX, dir1, dir1, dir2, dir3, "executable_template", NULL);
  exe_file = fopen(exe_fname, "r");
  if (exe_file == NULL) {
    coco_error("rw_gan_problem_allocate(): failed to open file '%s'.", exe_fname);
    return NULL; /* Never reached */
  }
  if (fgets(command_template, COCO_PATH_MAX, exe_file) == NULL) {
    coco_error("rw_gan_problem_allocate(): failed to read file '%s'.", exe_fname);
    return NULL; /* Never reached */
  }
  coco_free_memory(exe_fname);
  replace = coco_strdupf("%lu", (unsigned long)dimension);
  tmp = coco_string_replace(command_template, "<dim>", replace);
  coco_free_memory(replace);
  replace = coco_strdupf("%lu", (unsigned long)function);
  data->command = coco_string_replace(tmp, "<fun>", replace);
  coco_free_memory(replace);
  coco_free_memory(tmp);
  fclose(exe_file);

  problem->data = data;

  /* The best parameter and value are not known yet (TODO) */
  problem->best_value[0] = 0;
  if (problem->best_parameter != NULL) {
    coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
  }

  return problem;
}
