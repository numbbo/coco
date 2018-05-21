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

/**
 * @brief Data type for the rw-gan problem.
 */
typedef struct {
  char command[COCO_PATH_MAX]; /* Abusing COCO_PATH_MAX... */
  char *var_fname;
  char *obj_fname;
} rw_gan_data_t;

/**
 * @brief Writes x to var_fname, calls the process given in exe_fname and then reads the result
 * from obj_fname.
 *
 * The first line in var_fname contains the number of variables. Each next line contains values
 * for each variable.
 * The first line in obj_fname should contain the number of objectives. Each next line contains
 * values for each objective.
 */
static double rw_external_evaluate(const double *x,
                                   const size_t number_of_variables,
                                   const char *command,
                                   const char *var_fname,
                                   const char *obj_fname) {

  size_t i = 0;
  double result;
  FILE *var_file;
  FILE *obj_file;
  const int precision_x = 8;
  int system_result, scan_result, num_objs;

  /* Writes variables to file */
  var_file = fopen(var_fname, "w");
  if (var_file == NULL) {
    coco_error("rw_external_evaluate(): failed to open file '%s'.", var_fname);
    return 0; /* Never reached */
  }
  fprintf(var_file,"%lu\n", (unsigned long)number_of_variables);
  for (i = 0; i < number_of_variables; ++i) {
    fprintf(var_file, "%.*e\n", precision_x, x[i]);
  }
  fclose(var_file);

  /* Executes external evaluation with system() although
     https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152177
     warns against using system() */
  system_result = system(command);
  if (system_result == -1) {
    coco_error("rw_external_evaluate(): failed to execute '%s'.", command);
    return 0; /* Never reached */
  }
  else if (system_result != 0) {
    coco_error("rw_external_evaluate(): '%s' completed with error '%d'.", command, system_result);
    return 0; /* Never reached */
  }

  /* Reads the objective from file */
  obj_file = fopen(obj_fname, "r");
  if (obj_file == NULL) {
    coco_error("rw_external_evaluate(): failed to open file '%s'.", obj_fname);
    return 0; /* Never reached */
  }
  scan_result = fscanf(obj_file, "%d\n", &num_objs);
  if (scan_result != 1) {
    coco_error("rw_external_evaluate(): failed to read from '%s'.", obj_fname);
    return 0; /* Never reached */
  }
  assert(num_objs == 1);
  scan_result = fscanf(obj_file, "%lf\n", &result);
  if (scan_result != 1) {
    coco_error("rw_external_evaluate(): failed to read from '%s'.", obj_fname);
    return 0; /* Never reached */
  }
  fclose(obj_file);

  return result;
}

/**
 * @brief Calls the external function to evaluate the problem.
 */
static void rw_gan_evaluate(coco_problem_t *problem, const double *x, double *y) {

  rw_gan_data_t *data;
  data = (rw_gan_data_t *) problem->data;

  assert(problem->number_of_objectives == 1);
  if (coco_vector_contains_nan(x, problem->number_of_variables))
    y[0] = NAN;

  y[0] = rw_external_evaluate(x, problem->number_of_variables, data->command,
      data->var_fname, data->obj_fname);
}

/**
 * @brief Frees the rw_gan_data object.
 */
static void rw_gan_free(coco_problem_t *problem) {
  rw_gan_data_t *data;
  data = (rw_gan_data_t *) problem->data;
  coco_free_memory(data->var_fname);
  coco_free_memory(data->obj_fname);
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
  char *exe_fname;
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

  /* Store the command from exe_fname */
  exe_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(exe_fname, "", 1);
  coco_join_path(exe_fname, COCO_PATH_MAX, dir1, dir1, dir2, dir3, "executable.txt", NULL);
  exe_file = fopen(exe_fname, "r");
  if (exe_file == NULL) {
    coco_error("rw_gan_problem_allocate(): failed to open file '%s'.", exe_fname);
    return 0; /* Never reached */
  }
  if (fgets(data->command, COCO_PATH_MAX, exe_file) == NULL) {
    coco_error("rw_gan_problem_allocate(): failed to read file '%s'.", exe_fname);
    return 0; /* Never reached */
  }
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
