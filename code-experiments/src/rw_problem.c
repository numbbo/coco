/**
 * @file rw_problem.c
 *
 * @brief Implementation of the real-world problem data type and related functions.
 */
#include "coco.h"
#include "coco_platform.h"

/**
 * @brief Data type used by the real-world problems.
 */
typedef struct {
  char *command;
  char *var_fname;
  char *obj_fname;
} rw_problem_data_t;


/**
 * @brief Frees the rw_problem_data_t object.
 */
static void rw_problem_data_free(coco_problem_t *problem) {
  rw_problem_data_t *data;
  data = (rw_problem_data_t *) problem->data;
  coco_free_memory(data->var_fname);
  coco_free_memory(data->obj_fname);
  coco_free_memory(data->command);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Calls the external function to evaluate the problem.
 */
static void rw_problem_evaluate(coco_problem_t *problem, const double *x, double *y) {

  rw_problem_data_t *data;
  size_t i;
  data = (rw_problem_data_t *) problem->data;

  if (coco_vector_contains_nan(x, problem->number_of_variables))
    for (i = 0; i < problem->number_of_objectives; i++)
      y[i] = NAN;

  coco_external_evaluate(x, problem->number_of_variables, data->command, data->var_fname,
      data->obj_fname, y, problem->number_of_objectives);
}

static rw_problem_data_t *get_rw_problem_data(const char *folder_name,
                                              const size_t objectives,
                                              const size_t function,
                                              const size_t dimension,
                                              const size_t instance) {

  rw_problem_data_t *data;
  char dir1[] = "..";
  char dir2[] = "rw-problems";
  char *str1, *str2, *str3;
  char *exe_fname;
  FILE *exe_file;

  data = (rw_problem_data_t *) coco_allocate_memory(sizeof(*data));
  data->var_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(data->var_fname, "", 1);
  coco_join_path(data->var_fname, COCO_PATH_MAX, dir1, dir1, dir2, folder_name, "variables.txt", NULL);
  data->obj_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(data->obj_fname, "", 1);
  coco_join_path(data->obj_fname, COCO_PATH_MAX, dir1, dir1, dir2, folder_name, "objectives.txt", NULL);

  /* Load the command template from exe_fname, replace:
   * <obj> with the problem objectives,
   * <dim> with the problem dimension,
   * <fun> with the problem function and
   * <inst> with the problem instance
   * and store the resulting command to data->command */
  exe_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(exe_fname, "", 1);
  coco_join_path(exe_fname, COCO_PATH_MAX, dir1, dir1, dir2, folder_name, "evaluate_function_template", NULL);
  exe_file = fopen(exe_fname, "r");
  if (exe_file == NULL) {
    coco_error("rw_gan_mario_problem_allocate(): failed to open file '%s'.", exe_fname);
    return NULL; /* Never reached */
  }
  str2 = coco_allocate_string(COCO_PATH_MAX + 1);
  /* Store the contents of the exe_file to template2 */
  str1 = fgets(str2, COCO_PATH_MAX, exe_file);
  if (str1 == NULL) {
    coco_error("rw_gan_mario_problem_allocate(): failed to read file '%s'.", exe_fname);
    return NULL; /* Never reached */
  }
  assert(str1 == str2);
  /* Replace <obj> with objectives */
  str3 = coco_strdupf("%lu", (unsigned long)objectives);
  str1 = coco_string_replace(str2, "<obj>", str3);
  coco_free_memory(str3);
  coco_free_memory(str2);
  /* Replace <dim> with dimension */
  str3 = coco_strdupf("%lu", (unsigned long)dimension);
  str2 = coco_string_replace(str1, "<dim>", str3);
  coco_free_memory(str3);
  coco_free_memory(str1);
  /* Replace <fun> with function */
  str3 = coco_strdupf("%lu", (unsigned long)function);
  str1 = coco_string_replace(str2, "<fun>", str3);
  coco_free_memory(str3);
  coco_free_memory(str2);
  /* Replace <inst> with instance */
  str3 = coco_strdupf("%lu", (unsigned long)instance);
  data->command = coco_string_replace(str1, "<inst>", str3);
  coco_free_memory(str3);
  coco_free_memory(str1);

  fclose(exe_file);
  coco_free_memory(exe_fname);
  return data;
}
