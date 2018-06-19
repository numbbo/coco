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
  char *path;
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
  coco_free_memory(data->path);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Performs external evaluation of x (writes x to in_fname, calls the process given in
 * path/command, reads the result from out_fname and saves it to y).
 *
 * The first line in in_fname contains the size of x. Each next line contains the values of x.
 * The first line in out_fname should contain the size of y. Each next line contains the values of y.
 * If the first line in out_fname is 0, all elements of y are set to NAN.
 */
void rw_problem_external_evaluate(const double *x,
                                  const size_t size_of_x,
                                  const char *path,
                                  const char *command,
                                  const char *in_fname,
                                  const char *out_fname,
                                  double *y,
                                  const size_t expected_size_of_y) {
  size_t i = 0;
  double result;
  FILE *in_file;
  FILE *out_file;
  const int precision_x = 8;
  int process_result;
  int scan_result, read_size_of_y;

#if defined(USES_CREATEPROCESS)
  STARTUPINFO StartupInfo;
  PROCESS_INFORMATION ProcessInfo;
  StartupInfo.cb = sizeof(STARTUPINFO);
  StartupInfo.dwFlags = STARTF_USESHOWWINDOW || STARTF_FORCEONFEEDBACK;
  StartupInfo.wShowWindow = SW_HIDE;
  StartupInfo.lpReserved = NULL;
  StartupInfo.lpDesktop = NULL;
  StartupInfo.lpTitle = NULL;
  StartupInfo.cbReserved2 = 0;
  StartupInfo.lpReserved2 = NULL;
  char *path_dup = coco_strdup(path);
#elif defined(USES_EXECVP)
  pid_t process_id;
  int status;
  char **argv = coco_string_split(command, ' ');
#else
  char *entire_command;
#endif

  /* Writes x to file */
  in_file = fopen(in_fname, "w");
  if (in_file == NULL) {
    coco_error("rw_problem_run_process(): failed to open file '%s'.", in_fname);
  }
  fprintf(in_file,"%lu\n", (unsigned long)size_of_x);
  for (i = 0; i < size_of_x; ++i) {
    fprintf(in_file, "%.*e\n", precision_x, x[i]);
  }
  fclose(in_file);

#ifdef USES_CREATEPROCESS
  /* Call the process Windows-style */
  process_result = CreateProcess(NULL, command, NULL, NULL, 0,
    NORMAL_PRIORITY_CLASS, NULL, path_dup, &StartupInfo, &ProcessInfo);

  do {} while (WaitForSingleObject(ProcessInfo.hProcess, 100) == WAIT_TIMEOUT);

  if (!process_result) {
    coco_error("rw_problem_external_evaluate(): failed to execute '%s'. Error: %d", command,
        GetLastError());
  }

  CloseHandle(ProcessInfo.hProcess);
  CloseHandle(ProcessInfo.hThread);
  coco_free_memory(path_dup);

#elif defined(USES_EXECVP)
  /* Call the process Linux-style */
  if ((process_id = fork()) == 0) {

    chdir(path);
    process_result = execvp(argv[0], argv);

    /* Free argv */
    for (i = 0; *(argv + i); i++)
      coco_free_memory(*(argv + i));
    coco_free_memory(argv);

    if (process_result < 0) {
      coco_error("rw_problem_external_evaluate(): failed to execute '%s'. Error: %d", command,
          errno);
    }

  } else if (process_id < 0) {
    coco_error("rw_problem_external_evaluate(): fork error");
  } else {
    /* Wait for the process to complete */
    while (wait(&status) != process_id) {}
  }
#else
  /* Executes external evaluation with system() although
     https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152177
     warns against using system() */
  entire_command = coco_strdupf("cd %s; %s", path, command);
  process_result = system(entire_command);
  if (process_result == -1) {
    coco_error("rw_problem_external_evaluate(): failed to execute '%s'.", command);
  }
  else if (process_result != 0) {
    coco_error("rw_problem_external_evaluate(): '%s' completed with error '%d'.", command,
        process_result);
  }
  coco_free_memory(entire_command);
#endif

  /* Reads the values of y from file */
  out_file = fopen(out_fname, "r");
  if (out_file == NULL) {
    coco_error("rw_problem_external_evaluate(): failed to open file '%s'.", out_fname);
  }
  scan_result = fscanf(out_file, "%d\n", &read_size_of_y);
  if (scan_result != 1) {
    coco_error("rw_problem_external_evaluate(): failed to read from '%s'.", out_fname);
  }

  if (read_size_of_y == 0) {
    /* x could not be evaluated */
    coco_vector_set_to_nan(y, expected_size_of_y);
    return;
  }

  if (read_size_of_y != expected_size_of_y) {
    coco_error("rw_problem_external_evaluate(): '%s' contains %lu elements instead of %lu.",
        out_fname, (unsigned long)read_size_of_y, (unsigned long)expected_size_of_y);
  }

  for (i = 0; i < read_size_of_y; i++) {
    scan_result = fscanf(out_file, "%lf\n", &result);
    if (scan_result != 1) {
      coco_error("rw_problem_external_evaluate(): failed to read from '%s'.", out_fname);
    }
    y[i] = result;
  }
  fclose(out_file);
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

  rw_problem_external_evaluate(x, problem->number_of_variables, data->path, data->command,
      data->var_fname, data->obj_fname, y, problem->number_of_objectives);
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
  data->path = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(data->path, "", 1);
  coco_join_path(data->path, COCO_PATH_MAX, dir1, dir1, dir2, folder_name, NULL);
  data->var_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(data->var_fname, "", 1);
  coco_join_path(data->var_fname, COCO_PATH_MAX, data->path, "variables.txt", NULL);
  data->obj_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(data->obj_fname, "", 1);
  coco_join_path(data->obj_fname, COCO_PATH_MAX, data->path, "objectives.txt", NULL);

  /* Load the command template from exe_fname, replace:
   * <obj> with the problem objectives,
   * <dim> with the problem dimension,
   * <fun> with the problem function and
   * <inst> with the problem instance
   * and store the resulting command to data->command */
  exe_fname = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(exe_fname, "", 1);
  coco_join_path(exe_fname, COCO_PATH_MAX, data->path, "evaluate_function_template", NULL);
  exe_file = fopen(exe_fname, "r");
  if (exe_file == NULL) {
    coco_error("rw_problem_allocate(): failed to open file '%s'.", exe_fname);
    return NULL; /* Never reached */
  }
  str2 = coco_allocate_string(COCO_PATH_MAX + 1);
  /* Store the contents of the exe_file to template2 */
  str1 = fgets(str2, COCO_PATH_MAX, exe_file);
  if (str1 == NULL) {
    coco_error("rw_problem_allocate(): failed to read file '%s'.", exe_fname);
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
