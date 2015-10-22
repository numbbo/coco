#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <errno.h>

#include "coco.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

static int consprob2015_logger_verbosity = 3;  /* TODO: make this an option the user can modify */
static int raisedOptValWarning;
static int consprob2015_get_function_id(const coco_problem_t *problem);
static int consprob2015_get_instance_id(const coco_problem_t *problem);

/* FIXME: these names could easily created conflicts with other coco.c-global names. Use consprob2015 as prefix to prevent conflicts. */
static const size_t consprob2015_nbpts_nbevals = 20;
static const size_t consprob2015_nbpts_fval = 5;
static size_t consprob2015_current_dim = 0;
static long consprob2015_current_funId = 0;
static int consprob2015_infoFile_firstInstance = 0;
char consprob2015_infoFile_firstInstance_char[3];
/*a possible solution: have a list of dims that are already in the file, if the ones we're about to log is != consprob2015_current_dim and the funId is currend_funId, create a new .info file with as suffix the number of the first instance */
static const int consprob2015_number_of_dimensions = 6;
static size_t consprob2015_dimensions_in_current_infoFile[6] = {0,0,0,0,0,0}; /*TODO should use BBOB2009_NUMBER_OF_DIMENSIONS*/


/* The current_... mechanism fails if several problems are open. 
 * For the time being this should lead to an error.
 *
 * A possible solution: consprob2015_logger_is_open becomes a reference
 * counter and as long as another logger is open, always a new info
 * file is generated. 
 */
static int consprob2015_logger_is_open = 0;  /* this could become lock-list of .info files */

/*TODO: add possibility of adding a prefix to the index files*/

typedef struct {
  int is_initialized;
  char *path; /*relative path to the data folder. Simply the Algname*/
  const char *
      alg_name;      /*the alg name, for now, temporarly the same as the path*/
  FILE *index_file;  /*index file*/
  FILE *fdata_file;  /*function value aligned data file*/
  FILE *tdata_file;  /*number of function evaluations aligned data file*/
  FILE *rdata_file;  /*restart info data file*/
  double f_trigger;  /* next upper bound on the fvalue to trigger a log in the
                        .dat file*/
  long t_trigger;    /* next lower bound on nb fun evals to trigger a log in the
                        .tdat file*/
  int idx_f_trigger; /* allows to track the index i in logging target =
                        {10**(i/consprob2015_nbpts_fval), i \in Z} */
  int idx_t_trigger; /* allows to track the index i in logging nbevals  =
                        {int(10**(i/consprob2015_nbpts_nbevals)), i \in Z} */
  int idx_tdim_trigger; /* allows to track the index i in logging nbevals  =
                           {dim * 10**i, i \in Z} */
  long number_of_evaluations;
  double best_fvalue;
  double last_fvalue;
  short written_last_eval; /*allows writing the the data of the final fun eval
                              in the .tdat file if not already written by the
                              t_trigger*/
  double *best_solution;
  /*the following are to only pass data as a parameter in the free function. The
   * interface should probably be the same for all free functions so passing the
   * problem as a second parameter is not an option even though we need info
   * form it.*/
  int function_id; /*TODO: consider changing name*/
  int instance_id;
  size_t number_of_variables;
  double optimal_fvalue;
} consprob2015_logger_t; 

static const char *consprob2015_file_header_str = "%% function evaluation | "
                                      "noise-free fitness - Fopt (%13.12e) | "
                                      "best noise-free fitness - Fopt | "
                                      "measured fitness | "
                                      "best measured fitness | "
                                      "x1 | "
                                      "x2...\n";

static void _consprob2015_logger_update_f_trigger(consprob2015_logger_t *data,
                                              double fvalue) {
  /* "jump" directly to the next closest (but larger) target to the
   * current fvalue from the initial target
   */

  if (fvalue - data->optimal_fvalue <= 0.) {
    data->f_trigger = -DBL_MAX;
  } else {
    if (data->idx_f_trigger == INT_MAX) { /* first time*/
      data->idx_f_trigger =
          (int)(ceil(log10(fvalue - data->optimal_fvalue)) * consprob2015_nbpts_fval);
    } else { /* We only call this function when we reach the current f_trigger*/
      data->idx_f_trigger--;
    }
    data->f_trigger = pow(10, data->idx_f_trigger * 1.0 / consprob2015_nbpts_fval);
    while (fvalue - data->optimal_fvalue <= data->f_trigger) {
      data->idx_f_trigger--;
      data->f_trigger = pow(10, data->idx_f_trigger * 1.0 / consprob2015_nbpts_fval);
    }
  }
}

static void _consprob2015_logger_update_t_trigger(consprob2015_logger_t *data,
                                              size_t number_of_variables) {
  while (data->number_of_evaluations >=
         floor(pow(10, (double)data->idx_t_trigger / (double)consprob2015_nbpts_nbevals)))
    data->idx_t_trigger++;

  while (data->number_of_evaluations >=
         number_of_variables * pow(10, (double)data->idx_tdim_trigger))
    data->idx_tdim_trigger++;

  data->t_trigger =
      (long)fmin(floor(pow(10, (double)data->idx_t_trigger / (double)consprob2015_nbpts_nbevals)),
           number_of_variables * pow(10, (double)data->idx_tdim_trigger));
}

/**
 * adds a formated line to a data file
 */
static void _consprob2015_logger_write_data(FILE *target_file,
                                        long number_of_evaluations,
                                        double fvalue, double best_fvalue,
                                        double best_value, const double *x,
                                        size_t number_of_variables) {
  /* for some reason, it's %.0f in the old code instead of the 10.9e
   * in the documentation
   */
  fprintf(target_file, "%ld %+10.9e %+10.9e %+10.9e %+10.9e",
          number_of_evaluations, fvalue - best_value, best_fvalue - best_value,
          fvalue, best_fvalue);
  if (number_of_variables < 22) {
    size_t i;
    for (i = 0; i < number_of_variables; i++) {
      fprintf(target_file, " %+5.4e", x[i]);
    }
  }
  fprintf(target_file, "\n");
}

/**
 * Error when trying to create the file "path"
 */
static void _consprob2015_logger_error_io(FILE *path, int errnum) {
  char *buf;
  const char *error_format = "Error opening file: %s\n ";
                             /*"consprob2015_logger_prepare() failed to open log "
                             "file '%s'.";*/
  size_t buffer_size = (size_t)(snprintf(NULL, 0, error_format, path));/*to silence warning*/
  buf = (char *)coco_allocate_memory(buffer_size);
  snprintf(buf, buffer_size, error_format, strerror(errnum), path);
  coco_error(buf);
  coco_free_memory(buf);
}

/**
 * Creates the data files or simply opens it
 */

/*
 calling sequence:
 _consprob2015_logger_open_dataFile(&(data->fdata_file), data->path, dataFile_path,
 ".dat");
 */

static void _consprob2015_logger_open_dataFile(FILE **target_file, const char *path,
                                           const char *dataFile_path,
                                           const char *file_extension) {
    char file_path[NUMBBO_PATH_MAX] = {0};
    char relative_filePath[NUMBBO_PATH_MAX] = {0};
    int errnum;
    strncpy(relative_filePath, dataFile_path,
            NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
    strncat(relative_filePath, file_extension,
            NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
    coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
    if (*target_file == NULL) {
        *target_file = fopen(file_path, "a+");
        errnum = errno;
        if (*target_file == NULL) {
            _consprob2015_logger_error_io(*target_file, errnum);
        }
    }
}

/*
static void _consprob2015_logger_open_dataFile(FILE **target_file, const char *path,
                                           const char *dataFile_path,
                                           const char *file_extension) {
  char file_path[NUMBBO_PATH_MAX] = {0};
  char relative_filePath[NUMBBO_PATH_MAX] = {0};
  int errnum;
  strncpy(relative_filePath, dataFile_path,
          NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
  strncat(relative_filePath, file_extension,
          NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
  if (*target_file == NULL) {
    *target_file = fopen(file_path, "a+");
    errnum = errno;
    if (*target_file == NULL) {
      _consprob2015_logger_error_io(*target_file, errnum);
    }
  }
}*/

/**
 * Creates the index file fileName_prefix+problem_id+file_extension in
 * folde_path
 */
static void _consprob2015_logger_openIndexFile(consprob2015_logger_t *data,
                                           const char *folder_path,
                                           const char *indexFile_prefix,
                                           const char *function_id,
                                           const char *dataFile_path) {
    /*to add the instance number TODO: this should be done outside to avoid redoing this for the .*dat files */
    char used_dataFile_path[NUMBBO_PATH_MAX] = {0};
    int errnum, newLine;/*newLine is at 1 if we need a new line in the info file*/
    char function_id_char[3];/*TODO: consider adding them to data*/
    char file_name[NUMBBO_PATH_MAX] = {0};
    char file_path[NUMBBO_PATH_MAX] = {0};
    FILE **target_file;
    FILE *tmp_file;
    strncpy(used_dataFile_path, dataFile_path, NUMBBO_PATH_MAX - strlen(used_dataFile_path) - 1);
    if (consprob2015_infoFile_firstInstance == 0) {
        consprob2015_infoFile_firstInstance = data->instance_id;
    }
    sprintf(function_id_char, "%d", data->function_id);
    sprintf(consprob2015_infoFile_firstInstance_char, "%d", consprob2015_infoFile_firstInstance);
    target_file = &(data->index_file);
    tmp_file = NULL; /*to check whether the file already exists. Don't want to use
           target_file*/
    strncpy(file_name, indexFile_prefix, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, "_f", NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, function_id_char, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, "_i", NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, consprob2015_infoFile_firstInstance_char, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, ".info", NUMBBO_PATH_MAX - strlen(file_name) - 1);
    coco_join_path(file_path, sizeof(file_path), folder_path, file_name, NULL);
    if (*target_file == NULL) {
        tmp_file = fopen(file_path, "r");/*to check for existance*/
        if ((tmp_file ) &&
            (consprob2015_current_dim == data->number_of_variables) &&
            (consprob2015_current_funId == data->function_id)) {/*new instance of current funId and current dim*/
            newLine = 0;
            *target_file = fopen(file_path, "a+");
            if (*target_file == NULL) {
                errnum = errno;
                _consprob2015_logger_error_io(*target_file, errnum);
            }
            fclose(tmp_file);
        }
        else { /* either file doesn't exist (new funId) or new Dim*/
            /*check that the dim was not already present earlier in the file, if so, create a new info file*/
            if (consprob2015_current_dim != data->number_of_variables) {
                int i, j;
                for (i=0; i<consprob2015_number_of_dimensions && consprob2015_dimensions_in_current_infoFile[i]!=0 &&
                     consprob2015_dimensions_in_current_infoFile[i]!=data->number_of_variables;i++) {
                    ;/*checks whether dimension already present in the current infoFile*/
                }
                if (i<consprob2015_number_of_dimensions && consprob2015_dimensions_in_current_infoFile[i]==0) {
                    /*new dimension seen for the first time*/
                    consprob2015_dimensions_in_current_infoFile[i]=data->number_of_variables;
                    newLine = 1;
                }
                else{
                        if (i<consprob2015_number_of_dimensions) {/*dimension already present, need to create a new file*/
                            newLine = 0;
                            file_path[strlen(file_path)-strlen(consprob2015_infoFile_firstInstance_char) - 7] = 0;/*truncate the instance part*/
                            consprob2015_infoFile_firstInstance = data->instance_id;
                            sprintf(consprob2015_infoFile_firstInstance_char, "%d", consprob2015_infoFile_firstInstance);
                            strncat(file_path, "_i", NUMBBO_PATH_MAX - strlen(file_name) - 1);
                            strncat(file_path, consprob2015_infoFile_firstInstance_char, NUMBBO_PATH_MAX - strlen(file_name) - 1);
                            strncat(file_path, ".info", NUMBBO_PATH_MAX - strlen(file_name) - 1);
                        }
                        else{/*we have all dimensions*/
                            newLine = 1;
                        }
                        for (j=0; j<consprob2015_number_of_dimensions;j++){/*new info file, reinitilize list of dims*/
                            consprob2015_dimensions_in_current_infoFile[j]= 0;
                        }
                    consprob2015_dimensions_in_current_infoFile[i]=data->number_of_variables;
                }
            }
            *target_file = fopen(file_path, "a+");/*in any case, we append*/
            if (*target_file == NULL) {
                errnum = errno;
                _consprob2015_logger_error_io(*target_file, errnum);
            }
            if (tmp_file) { /*File already exists, new dim so just a new line. ALso, close the tmp_file*/
                if (newLine) {
                    fprintf(*target_file, "\n");
                }
                
                fclose(tmp_file);
            }
            
            fprintf(*target_file,
                    /* TODO: z-modifier is bound to fail as being incompatible to standard C */
                    "funcId = %d, DIM = %ld, Precision = %.3e, algId = '%s'\n",
                    (int)strtol(function_id, NULL, 10), (long)data->number_of_variables,
                    pow(10, -8), data->alg_name);
            fprintf(*target_file, "%%\n");
            strncat(used_dataFile_path, "_i", NUMBBO_PATH_MAX - strlen(used_dataFile_path) - 1);
            strncat(used_dataFile_path, consprob2015_infoFile_firstInstance_char,
                    NUMBBO_PATH_MAX - strlen(used_dataFile_path) - 1);
            fprintf(*target_file, "%s.dat",
                    used_dataFile_path); /*dataFile_path does not have the extension*/
            consprob2015_current_dim = data->number_of_variables;
            consprob2015_current_funId = data->function_id;
        }
    }
}


/**
 * Generates the different files and folder needed by the logger to store the
 * data if theses don't already exist
 */
static void _consprob2015_logger_initialize(consprob2015_logger_t *data,
                                        coco_problem_t *inner_problem) {
  /*
    Creates/opens the data and index files
  */
  char dataFile_path[NUMBBO_PATH_MAX] = {
      0}; /*relative path to the .dat file from where the .info file is*/
  char folder_path[NUMBBO_PATH_MAX] = {0};
  char tmpc_funId[3]; /*servs to extract the function id as a char *. There
                         should be a better way of doing this! */
  char tmpc_dim[3];   /*servs to extract the dimension as a char *. There should
                         be a better way of doing this! */
  char indexFile_prefix[10] = "bbobexp"; /* TODO (minor): make the prefix bbobexp a
                                            parameter that the user can modify */  
  assert(data != NULL);
  assert(inner_problem != NULL);
  assert(inner_problem->problem_id != NULL);

  sprintf(tmpc_funId, "%d", consprob2015_get_function_id(inner_problem));
  sprintf(tmpc_dim, "%lu", (unsigned long) inner_problem->number_of_variables);
  
  /* prepare paths and names*/
  strncpy(dataFile_path, "data_f", NUMBBO_PATH_MAX);
  strncat(dataFile_path, tmpc_funId,
          NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  coco_join_path(folder_path, sizeof(folder_path), data->path, dataFile_path,
                 NULL);
  coco_create_path(folder_path);
  strncat(dataFile_path, "/bbobexp_f",
          NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_funId,
          NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, "_DIM", NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_dim, NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);

  /* index/info file*/
  _consprob2015_logger_openIndexFile(data, data->path, indexFile_prefix, tmpc_funId,
                                 dataFile_path);
  fprintf(data->index_file, ", %d", consprob2015_get_instance_id(inner_problem));
  /* data files*/
  /*TODO: definitely improvable but works for now*/
  strncat(dataFile_path, "_i", NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, consprob2015_infoFile_firstInstance_char,
            NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  _consprob2015_logger_open_dataFile(&(data->fdata_file), data->path, dataFile_path,
                                 ".dat");
  fprintf(data->fdata_file, consprob2015_file_header_str, data->optimal_fvalue);

  _consprob2015_logger_open_dataFile(&(data->tdata_file), data->path, dataFile_path,
                                 ".tdat");
  fprintf(data->tdata_file, consprob2015_file_header_str, data->optimal_fvalue);

  _consprob2015_logger_open_dataFile(&(data->rdata_file), data->path, dataFile_path,
                                 ".rdat");
  fprintf(data->rdata_file, consprob2015_file_header_str, data->optimal_fvalue);
  /* TODO: manage duplicate filenames by either using numbers or raising an
   * error */
  data->is_initialized = 1;
}

/**
 * Layer added to the transformed-problem evaluate_function by the logger
 */
static void _consprob2015_logger_evaluate_function(coco_problem_t *self, const double *x,
                                               double *y) {
  consprob2015_logger_t *data = coco_get_transform_data(self);
  coco_problem_t * inner_problem = coco_get_transform_inner_problem(self);
  
  if (!data->is_initialized) {
    _consprob2015_logger_initialize(data, inner_problem);
  }
  if (consprob2015_logger_verbosity > 2 && data->number_of_evaluations == 0) {
    if (inner_problem->index >= 0) {
      printf("\n ***** %d: ", inner_problem->index);
    }
    printf("on problem %s", coco_get_problem_id(inner_problem));
  }
  coco_evaluate_function(inner_problem, x, y);
  data->last_fvalue = y[0];
  data->written_last_eval = 0;
  if (data->number_of_evaluations == 0 || y[0] < data->best_fvalue) {
    size_t i;
    data->best_fvalue = y[0];
    for (i = 0; i < self->number_of_variables; i++)
      data->best_solution[i] = x[i];
  }
  data->number_of_evaluations++;

  /* Add sanity check for optimal f value */
  /*assert(y[0] >= data->optimal_fvalue);*/
  if (!raisedOptValWarning && y[0] < data->optimal_fvalue) {
      coco_warning("Observed fitness is smaller than supposed optimal fitness.");
      raisedOptValWarning = 1;
  }

  /* Add a line in the .dat file for each logging target reached. */
  if (y[0] - data->optimal_fvalue <= data->f_trigger) {

    _consprob2015_logger_write_data(data->fdata_file, data->number_of_evaluations,
                                y[0], data->best_fvalue, data->optimal_fvalue,
                                x, self->number_of_variables);
    _consprob2015_logger_update_f_trigger(data, y[0]);
  }

  /* Add a line in the .tdat file each time an fevals trigger is reached. */
  if (data->number_of_evaluations >= data->t_trigger) {
    data->written_last_eval = 1;
    _consprob2015_logger_write_data(data->tdata_file, data->number_of_evaluations,
                                y[0], data->best_fvalue, data->optimal_fvalue,
                                x, self->number_of_variables);
    _consprob2015_logger_update_t_trigger(data, self->number_of_variables);
  }

  /* Flush output so that impatient users can see progress. */
  fflush(data->fdata_file);
}

/**
 * Also serves as a finalize run method so. Must be called at the end
 * of Each run to correctly fill the index file
 *
 * TODO: make sure it is called at the end of each run or move the
 * writing into files to another function
 */
static void _consprob2015_logger_free_data(void *stuff) {
  /*TODO: do all the "non simply freeing" stuff in another function
   * that can have problem as input
   */
  consprob2015_logger_t *data = stuff;

  if (consprob2015_logger_verbosity > 2 && data && data->number_of_evaluations > 0) {
    printf("\n\nDone observing after %ld fevals.", (long)data->number_of_evaluations);
    printf("\n\nbest objective function value = %e", data->best_fvalue);
    printf("\ninfeasibility = (retrieve it from somewhere)\n\n");
  }
  if (data->alg_name != NULL) {
    coco_free_memory((void*)data->alg_name);
    data->alg_name = NULL;
  }
    
  if (data->path != NULL) {
    coco_free_memory(data->path);
    data->path = NULL;
  }
  if (data->index_file != NULL) {
    fprintf(data->index_file, ":%ld|%.1e", (long)data->number_of_evaluations,
            data->best_fvalue - data->optimal_fvalue);
    fclose(data->index_file);
    data->index_file = NULL;
  }
  if (data->fdata_file != NULL) {
    fclose(data->fdata_file);
    data->fdata_file = NULL;
  }
  if (data->tdata_file != NULL) {
    /* TODO: make sure it handles restarts well. i.e., it writes
     * at the end of a single run, not all the runs on a given
     * instance. Maybe start with forcing it to generate a new
     * "instance" of problem for each restart in the beginning
     */
    if (!data->written_last_eval) {
      _consprob2015_logger_write_data(data->tdata_file, data->number_of_evaluations,
                                  data->last_fvalue, data->best_fvalue,
                                  data->optimal_fvalue, data->best_solution,
                                  data->number_of_variables);
    }
    fclose(data->tdata_file);
    data->tdata_file = NULL;
  }

  if (data->rdata_file != NULL) {
    fclose(data->rdata_file);
    data->rdata_file = NULL;
  }

  if (data->best_solution != NULL) {
    coco_free_memory(data->best_solution);
    data->best_solution = NULL;
  }
  consprob2015_logger_is_open = 0;
}

static coco_problem_t *consprob2015_logger(coco_problem_t *inner_problem,
                                const char *alg_name) {
  consprob2015_logger_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->alg_name = coco_strdup(alg_name);
  if (consprob2015_logger_is_open)
    coco_error("The current consprob2015_logger (observer) must be closed before a new one is opened");
  /* This is the name of the folder which happens to be the algName */
  data->path = coco_strdup(alg_name);
  data->index_file = NULL;
  data->fdata_file = NULL;
  data->tdata_file = NULL;
  data->rdata_file = NULL;
  data->number_of_variables = inner_problem->number_of_variables;
  if (inner_problem->best_value == NULL) {
      /*coco_error("Optimal f value must be defined for each problem in order for the logger to work propertly");*/
      /*Setting the value to 0 results in the assertion y>=optimal_fvalue being susceptible to failure*/
      coco_warning("undefined optimal f value. Set to 0");
      data->optimal_fvalue = 0;
  }
  else
  {
      data->optimal_fvalue = *(inner_problem->best_value);
  }
  raisedOptValWarning = 0;

  data->idx_f_trigger = INT_MAX;
  data->idx_t_trigger = 0;
  data->idx_tdim_trigger = 0;
  data->f_trigger = DBL_MAX;
  data->t_trigger = 0;
  data->number_of_evaluations = 0;
  data->best_solution =
      coco_allocate_vector(inner_problem->number_of_variables);
  /* TODO: the following inits are just to be in the safe side and
   * should eventually be removed. Some fileds of the consprob2015_logger struct
   * might be useless
   */
  data->function_id = consprob2015_get_function_id(inner_problem);
  data->instance_id = consprob2015_get_instance_id(inner_problem);
  data->written_last_eval = 1;
  data->last_fvalue = DBL_MAX;
  data->is_initialized = 0;

  self = coco_allocate_transformed_problem(inner_problem, data,
                                           _consprob2015_logger_free_data);
  self->evaluate_function = _consprob2015_logger_evaluate_function;
  consprob2015_logger_is_open = 1;
  return self;
}
