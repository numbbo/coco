#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>

#include "coco.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

static const size_t nbpts_nbevals = 20;
static const size_t nbpts_fval = 5;

/*TODO: add possibility of adding a prefix to the index files*/

typedef struct {
    char *path;/*path to the data folder. Simply the Algname*/
    const char *alg_name;/*path to the data folder. Simply the Algname*/
    FILE *index_file;/*index file*/
    FILE *fdata_file;/*function value aligned data file*/
    FILE *tdata_file;/*number of function evaluations aligned data file*/
    FILE *rdata_file;/*restart info data file*/
    double f_trigger; /* next upper bound on the fvalue to trigger a log in the .dat file*/
    long t_trigger; /* next lower bound on nb fun evals to trigger a log in the .tdat file*/
    long idx_f_trigger; /* allows to track the index i in logging target = {10**(i/nbpts_fval), i \in Z} */
    long idx_t_trigger; /* allows to track the index i in logging nbevals  = {int(10**(i/nbpts_nbevals)), i \in Z} */
    long idx_tdim_trigger; /* allows to track the index i in logging nbevals  = {dim * 10**i, i \in Z} */
    long number_of_evaluations;
    double best_fvalue;
    double last_fvalue;
    short written_last_eval; /*allows writing the the data of the final fun eval in the .tdat file if not already written by the t_trigger*/
    double * best_solution;
    /*the following are to only pass data as a parameter in the free function*/
    long number_of_variables;
    double optimal_fvalue;
} _bbob2009_logger_t;

char* _file_header_str="%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n";

static void _bbob2009_logger_update_f_trigger(_bbob2009_logger_t *data, double fvalue) {
    
    /* "jump" directly to the next closest (but larger) target to the current fvalue from the initial target*/
    if (data->idx_f_trigger == INT_MAX)
        data->idx_f_trigger = ceil(log10(fvalue)) * nbpts_fval;
    else
        data->idx_f_trigger--;
    
    data->f_trigger = pow(10, data->idx_f_trigger * 1.0 / nbpts_fval);
    while (fvalue <= data->f_trigger) {
        data->idx_f_trigger--;
        data->f_trigger = pow(10, data->idx_f_trigger * 1.0 / nbpts_fval);
    }
}


static void _bbob2009_logger_update_t_trigger(_bbob2009_logger_t *data, long number_of_variables) {
    while(data->number_of_evaluations >= floor(pow(10, (double)data->idx_t_trigger / (double) nbpts_nbevals)))
        data->idx_t_trigger++;
    
    while(data->number_of_evaluations >= number_of_variables * pow(10, (double) data->idx_tdim_trigger))
        data->idx_tdim_trigger++;

    data->t_trigger = fmin(floor(pow(10, (double)data->idx_t_trigger / (double)nbpts_nbevals)), number_of_variables * pow(10, (double) data->idx_tdim_trigger));

}


/**
 * adds a formated line to a data file
 */
static void _bbob2009_logger_write_data(FILE * target_file, long number_of_evaluations, double fvalue, double best_fvalue, double best_value, double * x, long number_of_variables){
    /* for some reason, it's %.0f in the old code instead of the 10.9e in the documentation*/
    fprintf(target_file, "%ld %+10.9e %+10.9e %+10.9e %+10.9e", number_of_evaluations, fvalue - best_value, best_fvalue - best_value, fvalue, best_fvalue);
    if (number_of_variables<22) {
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
static void _bbob2009_logger_error_io(FILE *path) {
    char *buf;
    const char *error_format =
        "bbob2009_logger_prepare() failed to open log file '%s'.";
    size_t buffer_size = snprintf(NULL, 0, error_format, path);
    buf = (char *) coco_allocate_memory(buffer_size);
    snprintf(buf, buffer_size, error_format, path);
    coco_error(buf);
    coco_free_memory(buf);
}


/**
 * Creates the the file fileName_prefix+problem_id+file_extension in folde_path
 */
static void _bbob2009_logger_createFile(FILE ** target_file, 
                                        const char* folder_path, 
                                        const char* problem_id,
                                        const char* fileName_prefix, 
                                        const char* file_extension) {
    char file_name[NUMBBO_PATH_MAX];
    char file_path[NUMBBO_PATH_MAX] = {0};
    /*file name */
    strncpy(file_name, fileName_prefix, NUMBBO_PATH_MAX);
    strncat(file_name, problem_id, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, file_extension, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    coco_join_path(file_path, sizeof(file_path), folder_path, file_name, NULL);
    /*TODO: use the correct folder name (no dimension) once we can get
     * function type (sphere/F1...)*/
    if (*target_file == NULL) {
        *target_file = fopen(file_path, "a+");
        if (target_file == NULL) {
            _bbob2009_logger_error_io(*target_file);
        }
    }
}


/**
 * Generates the different files and folder needed by the logger to store the data
 */
static void _bbob2009_logger_initialize(_bbob2009_logger_t *data,
                                           double optimal_fvalue,
                                           const char * problem_id) {
    /*
      Creates/opens the data and index files
    */
    char folder_name[NUMBBO_PATH_MAX];
    char folder_path[NUMBBO_PATH_MAX]={0};

    assert(data != NULL);
    assert(problem_id != NULL);

    /*generate folder name and path for current function*/
    strncpy(folder_name,"data_", NUMBBO_PATH_MAX);
    strncat(folder_name, problem_id, NUMBBO_PATH_MAX - strlen(folder_name) - 1);
    coco_join_path(folder_path, sizeof(folder_path), data->path, folder_name, NULL);
    coco_create_path(folder_path);
    

    _bbob2009_logger_createFile(&(data->index_file), data->path, problem_id, "bbobexp_", ".info");

    _bbob2009_logger_createFile(&(data->fdata_file), folder_path, problem_id, "bbobexp_", ".dat");
    fprintf(data->fdata_file,_file_header_str,optimal_fvalue);

    _bbob2009_logger_createFile(&(data->tdata_file), folder_path, problem_id, "bbobexp_", ".tdat");
    fprintf(data->tdata_file,_file_header_str,optimal_fvalue);
    
    _bbob2009_logger_createFile(&(data->rdata_file), folder_path, problem_id, "bbobexp_", ".rdat");
    fprintf(data->rdata_file,_file_header_str,optimal_fvalue);
    
    /* TODO: manage duplicate filenames by either using numbers or raising an error */
}

/**
 * Layer added to the transformed-problem evaluate_function by the logger
 */
static void _bbob2009_logger_evaluate_function(coco_problem_t *self,
                                               double *x, double *y) {
    _bbob2009_logger_t *data;
    data = coco_get_transform_data(self);
    coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
    data->last_fvalue = y[0];
    data->written_last_eval = 0;
    if (data->number_of_evaluations == 0 || y[0]<data->best_fvalue){
        data->best_fvalue=y[0];
        size_t i;
        for (i=0; i<self->number_of_variables;i++)
            data->best_solution[i]=x[i];
    }
    data->number_of_evaluations++;

    /* Add a line in the .dat file for each logging target reached. */
    if (y[0] <= data->f_trigger) {
        
        _bbob2009_logger_write_data(data->fdata_file, data->number_of_evaluations, y[0], data->best_fvalue, data->optimal_fvalue , x, self->number_of_variables);
        
        _bbob2009_logger_update_f_trigger(data, y[0]);
    }
    
    /* Add a line in the .tdat file each time an fevals trigger is reached. */
    if (data->number_of_evaluations >= data->t_trigger){
        data->written_last_eval = 1;
        _bbob2009_logger_write_data(data->tdata_file, data->number_of_evaluations, y[0], data->best_fvalue, data->optimal_fvalue , x, self->number_of_variables);
        _bbob2009_logger_update_t_trigger(data, self->number_of_variables);
        
        
    }
    

    /* Flush output so that impatient users can see progress. */
    fflush(data->fdata_file);
}

static void _bbob2009_logger_free_data(void *stuff) {
    _bbob2009_logger_t *data = stuff;
    coco_free_memory(data->path);
    if (data->index_file != NULL) {
        fclose(data->index_file);
        data->index_file = NULL;
    }
    if (data->fdata_file != NULL) {
        fclose(data->fdata_file);
        data->fdata_file = NULL;
    }
    if (data->tdata_file != NULL) {
        if (! data->written_last_eval) /* TODO: make sure it handles restarts well. i.e., it writes at the end of a single run, not all the runs on a given instance. Maybe start with forcing it to generate a new "instance" of problem for each restart in the beginning*/
            _bbob2009_logger_write_data(data->tdata_file, data->number_of_evaluations, data->last_fvalue, data->best_fvalue, data->optimal_fvalue , data->best_solution, data->number_of_variables);
        fclose(data->tdata_file);
        data->tdata_file = NULL;
    }
    
    if (data->rdata_file != NULL) {
        fclose(data->rdata_file);
        data->rdata_file = NULL;
    }
}

coco_problem_t *bbob2009_logger(coco_problem_t *inner_problem, const char *alg_name) {
    _bbob2009_logger_t *data;
    coco_problem_t *self;

    data = coco_allocate_memory(sizeof(*data));
    data->alg_name = alg_name;
    data->path = coco_strdup(alg_name);/* This is the name of the folder which happens to be the algName*/
    data->index_file = NULL;
    data->fdata_file = NULL;
    data->tdata_file = NULL;
    data->rdata_file = NULL;
    _bbob2009_logger_initialize(data,
                                   *(inner_problem->best_value),
                                   inner_problem->problem_id);

    fprintf(data->index_file, "funcId = %s, DIM = %zu, Precision = %.3e, algId = '%s'\n", inner_problem->problem_name, inner_problem->number_of_variables, pow(10,-8), data->alg_name);/*TODO: move into a function */
    
    
    
    data->number_of_variables = inner_problem->number_of_variables;
    data->optimal_fvalue = *(inner_problem->best_value);
    data->idx_f_trigger = INT_MAX;
    data->idx_t_trigger = 0;
    data->idx_tdim_trigger = 0;
    data->f_trigger = DBL_MAX;
    data->t_trigger = 0;
    data->number_of_evaluations = 0;
    data->best_solution = coco_allocate_vector(inner_problem->number_of_variables);
    /* TODO: the following inits are just to be in the safe side and should eventually be removed*/
    data->written_last_eval = 1;
    data->last_fvalue = DBL_MAX ;
    self = coco_allocate_transformed_problem(inner_problem, data,
                                             _bbob2009_logger_free_data);
    self->evaluate_function = _bbob2009_logger_evaluate_function;
    return self;
}

