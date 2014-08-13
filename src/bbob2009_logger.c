#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>

#include "coco.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

/* static const size_t nbpts_nbevals = 20; */
static const size_t nbpts_fval = 5;

/*TODO: add possibility of adding a prefix to the index files*/

typedef struct {
    char *path;/*path to the data folder*/
    FILE *index_file;/*index file*/
    FILE *fdata_file;/*function value aligned data file*/
    FILE *tdata_file;/*number of function evaluations aligned data file*/
    long idx_fval_trigger; /* logging target = {10**(i/nbPtsF), i \in Z} */
    double next_target;
    long idx_nbevals_trigger;
    long idx_dim_nbevals_trigger;
    double fTrigger;
    double evalsTrigger;
    long number_of_evaluations;
} _bbob2009_logger_t;

char* _file_header_str="%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n";

static void _bbob2009_logger_update_next_target(_bbob2009_logger_t *data) {
    data->next_target = pow(10, data->idx_fval_trigger * 1.0 / nbpts_fval);
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

static void _bbob2009_logger_prepare_files(_bbob2009_logger_t *data,
                                           double best_value,
                                           const char * problem_id) {
    /*
      Creates/opens the data files
      best_value if printed in the header
    */
    /*TODO: probably doable with less variables and less string function calls*/
    char folder_name[NUMBBO_PATH_MAX];
    char folder_path[NUMBBO_PATH_MAX]={0};
    char ffile_name[NUMBBO_PATH_MAX];
    char ffile_path[NUMBBO_PATH_MAX]= {0};
    char tfile_name[NUMBBO_PATH_MAX];
    char tfile_path[NUMBBO_PATH_MAX]= {0};
    char ifile_name[NUMBBO_PATH_MAX];
    char ifile_path[NUMBBO_PATH_MAX]= {0};

    assert(data != NULL);
    assert(problem_id != NULL);

    /*folder name and path for current function*/
    strncpy(folder_name,"data_", NUMBBO_PATH_MAX);
    strncat(folder_name, problem_id, NUMBBO_PATH_MAX - strlen(folder_name) - 1);
    coco_join_path(folder_path, sizeof(folder_path), data->path, folder_name, NULL);
    coco_create_path(folder_path);

    /*file name for the index file*/
    strncpy(ifile_name,"bbobexp_", NUMBBO_PATH_MAX);
    strncat(ifile_name,problem_id, NUMBBO_PATH_MAX - strlen(ifile_name) - 1);
    strncat(ifile_name,".info", NUMBBO_PATH_MAX - strlen(ifile_name) - 1);
    coco_join_path(ifile_path, sizeof(ifile_name), data->path, ifile_name, NULL);
    /* OME: Do not output anything to stdout! */
    /* printf("%s\n",ifile_path); */
    if (data->index_file == NULL) {
        data->index_file = fopen(ifile_path, "a+");
        if (data->index_file == NULL) {
            _bbob2009_logger_error_io(data->index_file);
        }
    }

    /*file name for the .dat file*/
    strncpy(ffile_name,"bbobexp_", NUMBBO_PATH_MAX);
    strncat(ffile_name,problem_id, NUMBBO_PATH_MAX - strlen(ffile_name) - 1);
    strncat(ffile_name,".dat", NUMBBO_PATH_MAX - strlen(ffile_name) - 1);
    coco_join_path(ffile_path, sizeof(ffile_name), folder_path, ffile_name, NULL);
    /*TODO: use the correct folder name (no dimension) once we can get
     * function type (sphere/F1....)*/
    if (data->fdata_file == NULL) {
        data->fdata_file = fopen(ffile_path, "a+");
        if (data->fdata_file == NULL) {
            _bbob2009_logger_error_io(data->fdata_file);
        }
    }
    fprintf(data->fdata_file,_file_header_str,best_value);

    /*file name for the .tdat file*/
    strncpy(tfile_name,"bbobexp_", NUMBBO_PATH_MAX);
    strncat(tfile_name,problem_id, NUMBBO_PATH_MAX - strlen(tfile_name) - 1);
    strncat(tfile_name,".tdat", NUMBBO_PATH_MAX - strlen(tfile_name) - 1);
    coco_join_path(tfile_path, sizeof(tfile_name), folder_path, tfile_name, NULL);

    if (data->tdata_file == NULL) {
        data->tdata_file = fopen(tfile_path, "a+");
        if (data->tdata_file == NULL) {
            _bbob2009_logger_error_io(data->tdata_file);
        }
    }
    fprintf(data->tdata_file,_file_header_str,best_value);
}

static void _bbob2009_logger_evaluate_function(coco_problem_t *self,
                                               double *x, double *y) {
    _bbob2009_logger_t *data;
    data = coco_get_transform_data(self);
    coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
    data->number_of_evaluations++;

    /* Add a line for each hitting level reached. */
    if (y[0] <= data->next_target) {
        size_t i;
        fprintf(data->fdata_file, "%ld", data->number_of_evaluations);/*for some reason, it's %.0f in the old code */
        fprintf(data->fdata_file, " %+10.9e %+10.9e %+10.9e %+10.9e", y[0],
                y[0], y[0], y[0]);
        if (self->number_of_variables<22) {
            for (i = 0; i < self->number_of_variables; i++) {
                fprintf(data->fdata_file, " %+5.4e", x[i]);
            }
        }

        fprintf(data->fdata_file, "\n");
        if (data->idx_fval_trigger == INT_MAX)
            /* "jump" directly to the next closest target (to the
             * actual fvalue) from the initial target
             */
            data->idx_fval_trigger = ceil(log10(y[0])) * nbpts_fval;
        else
            data->idx_fval_trigger--;
        _bbob2009_logger_update_next_target(data);
        while (y[0] <= data->next_target) {
            data->idx_fval_trigger--;
            _bbob2009_logger_update_next_target(data);
        }

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
        fclose(data->tdata_file);
        data->tdata_file = NULL;
    }
}

coco_problem_t *bbob2009_logger(coco_problem_t *inner_problem, const char *path) {
    _bbob2009_logger_t *data;
    coco_problem_t *self;

    data = coco_allocate_memory(sizeof(*data));
    data->path = coco_strdup(path);
    data->index_file = NULL;
    data->fdata_file = NULL;
    data->tdata_file = NULL;
    _bbob2009_logger_prepare_files(data,
                                   *(inner_problem->best_value),
                                   inner_problem->problem_id);
    data->idx_fval_trigger = INT_MAX;
    data->next_target = DBL_MAX;
    data->number_of_evaluations = 0;

    self = coco_allocate_transformed_problem(inner_problem, data,
                                             _bbob2009_logger_free_data);
    self->evaluate_function = _bbob2009_logger_evaluate_function;
    return self;
}

