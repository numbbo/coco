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

typedef struct {
    char *path;/*path to the data folder*/
    FILE *logfile;/*TODO: eventually removed*/
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
    char file_name[NUMBBO_PATH_MAX];
    char file_path[NUMBBO_PATH_MAX]= {0};
    char file_name_t[NUMBBO_PATH_MAX];
    char file_path_t[NUMBBO_PATH_MAX]= {0};
    char index_name[NUMBBO_PATH_MAX];
    char index_path[NUMBBO_PATH_MAX]= {0};

    assert(data != NULL);
    assert(problem_id != NULL);

    /*folder name and path for current function*/
    strncpy(folder_name,"data_", NUMBBO_PATH_MAX);
    strncat(folder_name, problem_id, NUMBBO_PATH_MAX - strlen(folder_name) - 1);
    coco_join_path(folder_path, sizeof(folder_path), data->path, folder_name, NULL);
    coco_create_path(folder_path);

    /*file name for the index file*/
    strncpy(index_name,"bbobexp_", NUMBBO_PATH_MAX);
    strncat(index_name,problem_id, NUMBBO_PATH_MAX - strlen(index_name) - 1);
    strncat(index_name,".info", NUMBBO_PATH_MAX - strlen(index_name) - 1);
    coco_join_path(index_path, sizeof(index_name), data->path, index_name, NULL);
    /* OME: Do not output anything to stdout! */
    /* printf("%s\n",index_path); */
    if (data->index_file == NULL) {
        data->index_file = fopen(index_path, "a+");
        if (data->index_file == NULL) {
            _bbob2009_logger_error_io(data->index_file);
        }
    }

    /*file name for the .dat file*/
    strncpy(file_name,"bbobexp_", NUMBBO_PATH_MAX);
    strncat(file_name,problem_id, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name,".dat", NUMBBO_PATH_MAX - strlen(file_name) - 1);
    coco_join_path(file_path, sizeof(file_name), folder_path, file_name, NULL);
    /*TODO: use the correct folder name (no dimension) once we can get
     * function type (sphere/F1....)*/
    if (data->fdata_file == NULL) {
        data->fdata_file = fopen(file_path, "a+");
        if (data->fdata_file == NULL) {
            _bbob2009_logger_error_io(data->fdata_file);
        }
    }
    fprintf(data->fdata_file,
            "%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n",
            best_value);

    /*file name for the .tdat file*/
    strncpy(file_name_t,"bbobexp_", NUMBBO_PATH_MAX);
    strncat(file_name_t,problem_id, NUMBBO_PATH_MAX - strlen(file_name_t) - 1);
    strncat(file_name_t,".tdat", NUMBBO_PATH_MAX - strlen(file_name_t) - 1);
    coco_join_path(file_path_t, sizeof(file_name), folder_path, file_name_t, NULL);

    if (data->tdata_file == NULL) {
        data->tdata_file = fopen(file_path_t, "a+");
        if (data->tdata_file == NULL) {
            _bbob2009_logger_error_io(data->tdata_file);
        }
    }
    fprintf(data->tdata_file,
            "%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n",
            best_value);
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
    if (data->logfile != NULL) {
        fclose(data->logfile);
        data->logfile = NULL;
    }
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
    data->logfile = NULL; /* Open lazily in logger_evaluate_function(). */
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

