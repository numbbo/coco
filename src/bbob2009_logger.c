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
} bbob2009_logger_t;

static void bbob2009_logger_update_next_target(bbob2009_logger_t * state) {
	state->next_target = pow(10,
			(double) state->idx_fval_trigger / (double) nbpts_fval);
	return;
}

/**
 * bbob2009_logger_error_io(FILE *path)
 *
 * Error when trying to create the file "path"
 *
 */
static void bbob2009_logger_error_io(FILE *path) {
	char *buf;
	const char *error_format =
			"bbob2009_logger_prepare() failed to open log file '%s'.";
	size_t buffer_size = snprintf(NULL, 0, error_format, path);
	buf = (char *) coco_allocate_memory(buffer_size);
	snprintf(buf, buffer_size, error_format, path);
	coco_error(buf);
	coco_free_memory(buf);
}

static void bbob2009_logger_prepare_files(bbob2009_logger_t *state, double best_value, const char * problem_id) {
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
    /*folder name and path for current function*/
    strcpy(folder_name,"data_");
    strcat(folder_name,problem_id);
    coco_join_path(folder_path, sizeof(folder_path), state->path, folder_name, NULL);
	coco_create_path(folder_path);
    
    /*file name for the index file*/
    strcpy(index_name,"bbobexp_");
    strcat(index_name,problem_id);
    strcat(index_name,".info");
    coco_join_path(index_path, sizeof(index_name), state->path, index_name, NULL);
    printf("%s\n",index_path);
    if (state->index_file == NULL) {
		state->index_file = fopen(index_path, "a+");
		if (state->index_file == NULL) {
			bbob2009_logger_error_io(state->index_file);
            
		}
	}
    
    /*file name for the .dat file*/
    strcpy(file_name,"bbobexp_");
    strcat(file_name,problem_id);
    strcat(file_name,".dat");
    coco_join_path(file_path, sizeof(file_name), folder_path, file_name, NULL);
    /*TODO: use the correct folder name (no dimension) once we can get function type (sphere/F1....)*/
	if (state->fdata_file == NULL) {
		state->fdata_file = fopen(file_path, "a+");
		if (state->fdata_file == NULL) {
			bbob2009_logger_error_io(state->fdata_file);

		}
	}
	fprintf(state->fdata_file,
			"%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n",
			best_value);
	
    /*file name for the .tdat file*/
    strcpy(file_name_t,"bbobexp_");
    strcat(file_name_t,problem_id);
    strcat(file_name_t,".tdat");
    coco_join_path(file_path_t, sizeof(file_name), folder_path, file_name_t, NULL);
    
	if (state->tdata_file == NULL) {
		state->tdata_file = fopen(file_path_t, "a+");
		if (state->tdata_file == NULL) {
			bbob2009_logger_error_io(state->tdata_file);
		}
	}
	fprintf(state->tdata_file,
			"%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n",
			best_value);
    
    
    
	return;
}

static void bbob2009_logger_evaluate_function(coco_problem_t *self, double *x,
		double *y) {
	coco_transformed_problem_t *obj = (coco_transformed_problem_t *) self;
	bbob2009_logger_t *state = (bbob2009_logger_t *) obj->state;
	assert(obj->state != NULL);
	coco_evaluate_function(obj->inner_problem, x, y);
	state->number_of_evaluations++;

	/* Add a line for each hitting level reached. */
	if (y[0] <= state->next_target) {
		size_t i;
		fprintf(state->fdata_file, "%ld", state->number_of_evaluations);/*for some reason, it's %.0f in the old code */
		fprintf(state->fdata_file, " %+10.9e %+10.9e %+10.9e %+10.9e", y[0],
				y[0], y[0], y[0]);
        if (self->number_of_variables<22) {
            for (i = 0; i < self->number_of_variables; i++) {
                fprintf(state->fdata_file, " %+5.4e", x[i]);
            }
		}

		fprintf(state->fdata_file, "\n");
		if (state->idx_fval_trigger == INT_MAX)
			/* "jump" directly to the next closest target (to the actual fvalue) from the initial target*/
			state->idx_fval_trigger = ceil(log10(y[0])) * nbpts_fval;
		else
			state->idx_fval_trigger--;
		bbob2009_logger_update_next_target(state);
		while (y[0] <= state->next_target) {
			state->idx_fval_trigger--;
			bbob2009_logger_update_next_target(state);
		}

	}
	/* Flush output so that impatient users can see progress. */
	fflush(state->fdata_file);
}

static void bbob2009_logger_free_problem(coco_problem_t *self) {
	coco_transformed_problem_t *obj = (coco_transformed_problem_t *) self;
	coco_problem_t *problem = (coco_problem_t *) obj;
	bbob2009_logger_t *state;
	assert(self != NULL);
	assert(obj->state != NULL);

	state = (bbob2009_logger_t *) obj->state;

	coco_free_memory(state->path);
	if (state->logfile != NULL) {
		fclose(state->logfile);
		state->logfile = NULL;
	}
	if (state->index_file != NULL) {
		fclose(state->index_file);
		state->index_file = NULL;
	}
	if (state->fdata_file != NULL) {
		fclose(state->fdata_file);
		state->fdata_file = NULL;
	}
	if (state->tdata_file != NULL) {
		fclose(state->tdata_file);
		state->tdata_file = NULL;
	}
	coco_free_memory(obj->state);
	if (obj->inner_problem != NULL) {
		coco_free_problem(obj->inner_problem);
		obj->inner_problem = NULL;
	}
	if (problem->problem_id != NULL)
		coco_free_memory(problem->problem_id);
	if (problem->problem_name != NULL)
		coco_free_memory(problem->problem_name);
	coco_free_memory(obj);
}

coco_problem_t *bbob2009_logger(coco_problem_t *inner_problem, const char *path) {
	static const char *problem_id;
	coco_transformed_problem_t *obj = coco_allocate_transformed_problem(
			inner_problem);
	coco_problem_t *problem = (coco_problem_t *) obj;
	bbob2009_logger_t *state = (bbob2009_logger_t *) coco_allocate_memory(
			sizeof(*state));

	problem->evaluate_function = bbob2009_logger_evaluate_function;
	problem->free_problem = bbob2009_logger_free_problem;
	state->path = coco_strdup(path);
	state->logfile = NULL; /* Open lazily in logger_evaluate_function(). */
	problem_id = coco_get_id(problem);
	bbob2009_logger_prepare_files(state, *(problem->best_value), problem_id);

	state->idx_fval_trigger = INT_MAX;
	state->next_target = DBL_MAX;
	state->number_of_evaluations = 0;
	obj->state = state;
	return problem;
}

