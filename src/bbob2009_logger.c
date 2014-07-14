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

static void bbob2009_logger_update_next_target(bbob2009_logger_t * state){
    state->next_target = pow(10, (double)state->idx_fval_trigger/(double)nbpts_fval);
    return;
}

static void bbob2009_logger_prepare_files(bbob2009_logger_t *state, double best_value){
    /*
        Creates/opens the data files
        best_value if printed in the header
     */
    static char * tmp;
    tmp = coco_strdup(state->path);
    if(state->fdata_file==NULL){
        state->fdata_file = fopen(strcat(tmp,".dat"), "w");
        if (state->fdata_file == NULL) {
            char *buf;
            const char *error_format = "bbob2009_logger_prepare() failed to open log file '%s'.";
            size_t buffer_size = snprintf(NULL, 0, error_format, state->fdata_file);
            buf = (char *)coco_allocate_memory(buffer_size);
            snprintf(buf, buffer_size, error_format, state->fdata_file);
            coco_error(buf);
            coco_free_memory(buf); /* Never reached */
        }
    }
    fprintf(state->fdata_file,"%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n", best_value);
    tmp = coco_strdup(state->path);
    
    if(state->tdata_file==NULL){
        state->tdata_file = fopen(strcat(tmp,".tdat"), "w");
        if (state->tdata_file == NULL) {
            char *buf;
            const char *error_format = "bbob2009_logger_prepare() failed to open log file '%s'.";
            size_t buffer_size = snprintf(NULL, 0, error_format, state->tdata_file);
            buf = (char *)coco_allocate_memory(buffer_size);
            snprintf(buf, buffer_size, error_format, state->tdata_file);
            coco_error(buf);
            coco_free_memory(buf); /* Never reached */
        }
    }
    fprintf(state->tdata_file,"%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n", best_value);
    
    
    return;
}



static void bbob2009_logger_evaluate_function(coco_problem_t *self, 
                                              double *x, double *y) {
    coco_transformed_problem_t *obj = (coco_transformed_problem_t *)self;
    bbob2009_logger_t *state = (bbob2009_logger_t *)obj->state;
    assert(obj->state != NULL);
    coco_evaluate_function(obj->inner_problem, x, y);
    state->number_of_evaluations++;

    /*if (state->logfile == NULL) {
        state->logfile = fopen(state->path, "w");
        if (state->logfile == NULL) {
            char *buf;
            const char *error_format = 
                "bbob2009_logger_evaluate_function() failed to open log file '%s'.";
            size_t buffer_size = 
                snprintf(NULL, 0, error_format, state->path);
            buf = (char *)coco_allocate_memory(buffer_size);
            snprintf(buf, buffer_size, error_format, state->path);
            coco_error(buf);
            coco_free_memory(buf);
        }
        fprintf(state->logfile,"%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n", *(self->best_value));
    }*/
    
    /* Add a line for each hitting level reached. */
    if (y[0] <= state->next_target ){
        size_t i;
        fprintf(state->fdata_file, "%ld", state->number_of_evaluations);/*for some reason, it's %.0f in the old code */
        fprintf(state->fdata_file, " %+10.9e %+10.9e %+10.9e %+10.9e", y[0], y[0], y[0], y[0]);
        for (i=0; i < self->number_of_variables; i++) {
            fprintf(state->fdata_file, " %+5.4e",x[i]);
        }
        
        fprintf(state->fdata_file, "\n");
        if (state->idx_fval_trigger==INT_MAX)
            /* "jump" directly to the next closest target (to the actual fvalue) from the initial target*/
            state->idx_fval_trigger=ceil(log10(y[0]))*nbpts_fval;
        else
            state->idx_fval_trigger--;
        bbob2009_logger_update_next_target(state);
        while ( y[0] <= state->next_target){
            state->idx_fval_trigger--;
            bbob2009_logger_update_next_target(state);
        }
        
    }
    /* Flush output so that impatient users can see progress. */
    fflush(state->fdata_file);
}

static void bbob2009_logger_free_problem(coco_problem_t *self) {
    coco_transformed_problem_t *obj = (coco_transformed_problem_t *)self;
    coco_problem_t *problem = (coco_problem_t *)obj;
    assert(obj->state != NULL);
    bbob2009_logger_t *state = (bbob2009_logger_t *)obj->state;
    
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
    coco_transformed_problem_t *obj = coco_allocate_transformed_problem(inner_problem);
    coco_problem_t *problem = (coco_problem_t *)obj;
    bbob2009_logger_t *state = (bbob2009_logger_t *)coco_allocate_memory(sizeof(*state));

    problem->evaluate_function = bbob2009_logger_evaluate_function;
    problem->free_problem = bbob2009_logger_free_problem;
    state->path = coco_strdup(path);
    state->logfile = NULL; /* Open lazily in logger_evaluate_function(). */

    bbob2009_logger_prepare_files(state, *(problem->best_value));
    
    
    state->idx_fval_trigger= INT_MAX;
    state->next_target=DBL_MAX;
    state->number_of_evaluations = 0;
    obj->state = state;
    return problem;
}



