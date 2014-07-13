#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>

#include "logger.h"
#include "numbbo.h"

#include "numbbo_utilities.c"
#include "numbbo_problem.c"
#include "numbbo_strdup.c"

static void logger_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    numbbo_transformed_problem_t *obj = (numbbo_transformed_problem_t *)self;
    logger_t *state = (logger_t *)obj->state;
    assert(obj->state != NULL);
    numbbo_evaluate_function(obj->inner_problem, x, y);
    state->number_of_evaluations++;

    /* Open logfile if it is not alread open */
    if (state->logfile == NULL) {
        state->logfile = fopen(state->path, "w");
        if (state->logfile == NULL) {
            const char *error_format = 
                "logger_evaluate_function() failed to open log file '%s'.";
            size_t buffer_size = 
                snprintf(NULL, 0, error_format, state->path);
            char buf[buffer_size];
            snprintf(buf, buffer_size, error_format, state->path);
            numbbo_error(buf);
        }
        fprintf(state->logfile,"%% function evaluation | noise-free fitness - Fopt (%13.12e) | best noise-free fitness - Fopt | measured fitness | best measured fitness | x1 | x2...\n", *(self->best_value));
        //fputs("target_value function_value number_of_evaluations\n",state->logfile);
    }
    
    /* Add a line for each hitting level we have reached. */
    if (y[0] <= state->next_target ){
        fprintf(state->logfile, " %+10.9e %+10.9e %+10.9e %+10.9e", y[0], y[0], y[0], y[0]);
        for (size_t i=0; i < self->number_of_variables; i++) {
            fprintf(state->logfile, " %+5.4e",x[i]);
        }
        
        fprintf(state->logfile, "\n");
//        fprintf(fout, " %+10.9e %+10.9e %+10.9e %+10.9e", F-Fopt, bestF-Fopt, Fnoisy, bestFnoisy);
//        fprintf(state->logfile, "%e %e %li\n",state->next_target,y[0],state->number_of_evaluations);
        if (state->idx_fval_trigger==INT_MAX)
            state->idx_fval_trigger=ceil(log10(y[0]))*nbpts_fval;
        else
            state->idx_fval_trigger--;
        update_next_target(state);
        //state->next_target=pow(10, (double)state->idx_fval_trigger/(double)nbpts_fval);
        //printf("%f <= %d : %f \n",y[0],state->idx_fval_trigger, state->next_target);
        while ( y[0] <= state->next_target){
            state->idx_fval_trigger--;
            update_next_target(state);
            //state->next_target=pow(10, (double)state->idx_fval_trigger/(double)nbpts_fval);
            //printf("%f %f %f \n",(double)state->idx_fval_trigger, (double)nbpts_fval,(double)state->idx_fval_trigger/(double)nbpts_fval);
        }
        
    }
    /* Flush output so that impatient users can see progress. */
    fflush(state->logfile);
}

static void logger_free_problem(numbbo_problem_t *self) {
    numbbo_transformed_problem_t *obj = (numbbo_transformed_problem_t *)self;
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;
    assert(obj->state != NULL);
    logger_t *state = (logger_t *)obj->state;
    
    numbbo_free_memory(state->path);
    if (state->logfile != NULL) {
        fclose(state->logfile);
        state->logfile = NULL;
    }
    numbbo_free_memory(obj->state);
    if (obj->inner_problem != NULL) {
        numbbo_free_problem(obj->inner_problem);
        obj->inner_problem = NULL;
    }
    if (problem->problem_id != NULL)
        numbbo_free_memory(problem->problem_id);
    if (problem->problem_name != NULL)
        numbbo_free_memory(problem->problem_name);
    numbbo_free_memory(obj);
}

               
numbbo_problem_t *logger(numbbo_problem_t *inner_problem, const char *path) {//TODO: consider renaming
    numbbo_transformed_problem_t *obj = 
        numbbo_allocate_transformed_problem(inner_problem);
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;
    logger_t *state = (logger_t *)numbbo_allocate_memory(sizeof(*state));

    problem->evaluate_function = logger_evaluate_function;
    problem->free_problem = logger_free_problem;

    state->path = numbbo_strdup(path);
    state->logfile = NULL; /* Open lazily in lht_evaluate_function(). */
    
    state->idx_fval_trigger= INT_MAX;
    state->next_target=DBL_MAX;
    state->number_of_evaluations = 0;
    obj->state = state;
    return problem;
}

void update_next_target(logger_t * state){
    state->next_target=pow(10, (double)state->idx_fval_trigger/(double)nbpts_fval);
    return;
}
