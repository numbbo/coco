#include <stdio.h>
#include <assert.h>

#include "numbbo.h"

#include "numbbo_utilities.c"
#include "numbbo_problem.c"
#include "numbbo_strdup.c"

typedef struct {
    char *path;
    FILE *logfile;
    double *target_values;
    size_t number_of_target_values;
    size_t next_target_value;
    long number_of_evaluations;
} log_hitting_time_t;

static void lht_evaluate_function(numbbo_problem_t *self, double *x, double *y) {
    numbbo_transformed_problem_t *obj = (numbbo_transformed_problem_t *)self;
    log_hitting_time_t *state = (log_hitting_time_t *)obj->state;
    assert(obj->state != NULL);
    
    numbbo_evaluate_function(obj->inner_problem, x, y);
    state->number_of_evaluations++;

    /* Open logfile if it is not alread open */
    if (state->logfile == NULL) {
        state->logfile = fopen(state->path, "w");
        if (state->logfile == NULL) {
            char buf[4096];
            snprintf(buf, sizeof(buf), 
                     "lht_evaluate_function() failed to open log file '%s'.",
                     state->path);
            numbbo_error(buf);
        }
        fputs("target_value function_value number_of_evaluations\n",
              state->logfile);                    
    }
    
    /* Add a line for each hitting level we have reached. */
    while (y[0] <= state->target_values[state->next_target_value] &&
           state->next_target_value < state->number_of_target_values) {
        fprintf(state->logfile, "%e %e %li\n",
                state->target_values[state->next_target_value],
                y[0],
                state->number_of_evaluations);
        state->next_target_value++;
    }
    /* Flush output so that impatient users can see progress. */
    fflush(state->logfile);
}

static void lht_free_problem(numbbo_problem_t *self) {
    numbbo_transformed_problem_t *obj = (numbbo_transformed_problem_t *)self;
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;
    assert(obj->state != NULL);
    log_hitting_time_t *state = (log_hitting_time_t *)obj->state;
    
    numbbo_free_memory(state->path);
    if (state->logfile != NULL) {
        fclose(state->logfile);
        state->logfile = NULL;
    }
    numbbo_free_memory(obj->state);
    if (obj->inner_problem != NULL) {
        numbbo_free_memory(obj->inner_problem);
        obj->inner_problem = NULL;
    }
    if (problem->problem_id != NULL)
        numbbo_free_memory(problem->problem_id);
    if (problem->problem_name != NULL)
        numbbo_free_memory(problem->problem_name);
    numbbo_free_memory(obj);
}

numbbo_problem_t *log_hitting_times(numbbo_problem_t *inner_problem,
                                    const double *target_values,
                                    const size_t number_of_target_values,
                                    const char *path) {
    numbbo_transformed_problem_t *obj = 
        numbbo_allocate_transformed_problem(inner_problem);
    numbbo_problem_t *problem = (numbbo_problem_t *)obj;
    log_hitting_time_t *state = numbbo_allocate_memory(sizeof(log_hitting_time_t));

    problem->evaluate_function = lht_evaluate_function;
    problem->free_problem = lht_free_problem;

    state->number_of_evaluations = 0;
    state->path = numbbo_strdup(path);
    state->logfile = NULL; /* Open lazily in lht_evaluate_function(). */
    state->target_values = numbbo_duplicate_vector(target_values, 
                                                   number_of_target_values);
    state->number_of_target_values = number_of_target_values;
    state->next_target_value = 0;

    obj->state = state;
    return problem;
}
