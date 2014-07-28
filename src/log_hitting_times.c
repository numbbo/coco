#include <stdio.h>
#include <assert.h>

#include "coco.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

typedef struct {
    char *path;
    FILE *logfile;
    double *target_values;
    size_t number_of_target_values;
    size_t next_target_value;
    long number_of_evaluations;
} _log_hitting_time_t;

static void _lht_evaluate_function(coco_problem_t *self, double *x, double *y) {
    _log_hitting_time_t *data;
    data = coco_get_transform_data(self);

    coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
    data->number_of_evaluations++;

    /* Open logfile if it is not alread open */
    if (data->logfile == NULL) {
        data->logfile = fopen(data->path, "w");
        if (data->logfile == NULL) {
            char *buf;
            const char *error_format =
                "lht_evaluate_function() failed to open log file '%s'.";
            size_t buffer_size =
                snprintf(NULL, 0, error_format, data->path);
            buf = (char *)coco_allocate_memory(buffer_size);
            snprintf(buf, buffer_size, error_format, data->path);
            coco_error(buf);
            coco_free_memory(buf); /* Never reached */
        }
        fputs("target_value function_value number_of_evaluations\n",
              data->logfile);
    }

    /* Add a line for each hitting level we have reached. */
    while (y[0] <= data->target_values[data->next_target_value] &&
           data->next_target_value < data->number_of_target_values) {
        fprintf(data->logfile, "%e %e %li\n",
                data->target_values[data->next_target_value],
                y[0],
                data->number_of_evaluations);
        data->next_target_value++;
    }
    /* Flush output so that impatient users can see progress. */
    fflush(data->logfile);
}

static void _lht_free_data(void *stuff) {
    _log_hitting_time_t *data;
    assert(stuff != NULL);
    data = stuff;

    if (data->path != NULL) {
        coco_free_memory(data->path);
        data->path = NULL;
    }
    if (data->target_values != NULL) {
        coco_free_memory(data->target_values);
        data->target_values = NULL;
    }
    if (data->logfile != NULL) {
        fclose(data->logfile);
        data->logfile = NULL;
    }
}

coco_problem_t *log_hitting_times(coco_problem_t *inner_problem,
                                  const double *target_values,
                                  const size_t number_of_target_values,
                                  const char *path) {
    _log_hitting_time_t *data;
    coco_problem_t *self;

    data = coco_allocate_memory(sizeof(*data));
    data->number_of_evaluations = 0;
    data->path = coco_strdup(path);
    data->logfile = NULL; /* Open lazily in lht_evaluate_function(). */
    data->target_values = coco_duplicate_vector(target_values,
                                                   number_of_target_values);
    data->number_of_target_values = number_of_target_values;
    data->next_target_value = 0;

    self = coco_allocate_transformed_problem(inner_problem, data, _lht_free_data);
    self->evaluate_function = _lht_evaluate_function;
    return self;
}
