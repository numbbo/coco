#include "numbbo.h"

#include "log_hitting_times.c"

numbbo_problem_t *toy_observer(numbbo_problem_t *problem, const char *options) {
    /* Calculate target levels for first hitting times */
    size_t i;
    static const size_t number_of_targets = 20;
    double targets[number_of_targets];

    /* Calculate target levels: */
    for (i = number_of_targets; i > 0; --i) {
        targets[i - 1] = pow(10.0, (number_of_targets - i) - 9.0);
    }

    char base_path[1024] = {0};
    char filename[1024] = {0};
    numbbo_join_path(base_path, sizeof(base_path),
                     options, "toy_so", numbbo_get_id(problem), NULL);
    if (numbbo_path_exists(base_path)) {
        numbbo_error("Result directory exists.");
        return NULL; /* never reached */
    }
    numbbo_create_path(base_path);
    numbbo_join_path(filename, sizeof(filename), 
                     base_path, "first_hitting_times.txt", NULL);
    problem = log_hitting_times(problem, targets, number_of_targets, filename);
    return problem;
}
