#include "numbbo.h"

#include "logger.c"

numbbo_problem_t *logger_observer(numbbo_problem_t *problem, const char *options) {

    char base_path[NUMBBO_PATH_MAX] = {0};
    char filename[NUMBBO_PATH_MAX] = {0};
    numbbo_join_path(base_path, sizeof(base_path),
                     options, "toy_so", numbbo_get_id(problem), NULL);
    if (numbbo_path_exists(base_path)) {
        numbbo_error("Result directory exists.");
        return NULL; /* never reached */
    }
    numbbo_create_path(base_path);
    numbbo_join_path(filename, sizeof(filename), 
                     base_path, "first_hitting_times.txt", NULL);
    problem = logger(problem, filename);
    return problem;
}
