#include "coco.h"
#include "coco_utilities.c"
#include "log_nondominating.c"

/**
 * Multiobjective observer for logging all nondominated solutions found when
 * a new solution is generated and evaluated.
 */
static coco_problem_t *mo_toy_observer(coco_problem_t *problem, const char *options) {
    /* Calculate target levels for first hitting times */
    static const size_t max_size_of_archive = 100000;
    char base_path[COCO_PATH_MAX] = {0};
    char filename[COCO_PATH_MAX] = {0};
    coco_join_path(base_path, sizeof(base_path), options, "log_nondominated_solutions",
                   coco_get_problem_id(problem), NULL);
    if (coco_path_exists(base_path)) {
        coco_error("Result directory exists.");
        return NULL; /* never reached */
    }
    coco_create_path(base_path);
    coco_join_path(filename, sizeof(filename), 
                   base_path, "nondominated_at_birth.txt", NULL);
    problem = log_nondominating(problem, max_size_of_archive, filename);
    /* To control which information to be logged at each func eval, modify
     * the function 'lht_evaluate_function' in the file 'log_hitting_times.c' */
    return problem;
}

