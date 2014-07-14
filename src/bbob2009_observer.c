#include "coco.h"

#include "bbob2009_logger.c"

coco_problem_t *bbob2009_observer(coco_problem_t *problem, const char *options) {

    char base_path[NUMBBO_PATH_MAX] = {0};
    char filename[NUMBBO_PATH_MAX] = {0};
    coco_join_path(base_path, sizeof(base_path),
                     options, "toy_so", coco_get_id(problem), NULL);
    if (coco_path_exists(base_path)) {
        coco_error("Result directory exists.");
        return NULL; /* never reached */
    }
    coco_create_path(base_path);
    coco_join_path(filename, sizeof(filename), 
                     base_path, "first_hitting_times.txt", NULL);
    problem = bbob2009_logger(problem, filename);
    return problem;
}
