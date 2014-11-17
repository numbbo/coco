#include <string.h>
#include <stdio.h>

#include "coco.h"

#include "toy_suit.c"
#include "toy_observer.c"

#include "bbob2009_suit.c"
#include "bbob2009_observer.c"

coco_problem_t *coco_get_problem(const char *problem_suit,
                                 const int function_index) {
    if (0 == strcmp(problem_suit, "toy_suit")) {
        return toy_suit(function_index);
    } else if (0 == strcmp(problem_suit, "bbob2009")) {
        return bbob2009_suit(function_index);
    } else {
        coco_warning("Unknown problem suit.");
        return NULL;
    }
}

coco_problem_t *coco_observe_problem(const char *observer,
                                     coco_problem_t *problem,
                                     const char *options) {
    if (0 == strcmp(observer, "toy_observer")) {
        return toy_observer(problem, options);
    } else if (0 == strcmp(observer, "bbob2009_observer")) {
        return bbob2009_observer(problem, options);
    } else if (0 == strcmp(observer, "")
               || 0 == strcmp(observer, "no_observer")) {
        return problem;
    } else {
        coco_error("Unknown observer.");
        return NULL; /* Never reached */
    }
}


void coco_benchmark(const char *problem_suit,
                    const char *observer,
                    const char *options,
                    coco_optimizer_t optimizer) {
    int function_index;
    coco_problem_t *problem;
    for (function_index = 0;function_index<10; ++function_index) {
        problem = coco_get_problem(problem_suit, function_index);
        if (NULL == problem)
            break;
        problem = coco_observe_problem(observer, problem, options);
        optimizer(problem);
        /* Free problem after optimization. */
        coco_free_problem(problem);
    }
}
