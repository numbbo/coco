#include <string.h>
#include <stdio.h>

#include "numbbo.h"

#include "toy_suit.c"
#include "toy_observer.c"

numbbo_problem_t *numbbo_get_problem(const char *problem_suit,
                                     const int function_index) {
    if (0 == strcmp(problem_suit, "toy_suit")) {
        return toy_suit(function_index);
    } else {
        numbbo_error("Unknown problem suit.");
        return NULL; /* Never reached */        
    }
}

numbbo_problem_t *numbbo_observe_problem(const char *observer,
                                         numbbo_problem_t *problem,
                                         const char *options) {
    if (0 == strcmp(observer, "toy_observer")) {
        return toy_observer(problem, options);
    } else {
        numbbo_error("Unknown observer.");
        return NULL; /* Never reached */        
    }
}

void numbbo_benchmark(const char *problem_suit,
                      const char *observer,
                      const char *options,
                      numbbo_optimizer_t optimizer) {
    int function_index;
    numbbo_problem_t *problem;
    for (function_index = 0;; ++function_index) {
        problem = numbbo_get_problem(problem_suit, function_index);
        if (NULL == problem)
            break;
        problem = numbbo_observe_problem(observer, problem, options);
        optimizer(problem);
    }
}
