#include <stdlib.h>
#include <stdio.h>

#include "coco.h"

void my_optimizer(coco_problem_t *problem) {
    int i;
    static const int budget = 100000;
    const size_t number_of_variables = coco_get_number_of_variables(problem);
    coco_random_state_t *rng = coco_new_random(0xdeadbeef);
    double *x = coco_allocate_vector(number_of_variables);
    const double *lower = coco_get_smallest_values_of_interest(problem);
    const double *upper = coco_get_largest_values_of_interest(problem);
    double y;

    /* Skip any problems with more than 20 variables */
    if (number_of_variables > 20) 
        return;
    for (i = 0; i < budget; ++i) {
        size_t j;
        for (j = 0; j < number_of_variables; ++j) {
            const double range = upper[j] - lower[j];
            x[j] = lower[j] + coco_uniform_random(rng) * range;
        }
        coco_evaluate_function(problem, x, &y);
    }
    coco_free_random(rng);
    coco_free_memory(x);
    /*if (problem != NULL) { //TODO: make this work with no seg faults. Needed for logging the best value at the end of a run
        coco_free_problem(problem);
    }*/
    
}

int main() {
    coco_benchmark("toy_suit", "bbob2009_observer", "random_search", my_optimizer);
    return 0;
}
