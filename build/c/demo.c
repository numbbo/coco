#include <stdlib.h>
#include <stdio.h>

#include "numbbo.h"

void my_optimizer(numbbo_problem_t *problem) {
    static const int budget = 100000;
    numbbo_random_state_t *rng = numbbo_new_random(0xdeadbeef);
    /* Skip any problems with more than 20 parameters */
    if (problem->number_of_parameters > 20) 
        return;
    double *x = (double *)malloc(problem->number_of_parameters * sizeof(double));
    double y;
    for (int i = 0; i < budget; ++i) {
        for (int j = 0; j < problem->number_of_parameters; ++j) {
            const double range = problem->upper_bounds[j] - problem->lower_bounds[j];
            x[j] = problem->lower_bounds[j] + numbbo_uniform_random(rng) * range;
        }
        numbbo_evaluate_function(problem, x, &y);
    }
    numbbo_free_random(rng);
    free(x);
}

int main(int argc, char **argv) {
    numbbo_benchmark("toy_suit", "toy_observer", 
                     "random_search", my_optimizer);
}
