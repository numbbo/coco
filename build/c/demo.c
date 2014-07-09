#include <stdlib.h>
#include <stdio.h>

#include "numbbo.h"

void my_optimizer(numbbo_problem_t *problem) {
    static const int budget = 10000;
    numbbo_random_state_t *rng = numbbo_new_random(0xdeadbeef);
    const double *lower, *upper;
    lower = numbbo_get_smallest_values_of_interest(problem);
    upper = numbbo_get_largest_values_of_interest(problem);

    /* Skip any problems with more than 20 variables */
    if (numbbo_get_number_of_variables(problem) > 20) 
        return;
    double *x = (double *)malloc(numbbo_get_number_of_variables(problem) * sizeof(double));
    double y;
    for (int i = 0; i < budget; ++i) {
        for (int j = 0; j < numbbo_get_number_of_variables(problem); ++j) {
            const double range = upper[j] - lower[j];
            x[j] = lower[j] + numbbo_uniform_random(rng) * range;
        }
        numbbo_evaluate_function(problem, x, &y);
    }
    numbbo_free_random(rng);
    free(x);
}

int main(int argc, char **argv) {
    numbbo_benchmark("toy_suit", "logger_observer",//"toy_observer",
                     "random_search", my_optimizer);
}
