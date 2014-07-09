#include <iostream>

#include "numbbo.h"

using namespace std;

void my_optimizer(numbbo_problem_t *problem) {
    static const int budget = 100000;
    const size_t number_of_variables = numbbo_get_number_of_variables(problem);
    numbbo_random_state_t *rng = numbbo_new_random(0xdeadbeef);
    double *x = (double *)malloc(number_of_variables * sizeof(double));
    double y;
    for (int i = 0; i < budget; ++i) {
        for (unsigned int j = 0; j < number_of_variables; ++j) {
            const double range = problem->upper_bounds[j] - problem->lower_bounds[j];
            x[j] = problem->lower_bounds[j] + numbbo_uniform_random(rng) * range;
        }
        numbbo_evaluate_function(problem, x, &y);
    }
    numbbo_free_random(rng);
    free(x);
}

int main(int argc, char **argv) {
    cout << "Greetings from C++ land." << endl;
    numbbo_benchmark("toy_suit", "toy_observer", "random_search", my_optimizer);
    return 0;
}
