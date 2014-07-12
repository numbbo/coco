#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>

#include "numbbo.h"
#include "test_bbob2009.h"

static bool about_equal(const double a, const double b) {
    /* Shortcut to avoid the case where a - b is tiny and both a and b
     * are close to or equal to 0. 
     *
     * Example: a = +EPS and b = -EPS then the relative error is 2 but
     * in fact the two numbers are both for all practical purposes 0.
     */
    const double absolute_error = fabs(a - b);
    if (absolute_error < 2 * DBL_MIN) return true;
    
    const double larger = fabs(a) > fabs(b) ? a : b;
    const double relative_error = fabs((a - b) / larger);
    return relative_error < 0.0000001;
}

int main(int argc, char **argv) {    
    bool header_shown = false;
    int number_of_failures = 0;
    const size_t number_of_testcases = sizeof(testcases) / sizeof(testcases[0]);
    
    for (int i = 0; i < number_of_testcases; ++i) {
        double y;
        numbbo_problem_t *problem = numbbo_get_problem("bbob2009", 
                                                       testcases[i].function_index);
        double *x = testvectors[testcases[i].testvector_index].x;
        numbbo_evaluate_function(problem, x, &y);
        numbbo_free_problem(problem);
        if (!about_equal(testcases[i].y, y)) {
            ++number_of_failures;
            if (!header_shown) {
                fprintf(stdout, "Function Testcase Status Message\n");
                header_shown = true;
            }
            fprintf(stdout, "%8i %8i FAILED expected=%.8lf observed=%.8lf\n",
                    testcases[i].function_index, 
                    testcases[i].testvector_index,
                    testcases[i].y, y);
        }
    }
    fprintf(stderr, "%lu of %i tests passed (failure rate %.2f%%)\n",
            number_of_testcases - number_of_failures, (int)number_of_testcases,
            (100.0 * number_of_failures) / number_of_testcases);
    return 0;
}
