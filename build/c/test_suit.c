#include <stdio.h>
#include <float.h>
#include <math.h>

#include "coco.h"

typedef struct {
  double x[40];
} testvector_t;

static int about_equal(const double a, const double b) {
    /* Shortcut to avoid the case where a - b is tiny and both a and b
     * are close to or equal to 0. 
     *
     * Example: a = +EPS and b = -EPS then the relative error is 2 but
     * in fact the two numbers are both for all practical purposes 0.
     */
    const double absolute_error = fabs(a - b);
    const double larger = fabs(a) > fabs(b) ? a : b;
    const double relative_error = fabs((a - b) / larger);

    if (absolute_error < 2 * DBL_MIN) return 1;
    return relative_error < 0.0000001;
}

int main(int argc, char **argv) {
    int header_shown = 0, number_of_failures = 0;
    int number_of_testvectors = 0, number_of_testcases = 0, i, j;
    testvector_t *testvectors;
    int function_id, testvector_id, ret;
    double expected_value, *x, y;
    coco_problem_t *problem;
    char suit_name[128];
    FILE *testfile;
    
    if (argc != 2) {
        fprintf(stderr, "Usage:\n  %s <testcasefile>\n", argv[0]);
        return 1;
    }
    testfile = fopen(argv[1], "r");
    if (testfile == NULL) {
        fprintf(stderr, "Failed to open testcases file %s.\n", argv[1]);
        return 2;
    }

    ret = fscanf(testfile, "%s", suit_name);
    if (ret != 1) {
        fprintf(stderr, "Failed to read suit name from testcases file.\n");
        return 3;
    }
    
    ret = fscanf(testfile, "%i", &number_of_testvectors);
    if (ret != 1) {
        fprintf(stderr, "Failed to read number of test vectors from testcases file.\n");
        return 3;
    }

    testvectors = malloc(number_of_testvectors * sizeof(*testvectors));
    if (NULL == testvectors) {
        fprintf(stderr, "Failed to allocate memory for testvectors.\n");
        return 4;
    }

    for (i = 0; i < number_of_testvectors; ++i) {
        for (j = 0; j < 40; ++j) {
            ret = fscanf(testfile, "%lf", &testvectors[i].x[j]);
            if (ret != 1) {
                fprintf(stderr, "ERROR: Failed to parse testvector %i element %i.\n", 
                        i + 1, j + 1);
            }
        }
    }

    while (1) {
        ret = fscanf(testfile, "%i %i %lf", 
                     &function_id, &testvector_id, &expected_value);
        if (ret != 3) 
            break;
        ++number_of_testcases;
        problem = coco_get_problem(suit_name, function_id);
        x = testvectors[testvector_id].x;
        coco_evaluate_function(problem, x, &y);
        coco_free_problem(problem);
        if (!about_equal(expected_value, y)) {
            ++number_of_failures;
            if (!header_shown) {
                fprintf(stdout, "Function Testcase Status Message\n");
                header_shown = 1;
            }
            fprintf(stdout, "%8i %8i FAILED expected=%.8f observed=%.8f\n",
                    function_id, testvector_id, expected_value, y);
        }
    }
    fclose(testfile);
    fprintf(stderr, "%i of %i tests passed (failure rate %.2f%%)\n",
            number_of_testcases - number_of_failures, (int)number_of_testcases,
            (100.0 * number_of_failures) / number_of_testcases);
    return 0;
}
