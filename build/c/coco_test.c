#include <stdio.h>
#include <float.h>
#include <math.h>

#include "coco.h"

typedef struct { double x[40]; } testvector_t;

static int about_equal(const double a, const double b) {
  /* Shortcut to avoid the case where a - b is tiny and both a and b
   * are close to or equal to 0.
   *
   * Example: a = +EPS and b = -EPS then the relative error is 2 but
   * in fact the two numbers are both for all practical purposes 0.
   */
  if (a == b)
    return 1;
  {
    const double absolute_error = fabs(a - b);
    const double larger = fabs(a) > fabs(b) ? a : b;
    const double relative_error = fabs((a - b) / larger);
  
    if (absolute_error < 2 * DBL_MIN)
      return 1;
    return relative_error < 1e-6;
  }
}

static void usage(const char *program_name) {
  fprintf(
      stderr,
      "COCO function test suit runner\n"
      "\n"
      "Usage:\n"
      "  %s <testcasefile>\n"
      "\n"
      "This program tests the numerical accuracy of the functions in a\n"
      "particular COCO test suit. Its sole argument is the name of a\n"
      "text file that contains the test cases. The COCO distribution contains\n"
      "(at least) the following test cases:\n"
      "\n"
      "  bbob2009_testcases.txt - Covers the 24 noise-free BBOB 2009\n"
      "    functions in 2D, 3D, 5D, 10D, 20D and 40D.\n",
      program_name);
}

int main(int argc, char **argv) {
  int header_shown = 0, number_of_failures = 0, shown_failures = 0;
  int number_of_testvectors = 0, number_of_testcases = 0, i, j;
  testvector_t *testvectors = NULL;
  int previous_function_id = -1, function_id, testvector_id, ret;
  coco_problem_t *problem = NULL;
  char suit_name[128];
  FILE *testfile = NULL;

  if (argc != 2) {
    usage(argv[0]);
    goto err;
  }

  testfile = fopen(argv[1], "r");
  if (testfile == NULL) {
    fprintf(stderr, "Failed to open testcases file %s.\n", argv[1]);
    goto err;
  }

  ret = fscanf(testfile, "%127s", suit_name);
  if (ret != 1) {
    fprintf(stderr, "Failed to read suit name from testcases file.\n");
    goto err;
  }

  ret = fscanf(testfile, "%30i", &number_of_testvectors);
  if (ret != 1) {
    fprintf(stderr,
            "Failed to read number of test vectors from testcases file.\n");
    goto err;
  }

  testvectors = malloc(number_of_testvectors * sizeof(*testvectors));
  if (NULL == testvectors) {
    fprintf(stderr, "Failed to allocate memory for testvectors.\n");
    goto err;
  }

  for (i = 0; i < number_of_testvectors; ++i) {
    for (j = 0; j < 40; ++j) {
      ret = fscanf(testfile, "%30lf", &testvectors[i].x[j]);
      if (ret != 1) {
        fprintf(stderr, "ERROR: Failed to parse testvector %i element %i.\n",
                i + 1, j + 1);
      }
    }
  }

  while (1) {
    double expected_value, *x, y;
    ret = fscanf(testfile, "%30i %30i %30lf", &function_id, &testvector_id,
                 &expected_value);
    if (ret != 3)
      break;
    ++number_of_testcases;
    /* We cache the problem object to save time. Instantiating
     * some functions is expensive because we have to generate
     * large rotation matrices.
     */
    if (previous_function_id != function_id) {
      if (NULL != problem)
        coco_free_problem(problem);
      problem = coco_get_problem(suit_name, function_id);
      previous_function_id = function_id;
    }
    x = testvectors[testvector_id].x;

    coco_evaluate_function(problem, x, &y);
    if (!about_equal(expected_value, y)) {
      ++number_of_failures;
      if (!header_shown) {
        fprintf(stdout, "Function Testcase Status Message\n");
        header_shown = 1;
      }
      if (shown_failures < 100) {
        fprintf(stdout,
                "%8i %8i FAILED expected=%.8e observed=%.8e function_id=%s\n",
                function_id, testvector_id, expected_value, y,
                coco_get_problem_id(problem));
        fflush(stdout);
        ++shown_failures;
      } else if (shown_failures == 100) {
        fprintf(stdout, "... further failed tests suppressed ...\n");
        fflush(stdout);
        ++shown_failures;
      }
    }
  }
  fclose(testfile);
  /* Output summary statistics */
  fprintf(stderr, "%i of %i tests passed (failure rate %.2f%%)\n",
          number_of_testcases - number_of_failures, (int)number_of_testcases,
          (100.0 * number_of_failures) / number_of_testcases);

  /* Free any remaining allocated memory so that we pass valgrind checks. */
  if (NULL != problem)
    coco_free_problem(problem);
  free(testvectors);

  return number_of_failures == 0 ? 0 : 1;

err:
  if (testfile != NULL)
    fclose(testfile);
  if (testvectors != NULL)
    free(testvectors);
  return 2;
}
