#include "minunit.h"
#include "minunit.h"
#include "coco.c"

/**
 * Tests coco_archive-related functions.
 */
MU_TEST(test_coco_archive) {

  unsigned long number_of_evaluations;
  size_t i;
  char file_name[] = "test_hypervolume.txt";
  double *x = coco_allocate_vector(2);
  double *y = coco_allocate_vector(2);
  double hypervolume_read, hypervolume_computed;
  int scan_return;
  coco_archive_t *archive;

  const char *text;
  char *line;
  size_t number_of_solutions;
  size_t number;
  size_t numbers[11] = { 0, 0, 199, 124, 48, 84, 166, 191, 168, 187, 161 };

  FILE *f_results = fopen(file_name, "r");
  if (f_results == NULL) {
    coco_error("test_coco_archive() failed to open file '%s'.", file_name);
  }

  archive = coco_archive("bbob-biobj", 23, 2, 5);

  while (f_results) {
    /* Reads the values from the file */
    scan_return = fscanf(f_results, "%lu\t%lf\t%lf\t%lf\t%lf\t%lf\n", &number_of_evaluations, &x[0], &x[1],
        &y[0], &y[1], &hypervolume_read);

    if (scan_return != 6)
      break;

    /* Add solution to the archive */
    line = coco_strdupf("%lu\t%f\t%f\t%f\t%f\t%f\n", number_of_evaluations, x[0], x[1], y[0],
    		y[1], hypervolume_read);
    coco_archive_add_solution(archive, y[0], y[1], line);
    coco_free_memory(line);
  }
  fclose(f_results);

  coco_free_memory(x);
  coco_free_memory(y);

  /* Check if the values are correct */
  number_of_solutions = coco_archive_get_number_of_solutions(archive);
  mu_check(number_of_solutions == 11);

  /* Checks that the computed hypervolume is correct */
  hypervolume_computed = coco_archive_get_hypervolume(archive);
  mu_assert_double_eq(hypervolume_read, hypervolume_computed);

  i = 0;
  while (strcmp(text = coco_archive_get_next_solution_text(archive), "") != 0) {
    number = (size_t) strtol(text, NULL, 10);
    mu_check(numbers[i] == number);
    i++;
  }

  coco_archive_free(archive);
}

/**
 * Tests updating the coco_archive with solutions better than the extremes.
 */
MU_TEST(test_coco_archive_extreme_solutions) {

  size_t number_of_evaluations;
  double *y = coco_allocate_vector(2);
  char *line;
  coco_archive_t *archive;
  double hypervolume;

  archive = coco_archive("bbob-biobj", 18, 2, 3);

  number_of_evaluations = 0;
  y[0] = 5.222459558139120e+03;
  y[1] = -1.054000000000000e+01;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  number_of_evaluations = 0;
  y[0] = 2.070000000000000e+001;
  y[1] = 3.604904192800397e+002;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  number_of_evaluations = 1;
  y[0] = 5.2222140666e+03;
  y[1] = -1.0540000000e+01;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  /* Checks that the computed hypervolume is correct */
  mu_check(coco_archive_get_number_of_solutions(archive) == 2);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_assert_double_eq(0.0, hypervolume);

  number_of_evaluations = 1;
  y[0] = 5.2220345785e+03;
  y[1] = -1.0539999999e+01;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  /* Checks that the computed hypervolume is correct */
  mu_check(coco_archive_get_number_of_solutions(archive) == 3);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_assert_double_eq(8.1699208579037619e-05, hypervolume);

  coco_free_memory(y);
  coco_archive_free(archive);
}

/**
 * Tests updating the coco_archive with similar solutions.
 */
MU_TEST(test_coco_archive_precision_issues) {

  size_t number_of_evaluations, count;
  double *y = coco_allocate_vector(2);
  char *line;
  coco_archive_t *archive;
  double hypervolume;

  /* First example */

  archive = coco_archive("bbob-biobj", 1, 2, 1);

  number_of_evaluations = 0;
  y[0] = 4.262796608000000e+02;
  y[1] = -1.520400000000000e+02;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_check(count == 2);
  mu_assert_double_eq(0.0, hypervolume);

  number_of_evaluations = 0;
  y[0] = 3.944800000000000e+02;
  y[1] = -1.202403392000000e+02;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_check(count == 2);
  mu_assert_double_eq(0.0, hypervolume);

  number_of_evaluations = 342;
  y[0] = 4.262796567880355e+02;
  y[1] = -1.520399999999999e+02;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_check(count == 2);
  mu_assert_double_eq(0.0, hypervolume);

  number_of_evaluations = 351;
  y[0] = 4.262796555526893e+02;
  y[1] = -1.520399999999998e+02;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  mu_check(count == 2);
  mu_assert_double_eq(0.0, hypervolume);

  number_of_evaluations = 2240;
  y[0] = 4.262796544864155e+02;
  y[1] = -1.520399999999997e+02;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  mu_check(count == 2);
  mu_assert_double_eq(0.0, hypervolume);

  coco_archive_free(archive);

  /* Second example */

  archive = coco_archive("bbob-biobj", 12, 10, 7);

  number_of_evaluations = 1;
  y[0] = 2815;
  y[1] = 100;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_check(count == 3);
  mu_assert_double_eq(0.9998501993077, hypervolume);

  number_of_evaluations = 1;
  y[0] = 2790;
  y[1] = 306;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_check(count == 4);
  mu_assert_double_eq(9.998510436916014e-01, hypervolume);

  number_of_evaluations = 5118865;
  y[0] = 2.809875541591976e+03;
  y[1] = 3.059972106158142e+02 ;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  hypervolume = coco_archive_get_hypervolume(archive);
  mu_check(count == 5);
  mu_assert_double_eq(9.998510436916023e-01, hypervolume);

  number_of_evaluations = 2173490;
  y[0] = 2.814996411168355e+03;
  y[1] = 3.047803494075471e+02;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  mu_check(count == 6);

  number_of_evaluations = 361541;
  y[0] = 2.940001451714048e+03;
  y[1] = 2.886826783155979e+02;
  line = coco_strdupf("%lu\t%f\t%f\n", (unsigned long) number_of_evaluations, y[0], y[1]);
  coco_archive_add_solution(archive, y[0], y[1], line);
  coco_free_memory(line);

  count = coco_archive_get_number_of_solutions(archive);
  mu_check(count == 6);

  coco_free_memory(y);
  coco_archive_free(archive);
}

int main(void) {
  MU_RUN_TEST(test_coco_archive);
  MU_RUN_TEST(test_coco_archive_extreme_solutions);
  MU_RUN_TEST(test_coco_archive_precision_issues);
	
	MU_REPORT();

  return minunit_status;
}
