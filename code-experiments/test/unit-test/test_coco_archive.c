#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include "cmocka.h"
#include "coco.h"

static int about_equal_value(const double a, const double b);

/**
 * Tests coco_archive-related functions.
 */
static void test_coco_archive(void **state) {

  size_t number_of_evaluations, i;
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
    line = coco_strdupf("%lu\t%f\t%f\t%f\t%f\t%f\n", (unsigned long) number_of_evaluations, x[0], x[1], y[0],
    		y[1], hypervolume_read);
    coco_archive_add_solution(archive, y[0], y[1], line);
    coco_free_memory(line);
  }
  fclose(f_results);

  coco_free_memory(x);
  coco_free_memory(y);

  /* Check if the values are correct */
  number_of_solutions = coco_archive_get_number_of_solutions(archive);
  assert(number_of_solutions == 11);

  /* Checks that the computed hypervolume is correct */
  hypervolume_computed = coco_archive_get_hypervolume(archive);
  assert(about_equal_value(hypervolume_computed, hypervolume_read));

  i = 0;
  while ((text = coco_archive_get_next_solution_text(archive)) != "") {
    number = (size_t) strtol(text, NULL, 10);
    assert(numbers[i] == number);
    i++;
  }

  coco_archive_free(archive);

  (void)state; /* unused */
}

static int test_all_coco_archive(void) {

  const struct CMUnitTest tests[] = {
  cmocka_unit_test(test_coco_archive) };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
