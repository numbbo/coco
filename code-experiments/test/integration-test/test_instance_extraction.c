/*
 * Test instance extraction for the bbob and biobj problem suites.
 */

#include <stdio.h>
#include <float.h>
#include <math.h>

#include "coco.h"

int test_instance_extraction(char *suite_name) {

  coco_suite_t *suite;
  coco_problem_t *problem = NULL;
  size_t index;
  size_t max_index;

  suite = coco_suite(suite_name, NULL, NULL);
  max_index = coco_suite_get_number_of_problems(suite) - 1;

  while ((problem = coco_suite_get_next_problem(suite, NULL)) != NULL) {
    index = coco_problem_get_suite_dep_index(problem);
    printf("Problem %4lu: %s found!\n", (unsigned long) index, coco_problem_get_id(problem));
  }
  coco_suite_free(suite);

  if (index < max_index) {
    printf("Only %lu out of all %lu problems extracted from suite %s\n", (unsigned long) index,
    		(unsigned long) max_index, suite_name);
    return 1;
  }
  return 0;
}

int main(void) {

  /* Mute output that is not error */
  coco_set_log_level("error");

  if (test_instance_extraction("bbob") != 0)
    return 1;

  if (test_instance_extraction("bbob-biobj") != 0)
    return 1;
    
  if (test_instance_extraction("bbob-constrained") != 0)
    return 1;

  if (test_instance_extraction("bbob-mixint") != 0)
    return 1;

  return 0;
}
