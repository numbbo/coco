/**
 * @file rw_top_trumps.c
 *
 * @brief Implementation of the real-world problems about balancing top trumps decks.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "rw_problem.c"

/**
 * @brief Creates a single- or bi-objective rw_top_trumps problem.
 */
static coco_problem_t *rw_top_trumps_problem_allocate(const char *suite_name,
                                                      const size_t objectives,
                                                      const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance) {

  coco_problem_t *problem = NULL;
  char *str1, *str2, *str3;
  size_t i, num;

  if ((objectives != 1) && (objectives != 2))
    coco_error("rw_top_trumps_problem_allocate(): %lu objectives are not supported (only 1 or 2)",
        (unsigned long)objectives);

  problem = coco_problem_allocate(dimension, objectives, 0);
  for (i = 0; i < dimension; ++i) {
    problem->smallest_values_of_interest[i] = 1;
    problem->largest_values_of_interest[i] = 100;
  }
  problem->number_of_integer_variables = dimension;
  problem->evaluate_function = rw_problem_evaluate;
  problem->problem_free_function = rw_problem_data_free;

  coco_problem_set_id(problem, "%s_f%03lu_i%02lu_d%02lu", suite_name, (unsigned long) function,
      (unsigned long) instance, (unsigned long) dimension);

  if (objectives == 1) {
    coco_problem_set_name(problem, "real-world Top Trumps single-objective problem f%lu instance %lu in %luD",
        function, instance, dimension);
    coco_problem_set_type(problem, "single-objective");
  }
  else if (objectives == 2) {
    coco_problem_set_name(problem, "real-world Top Trumps bi-objective problem f%lu instance %lu in %luD",
        function, instance, dimension);
    coco_problem_set_type(problem, "bi-objective");
  }

  /* TODO Add realistic best values */
  problem->best_value[0] = 0;
  if (problem->best_parameter != NULL) {
    coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
  }

  problem->data = get_rw_problem_data("top-trumps", objectives, function, dimension, instance);

  return problem;
}
