/**
 * @file rw_top_trumps.c
 *
 * @brief Implementation of the real-world problems about balancing top trumps decks.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "rw_top_trumps.h"

/**
 * @brief Data type used by the rw-top-trumps problem.
 */
typedef struct {
  size_t function;
  size_t instance;
} rw_top_trumps_data_t;

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Calls the top_trumps_evaluate function from the rw_top_trumps library to evaluate the problem.
 */
static void rw_top_trumps_evaluate(coco_problem_t *problem, const double *x, double *y) {

  rw_top_trumps_data_t *data = (rw_top_trumps_data_t *) problem->data;
  size_t i;

  coco_debug("evaluation #%lu", (unsigned long)problem->evaluations);

  if (coco_vector_contains_nan(x, problem->number_of_variables)) {
    for (i = 0; i < problem->number_of_objectives; i++)
      y[i] = NAN;
    return;
  }

  top_trumps_evaluate(data->function, data->instance, problem->number_of_variables,
      (double *) x, problem->number_of_objectives, y);
}
#ifdef __cplusplus
}
#endif


/**
 * @brief Creates a single- or bi-objective rw_top_trumps problem.
 */
static coco_problem_t *rw_top_trumps_problem_allocate(const char *suite_name,
                                                      const size_t objectives,
                                                      const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance) {

  rw_top_trumps_data_t *data;
  coco_problem_t *problem = NULL;
  size_t i;

  data = (rw_top_trumps_data_t *) coco_allocate_memory(sizeof(*data));
  data->function = function;
  data->instance = instance;
  if (objectives == 2)
    data->function = function + 5;

  if ((objectives != 1) && (objectives != 2))
    coco_error("rw_top_trumps_problem_allocate(): %lu objectives are not supported (only 1 or 2)",
        (unsigned long)objectives);

  problem = coco_problem_allocate(dimension, objectives, 0);
  for (i = 0; i < dimension; ++i) {
    problem->smallest_values_of_interest[i] = 1;
    problem->largest_values_of_interest[i] = 100;
  }
  problem->number_of_integer_variables = dimension;
  problem->evaluate_function = rw_top_trumps_evaluate;
  problem->problem_free_function = NULL;

  coco_problem_set_id(problem, "%s_f%03lu_i%02lu_d%02lu", suite_name, (unsigned long) function,
      (unsigned long) instance, (unsigned long) dimension);

  if (objectives == 1) {
    coco_problem_set_name(problem, "real-world Top Trumps single-objective problem f%lu instance %lu in %luD",
        function, instance, dimension);
    coco_problem_set_type(problem, "single-objective");
    /* TODO Add realistic best values */
    problem->best_value[0] = 0;
  }
  else if (objectives == 2) {
    coco_problem_set_name(problem, "real-world Top Trumps bi-objective problem f%lu instance %lu in %luD",
        function, instance, dimension);
    coco_problem_set_type(problem, "bi-objective");
    /* TODO Add realistic best values */
    problem->best_value[0] = 0;
    problem->best_value[1] = 0;
    problem->nadir_value[0] = 1000;
    problem->nadir_value[1] = 1000;
  }

  if (problem->best_parameter != NULL) {
    coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
  }

  problem->data = data;

  return problem;
}
