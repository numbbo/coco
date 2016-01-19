#include "coco.h"
#include "mo_generics.c"
#include "suite_bbob.c"
#include "suite_biobj_best_values_hyp.c"

/* An array of triples biobj_instance - problem1_instance - problem2_instance that should be updated
 * with new instances when they are selected. */
static const size_t suite_biobj_instances[][3] = {
    { 1, 2, 4 },
    { 2, 3, 5 },
    { 3, 7, 8 },
    { 4, 9, 10 },
    { 5, 11, 12 },
    { 6, 13, 14 },
    { 7, 15, 16 },
    { 8, 17, 18 },
    { 9, 19, 21 },
    { 10, 21, 22 }
};

/* Data for the biobjective suite */
typedef struct {

  /* A matrix of new instances (equal in form to suite_biobj_instances) that needs to be used only when
   * an instance that is not in suite_biobj_instances is being invoked. */
  size_t **new_instances;
  size_t max_new_instances;

} suite_biobj_t;

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);
static void suite_biobj_free(void *stuff);
static size_t suite_biobj_get_new_instance(coco_suite_t *suite,
                                           const size_t instance,
                                           const size_t instance1,
                                           const size_t num_bbob_functions,
                                           const size_t *bbob_functions);

static coco_suite_t *suite_biobj_allocate(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("bbob-biobj", 55, 6, dimensions, "instances:1-10");

  return suite;
}

static char *suite_biobj_get_instances_by_year(const int year) {

  if (year == 2016) {
    return "1-10";
  }
  else {
    coco_error("suite_biobj_get_instances_by_year(): year %d not defined for suite_biobj", year);
    return NULL;
  }
}

static coco_problem_t *suite_biobj_get_problem(coco_suite_t *suite,
                                               const size_t function_idx,
                                               const size_t dimension_idx,
                                               const size_t instance_idx) {

  const size_t num_bbob_functions = 10;
  const size_t bbob_functions[] = { 1, 2, 6, 8, 13, 14, 15, 17, 20, 21 };

  coco_problem_t *problem1, *problem2, *problem = NULL;
  size_t function1_idx, function2_idx;
  size_t instance1 = 0, instance2 = 0;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  suite_biobj_t *data = (suite_biobj_t *) suite->data;
  size_t i, j;
  const size_t num_existing_instances = sizeof(suite_biobj_instances) / sizeof(suite_biobj_instances[0]);
  int instance_found = 0;

  /* A "magic" formula to compute the BBOB function index from the bi-objective function index */
  function1_idx = num_bbob_functions -
      (size_t) (-0.5 + sqrt(0.25 + 2.0 * (double) (suite->number_of_functions - function_idx - 1))) - 1;
  function2_idx = function_idx - (function1_idx * num_bbob_functions) +
      (function1_idx * (function1_idx + 1)) / 2;

  /* First search for instance in suite_biobj_instances */
  for (i = 0; i < num_existing_instances; i++) {
    if (suite_biobj_instances[i][0] == instance) {
      /* The instance has been found in suite_biobj_instances */
      instance1 = suite_biobj_instances[i][1];
      instance2 = suite_biobj_instances[i][2];
      instance_found = 1;
      break;
    }
  }

  if ((!instance_found) && (data)) {
    /* Next, search for instance in new_instances */
    for (i = 0; i < data->max_new_instances; i++) {
      if (data->new_instances[i][0] == 0)
        break;
      if (data->new_instances[i][0] == instance) {
        /* The instance has been found in new_instances */
        instance1 = data->new_instances[i][1];
        instance2 = data->new_instances[i][2];
        instance_found = 1;
        break;
      }
    }
  }

  if (!instance_found) {
    /* Finally, if the instance is not found, create a new one */

    if (!data) {
      /* Allocate space needed for saving new instances */
      data = (suite_biobj_t *) coco_allocate_memory(sizeof(*data));

      /* Most often the actual number of new instances will be lower than max_new_instances, because
       * some of them are already in suite_biobj_instances. However, in order to avoid iterating over
       * suite_biobj_instances, the allocation uses max_new_instances. */
      data->max_new_instances = suite->number_of_instances;

      data->new_instances = coco_allocate_memory(data->max_new_instances * sizeof(size_t *));
      for (i = 0; i < data->max_new_instances; i++) {
        data->new_instances[i] = malloc(3 * sizeof(size_t));
        for (j = 0; j < 3; j++) {
          data->new_instances[i][j] = 0;
        }
      }
      suite->data_free_function = suite_biobj_free;
      suite->data = data;
    }

    /* A simple formula to set the first instance */
    instance1 = 2 * instance + 1;
    instance2 = suite_biobj_get_new_instance(suite, instance, instance1, num_bbob_functions, bbob_functions);
  }

  problem1 = get_bbob_problem(bbob_functions[function1_idx], dimension, instance1);
  problem2 = get_bbob_problem(bbob_functions[function2_idx], dimension, instance2);

  problem = coco_stacked_problem_allocate(problem1, problem2);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  /* Use the standard stacked problem_id as problem_name and construct a new suite-specific problem_id */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "bbob-biobj_f%02d_i%02ld_d%02d", function, instance, dimension);

  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem1->problem_type, problem2->problem_type);

  return problem;
}

static size_t suite_biobj_get_new_instance(coco_suite_t *suite,
                                           const size_t instance,
                                           const size_t instance1,
                                           const size_t num_bbob_functions,
                                           const size_t *bbob_functions) {
  size_t instance2 = 0;
  size_t num_tries = 0;
  const size_t max_tries = 1000;
  const double apart_enough = 1e-4;
  int appropriate_instance_found = 0, break_search, warning_produced = 0;
  coco_problem_t *problem1, *problem2, *problem = NULL;
  size_t d, f1, f2, i;
  size_t function1, function2, dimension;
  double norm;

  suite_biobj_t *data;
  assert(suite->data);
  data = (suite_biobj_t *) suite->data;

  while ((!appropriate_instance_found) && (num_tries < max_tries)) {
    num_tries++;
    instance2 = instance1 + num_tries;
    break_search = 0;

    /* An instance is "appropriate" if the ideal and nadir points in the objective space and the two
     * extreme optimal points in the decisions space are apart enough for all problems (all dimensions
     * and function combinations); therefore iterate over all dimensions and function combinations  */
    for (f1 = 0; (f1 < num_bbob_functions) && !break_search; f1++) {
      function1 = bbob_functions[f1];
      for (f2 = f1; (f2 < num_bbob_functions) && !break_search; f2++) {
        function2 = bbob_functions[f2];
        for (d = 0; (d < suite->number_of_dimensions) && !break_search; d++) {
          dimension = suite->dimensions[d];

          if (dimension == 0) {
            if (!warning_produced)
              coco_warning("suite_biobj_get_new_instance(): remove filtering of dimensions to get generally acceptable instances!");
            warning_produced = 1;
            continue;
          }

          problem1 = get_bbob_problem(function1, dimension, instance1);
          problem2 = get_bbob_problem(function2, dimension, instance2);
          if (problem) {
            coco_stacked_problem_free(problem);
            problem = NULL;
          }
          problem = coco_stacked_problem_allocate(problem1, problem2);

          /* Check whether the ideal and reference points are too close in the objective space */
          norm = mo_get_norm(problem->best_value, problem->nadir_value, 2);
          if (norm < 1e-1) { /* TODO How to set this value in a sensible manner? */
            coco_debug(
                "suite_biobj_get_new_instance(): The ideal and nadir points of %s are too close in the objective space",
                problem->problem_id);
            coco_debug("norm = %e, ideal = %e\t%e, nadir = %e\t%e", norm, problem->best_value[0],
                problem->best_value[1], problem->nadir_value[0], problem->nadir_value[1]);
            break_search = 1;
          }

          /* Check whether the extreme optimal points are too close in the decision space */
          norm = mo_get_norm(problem1->best_parameter, problem2->best_parameter, problem->number_of_variables);
          if (norm < apart_enough) {
            coco_debug(
                "suite_biobj_get_new_instance(): The extremal optimal points of %s are too close in the decision space",
                problem->problem_id);
            coco_debug("norm = %e", norm);
            break_search = 1;
          }
        }
      }
    }
    /* Clean up */
    if (problem) {
      coco_stacked_problem_free(problem);
      problem = NULL;
    }

    if (break_search) {
      /* The search was broken, continue with next instance2 */
      continue;
    } else {
      /* An appropriate instance was found */
      appropriate_instance_found = 1;
      coco_info("suite_biobj_set_new_instance(): Instance %lu created from instances %lu and %lu", instance,
          instance1, instance2);

      /* Save the instance to new_instances */
      for (i = 0; i < data->max_new_instances; i++) {
        if (data->new_instances[i][0] == 0) {
          data->new_instances[i][0] = instance;
          data->new_instances[i][1] = instance1;
          data->new_instances[i][2] = instance2;
          break;
        };
      }
    }
  }

  if (!appropriate_instance_found) {
    coco_error("suite_biobj_get_new_instance(): Could not find suitable instance %lu in $lu tries", instance, num_tries);
    return 0; /* Never reached */
  }

  return instance2;
}

/**
 * Frees the memory of the given biobjective suite.
 */
static void suite_biobj_free(void *stuff) {

  suite_biobj_t *data;
  size_t i;

  assert(stuff != NULL);
  data = stuff;

  if (data->new_instances) {
    for (i = 0; i < data->max_new_instances; i++) {
      if (data->new_instances[i]) {
        coco_free_memory(data->new_instances[i]);
        data->new_instances[i] = NULL;
      }
    }
  }
  coco_free_memory(data->new_instances);
  data->new_instances = NULL;
}

/**
 * Returns the best known value for indicator_name matching the given key if the key is found, and raises an
 * error otherwise.  */
static double suite_biobj_get_best_value(const char *indicator_name, const char *key) {

  size_t i, count;
  double best_value = 0;
  char *curr_key;

  if (strcmp(indicator_name, "hyp") == 0) {

    curr_key = coco_allocate_memory(COCO_PATH_MAX * sizeof(char));
    count = sizeof(suite_biobj_best_values_hyp) / sizeof(char *);
    for (i = 0; i < count; i++) {
      sscanf(suite_biobj_best_values_hyp[i], "%s %lf", curr_key, &best_value);
      if (strcmp(curr_key, key) == 0) {
        coco_free_memory(curr_key);
        return best_value;
      }
    }

    coco_free_memory(curr_key);
    coco_warning("suite_biobj_get_best_value(): best value of %s could not be found; set to 1.0", key);
    return 1.0;

  } else {
    coco_error("suite_biobj_get_best_value(): indicator %s not supported", indicator_name);
    return 0; /* Never reached */
  }

  coco_error("suite_biobj_get_best_value(): unexpected exception");
  return 0; /* Never reached */
}
