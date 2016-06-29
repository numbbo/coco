/**
 * @file suite_biobj.c
 * @brief Implementation of the bbob-biobj suite containing 55 functions and 6 dimensions.
 *
 * The bi-objective suite was created by combining two single-objective problems from the bbob suite.
 *
 * @note Because some bi-objective problems constructed from two single-objective ones have a single optimal
 * value, some care must be taken when selecting the instances. The already verified instances are stored in
 * suite_biobj_instances. If a new instance of the problem is called, a check ensures that the two underlying
 * single-objective instances create a true bi-objective problem. However, these new instances need to be
 * manually added to suite_biobj_instances, otherwise they will be computed each time the suite constructor
 * is invoked with these instances.
 */

#include "coco.h"
#include "mo_utilities.c"
#include "suite_bbob.c"
#include "suite_biobj_best_values_hyp.c"

/**
 * @brief The array of triples biobj_instance - problem1_instance - problem2_instance connecting bi-objective
 * suite instances to the instances of the bbob suite.
 *
 * It should be updated with new instances when they are chosen.
 */
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

/**
 * @brief The bbob-biobj suite data type.
 */
typedef struct {

  size_t **new_instances;    /**< @brief A matrix of new instances (equal in form to suite_biobj_instances)
                                   that needs to be used only when an instance that is not in
                                   suite_biobj_instances is being invoked. */

  size_t max_new_instances;  /**< @brief The maximal number of new instances. */

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

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj suite.
 */
static coco_suite_t *suite_biobj_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("bbob-biobj", 55, 6, dimensions, "year: 2016");

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj suite.
 */
static const char *suite_biobj_get_instances_by_year(const int year) {

  if (year == 2016) {
    return "1-10";
  }
  else {
    coco_error("suite_biobj_get_instances_by_year(): year %d not defined for suite_biobj", year);
    return NULL;
  }
}

/**
 * @brief Returns the problem from the bbob-biobj suite that corresponds to the given parameters.
 *
 * Creates the bi-objective problem by constructing it from two single-objective problems from the bbob
 * suite. If the invoked instance number is not in suite_biobj_instances, the function uses the following
 * formula to construct a new appropriate instance:
 *
 *   problem1_instance = 2 * biobj_instance + 1
 *
 *   problem2_instance = problem1_instance + 1
 *
 * If needed, problem2_instance is increased (see also the explanation of suite_biobj_get_new_instance).
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_biobj_get_problem(coco_suite_t *suite,
                                               const size_t function_idx,
                                               const size_t dimension_idx,
                                               const size_t instance_idx) {

  const size_t num_bbob_functions = 10;
  /* Functions from the bbob suite that are used to construct the bi-objective problem. */
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

  double *smallest_values_of_interest = coco_allocate_vector_with_value(dimension, -100);
  double *largest_values_of_interest = coco_allocate_vector_with_value(dimension, 100);

  /* A "magic" formula to compute the BBOB function index from the bi-objective function index */
  function1_idx = num_bbob_functions
      - coco_double_to_size_t(
          floor(-0.5 + sqrt(0.25 + 2.0 * (double) (suite->number_of_functions - function_idx - 1)))) - 1;
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

      data->new_instances = (size_t **) coco_allocate_memory(data->max_new_instances * sizeof(size_t *));
      for (i = 0; i < data->max_new_instances; i++) {
        data->new_instances[i] = (size_t *) malloc(3 * sizeof(size_t));
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

  problem1 = coco_get_bbob_problem(bbob_functions[function1_idx], dimension, instance1);
  problem2 = coco_get_bbob_problem(bbob_functions[function2_idx], dimension, instance2);

  problem = coco_problem_stacked_allocate(problem1, problem2, smallest_values_of_interest, largest_values_of_interest);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  /* Use the standard stacked problem_id as problem_name and construct a new suite-specific problem_id */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "bbob-biobj_f%02lu_i%02lu_d%02lu", (unsigned long) function,
  		(unsigned long) instance, (unsigned long) dimension);

  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem1->problem_type, problem2->problem_type);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}

/**
 * @brief Computes the instance number of the second problem/objective so that the resulting bi-objective
 * problem has more than a single optimal solution.
 *
 * Starts by setting instance2 = instance1 + 1 and increases this number until an appropriate instance has
 * been found (or until a maximum number of tries has been reached, in which case it throws a coco_error).
 * An appropriate instance is the one for which the resulting bi-objective problem (in any considered
 * dimension) has the ideal and nadir points apart enough in the objective space and the extreme optimal
 * points apart enough in the decision space. When the instance has been found, it is output through
 * coco_warning, so that the user can see it and eventually manually add it to suite_biobj_instances.
 */
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
  double *smallest_values_of_interest, *largest_values_of_interest;

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

          problem1 = coco_get_bbob_problem(function1, dimension, instance1);
          problem2 = coco_get_bbob_problem(function2, dimension, instance2);
          if (problem) {
            coco_problem_stacked_free(problem);
            problem = NULL;
          }

          /* Set smallest and largest values of interest to some value (not important which, it just needs to be a
           * vector of doubles of the right dimension) */
          smallest_values_of_interest = coco_allocate_vector_with_value(dimension, -100);
          largest_values_of_interest = coco_allocate_vector_with_value(dimension, 100);
          problem = coco_problem_stacked_allocate(problem1, problem2, smallest_values_of_interest,
          		largest_values_of_interest);
          coco_free_memory(smallest_values_of_interest);
          coco_free_memory(largest_values_of_interest);

          /* Check whether the ideal and nadir points are too close in the objective space */
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
                "suite_biobj_get_new_instance(): The extreme points of %s are too close in the decision space",
                problem->problem_id);
            coco_debug("norm = %e", norm);
            break_search = 1;
          }
        }
      }
    }
    /* Clean up */
    if (problem) {
      coco_problem_stacked_free(problem);
      problem = NULL;
    }

    if (break_search) {
      /* The search was broken, continue with next instance2 */
      continue;
    } else {
      /* An appropriate instance was found */
      appropriate_instance_found = 1;
      coco_info("suite_biobj_set_new_instance(): Instance %lu created from instances %lu and %lu",
      		(unsigned long) instance, (unsigned long) instance1, (unsigned long) instance2);

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
    coco_error("suite_biobj_get_new_instance(): Could not find suitable instance %lu in %lu tries",
    		(unsigned long) instance, (unsigned long) num_tries);
    return 0; /* Never reached */
  }

  return instance2;
}

/**
 * @brief  Frees the memory of the given bi-objective suite.
 */
static void suite_biobj_free(void *stuff) {

  suite_biobj_t *data;
  size_t i;

  assert(stuff != NULL);
  data = (suite_biobj_t *) stuff;

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
 * @brief Returns the best known value for indicator_name matching the given key if the key is found, and
 * throws a coco_error otherwise.
 */
static double suite_biobj_get_best_value(const char *indicator_name, const char *key) {

  size_t i, count;
  double best_value = 0;
  char *curr_key;

  if (strcmp(indicator_name, "hyp") == 0) {

    curr_key = coco_allocate_string(COCO_PATH_MAX);
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
