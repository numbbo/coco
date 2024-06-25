/**
 * @file suite_biobj_ext.c
 * @brief Implementation of the extended biobjective bbob-biobj-ext suite containing 92 functions and 6 dimensions.
 *
 * The bbob-biobj-ext suite is created by combining two single-objective problems from the bbob suite.
 * The first 55 functions are the same as in the original bbob-biobj test suite to which 37 functions are added.
 * Those additional functions are constructed by combining all not yet contained in-group combinations (i,j) of
 * single-objective bbob functions i and j such that i<j (i.e. in particular not all combinations (i,i) are
 * included in this bbob-biobj-ext suite), with the exception of the Weierstrass function (f16) for which
 * the optimum is not unique and thus a nadir point is difficult to compute, see
 * http://numbbo.github.io/coco-doc/bbob-biobj/functions/ for details.
 *
 * @note Because some bi-objective problems constructed from two single-objective ones have a single optimal
 * value, some care must be taken when selecting the instances. The already verified instances are stored in
 * suite_biobj_ext_instances. If a new instance of the problem is called, a check ensures that the two underlying
 * single-objective instances create a true bi-objective problem. However, these new instances need to be
 * manually added to suite_biobj_ext_instances, otherwise they will be computed each time the suite constructor
 * is invoked with these instances.
 *
 * @note This file is based on the original suite_bbob_biobj.c and extends it by 37 functions in 6 dimensions.
 */

#include "coco.h"
#include "mo_utilities.c"
#include "suite_bbob.c"


/**
 * @brief The array of triples biobj_instance - problem1_instance - problem2_instance connecting bi-objective
 * suite instances to the instances of the bbob suite.
 *
 * It should be updated with new instances when they are chosen.
 */
static const size_t suite_biobj_ext_instances[][3] = {
    { 1, 2, 4 },
    { 2, 3, 5 },
    { 3, 7, 8 },
    { 4, 9, 10 },
    { 5, 11, 12 },
    { 6, 13, 14 },
    { 7, 15, 16 },
    { 8, 17, 18 },
    { 9, 19, 21 },
    { 10, 21, 22 },
    { 11, 23, 24 },
    { 12, 25, 26 },
    { 13, 27, 28 },
    { 14, 29, 30 },
    { 15, 31, 34 }
}; 
 
/**
 * @brief The bbob-biobj-ext suite data type.
 */
typedef struct {

  size_t **new_instances;    /**< @brief A matrix of new instances (equal in form to suite_biobj_ext_instances)
                                   that needs to be used only when an instance that is not in
                                   suite_biobj_ext_instances is being invoked. */

  size_t max_new_instances;  /**< @brief The maximal number of new instances. */

} suite_biobj_ext_t;

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances,
                                         const int known_optima);
static void suite_biobj_ext_free(void *stuff);
static size_t suite_biobj_ext_get_new_instance(coco_suite_t *suite,
                                           const size_t instance,
                                           const size_t instance1,
                                           const size_t num_bbob_functions,
                                           const size_t *bbob_functions);

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj-ext suite.
 */
static coco_suite_t *suite_biobj_ext_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };
  const size_t num_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("bbob-biobj-ext", 55+37, num_dimensions, dimensions, "year: 2018", 1);

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj-ext suite.
 */
static const char *suite_biobj_ext_get_instances_by_year(const int year) {

  if (year == 0000) { /* default/test case */
    return "1-10";
  }
  else if (year >= 2017) {
    return "1-15";
  }
  else if (year >= 2009) {
    return "1-15";
  }
  else {
    coco_error("suite_biobj_ext_get_instances_by_year(): year %d not defined for suite_biobj_ext", year);
    return NULL;
  }
}

/**
 * @brief Returns the problem from the bbob-biobj-ext suite that corresponds to the given parameters.
 *
 * Creates the bi-objective problem by constructing it from two single-objective problems from the bbob
 * suite. If the invoked instance number is not in suite_biobj_ext_instances, the function uses the following
 * formula to construct a new appropriate instance:
 *
 *   problem1_instance = 2 * biobj_instance + 1
 *
 *   problem2_instance = problem1_instance + 1
 *
 * If needed, problem2_instance is increased (see also the explanation of suite_biobj_ext_get_new_instance).
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 * @note Copied from suite_bbob_biobj.c and extended.
 */
static coco_problem_t *suite_biobj_ext_get_problem(coco_suite_t *suite,
                                               const size_t function_idx,
                                               const size_t dimension_idx,
                                               const size_t instance_idx) {

  
  const size_t num_bbob_functions = 10;
  /* Functions from the bbob suite that are used to construct the original bbob-biobj suite. */
  const size_t bbob_functions[] = { 1, 2, 6, 8, 13, 14, 15, 17, 20, 21 };
  /* All functions from the bbob suite for later use during instance generation. */
  const size_t all_bbob_functions[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
  
  coco_problem_t *problem1, *problem2, *problem = NULL;
  size_t instance1 = 0, instance2 = 0;
  size_t function1_idx, function2_idx;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  suite_biobj_ext_t *data = (suite_biobj_ext_t *) suite->data;
  size_t i, j;
  const size_t num_existing_instances = sizeof(suite_biobj_ext_instances) / sizeof(suite_biobj_ext_instances[0]);
  int instance_found = 0;

  double *smallest_values_of_interest = coco_allocate_vector_with_value(dimension, -100);
  double *largest_values_of_interest = coco_allocate_vector_with_value(dimension, 100);
  
  /* First, create all problems from the bbob-biobj suite */
  if (function_idx < 55) {
    /* A "magic" formula to compute the BBOB function index from the bi-objective function index */
    function1_idx = num_bbob_functions
        - coco_double_to_size_t(
            floor(-0.5 + sqrt(0.25 + 2.0 * (double) (55 - function_idx - 1)))) - 1;
    function2_idx = function_idx - (function1_idx * num_bbob_functions) +
        (function1_idx * (function1_idx + 1)) / 2;
        
  } else {
    /* New 41 functions extending the bbob-biobj suite,
     * unfortunately, there is not a simple "magic" formula anymore. */
    if (function_idx == 55) {
        function1_idx = 0;
        function2_idx = 2;
    } else if (function_idx == 56) {
        function1_idx = 0;
        function2_idx = 3;
    } else if (function_idx == 57) {
        function1_idx = 0;
        function2_idx = 4;
    } else if (function_idx == 58) {
        function1_idx = 1;
        function2_idx = 2;
    } else if (function_idx == 59) {
        function1_idx = 1;
        function2_idx = 3;
    } else if (function_idx == 60) {
        function1_idx = 1;
        function2_idx = 4;
    } else if (function_idx == 61) {
        function1_idx = 2;
        function2_idx = 3;
    } else if (function_idx == 62) {
        function1_idx = 2;
        function2_idx = 4;
    } else if (function_idx == 63) {
        function1_idx = 3;
        function2_idx = 4;
    } else if (function_idx == 64) {
        function1_idx = 5;
        function2_idx = 6;
    } else if (function_idx == 65) {
        function1_idx = 5;
        function2_idx = 8;
    } else if (function_idx == 66) {
        function1_idx = 6;
        function2_idx = 7;
    } else if (function_idx == 67) {
        function1_idx = 6;
        function2_idx = 8;
    } else if (function_idx == 68) {
        function1_idx = 7;
        function2_idx = 8;
    } else if (function_idx == 69) {
        function1_idx = 9;
        function2_idx = 10;
    } else if (function_idx == 70) {
        function1_idx = 9;
        function2_idx = 11;
    } else if (function_idx == 71) {
        function1_idx = 9;
        function2_idx = 12;
    } else if (function_idx == 72) {
        function1_idx = 9;
        function2_idx = 13;
    } else if (function_idx == 73) {
        function1_idx = 10;
        function2_idx = 11;
    } else if (function_idx == 74) {
        function1_idx = 10;
        function2_idx = 12;
    } else if (function_idx == 75) {
        function1_idx = 10;
        function2_idx = 13;
    } else if (function_idx == 76) {
        function1_idx = 11;
        function2_idx = 12;
    } else if (function_idx == 77) {
        function1_idx = 11;
        function2_idx = 13;
    } else if (function_idx == 78) {
        function1_idx = 14;
        function2_idx = 17;
    } else if (function_idx == 79) {
        function1_idx = 14;
        function2_idx = 18;
    } else if (function_idx == 80) {
        function1_idx = 16;
        function2_idx = 17;
    } else if (function_idx == 81) {
        function1_idx = 16;
        function2_idx = 18;
    } else if (function_idx == 82) {
        function1_idx = 17;
        function2_idx = 18;
    } else if (function_idx == 83) {
        function1_idx = 19;
        function2_idx = 21;
    } else if (function_idx == 84) {
        function1_idx = 19;
        function2_idx = 22;
    } else if (function_idx == 85) {
        function1_idx = 19;
        function2_idx = 23;
    } else if (function_idx == 86) {
        function1_idx = 20;
        function2_idx = 21;
    } else if (function_idx == 87) {
        function1_idx = 20;
        function2_idx = 22;
    } else if (function_idx == 88) {
        function1_idx = 20;
        function2_idx = 23;
    } else if (function_idx == 89) {
        function1_idx = 21;
        function2_idx = 22;
    } else if (function_idx == 90) {
        function1_idx = 21;
        function2_idx = 23;
    } else if (function_idx == 91) {
        function1_idx = 22;
        function2_idx = 23;
    } 
  }
      
  /* First search for instance in suite_biobj_ext_instances */
  for (i = 0; i < num_existing_instances; i++) {
    if (suite_biobj_ext_instances[i][0] == instance) {
      /* The instance has been found in suite_biobj_ext_instances */
      instance1 = suite_biobj_ext_instances[i][1];
      instance2 = suite_biobj_ext_instances[i][2];
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
      data = (suite_biobj_ext_t *) coco_allocate_memory(sizeof(*data));

      /* Most often the actual number of new instances will be lower than max_new_instances, because
       * some of them are already in suite_biobj_ext_instances. However, in order to avoid iterating over
       * suite_biobj_ext_instances, the allocation uses max_new_instances. */
      data->max_new_instances = suite->number_of_instances;

      data->new_instances = (size_t **) coco_allocate_memory(data->max_new_instances * sizeof(size_t *));
      for (i = 0; i < data->max_new_instances; i++) {
        data->new_instances[i] = (size_t *) malloc(3 * sizeof(size_t));
        for (j = 0; j < 3; j++) {
          data->new_instances[i][j] = 0;
        }
      }
      suite->data_free_function = suite_biobj_ext_free;
      suite->data = data;
    }

    /* A simple formula to set the first instance, but instead of for 10 functions
     * as in bbob-biobj suite, now for all combinations of functions.
     */
    instance1 = 2 * instance + 1;
    instance2 = suite_biobj_ext_get_new_instance(suite, instance, instance1, 24, all_bbob_functions);
  }
  
  if (function_idx < 55) {  
    problem1 = coco_get_bbob_problem(bbob_functions[function1_idx], dimension, instance1);
    problem2 = coco_get_bbob_problem(bbob_functions[function2_idx], dimension, instance2);
  } else {
    problem1 = coco_get_bbob_problem(all_bbob_functions[function1_idx], dimension, instance1);
    problem2 = coco_get_bbob_problem(all_bbob_functions[function2_idx], dimension, instance2);
  }
  
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
 * @brief  Performs a few checks and returns whether the two problem
 * instances given should break the search for new instances in
 * suite_biobj_ext_get_new_instance(...).
 */
static int check_consistency_of_instances(const size_t dimension, 
                                          size_t function1,
                                          size_t instance1,
                                          size_t function2,
                                          size_t instance2) {
  coco_problem_t *problem = NULL;
  coco_problem_t *problem1, *problem2;
  int break_search = 0;
  double norm;
  double *smallest_values_of_interest, *largest_values_of_interest;
  const double apart_enough = 1e-4;
  
  problem1 = coco_get_bbob_problem(function1, dimension, instance1);
  problem2 = coco_get_bbob_problem(function2, dimension, instance2);

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
        "suite_biobj_ext_get_new_instance(): The ideal and nadir points of %s are too close in the objective space",
        problem->problem_id);
    coco_debug("norm = %e, ideal = %e\t%e, nadir = %e\t%e", norm, problem->best_value[0],
        problem->best_value[1], problem->nadir_value[0], problem->nadir_value[1]);
    break_search = 1;
  }

  /* Check whether the extreme optimal points are too close in the decision space */
  norm = mo_get_norm(problem1->best_parameter, problem2->best_parameter, problem->number_of_variables);
  if (norm < apart_enough) {
    coco_debug(
        "suite_biobj_ext_get_new_instance(): The extreme points of %s are too close in the decision space",
        problem->problem_id);
    coco_debug("norm = %e", norm);
    break_search = 1;
  }

  /* Clean up */
  if (problem) {
    coco_problem_stacked_free(problem);
    problem = NULL;
  }

  return break_search;

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
 * coco_warning, so that the user can see it and eventually manually add it to suite_biobj_ext_instances.
 *
 * @note Copied from suite_bbob_biobj.c.
 */
static size_t suite_biobj_ext_get_new_instance(coco_suite_t *suite,
                                           const size_t instance,
                                           const size_t instance1,
                                           const size_t num_bbob_functions,
                                           const size_t *bbob_functions) {
  size_t instance2 = 0;
  size_t num_tries = 0;
  const size_t max_tries = 1000;
  int appropriate_instance_found = 0, break_search, warning_produced = 0;
  size_t d, f1, f2, i;
  size_t function1, function2, dimension;
  const size_t reduced_bbob_functions[] = { 1, 2, 6, 8, 13, 14, 15, 17, 20, 21 };
  
  suite_biobj_ext_t *data;
  assert(suite->data);
  data = (suite_biobj_ext_t *) suite->data;

  while ((!appropriate_instance_found) && (num_tries < max_tries)) {
    num_tries++;
    instance2 = instance1 + num_tries;
    break_search = 0;

    /* An instance is "appropriate" if the ideal and nadir points in the objective space and the two
     * extreme optimal points in the decisions space are apart enough for all problems (all dimensions
     * and function combinations); therefore iterate over all dimensions and function combinations  */
     
    for (f1 = 0; (f1 < num_bbob_functions-1) && !break_search; f1++) {
      function1 = bbob_functions[f1];
      for (f2 = f1+1; (f2 < num_bbob_functions) && !break_search; f2++) {
        function2 = bbob_functions[f2];
        for (d = 0; (d < suite->number_of_dimensions) && !break_search; d++) {
          dimension = suite->dimensions[d];
          
          if (dimension == 0) {
            if (!warning_produced)
              coco_warning("suite_biobj_ext_get_new_instance(): remove filtering of dimensions to get generally acceptable instances!");
            warning_produced = 1;
            continue;
          }
          
          break_search = check_consistency_of_instances(dimension, function1, instance1, function2, instance2);
        }
      }
    }
    
    /* Finally, check all functions (f,f) with f in {f1, f2, f6, f8, f13, f14, f15, f17, f20, f21}: */
    for (f1 = 0; (f1 < 10) && !break_search; f1++) {
      function1 = reduced_bbob_functions[f1];
      function2 = reduced_bbob_functions[f1];
      for (d = 0; (d < suite->number_of_dimensions) && !break_search; d++) {
        dimension = suite->dimensions[d];
        
        if (dimension == 0) {
            if (!warning_produced)
              coco_warning("suite_biobj_ext_get_new_instance(): remove filtering of dimensions to get generally acceptable instances!");
            warning_produced = 1;
            continue;
          }
          
        break_search = check_consistency_of_instances(dimension, function1, instance1, function2, instance2);
      }
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
    coco_error("suite_biobj_ext_get_new_instance(): Could not find suitable instance %lu in %lu tries",
    		(unsigned long) instance, (unsigned long) num_tries);
    return 0; /* Never reached */
  }

  return instance2;
}

/**
 * @brief  Frees the memory of the given bi-objective suite.
 * @note   Copied from suite_bbob_biobj.c.
 */
static void suite_biobj_ext_free(void *stuff) {

  suite_biobj_ext_t *data;
  size_t i;

  assert(stuff != NULL);
  data = (suite_biobj_ext_t *) stuff;

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
