#include <time.h>

#include "coco.h"
#include "coco_internal.h"

#include "suite_bbob.c"
#include "suite_biobj.c"
#include "suite_toy.c"
#include "suite_largescale.c"

/**
 * TODO: Add instructions on how to implement a new suite!
 * TODO: Add asserts regarding input values!
 */

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances) {

  coco_suite_t *suite;
  size_t i;

  suite = (coco_suite_t *) coco_allocate_memory(sizeof(*suite));

  suite->suite_name = coco_strdup(suite_name);

  suite->number_of_dimensions = number_of_dimensions;
  suite->dimensions = coco_allocate_memory(suite->number_of_dimensions * sizeof(size_t));
  for (i = 0; i < suite->number_of_dimensions; i++) {
    suite->dimensions[i] = dimensions[i];
  }

  suite->number_of_functions = number_of_functions;
  suite->functions = coco_allocate_memory(suite->number_of_functions * sizeof(size_t));
  for (i = 0; i < suite->number_of_functions; i++) {
    suite->functions[i] = i + 1;
  }

  suite->default_instances = coco_strdup(default_instances);

  /* To be set when iterating through the suite */
  suite->current_dimension_idx = -1;
  suite->current_function_idx = -1;
  suite->current_instance_idx = -1;
  suite->current_problem = NULL;

  /* To be set in coco_suite_set_instance() */
  suite->number_of_instances = 0;
  suite->instances = NULL;

  /* To be set in particular suites if needed */
  suite->data = NULL;
  suite->data_free_function = NULL;

  return suite;
}

static void coco_suite_set_instance(coco_suite_t *suite,
                                    const size_t *instance_numbers) {

  size_t i;

  if (!instance_numbers) {
    coco_error("coco_suite_set_instance(): no instance given");
    return;
  }

  suite->number_of_instances = coco_numbers_count(instance_numbers, "suite instance numbers");
  suite->instances = coco_allocate_memory(suite->number_of_instances * sizeof(size_t));
  for (i = 0; i < suite->number_of_instances; i++) {
    suite->instances[i] = instance_numbers[i];
  }

}

static void coco_suite_filter_ids(size_t *items, const size_t number_of_items, const size_t *indices, const char *name) {

  size_t i, j;
  size_t count = coco_numbers_count(indices, name);
  int found;

  for (i = 1; i <= number_of_items; i++) {
    found = 0;
    for (j = 0; j < count; j++) {
      if (i == indices[j]) {
        found = 1;
        break;
      }
    }
    if (!found)
      items[i - 1] = 0;
  }

}

static void coco_suite_filter_dimensions(coco_suite_t *suite, const size_t *dims) {

  size_t i, j;
  size_t count = coco_numbers_count(dims, "dimensions");
  int found;

  for (i = 0; i < suite->number_of_dimensions; i++) {
    found = 0;
    for (j = 0; j < count; j++) {
      if (suite->dimensions[i] == dims[j])
        found = 1;
    }
    if (!found)
      suite->dimensions[i] = 0;
  }

}

size_t coco_suite_get_function_from_function_index(coco_suite_t *suite, size_t function_idx) {

  if (function_idx >= suite->number_of_functions) {
    coco_error("coco_suite_get_function_from_index(): function index exceeding the number of functions in the suite");
    return 0; /* Never reached*/
  }

 return suite->functions[function_idx];
}

size_t coco_suite_get_dimension_from_dimension_index(coco_suite_t *suite, size_t dimension_idx) {

  if (dimension_idx >= suite->number_of_dimensions) {
    coco_error("coco_suite_get_dimension_from_index(): dimensions index exceeding the number of dimensions in the suite");
    return 0; /* Never reached*/
  }

 return suite->dimensions[dimension_idx];
}

size_t coco_suite_get_instance_from_instance_index(coco_suite_t *suite, size_t instance_idx) {

  if (instance_idx >= suite->number_of_instances) {
    coco_error("coco_suite_get_instance_from_index(): instance index exceeding the number of instances in the suite");
    return 0; /* Never reached*/
  }

 return suite->functions[instance_idx];
}

void coco_suite_free(coco_suite_t *suite) {

  if (suite != NULL) {

    if (suite->suite_name) {
      coco_free_memory(suite->suite_name);
      suite->suite_name = NULL;
    }
    if (suite->dimensions) {
      coco_free_memory(suite->dimensions);
      suite->dimensions = NULL;
    }
    if (suite->functions) {
      coco_free_memory(suite->functions);
      suite->functions = NULL;
    }
    if (suite->instances) {
      coco_free_memory(suite->instances);
      suite->instances = NULL;
    }
    if (suite->default_instances) {
      coco_free_memory(suite->default_instances);
      suite->default_instances = NULL;
    }

    if (suite->current_problem) {
      coco_problem_free(suite->current_problem);
      suite->current_problem = NULL;
    }

    if (suite->data != NULL) {
      if (suite->data_free_function != NULL) {
        suite->data_free_function(suite->data);
      }
      coco_free_memory(suite->data);
      suite->data = NULL;
    }

    coco_free_memory(suite);
    suite = NULL;
  }
}

static coco_suite_t *coco_suite_intialize(const char *suite_name) {

  coco_suite_t *suite;

  if (strcmp(suite_name, "toy") == 0) {
    suite = suite_toy_allocate();
  } else if (strcmp(suite_name, "bbob") == 0) {
    suite = suite_bbob_allocate();
  } else if (strcmp(suite_name, "bbob-biobj") == 0) {
    suite = suite_biobj_allocate();
  } else if (strcmp(suite_name, "bbob-largescale") == 0) {
    suite = suite_largescale_allocate();
  }
  else {
    coco_error("coco_suite(): unknown problem suite");
    return NULL;
  }

  return suite;
}

static char *coco_suite_get_instances_by_year(coco_suite_t *suite, const int year) {

  char *year_string;

  if (strcmp(suite->suite_name, "bbob") == 0) {
    year_string = suite_bbob_get_instances_by_year(year);
  } else if (strcmp(suite->suite_name, "bbob-biobj") == 0) {
    year_string = suite_biobj_get_instances_by_year(year);
  } else {
    coco_error("coco_suite_get_instances_by_year(): suite '%s' has no years defined", suite->suite_name);
    return NULL;
  }

  return year_string;
}

static coco_problem_t *coco_suite_get_problem_from_indices(coco_suite_t *suite,
                                                           size_t function_idx,
                                                           size_t dimension_idx,
                                                           size_t instance_idx) {

  coco_problem_t *problem;

  if (strcmp(suite->suite_name, "toy") == 0) {
    problem = suite_toy_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob") == 0) {
    problem = suite_bbob_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob-biobj") == 0) {
    problem = suite_biobj_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob-largescale") == 0) {
    problem = suite_largescale_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else {
    coco_error("coco_suite_get_problem(): unknown problem suite");
    return NULL;
  }

  return problem;
}

coco_problem_t *coco_suite_get_problem(coco_suite_t *suite, size_t problem_index) {

  size_t function_idx = 0, instance_idx = 0, dimension_idx = 0;
  coco_suite_decode_problem_index(suite, problem_index, &function_idx, &dimension_idx, &instance_idx);

  return coco_suite_get_problem_from_indices(suite, function_idx, dimension_idx, instance_idx);
}

size_t coco_suite_get_number_of_problems(coco_suite_t *suite) {
  return (suite->number_of_instances * suite->number_of_functions * suite->number_of_dimensions);
}

static size_t *coco_suite_get_instance_indices(coco_suite_t *suite, const char *suite_instance) {

  int year = -1;
  char *instances = NULL;
  char *year_string = NULL;
  long year_found, instances_found;
  int parce_year = 1, parce_instances = 1;
  size_t *result = NULL;

  if (suite_instance == NULL)
    return NULL;

  year_found = coco_strfind(suite_instance, "year");
  instances_found = coco_strfind(suite_instance, "instances");

  if ((year_found < 0) && (instances_found < 0))
    return NULL;

  if ((year_found > 0) && (instances_found > 0)) {
    if (year_found < instances_found) {
      parce_instances = 0;
      coco_warning("coco_suite_parse_instance_string(): 'instances' suite option ignored because it follows 'year'");
    }
    else {
      parce_year = 0;
      coco_warning("coco_suite_parse_instance_string(): 'year' suite option ignored because it follows 'instances'");
    }
  }

  if ((year_found >= 0) && (parce_year == 1)) {
    if (coco_options_read_int(suite_instance, "year", &(year)) != 0) {
      year_string = coco_suite_get_instances_by_year(suite, year);
      result = coco_string_get_numbers_from_ranges(year_string, "instances", 1, 0);
    } else {
      coco_warning("coco_suite_parse_instance_string(): problems parsing the 'year' suite_instance option, ignored");
    }
  }

  instances = coco_allocate_memory(COCO_MAX_INSTANCES * sizeof(char));
  if ((instances_found >= 0) && (parce_instances == 1)) {
    if (coco_options_read_values(suite_instance, "instances", instances) > 0) {
      result = coco_string_get_numbers_from_ranges(instances, "instances", 1, 0);
    } else {
      coco_warning("coco_suite_parse_instance_string(): problems parsing the 'instance' suite_instance option, ignored");
    }
  }
  coco_free_memory(instances);

  return result;
}

/**
 * Iterates through the items from the current_item_id position on in search for the next positive item.
 * If such an item is found, current_item_id points to this item and the method returns 1. If such an
 * item cannot be found, current_item_id points to the first positive item and the method returns 0.
 */
static int coco_suite_is_next_item_found(const size_t number_of_items, const size_t *items, long *current_item_id) {

  if ((*current_item_id) != number_of_items - 1)  {
    /* Not the last item, iterate through items */
    do {
      (*current_item_id)++;
    } while (((*current_item_id) < number_of_items - 1) && (items[*current_item_id] == 0));

    assert((*current_item_id) < number_of_items);
    if (items[*current_item_id] != 0) {
      /* Next item is found, return true */
      return 1;
    }
  }

  /* Next item cannot be found, move to the first good item and return false */
  *current_item_id = -1;
  do {
    (*current_item_id)++;
  } while ((*current_item_id < number_of_items - 1) && (items[*current_item_id] == 0));
  if (items[*current_item_id] == 0)
    coco_error("coco_suite_is_next_item_found(): the chosen suite has no valid (positive) items");
  return 0;
}

/**
 * Iterates through the instances of the given suite from the current_instance_idx position on in search for
 * the next positive instance. If such an instance is found, current_instance_idx points to this instance and
 * the method returns 1. If such an instance cannot be found, current_instance_idx points to the first
 * positive instance and the method returns 0.
 */
static int coco_suite_is_next_instance_found(coco_suite_t *suite) {

  return coco_suite_is_next_item_found(suite->number_of_instances, suite->instances,
      &suite->current_instance_idx);
}

/**
 * Iterates through the functions of the given suite from the current_function_idx position on in search for
 * the next positive function. If such a function is found, current_function_idx points to this function and
 * the method returns 1. If such a function cannot be found, current_function_idx points to the first
 * positive function, current_instance_idx points to the first positive instance and the method returns 0.
 */
static int coco_suite_is_next_function_found(coco_suite_t *suite) {

  int result = coco_suite_is_next_item_found(suite->number_of_functions, suite->functions,
      &suite->current_function_idx);
  if (!result) {
    /* Reset the instances */
    suite->current_instance_idx = -1;
    coco_suite_is_next_instance_found(suite);
  }
  return result;
}

/**
 * Iterates through the dimensions of the given suite from the current_dimension_idx position on in search for
 * the next positive dimension. If such a dimension is found, current_dimension_idx points to this dimension
 * and the method returns 1. If such a dimension cannot be found, current_dimension_idx points to the first
 * positive dimension and the method returns 0.
 */
static int coco_suite_is_next_dimension_found(coco_suite_t *suite) {

  return coco_suite_is_next_item_found(suite->number_of_dimensions, suite->dimensions,
      &suite->current_dimension_idx);
}

coco_suite_t *coco_suite(const char *suite_name, const char *suite_instance, const char *suite_options) {

  coco_suite_t *suite;
  size_t *instances;
  char *option_string = NULL;
  char *ptr;
  size_t *indices = NULL;
  size_t *dimensions = NULL;
  long dim_found, dim_idx_found;
  int parce_dim = 1, parce_dim_idx = 1;

  /* Initialize the suite */
  suite = coco_suite_intialize(suite_name);

  /* Set the instance */
  if ((!suite_instance) || (strlen(suite_instance) == 0))
    instances = coco_suite_get_instance_indices(suite, suite->default_instances);
  else
    instances = coco_suite_get_instance_indices(suite, suite_instance);
  coco_suite_set_instance(suite, instances);
  coco_free_memory(instances);

  /* Apply filter if any given by the suite_options */
  if ((suite_options) && (strlen(suite_options) > 0)) {
    option_string = coco_allocate_memory(COCO_PATH_MAX * sizeof(char));
    if (coco_options_read_values(suite_options, "function_idx", option_string) > 0) {
      indices = coco_string_get_numbers_from_ranges(option_string, "function_idx", 1, suite->number_of_functions);
      if (indices != NULL) {
        coco_suite_filter_ids(suite->functions, suite->number_of_functions, indices, "function_idx");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    option_string = coco_allocate_memory(COCO_PATH_MAX * sizeof(char));
    if (coco_options_read_values(suite_options, "instance_idx", option_string) > 0) {
      indices = coco_string_get_numbers_from_ranges(option_string, "instance_idx", 1, suite->number_of_instances);
      if (indices != NULL) {
        coco_suite_filter_ids(suite->instances, suite->number_of_instances, indices, "instance_idx");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    dim_found = coco_strfind(suite_options, "dimensions");
    dim_idx_found = coco_strfind(suite_options, "dimension_idx");

    if ((dim_found > 0) && (dim_idx_found > 0)) {
      if (dim_found < dim_idx_found) {
        parce_dim_idx = 0;
        coco_warning("coco_suite(): 'dimension_idx' suite option ignored because it follows 'dimensions'");
      }
      else {
        parce_dim = 0;
        coco_warning("coco_suite(): 'dimensions' suite option ignored because it follows 'dimension_idx'");
      }
    }

    option_string = coco_allocate_memory(COCO_PATH_MAX * sizeof(char));
    if ((dim_idx_found >= 0) && (parce_dim_idx == 1) && (coco_options_read_values(suite_options, "dimension_idx", option_string) > 0)) {
      indices = coco_string_get_numbers_from_ranges(option_string, "dimension_idx", 1, suite->number_of_dimensions);
      if (indices != NULL) {
        coco_suite_filter_ids(suite->dimensions, suite->number_of_dimensions, indices, "dimension_idx");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    option_string = coco_allocate_memory(COCO_PATH_MAX * sizeof(char));
    if ((dim_found >= 0) && (parce_dim == 1) && (coco_options_read_values(suite_options, "dimensions", option_string) > 0)) {
      ptr = option_string;
      /* Check for disallowed characters */
      while (*ptr != '\0') {
        if ((*ptr != ',') && !isdigit((unsigned char )*ptr)) {
          coco_warning("coco_suite(): 'dimensions' suite option ignored because of disallowed characters");
          return NULL;
        } else
          ptr++;
      }
      dimensions = coco_string_get_numbers_from_ranges(option_string, "dimensions", suite->dimensions[0],
          suite->dimensions[suite->number_of_dimensions - 1]);
      if (dimensions != NULL) {
        coco_suite_filter_dimensions(suite, dimensions);
        coco_free_memory(dimensions);
      }
    }
    coco_free_memory(option_string);
  }

  /* Check that there are enough dimensions, functions and instances left */
  if ((suite->number_of_dimensions < 1)
      || (suite->number_of_functions < 1)
      || (suite->number_of_instances < 1)) {
    coco_error("coco_suite(): the suite does not contain at least one dimension, function and instance");
    return NULL;
  }

  /* Set the starting values of the current indices in such a way, that when the instance_idx is incremented,
   * this results in a valid problem */
  coco_suite_is_next_function_found(suite);
  coco_suite_is_next_dimension_found(suite);

  return suite;
}

/**
 * Returns next problem of the suite by iterating first through available instances, then functions and
 * lastly dimensions. The problem is wrapped with the observer's logger. If there is no next problem,
 * returns NULL.
 */
coco_problem_t *coco_suite_get_next_problem(coco_suite_t *suite, coco_observer_t *observer) {

  size_t function_idx;
  size_t dimension_idx;
  size_t instance_idx;
  coco_problem_t *problem;

  size_t previous_function = 0;

  /* Iterate through the suite by instances, then functions and lastly dimensions in search for the next
   * problem. Note that these functions set the values of suite fields current_instance_idx,
   * current_function_idx and current_dimension_idx. */
  if (!coco_suite_is_next_instance_found(suite)
      && !coco_suite_is_next_function_found(suite)
      && !coco_suite_is_next_dimension_found(suite))
    return NULL;

  if (suite->current_problem) {
    previous_function = coco_problem_get_suite_dep_function(suite->current_problem);
    coco_problem_free(suite->current_problem);
  }

  assert(suite->current_function_idx >= 0);
  assert(suite->current_dimension_idx >= 0);
  assert(suite->current_instance_idx >= 0);

  function_idx = (size_t) suite->current_function_idx;
  dimension_idx = (size_t) suite->current_dimension_idx;
  instance_idx = (size_t) suite->current_instance_idx;

  problem = coco_suite_get_problem_from_indices(suite, function_idx, dimension_idx, instance_idx);
  if (observer != NULL)
    problem = coco_problem_add_observer(problem, observer);
  suite->current_problem = problem;

  if (coco_problem_get_suite_dep_function(suite->current_problem) != previous_function) {
    time_t timer;
    char time_string[30];
    struct tm* tm_info;
    time(&timer);
    tm_info = localtime(&timer);
    strftime(time_string, 30, "%H:%M:%S %d-%m-%Y", tm_info);
    coco_info("Processing problems f%02lu_i*_d%02lu at %s",
        coco_problem_get_suite_dep_function(suite->current_problem),
        coco_problem_get_dimension(suite->current_problem), time_string);
  }

  return problem;
}

void coco_run_benchmark(const char *suite_name,
                        const char *suite_instance,
                        const char *suite_options,
                        const char *observer_name,
                        const char *observer_options,
                        coco_optimizer_t optimizer) {

  coco_suite_t *suite;
  coco_observer_t *observer;
  coco_problem_t *problem;

  suite = coco_suite(suite_name, suite_instance, suite_options);
  observer = coco_observer(observer_name, observer_options);

  while ((problem = coco_suite_get_next_problem(suite, observer)) != NULL) {

    optimizer(problem);

  }

  coco_observer_free(observer);
  coco_suite_free(suite);

}

/*
 * General schema for encoding/decoding a suite_dep_index. Note that the index depends on the number of
 * instances a suite is defined with (it is really a suite-instance-depended index...).
 * Also, while functions, instances and dimensions start from 1, function_idx, instance_idx and dimension_idx
 * as well as suite_dep_index start from 0!
 *
 * Showing an example with 2 dimensions (2, 3), 5 instances (6, 7, 8, 9, 10) and 2 functions (1, 2):
 *
 * index | instance | function | dimension
 * ------+----------+----------+-----------
 *     0 |        6 |        1 |         2
 *     1 |        7 |        1 |         2
 *     2 |        8 |        1 |         2
 *     3 |        9 |        1 |         2
 *     4 |       10 |        1 |         2
 *     5 |        6 |        2 |         2
 *     6 |        7 |        2 |         2
 *     7 |        8 |        2 |         2
 *     8 |        9 |        2 |         2
 *     9 |       10 |        2 |         2
 *    10 |        6 |        1 |         3
 *    11 |        7 |        1 |         3
 *    12 |        8 |        1 |         3
 *    13 |        9 |        1 |         3
 *    14 |       10 |        1 |         3
 *    15 |        6 |        2 |         2
 *    16 |        7 |        2 |         3
 *    17 |        8 |        2 |         3
 *    18 |        9 |        2 |         3
 *    19 |       10 |        2 |         3
 *
 * index | instance_idx | function_idx | dimension_idx
 * ------+--------------+--------------+---------------
 *     0 |            0 |            0 |             0
 *     1 |            1 |            0 |             0
 *     2 |            2 |            0 |             0
 *     3 |            3 |            0 |             0
 *     4 |            4 |            0 |             0
 *     5 |            0 |            1 |             0
 *     6 |            1 |            1 |             0
 *     7 |            2 |            1 |             0
 *     8 |            3 |            1 |             0
 *     9 |            4 |            1 |             0
 *    10 |            0 |            0 |             1
 *    11 |            1 |            0 |             1
 *    12 |            2 |            0 |             1
 *    13 |            3 |            0 |             1
 *    14 |            4 |            0 |             1
 *    15 |            0 |            1 |             1
 *    16 |            1 |            1 |             1
 *    17 |            2 |            1 |             1
 *    18 |            3 |            1 |             1
 *    19 |            4 |            1 |             1
 *
 */

/**
 * Computes the problem index from function_idx, dimension_idx and instance_idx of the given suite.
 * Note that the index depends on the number of instances a suite is defined with!
 */
size_t coco_suite_encode_problem_index(coco_suite_t *suite,
                                       const size_t function_idx,
                                       const size_t dimension_idx,
                                       const size_t instance_idx) {

  return instance_idx + (function_idx * suite->number_of_instances) +
      (dimension_idx * suite->number_of_instances * suite->number_of_functions);

}

/**
 * Computes the function_idx, dimension_idx and instance_idx of the problem with index problem_index of
 * the given suite. Note that the index depends on the number of instances a suite is defined with!
 */
void coco_suite_decode_problem_index(coco_suite_t *suite,
                                     const size_t problem_index,
                                     size_t *function_idx,
                                     size_t *dimension_idx,
                                     size_t *instance_idx) {

  if (problem_index > (suite->number_of_instances * suite->number_of_functions * suite->number_of_dimensions) - 1) {
    coco_warning("coco_suite_decode_problem_index(): problem_index too large");
    function_idx = 0;
    instance_idx = 0;
    dimension_idx = 0;
    return;
  }

  *instance_idx = problem_index % suite->number_of_instances;
  *function_idx = (problem_index / suite->number_of_instances) % suite->number_of_functions;
  *dimension_idx = problem_index / (suite->number_of_instances * suite->number_of_functions);

}
