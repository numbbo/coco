/**
 * @file coco_suite.c
 * @brief Definitions of functions regarding COCO suites.
 *
 * When a new suite is added, the functions coco_suite_intialize, coco_suite_get_instances_by_year and
 * coco_suite_get_problem_from_indices need to be updated.
 *
 * @see <a href="index.html">Instructions</a> on how to write new test functions and combine them into test
 * suites.
 */

#include <time.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_utilities.c"

#include "suite_bbob.c"
#include "suite_biobj.c"
#include "suite_toy.c"
#include "suite_largescale.c"

/** @brief The maximum number of different instances in a suite. */
#define COCO_MAX_INSTANCES 1000

/**
 * @brief Calls the initializer of the given suite.
 *
 * @note This function needs to be updated when a new suite is added to COCO.
 */
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
    coco_error("coco_suite_intialize(): unknown problem suite");
    return NULL;
  }

  return suite;
}

/**
 * @brief Calls the function that sets the instanced by year for the given suite.
 *
 * @note This function needs to be updated when a new suite is added to COCO.
 */
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

/**
 * @brief Calls the function that returns the problem corresponding to the given suite, function index,
 * dimension index and instance index.
 *
 * @note This function needs to be updated when a new suite is added to COCO.
 */
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
    coco_error("coco_suite_get_problem_from_indices(): unknown problem suite");
    return NULL;
  }

  return problem;
}

/**
 * @brief Allocates the space for a coco_suite_t instance.
 *
 * This function sets the functions and dimensions contained in the suite, while the instances are set by
 * the function coco_suite_set_instance.
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
  assert(number_of_dimensions > 0);
  suite->dimensions = coco_allocate_vector_size_t(suite->number_of_dimensions);
  for (i = 0; i < suite->number_of_dimensions; i++) {
    suite->dimensions[i] = dimensions[i];
  }

  suite->number_of_functions = number_of_functions;
  assert(number_of_functions > 0);
  suite->functions = coco_allocate_vector_size_t(suite->number_of_functions);
  for (i = 0; i < suite->number_of_functions; i++) {
    suite->functions[i] = i + 1;
  }

  assert(strlen(default_instances) > 0);
  suite->default_instances = coco_strdup(default_instances);

  /* Will be set to the first valid dimension index before the constructor ends */
  suite->current_dimension_idx = -1;
  /* Will be set to the first valid function index before the constructor ends  */
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

/**
 * @brief Returns the number of positive numbers pointed to by numbers (the count stops when the first
 * 0 is encountered).
 *
 * If there are more than COCO_MAX_INSTANCES numbers, a coco_error is raised. The name argument is used
 * only to provide more informative output in case of any problems.
 */
static size_t coco_suite_count_numbers(const size_t *numbers, const char *name) {

  size_t count = 0;
  while ((count < COCO_MAX_INSTANCES) && (numbers[count] != 0)) {
    count++;
  }
  if (count == COCO_MAX_INSTANCES) {
    coco_error("coco_suite_count_numbers(): over %lu numbers in %s", COCO_MAX_INSTANCES, name);
    return 0; /* Never reached*/
  }

  return count;
}

/**
 * @brief Sets the suite instance to the given instance_numbers.
 */
static void coco_suite_set_instance(coco_suite_t *suite,
                                    const size_t *instance_numbers) {

  size_t i;

  if (!instance_numbers) {
    coco_error("coco_suite_set_instance(): no instance given");
    return;
  }

  suite->number_of_instances = coco_suite_count_numbers(instance_numbers, "suite instance numbers");
  suite->instances = coco_allocate_vector_size_t(suite->number_of_instances);
  for (i = 0; i < suite->number_of_instances; i++) {
    suite->instances[i] = instance_numbers[i];
  }

}

/**
 * @brief Filters the given items w.r.t. the given indices (starting from 0).
 *
 * Sets items[i] to 0 for every i that cannot be found in indices.
 */
static void coco_suite_filter_idx(size_t *items, const size_t number_of_items, const size_t *indices, const char *name) {

  size_t i, j;
  size_t count = coco_suite_count_numbers(indices, name);
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

/**
 * @brief Filters dimensions w.r.t. the given dimension_numbers.
 *
 * Sets suite->dimensions[i] to 0 for every dimension value that cannot be found in dimension_numbers.
 */
static void coco_suite_filter_dimensions(coco_suite_t *suite, const size_t *dimension_numbers) {

  size_t i, j;
  size_t count = coco_suite_count_numbers(dimension_numbers, "dimensions");
  int found;

  for (i = 0; i < suite->number_of_dimensions; i++) {
    found = 0;
    for (j = 0; j < count; j++) {
      if (suite->dimensions[i] == dimension_numbers[j])
        found = 1;
    }
    if (!found)
      suite->dimensions[i] = 0;
  }

}

/**
 * @param suite The given suite.
 * @param function_idx The index of the function in question (starting from 0).
 *
 * @return The function number in position function_idx in the suite. If the function has been filtered out
 * through suite_options in the coco_suite function, the result is 0.
 */
size_t coco_suite_get_function_from_function_index(coco_suite_t *suite, size_t function_idx) {

  if (function_idx >= suite->number_of_functions) {
    coco_error("coco_suite_get_function_from_function_index(): function index exceeding the number of functions in the suite");
    return 0; /* Never reached*/
  }

 return suite->functions[function_idx];
}

/**
 * @param suite The given suite.
 * @param dimension_idx The index of the dimension in question (starting from 0).
 *
 * @return The dimension number in position dimension_idx in the suite. If the dimension has been filtered out
 * through suite_options in the coco_suite function, the result is 0.
 */
size_t coco_suite_get_dimension_from_dimension_index(coco_suite_t *suite, size_t dimension_idx) {

  if (dimension_idx >= suite->number_of_dimensions) {
    coco_error("coco_suite_get_dimension_from_dimension_index(): dimensions index exceeding the number of dimensions in the suite");
    return 0; /* Never reached*/
  }

 return suite->dimensions[dimension_idx];
}

/**
 * @param suite The given suite.
 * @param instance_idx The index of the instance in question (starting from 0).
 *
 * @return The instance number in position instance_idx in the suite. If the instance has been filtered out
 * through suite_options in the coco_suite function, the result is 0.
 */
size_t coco_suite_get_instance_from_instance_index(coco_suite_t *suite, size_t instance_idx) {

  if (instance_idx >= suite->number_of_instances) {
    coco_error("coco_suite_get_instance_from_instance_index(): instance index exceeding the number of instances in the suite");
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

/**
 * Note that the problem_index depends on the number of instances a suite is defined with.
 *
 * @param suite The given suite.
 * @param problem_index The index of the problem to be returned.
 *
 * @return The problem of the suite defined by problem_index.
 */
coco_problem_t *coco_suite_get_problem(coco_suite_t *suite, size_t problem_index) {

  size_t function_idx = 0, instance_idx = 0, dimension_idx = 0;
  coco_suite_decode_problem_index(suite, problem_index, &function_idx, &dimension_idx, &instance_idx);

  return coco_suite_get_problem_from_indices(suite, function_idx, dimension_idx, instance_idx);
}

/**
 * The number of problems in the suite is computed as a product of the number of instances, number of
 * functions and number of dimensions and therefore doesn't account for any filtering done through the
 * suite_options parameter of the coco_suite function.
 *
 * @param suite The given suite.
 *
 * @return The number of problems in the suite.
 */
size_t coco_suite_get_number_of_problems(coco_suite_t *suite) {
  return (suite->number_of_instances * suite->number_of_functions * suite->number_of_dimensions);
}

/**
 * @brief Returns the numbers defined by the ranges.
 *
 * Reads ranges from a string of positive ranges separated by commas. For example: "-3,5-6,8-". Returns the
 * numbers that are defined by the ranges if min and max are used as their extremes. If the ranges with open
 * beginning/end are not allowed, use 0 as min/max. The returned string has an appended 0 to mark its end.
 * A maximum of COCO_MAX_INSTANCES values is returned. If there is a problem with one of the ranges, the
 * parsing stops and the current result is returned. The memory of the returned object needs to be freed by
 * the caller.
 */
static size_t *coco_suite_parse_ranges(char *string, const char *name, size_t min, size_t max) {

  char *ptr, *dash = NULL;
  char **ranges, **numbers;
  size_t i, j, count;
  size_t num[2];

  size_t *result;
  size_t i_result = 0;

  /* Check for empty string */
  if ((string == NULL) || (strlen(string) == 0)) {
    coco_warning("coco_suite_parse_ranges(): cannot parse empty ranges");
    return NULL;
  }

  ptr = string;
  /* Check for disallowed characters */
  while (*ptr != '\0') {
    if ((*ptr != '-') && (*ptr != ',') && !isdigit((unsigned char )*ptr)) {
      coco_warning("coco_suite_parse_ranges(): problem parsing '%s' - cannot parse ranges with '%c'", string,
          *ptr);
      return NULL;
    } else
      ptr++;
  }

  /* Check for incorrect boundaries */
  if ((max > 0) && (min > max)) {
    coco_warning("coco_suite_parse_ranges(): incorrect boundaries");
    return NULL;
  }

  result = coco_allocate_vector_size_t(COCO_MAX_INSTANCES + 1);

  /* Split string to ranges w.r.t commas */
  ranges = coco_string_split(string, ',');

  if (ranges) {
    /* Go over the current range */
    for (i = 0; *(ranges + i); i++) {

      ptr = *(ranges + i);
      /* Count the number of '-' */
      count = 0;
      while (*ptr != '\0') {
        if (*ptr == '-') {
          if (count == 0)
            /* Remember the position of the first '-' */
            dash = ptr;
          count++;
        }
        ptr++;
      }
      /* Point again to the start of the range */
      ptr = *(ranges + i);

      /* Check for incorrect number of '-' */
      if (count > 1) {
        coco_warning("coco_suite_parse_ranges(): problem parsing '%s' - too many '-'s", string);
        /* Cleanup */
        for (j = i; *(ranges + j); j++)
          coco_free_memory(*(ranges + j));
        coco_free_memory(ranges);
        if (i_result == 0) {
          coco_free_memory(result);
          return NULL;
        }
        result[i_result] = 0;
        return result;
      } else if (count == 0) {
        /* Range is in the format: n (no range) */
        num[0] = (size_t) strtol(ptr, NULL, 10);
        num[1] = num[0];
      } else {
        /* Range is in one of the following formats: n-m / -n / n- / - */

        /* Split current range to numbers w.r.t '-' */
        numbers = coco_string_split(ptr, '-');
        j = 0;
        if (numbers) {
          /* Read the numbers */
          for (j = 0; *(numbers + j); j++) {
            assert(j < 2);
            num[j] = (size_t) strtol(*(numbers + j), NULL, 10);
            coco_free_memory(*(numbers + j));
          }
        }
        coco_free_memory(numbers);

        if (j == 0) {
          /* Range is in the format - (open ends) */
          if ((min == 0) || (max == 0)) {
            coco_warning("coco_suite_parse_ranges(): '%s' ranges cannot have an open ends; some ranges ignored", name);
            /* Cleanup */
            for (j = i; *(ranges + j); j++)
              coco_free_memory(*(ranges + j));
            coco_free_memory(ranges);
            if (i_result == 0) {
              coco_free_memory(result);
              return NULL;
            }
            result[i_result] = 0;
            return result;
          }
          num[0] = min;
          num[1] = max;
        } else if (j == 1) {
          if (dash - *(ranges + i) == 0) {
            /* Range is in the format -n */
            if (min == 0) {
              coco_warning("coco_suite_parse_ranges(): '%s' ranges cannot have an open beginning; some ranges ignored", name);
              /* Cleanup */
              for (j = i; *(ranges + j); j++)
                coco_free_memory(*(ranges + j));
              coco_free_memory(ranges);
              if (i_result == 0) {
                coco_free_memory(result);
                return NULL;
              }
              result[i_result] = 0;
              return result;
            }
            num[1] = num[0];
            num[0] = min;
          } else {
            /* Range is in the format n- */
            if (max == 0) {
              coco_warning("coco_suite_parse_ranges(): '%s' ranges cannot have an open end; some ranges ignored", name);
              /* Cleanup */
              for (j = i; *(ranges + j); j++)
                coco_free_memory(*(ranges + j));
              coco_free_memory(ranges);
              if (i_result == 0) {
                coco_free_memory(result);
                return NULL;
              }
              result[i_result] = 0;
              return result;
            }
            num[1] = max;
          }
        }
        /* if (j == 2), range is in the format n-m and there is nothing to do */
      }

      /* Make sure the boundaries are taken into account */
      if ((min > 0) && (num[0] < min)) {
        num[0] = min;
        coco_warning("coco_suite_parse_ranges(): '%s' ranges adjusted to be >= %lu", name, min);
      }
      if ((max > 0) && (num[1] > max)) {
        num[1] = max;
        coco_warning("coco_suite_parse_ranges(): '%s' ranges adjusted to be <= %lu", name, max);
      }
      if (num[0] > num[1]) {
        coco_warning("coco_suite_parse_ranges(): '%s' ranges not within boundaries; some ranges ignored", name);
        /* Cleanup */
        for (j = i; *(ranges + j); j++)
          coco_free_memory(*(ranges + j));
        coco_free_memory(ranges);
        if (i_result == 0) {
          coco_free_memory(result);
          return NULL;
        }
        result[i_result] = 0;
        return result;
      }

      /* Write in result */
      for (j = num[0]; j <= num[1]; j++) {
        if (i_result > COCO_MAX_INSTANCES - 1)
          break;
        result[i_result++] = j;
      }

      coco_free_memory(*(ranges + i));
      *(ranges + i) = NULL;
    }
  }

  coco_free_memory(ranges);

  if (i_result == 0) {
    coco_free_memory(result);
    return NULL;
  }

  result[i_result] = 0;
  return result;
}

/**
 * @brief Returns the instances read from either a "year: YEAR" or "instances: NUMBERS" string.
 *
 * If both "year" and "instances" are given, the second is ignored (and a warning is raised). See the
 * coco_suite function for more information about the required format.
 */
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
      coco_warning("coco_suite_get_instance_indices(): 'instances' suite option ignored because it follows 'year'");
    }
    else {
      parce_year = 0;
      coco_warning("coco_suite_get_instance_indices(): 'year' suite option ignored because it follows 'instances'");
    }
  }

  if ((year_found >= 0) && (parce_year == 1)) {
    if (coco_options_read_int(suite_instance, "year", &(year)) != 0) {
      year_string = coco_suite_get_instances_by_year(suite, year);
      result = coco_suite_parse_ranges(year_string, "instances", 1, 0);
    } else {
      coco_warning("coco_suite_get_instance_indices(): problems parsing the 'year' suite_instance option, ignored");
    }
  }

  instances = coco_allocate_string(COCO_MAX_INSTANCES);
  if ((instances_found >= 0) && (parce_instances == 1)) {
    if (coco_options_read_values(suite_instance, "instances", instances) > 0) {
      result = coco_suite_parse_ranges(instances, "instances", 1, 0);
    } else {
      coco_warning("coco_suite_get_instance_indices(): problems parsing the 'instance' suite_instance option, ignored");
    }
  }
  coco_free_memory(instances);

  return result;
}

/**
 * @brief Iterates through the items from the current_item_id position on in search for the next positive
 * item.
 *
 * If such an item is found, current_item_id points to this item and the method returns 1. If such an
 * item cannot be found, current_item_id points to the first positive item and the method returns 0.
 */
static int coco_suite_is_next_item_found(const size_t *items, const size_t number_of_items, long *current_item_id) {

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
 * @brief Iterates through the instances of the given suite from the current_instance_idx position on in
 * search for the next positive instance.
 *
 * If such an instance is found, current_instance_idx points to this instance and the method returns 1. If
 * such an instance cannot be found, current_instance_idx points to the first positive instance and the
 * method returns 0.
 */
static int coco_suite_is_next_instance_found(coco_suite_t *suite) {

  return coco_suite_is_next_item_found(suite->instances, suite->number_of_instances,
      &suite->current_instance_idx);
}

/**
 * @brief Iterates through the functions of the given suite from the current_function_idx position on in
 * search for the next positive function.
 *
 * If such a function is found, current_function_idx points to this function and the method returns 1. If
 * such a function cannot be found, current_function_idx points to the first positive function,
 * current_instance_idx points to the first positive instance and the method returns 0.
 */
static int coco_suite_is_next_function_found(coco_suite_t *suite) {

  int result = coco_suite_is_next_item_found(suite->functions, suite->number_of_functions,
      &suite->current_function_idx);
  if (!result) {
    /* Reset the instances */
    suite->current_instance_idx = -1;
    coco_suite_is_next_instance_found(suite);
  }
  return result;
}

/**
 * @brief Iterates through the dimensions of the given suite from the current_dimension_idx position on in
 * search for the next positive dimension.
 *
 * If such a dimension is found, current_dimension_idx points to this dimension and the method returns 1. If
 * such a dimension cannot be found, current_dimension_idx points to the first positive dimension and the
 * method returns 0.
 */
static int coco_suite_is_next_dimension_found(coco_suite_t *suite) {

  return coco_suite_is_next_item_found(suite->dimensions, suite->number_of_dimensions,
      &suite->current_dimension_idx);
}

/**
 * Currently, four suites are supported:
 * - "bbob" contains 24 <a href="http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf">
 * single-objective functions</a> in 6 dimensions (2, 3, 5, 10, 20, 40)
 * - "bbob-biobj" contains 55 <a href="http://numbbo.github.io/bbob-biobj-functions-doc">bi-objective
 * functions</a> in 6 dimensions (2, 3, 5, 10, 20, 40)
 * - "bbob-largescale" contains 24 <a href="http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf">
 * single-objective functions</a> in 6 large dimensions (40, 80, 160, 320, 640, 1280)
 * - "toy" contains 6 <a href="http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf">
 * single-objective functions</a> in 5 dimensions (2, 3, 5, 10, 20)
 *
 * Only the suite_name parameter needs to be non-empty. The suite_instance and suite_options can be "" or
 * NULL. In this case, default values are taken (default instances of a suite are those used in the last year
 * and the suite is not filtered by default).
 *
 * @param suite_name A string containing the name of the suite. Currently supported suite names are "bbob",
 * "bbob-biobj", "bbob-largescale" and "toy".
 * @param suite_instance A string used for defining the suite instances. Two ways are supported:
 * - "year: YEAR", where YEAR is the year of the BBOB workshop, includes the instances (to be) used in that
 * year's workshop;
 * - "instances: VALUES", where VALUES are instance numbers from 1 on written as a comma-separated list or a
 * range m-n.
 * @param suite_options A string of pairs "key: value" used to filter the suite (especially useful for
 * parallelizing the experiments). Supported options:
 * - "dimensions: LIST", where LIST is the list of dimensions to keep in the suite (range-style syntax is
 * not allowed here),
 * - "function_idx: VALUES", where VALUES is a list or a range of function indexes (starting from 1) to keep
 * in the suite, and
 * - "instance_idx: VALUES", where VALUES is a list or a range of instance indexes (starting from 1) to keep
 * in the suite.
 *
 * @return The constructed suite object.
 */
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
    option_string = coco_allocate_string(COCO_PATH_MAX);
    if (coco_options_read_values(suite_options, "function_idx", option_string) > 0) {
      indices = coco_suite_parse_ranges(option_string, "function_idx", 1, suite->number_of_functions);
      if (indices != NULL) {
        coco_suite_filter_idx(suite->functions, suite->number_of_functions, indices, "function_idx");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    option_string = coco_allocate_string(COCO_PATH_MAX);
    if (coco_options_read_values(suite_options, "instance_idx", option_string) > 0) {
      indices = coco_suite_parse_ranges(option_string, "instance_idx", 1, suite->number_of_instances);
      if (indices != NULL) {
        coco_suite_filter_idx(suite->instances, suite->number_of_instances, indices, "instance_idx");
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

    option_string = coco_allocate_string(COCO_PATH_MAX);
    if ((dim_idx_found >= 0) && (parce_dim_idx == 1) && (coco_options_read_values(suite_options, "dimension_idx", option_string) > 0)) {
      indices = coco_suite_parse_ranges(option_string, "dimension_idx", 1, suite->number_of_dimensions);
      if (indices != NULL) {
        coco_suite_filter_idx(suite->dimensions, suite->number_of_dimensions, indices, "dimension_idx");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    option_string = coco_allocate_string(COCO_PATH_MAX);
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
      dimensions = coco_suite_parse_ranges(option_string, "dimensions", suite->dimensions[0],
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
 * Iterates through the suite first by instances, then by functions and finally by dimensions.
 * The instances/functions/dimensions that have been filtered out using the suite_options of the coco_suite
 * function are skipped. Outputs some information regarding the current place in the iteration. The returned
 * problem is wrapped with the observer. If the observer is NULL, the returned problem is unobserved.
 *
 * @param suite The given suite.
 * @param observer The observer used to wrap the problem. If NULL, the problem is returned unobserved.
 *
 * @returns The next problem of the suite or NULL if there is no next problem left.
 */
coco_problem_t *coco_suite_get_next_problem(coco_suite_t *suite, coco_observer_t *observer) {

  size_t function_idx;
  size_t dimension_idx;
  size_t instance_idx;
  coco_problem_t *problem;

  long previous_function_idx = suite->current_function_idx;
  long previous_dimension_idx = suite->current_dimension_idx;
  long previous_instance_idx = suite->current_instance_idx;

  /* Iterate through the suite by instances, then functions and lastly dimensions in search for the next
   * problem. Note that these functions set the values of suite fields current_instance_idx,
   * current_function_idx and current_dimension_idx. */
  if (!coco_suite_is_next_instance_found(suite)
      && !coco_suite_is_next_function_found(suite)
      && !coco_suite_is_next_dimension_found(suite)) {
    coco_info_partial("done\n");
    return NULL;
  }

  if (suite->current_problem) {
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

  /* Output information regarding the current place in the iteration */
  if (coco_log_level >= COCO_INFO) {
    if (((long) dimension_idx != previous_dimension_idx) || (previous_instance_idx < 0)) {
      /* A new dimension started */
      time_t timer;
      char time_string[30];
      struct tm* tm_info;
      time(&timer);
      tm_info = localtime(&timer);
      strftime(time_string, 30, "%d.%m.%y %H:%M:%S", tm_info);
      if (dimension_idx > 0)
        coco_info_partial("done\n");
      else
        coco_info_partial("\n");
      coco_info_partial("COCO INFO: %s, d=%lu, running: f%02lu", time_string, suite->dimensions[dimension_idx], suite->functions[function_idx]);
    }
    else if ((long) function_idx != previous_function_idx){
      /* A new function started */
      coco_info_partial("f%02lu", suite->functions[function_idx]);
    }
    /* One dot for each instance */
    coco_info_partial(".", suite->instances[instance_idx]);
  }

  return problem;
}

/**
 * Constructs a suite and observer given their options and runs the optimizer on all the problems in the
 * suite.
 *
 * @param suite_name A string containing the name of the suite. See suite_name in the coco_suite function for
 * possible values.
 * @param suite_instance A string used for defining the suite instances. See suite_instance in the coco_suite
 * function for possible values ("" and NULL result in default suite instances).
 * @param suite_options A string of pairs "key: value" used to filter the suite. See suite_options in the
 * coco_suite function for possible values ("" and NULL result in a non-filtered suite).
 * @param observer_name A string containing the name of the observer. See observer_name in the coco_observer
 * function for possible values ("", "no_observer" and NULL result in not using an observer).
 * @param observer_options A string of pairs "key: value" used to pass the options to the observer. See
 * observer_options in the coco_observer function for possible values ("" and NULL result in default observer
 * options).
 * @param optimizer An optimization algorithm to be run on each problem in the suite.
 */
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

/* See coco.h for more information on encoding and decoding problem index */

/**
 * @param suite The suite.
 * @param function_idx Index of the function (starting with 0).
 * @param dimension_idx Index of the dimension (starting with 0).
 * @param instance_idx Index of the insatnce (starting with 0).
 *
 * @return The problem index in the suite computed from function_idx, dimension_idx and instance_idx.
 */
size_t coco_suite_encode_problem_index(coco_suite_t *suite,
                                       const size_t function_idx,
                                       const size_t dimension_idx,
                                       const size_t instance_idx) {

  return instance_idx + (function_idx * suite->number_of_instances) +
      (dimension_idx * suite->number_of_instances * suite->number_of_functions);

}

/**
 * @param suite The suite.
 * @param problem_index Index of the problem in the suite (starting with 0).
 * @param function_idx Pointer to the index of the function, which is set by this function.
 * @param dimension_idx Pointer to the index of the dimension, which is set by this function.
 * @param instance_idx Pointer to the index of the instance, which is set by this function.
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
