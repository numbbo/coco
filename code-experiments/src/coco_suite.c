#include "coco.h"
#include "coco_internal.h"

#define COCO_NUMBER_OF_SUITES 3
const char *COCO_SUITES[COCO_NUMBER_OF_SUITES] = { "bbob", "biobj", "toy" };

/**
 * TODO: Add instructions on how to implement a new suite!
 * TODO: Add asserts regarding input values!
 * TODO: Write getters!
 */

struct coco_suite {

  char *suite_name;

  size_t number_of_dimensions;
  size_t *dimensions;
  int current_dimension_id;

  size_t number_of_functions;
  size_t *functions;
  int current_function_id;

  size_t number_of_instances;
  size_t *instances;
  int current_instance_id;

};

typedef struct coco_suite coco_suite_t;

static coco_suite_t *coco_suite_allocate(const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions) {

  coco_suite_t *suite;
  size_t i;

  suite = (coco_suite_t *) coco_allocate_memory(sizeof(*suite));
  suite->number_of_dimensions = number_of_dimensions;
  suite->dimensions = coco_allocate_memory(suite->number_of_dimensions * sizeof(size_t));
  for (i = 0; i < suite->number_of_dimensions; i++) {
    suite->dimensions[i] = dimensions;
  }

  suite->number_of_functions = number_of_functions;
  suite->functions = coco_allocate_memory(suite->number_of_functions * sizeof(size_t));
  for (i = 0; i < suite->number_of_functions; i++) {
    suite->functions[i] = i + 1;
  }

  suite->current_dimension_id = -1;
  suite->current_function_id = -1;
  suite->current_instance_id = -1;

  /* To be set in coco_suite_set_instance() */
  suite->number_of_instances = 0;
  suite->instances = NULL;
  suite->suite_name = NULL;

  return suite;
}

static void coco_suite_set_instance(coco_suite_t *suite,
                                    const size_t number_of_instances,
                                    const size_t *instance_numbers,
                                    const char *suite_name) {

  size_t i;

  suite->number_of_instances = number_of_instances;
  suite->instances = coco_allocate_memory(suite->number_of_instances * sizeof(size_t));
  for (i = 0; i < suite->number_of_instances; i++) {
    suite->instances[i] = instance_numbers[i];
  }

  suite->suite_name = coco_strdup(suite_name);

}

void coco_suite_free(coco_suite_t *suite) {

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

  coco_free_memory(suite);
}

coco_suite_t *coco_suite(const char *suite_name, const char *suite_instance, const char *suite_options) {

  /* TODO: Implement these functions! */

  coco_suite_t *suite = coco_suite_allocate();

  coco_suite_set_instance(suite);

  coco_suite_apply_filter(suite);

  /* TODO: Check that at least one dimension/function/instance exists! */

  return suite;
}

static size_t *coco_suite_parse_instance_string(coco_suite_t *suite, const char *suite_instance) {

  int year = -1;
  char *instances = NULL;
  long year_ptr, instances_ptr;

  if (suite_instance == NULL)
    return NULL;

  year_ptr = coco_strfind(suite_instance, "year");
  instances_ptr = coco_strfind(suite_instance, "instances");

  if ((year_ptr < 0) && (instances_ptr < 0))
    return NULL;

  if ((year_ptr < instances_ptr) && (year_ptr >= 0)) {
    /* Stores the year */
    if (coco_options_read_int(suite_instance, "year", &(year)) != 0) {
      if (instances_ptr >= 0) {
        instances = NULL;
        coco_warning("coco_suite_parse_instance_string(): only the 'year' suite_instance option is taken into account, 'instances' is ignored");
      }
    } else {
      year = -1;
      coco_warning("coco_suite_parse_instance_string(): problems parsing the 'year' suite_instance option, ignored");
    }
  }
  else {
    /* Stores the instances */
    if (coco_options_read_int(suite_instance, "instances", instances) != 0) {
      if (year_ptr >= 0) {
        year = -1;
        coco_warning("coco_suite_parse_instance_string(): only the 'instances' suite_instance option is taken into account, 'year' is ignored");
      }
    } else {
      instances = NULL;
      coco_warning("coco_suite_parse_instance_string(): problems parsing the 'instance' suite_instance option, ignored");
    }
  }

  if (year > 0)
    return coco_suite_get_instances_matching_year(suite, year);

  if (instances != NULL) {
    return coco_options_read_ranges(instances, "instances", 1, NAN);
  }

  /* This should never happen */
  coco_warning("coco_suite_parse_instance_string(): a problem occurred when parsing suite_instance options, ignored");
  return NULL;
}

/**
 * Iterates through the items from the current_item_id position on in search for the next positive item.
 * If such an item is found, current_item_id points to this item and the method returns 1. If such an
 * item cannot be found, current_item_id points to the first positive item and the method returns 0.
 */
static int coco_suite_is_next_item_found(const size_t number_of_items, const size_t *items, int *current_item_id) {

  /* Iterate through items */
  do {
    (*current_item_id)++;
  } while ((*current_item_id < number_of_items - 1) && (items[*current_item_id] == 0));

  if (items[*current_item_id] != 0) {
    /* Next item is found, return true */
    return 1;
  } else {
    /* Next item cannot be found, move to the first good item and return false */
    *current_item_id = -1;
    do {
      (*current_item_id)++;
    } while ((*current_item_id < number_of_items - 1) && (items[*current_item_id] == 0));
    if (items[*current_item_id] == 0)
      coco_error("coco_suite_is_next_item_found(): the chosen suite has no valid (positive) items");
    return 0;
  }
}

/**
 * Iterates through the instances of the given suite from the current_instance_id position on in search for
 * the next positive instance. If such an instance is found, current_instance_id points to this instance and
 * the method returns 1. If such an instance cannot be found, current_instance_id points to the first
 * positive instance and the method returns 0.
 */
static int coco_suite_is_next_instance_found(coco_suite_t *suite) {

  return coco_suite_is_next_item_found(suite->number_of_instances, suite->instances,
      &suite->current_instance_id);
}

/**
 * Iterates through the functions of the given suite from the current_function_id position on in search for
 * the next positive function. If such a function is found, current_function_id points to this function and
 * the method returns 1. If such a function cannot be found, current_function_id points to the first
 * positive function, current_instance_id points to the first positive instance and the method returns 0.
 */
static int coco_suite_is_next_function_found(coco_suite_t *suite) {

  int result = coco_suite_is_next_item_found(suite->number_of_functions, suite->functions,
      &suite->current_function_id);
  if (!result) {
    /* Reset the instances */
    suite->current_instance_id = -1;
    coco_suite_is_next_instance_found(suite);
  }
  return result;
}

/**
 * Iterates through the dimensions of the given suite from the current_dimension_id position on in search for
 * the next positive dimension. If such a dimension is found, current_dimension_id points to this dimension
 * and the method returns 1. If such a dimension cannot be found, current_dimension_id points to the first
 * positive dimension and the method returns 0.
 */
static int coco_suite_is_next_dimension_found(coco_suite_t *suite) {

  return coco_suite_is_next_item_found(suite->number_of_dimensions, suite->dimensions,
      &suite->current_dimension_id);
}

/**
 * Returns next problem of the suite by iterating first through available instances, then functions and
 * lastly dimensions. The problem is wrapped with the observer's logger. If there is no next problem,
 * returns NULL.
 */
coco_problem_t *coco_suite_get_next_problem(coco_suite_t *suite, coco_observer_t *observer) {

  coco_problem_t *problem;
  size_t function, dimension, instance;

  if (!coco_suite_is_next_instance_found(suite)
      && !coco_suite_is_next_function_found(suite)
      && !coco_suite_is_next_dimension_found(suite))
    return NULL;

  function = suite->functions[suite->current_function_id];
  dimension = suite->dimensions[suite->current_dimension_id];
  instance = suite->instances[suite->current_instance_id];

  if (strcmp(suite->suite_name, "suite_toy") == 0) {
    problem = suite_toy_get_problem(function, dimension, instance);
  } else if (strcmp(suite->suite_name, "suite_bbob") == 0) {
    problem = suite_bbob_get_problem(function, dimension, instance);
  } else if (strcmp(suite->suite_name, "suite_biobj") == 0) {
    problem = suite_biobj_get_problem(function, dimension, instance);
  } else {
    coco_error("coco_suite_get_next_problem(): unknown problem suite");
    return NULL;
  }

  return coco_problem_add_observer(problem, observer);
}

void coco_benchmark(const char *suite_name,
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
    coco_problem_free(problem);

  }

  coco_observer_free(observer);
  coco_suite_free(suite);

}

/* TODO: Move code below to specific suite files */

static coco_problem_t *suite_bbob_get_problem(size_t function_id, size_t dimension, size_t instance_id) {

  coco_problem_t *problem = NULL;

  const char *problem_id_template = "bbob_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "BBOB f%lu instance %lu in %luD";

  const long rseed = (long) (function_id + 10000 * instance_id);
  const long rseed_3 = (long) (3 + 10000 * instance_id);
  const long rseed_17 = (long) (17 + 10000 * instance_id);

  if (function_id == 1) {
    problem = f_sphere_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 2) {
    problem = f_ellipsoid_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 3) {
    problem = f_rastrigin_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 4) {
    problem = f_bueche_rastrigin_bbob_problem_allocate(function_id, dimension, instance_id, rseed_3,
        problem_id_template, problem_name_template);
  } else if (function_id == 5) {
    problem = f_linear_slope_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 6) {
    problem = f_attractive_sector_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 7) {
    problem = f_step_ellipsoid_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 8) {
    problem = f_rosenbrock_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 9) {
    problem = f_rosenbrock_rotated_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 10) {
    problem = f_ellipsoid_rotated_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 11) {
    problem = f_discus_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 12) {
    problem = f_bent_cigar_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 13) {
    problem = f_sharp_ridge_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 14) {
    problem = f_different_powers_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 15) {
    problem = f_rastrigin_rotated_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 16) {
    problem = f_weierstrass_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 17) {
    problem = f_schaffers_bbob_problem_allocate(function_id, dimension, instance_id, rseed, 10,
        problem_id_template, problem_name_template);
  } else if (function_id == 18) {
    problem = f_schaffers_bbob_problem_allocate(function_id, dimension, instance_id, rseed_17, 1000,
        problem_id_template, problem_name_template);
  } else if (function_id == 19) {
    problem = f_griewank_rosenbrock_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 20) {
    problem = f_schwefel_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 21) {
    problem = f_gallagher_bbob_problem_allocate(function_id, dimension, instance_id, rseed, 101,
        problem_id_template, problem_name_template);
  } else if (function_id == 22) {
    problem = f_gallagher_bbob_problem_allocate(function_id, dimension, instance_id, rseed, 21,
        problem_id_template, problem_name_template);
  } else if (function_id == 23) {
    problem = f_katsuura_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  } else if (function_id == 24) {
    problem = f_lunacek_bi_rastrigin_bbob_problem_allocate(function_id, dimension, instance_id, rseed,
        problem_id_template, problem_name_template);
  }

  return problem;
}

