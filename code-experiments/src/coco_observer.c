/**
 * @file coco_observer.c
 * @brief Definitions of functions regarding COCO observers.
 */

#include "coco.h"
#include "coco_internal.h"
#include <limits.h>
#include <float.h>
#include <math.h>

/**
 * @brief The type for triggers based on target values.
 *
 * The target values that trigger logging are at every 10**(exponent/number_of_triggers) from positive
 * infinity down to precision, at 0, and from -precision on with step -10**(exponent/number_of_triggers) until
 * negative infinity.
 */
typedef struct {

  int exponent;               /**< @brief Value used to compare with the previously hit target. */
  double value;               /**< @brief Value of the currently hit target. */
  size_t number_of_triggers;  /**< @brief Number of target triggers between 10**i and 10**(i+1) for any i. */
  double precision;           /**< @brief Minimal precision of interest. */

} coco_observer_targets_t;

/**
 * @brief The type for triggers based on numbers of evaluations.
 *
 * The numbers of evaluations that trigger logging are any of the two:
 * - every 10**(exponent1/number_of_triggers) for exponent1 >= 0
 * - every base_evaluation * dimension * (10**exponent2) for exponent2 >= 0
 */
typedef struct {

  /* First trigger */
  size_t value1;              /**< @brief The next value for the first trigger. */
  size_t exponent1;           /**< @brief Exponent used to compute the first trigger. */
  size_t number_of_triggers;  /**< @brief Number of target triggers between 10**i and 10**(i+1) for any i. */

  /* Second trigger */
  size_t value2;              /**< @brief The next value for the second trigger. */
  size_t exponent2;           /**< @brief Exponent used to compute the second trigger. */
  size_t *base_evaluations;   /**< @brief The base evaluation numbers used to compute the actual evaluation
                                   numbers that trigger logging. */
  size_t base_count;          /**< @brief The number of base evaluations. */
  size_t base_index;          /**< @brief The next index of the base evaluations. */
  size_t dimension;           /**< @brief Dimension used in the calculation of the first trigger. */

} coco_observer_evaluations_t;

/**
 * @brief The maximum number of evaluations to trigger logging.
 *
 * @note This is not the maximal evaluation number to be logged, but the maximal number of times logging is
 * triggered by the number of evaluations.
 */
#define COCO_MAX_EVALS_TO_LOG 1000

/***********************************************************************************************************/

/**
 * @name Methods regarding triggers based on target values
 */
/**@{*/

/**
 * @brief Creates and returns a structure containing information on targets.
 *
 * @param number_of_targets The number of targets between 10**(i/n) and 10**((i+1)/n) for each i.
 * @param precision Minimal precision of interest.
 */
static coco_observer_targets_t *coco_observer_targets(const size_t number_of_targets,
                                                      const double precision) {

  coco_observer_targets_t *targets = (coco_observer_targets_t *) coco_allocate_memory(sizeof(*targets));
  targets->exponent = INT_MAX;
  targets->value = DBL_MAX;
  targets->number_of_triggers = number_of_targets;
  targets->precision = precision;

  return targets;
}

/**
 * @brief Computes and returns whether the given value should trigger logging.
 */
static int coco_observer_targets_trigger(coco_observer_targets_t *targets, const double given_value) {

  int update_performed = 0;

  const double number_of_targets_double = (double) (long) targets->number_of_triggers;

  double verified_value = 0;
  int last_exponent = 0;
  int current_exponent = 0;
  int adjusted_exponent = 0;

  assert(targets != NULL);

  /* The given_value is positive or zero */
  if (given_value >= 0) {

  	if (given_value == 0) {
  		/* If zero, use even smaller value than precision */
  		verified_value = targets->precision / 10.0;
  	} else if (given_value < targets->precision) {
      /* If close to zero, use precision instead of the given_value*/
      verified_value = targets->precision;
    } else {
      verified_value = given_value;
    }

    current_exponent = (int) (ceil(log10(verified_value) * number_of_targets_double));

    /* If this is the first time the update was called, set the last_exponent to some value greater than the
     * current exponent */
    if (last_exponent == INT_MAX) {
      last_exponent = current_exponent + 1;
    } else {
      last_exponent = targets->exponent;
    }

    if (current_exponent < last_exponent) {
      /* Update the target information */
      targets->exponent = current_exponent;
      if (given_value == 0)
      	targets->value = 0;
      else
      	targets->value = pow(10, (double) current_exponent / number_of_targets_double);
      update_performed = 1;
    }
  }
  /* The given_value is negative, therefore adjustments need to be made */
  else {

    /* If close to zero, use precision instead of the given_value*/
    if (given_value > -targets->precision) {
      verified_value = targets->precision;
    } else {
      verified_value = -given_value;
    }

    /* Adjustment: use floor instead of ceil! */
    current_exponent = (int) (floor(log10(verified_value) * number_of_targets_double));

    /* If this is the first time the update was called, set the last_exponent to some value greater than the
     * current exponent */
    if (last_exponent == INT_MAX) {
      last_exponent = current_exponent + 1;
    } else {
      last_exponent = targets->exponent;
    }

    /* Compute the adjusted_exponent in such a way, that it is always diminishing in value. The adjusted
     * exponent can only be used to verify if a new target has been hit. To compute the actual target
     * value, the current_exponent needs to be used. */
    adjusted_exponent = 2 * (int) (ceil(log10(targets->precision / 10.0) * number_of_targets_double))
        - current_exponent - 1;

    if (adjusted_exponent < last_exponent) {
      /* Update the target information */
      targets->exponent = adjusted_exponent;
      targets->value = - pow(10, (double) current_exponent / number_of_targets_double);
      update_performed = 1;
    }
  }

  return update_performed;
}

/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding triggers based on numbers of evaluations.
 */
/**@{*/

/**
 * @brief Creates and returns a structure containing information on triggers based on evaluation numbers.
 *
 * The numbers of evaluations that trigger logging are any of the two:
 * - every 10**(exponent1/number_of_triggers) for exponent1 >= 0
 * - every base_evaluation * dimension * (10**exponent2) for exponent2 >= 0
 *
 * @note The coco_observer_evaluations_t object instances need to be freed using the
 * coco_observer_evaluations_free function!
 *
 * @param base_evaluations Evaluation numbers formatted as a string, which are used as the base to compute
 * the second trigger. For example, if base_evaluations = "1,2,5", the logger will be triggered by
 * evaluations dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2, 10*dim*5, 100*dim*1, 100*dim*2, 100*dim*5, ...
 */
static coco_observer_evaluations_t *coco_observer_evaluations(const char *base_evaluations,
                                                              const size_t dimension) {

  coco_observer_evaluations_t *evaluations = (coco_observer_evaluations_t *) coco_allocate_memory(
      sizeof(*evaluations));

  /* First trigger */
  evaluations->value1 = 1;
  evaluations->exponent1 = 0;
  evaluations->number_of_triggers = 20;

  /* Second trigger */
  evaluations->base_evaluations = coco_string_parse_ranges(base_evaluations, 1, 0, "base_evaluations",
      COCO_MAX_EVALS_TO_LOG);
  evaluations->dimension = dimension;
  evaluations->base_count = coco_count_numbers(evaluations->base_evaluations, COCO_MAX_EVALS_TO_LOG,
      "base_evaluations");
  evaluations->base_index = 0;
  evaluations->value2 = dimension * evaluations->base_evaluations[0];
  evaluations->exponent2 = 0;

  return evaluations;
}

/**
 * @brief Computes and returns whether the given evaluation number triggers the first condition of the
 * logging based on the number of evaluations.
 *
 * The second condition is:
 * evaluation_number == 10**(exponent1/number_of_triggers)
 */
static int coco_observer_evaluations_trigger_first(coco_observer_evaluations_t *evaluations,
                                                   const size_t evaluation_number) {

  assert(evaluations != NULL);

  if (evaluation_number == evaluations->value1) {
    /* Compute the next value for the first trigger */
    while (coco_double_to_size_t(floor(pow(10, (double) evaluations->exponent1 / (double) evaluations->number_of_triggers)) <= evaluations->value1)) {
      evaluations->exponent1++;
    }
    evaluations->value1 = coco_double_to_size_t(floor(pow(10, (double) evaluations->exponent1 / (double) evaluations->number_of_triggers)));
    return 1;
  }
  return 0;
}

/**
 * @brief Computes and returns whether the given evaluation number triggers the second condition of the
 * logging based on the number of evaluations.
 *
 * The second condition is:
 * evaluation_number == base_evaluation[base_index] * dimension * (10**exponent2)
 */
static int coco_observer_evaluations_trigger_second(coco_observer_evaluations_t *evaluations,
                                                    const size_t evaluation_number) {

  assert(evaluations != NULL);

  if (evaluation_number == evaluations->value2) {
    /* Compute the next value for the second trigger */
    if (evaluations->base_index < evaluations->base_count - 1) {
      evaluations->base_index++;
    } else {
      evaluations->base_index = 0;
      evaluations->exponent2++;
    }
    evaluations->value2 = coco_double_to_size_t(pow(10, (double) evaluations->exponent2)
        * (double) (long) evaluations->dimension
        * (double) (long) evaluations->base_evaluations[evaluations->base_index]);
    return 1;
  }
  return 0;
}

/**
 * @brief Returns 1 if any of the two triggers based on the number of evaluations equal 1 and 0 otherwise.
 *
 * The numbers of evaluations that trigger logging are any of the two:
 * - every 10**(exponent1/number_of_triggers) for exponent1 >= 0
 * - every base_evaluation * dimension * (10**exponent2) for exponent2 >= 0
 */
static int coco_observer_evaluations_trigger(coco_observer_evaluations_t *evaluations,
                                             const size_t evaluation_number) {

  /* Both functions need to be called so that both triggers are correctly updated */
  int first = coco_observer_evaluations_trigger_first(evaluations, evaluation_number);
  int second = coco_observer_evaluations_trigger_second(evaluations, evaluation_number);

  return (first + second > 0) ? 1: 0;
}

/**
 * @brief Frees the given evaluations object.
 */
static void coco_observer_evaluations_free(coco_observer_evaluations_t *evaluations) {

  assert(evaluations != NULL);
  coco_free_memory(evaluations->base_evaluations);
  coco_free_memory(evaluations);
}

/**@}*/

/***********************************************************************************************************/

/**
 * @brief Allocates memory for a coco_observer_t instance.
 */
static coco_observer_t *coco_observer_allocate(const char *result_folder,
                                               const char *observer_name,
                                               const char *algorithm_name,
                                               const char *algorithm_info,
                                               const size_t number_target_triggers,
                                               const double target_precision,
                                               const size_t number_evaluation_triggers,
                                               const char *base_evaluation_triggers,
                                               const int precision_x,
                                               const int precision_f) {

  coco_observer_t *observer;
  observer = (coco_observer_t *) coco_allocate_memory(sizeof(*observer));
  /* Initialize fields to sane/safe defaults */
  observer->result_folder = coco_strdup(result_folder);
  observer->observer_name = coco_strdup(observer_name);
  observer->algorithm_name = coco_strdup(algorithm_name);
  observer->algorithm_info = coco_strdup(algorithm_info);
  observer->number_target_triggers = number_target_triggers;
  observer->target_precision = target_precision;
  observer->number_evaluation_triggers = number_evaluation_triggers;
  observer->base_evaluation_triggers = coco_strdup(base_evaluation_triggers);
  observer->precision_x = precision_x;
  observer->precision_f = precision_f;
  observer->data = NULL;
  observer->data_free_function = NULL;
  observer->logger_allocate_function = NULL;
  observer->logger_free_function = NULL;
  observer->is_active = 1;
  return observer;
}

void coco_observer_free(coco_observer_t *observer) {

  if (observer != NULL) {
    observer->is_active = 0;
    if (observer->observer_name != NULL)
      coco_free_memory(observer->observer_name);
    if (observer->result_folder != NULL)
      coco_free_memory(observer->result_folder);
    if (observer->algorithm_name != NULL)
      coco_free_memory(observer->algorithm_name);
    if (observer->algorithm_info != NULL)
      coco_free_memory(observer->algorithm_info);

    if (observer->base_evaluation_triggers != NULL)
      coco_free_memory(observer->base_evaluation_triggers);

    if (observer->data != NULL) {
      if (observer->data_free_function != NULL) {
        observer->data_free_function(observer->data);
      }
      coco_free_memory(observer->data);
      observer->data = NULL;
    }

    observer->logger_allocate_function = NULL;
    observer->logger_free_function = NULL;

    coco_free_memory(observer);
    observer = NULL;
  }
}

#include "logger_bbob.c"
#include "logger_biobj.c"
#include "logger_toy.c"

/**
 * Currently, three observers are supported:
 * - "bbob" is the observer for single-objective (both noisy and noiseless) problems with known optima, which
 * creates *.info, *.dat, *.tdat and *.rdat files and logs the distance to the optimum.
 * - "bbob-biobj" is the observer for bi-objective problems, which creates *.info, *.dat and *.tdat files for
 * the given indicators, as well as an archive folder with *.adat files containing nondominated solutions.
 * - "toy" is a simple observer that logs when a target has been hit.
 *
 * @param observer_name A string containing the name of the observer. Currently supported observer names are
 * "bbob", "bbob-biobj", "toy". Strings "no_observer", "" or NULL return NULL.
 * @param observer_options A string of pairs "key: value" used to pass the options to the observer. Some
 * observer options are general, while others are specific to some observers. Here we list only the general
 * options, see observer_bbob, observer_biobj and observer_toy for options of the specific observers.
 * - "result_folder: NAME" determines the folder within the "exdata" folder into which the results will be
 * output. If the folder with the given name already exists, first NAME_001 will be tried, then NAME_002 and
 * so on. The default value is "default".
 * - "algorithm_name: NAME", where NAME is a short name of the algorithm that will be used in plots (no
 * spaces are allowed). The default value is "ALG".
 * - "algorithm_info: STRING" stores the description of the algorithm. If it contains spaces, it must be
 * surrounded by double quotes. The default value is "" (no description).
 * - "number_target_triggers: VALUE" defines the number of targets between each 10**i and 10**(i+1)
 * (equally spaced in the logarithmic scale) that trigger logging. The default value is 100.
 * - "target_precision: VALUE" defines the precision used for targets (there are no targets for
 * abs(values) < target_precision). The default value is 1e-8.
 * - "number_evaluation_triggers: VALUE" defines the number of evaluations to be logged between each 10**i
 * and 10**(i+1). The default value is 20.
 * - "base_evaluation_triggers: VALUES" defines the base evaluations used to produce an additional
 * evaluation-based logging. The numbers of evaluations that trigger logging are every
 * base_evaluation * dimension * (10**i). For example, if base_evaluation_triggers = "1,2,5", the logger will
 * be triggered by evaluations dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2, 10*dim*5, 100*dim*1, 100*dim*2,
 * 100*dim*5, ... The default value is "1,2,5".
 * - "precision_x: VALUE" defines the precision used when outputting variables and corresponds to the number
 * of digits to be printed after the decimal point. The default value is 8.
 * - "precision_f: VALUE" defines the precision used when outputting f values and corresponds to the number of
 * digits to be printed after the decimal point. The default value is 15.
 *
 * @return The constructed observer object or NULL if observer_name equals NULL, "" or "no_observer".
 */
coco_observer_t *coco_observer(const char *observer_name, const char *observer_options) {

  coco_observer_t *observer;
  char *path, *result_folder, *algorithm_name, *algorithm_info;
  const char *outer_folder_name = "exdata";
  int precision_x, precision_f;

  size_t number_target_triggers;
  size_t number_evaluation_triggers;
  double target_precision;
  char *base_evaluation_triggers;

  coco_option_keys_t *known_option_keys, *given_option_keys, *additional_option_keys, *redundant_option_keys;

  /* Sets the valid keys for observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "result_folder", "algorithm_name", "algorithm_info",
      "number_target_triggers", "target_precision", "number_evaluation_triggers", "base_evaluation_triggers",
      "precision_x", "precision_f" };
  additional_option_keys = NULL; /* To be set by the chosen observer */

  if (0 == strcmp(observer_name, "no_observer")) {
    return NULL;
  } else if (strlen(observer_name) == 0) {
    coco_warning("Empty observer_name has no effect. To prevent this warning use 'no_observer' instead");
    return NULL;
  }

  result_folder = coco_allocate_string(COCO_PATH_MAX);
  algorithm_name = coco_allocate_string(COCO_PATH_MAX);
  algorithm_info = coco_allocate_string(5 * COCO_PATH_MAX);
  /* Read result_folder, algorithm_name and algorithm_info from the observer_options and use
   * them to initialize the observer */
  if (coco_options_read_string(observer_options, "result_folder", result_folder) == 0) {
    strcpy(result_folder, "default");
  }
  /* Create the result_folder inside the "exdata" folder */
  path = coco_allocate_string(COCO_PATH_MAX);
  memcpy(path, outer_folder_name, strlen(outer_folder_name) + 1);
  coco_join_path(path, COCO_PATH_MAX, result_folder, NULL);
  coco_create_unique_directory(&path);
  coco_info("Results will be output to folder %s", path);

  if (coco_options_read_string(observer_options, "algorithm_name", algorithm_name) == 0) {
    strcpy(algorithm_name, "ALG");
  }

  if (coco_options_read_string(observer_options, "algorithm_info", algorithm_info) == 0) {
    strcpy(algorithm_info, "");
  }

  number_target_triggers = 100;
  if (coco_options_read_size_t(observer_options, "number_target_triggers", &number_target_triggers) != 0) {
    if (number_target_triggers == 0)
      number_target_triggers = 100;
  }

  target_precision = 1e-8;
  if (coco_options_read_double(observer_options, "target_precision", &target_precision) != 0) {
    if ((target_precision > 1) || (target_precision <= 0))
      target_precision = 1e-8;
  }

  number_evaluation_triggers = 20;
  if (coco_options_read_size_t(observer_options, "number_evaluation_triggers", &number_evaluation_triggers) != 0) {
    if (number_evaluation_triggers < 4)
      number_evaluation_triggers = 20;
  }

  base_evaluation_triggers = coco_allocate_string(COCO_PATH_MAX);
  if (coco_options_read_string(observer_options, "base_evaluation_triggers", base_evaluation_triggers) == 0) {
    strcpy(base_evaluation_triggers, "1,2,5");
  }

  precision_x = 8;
  if (coco_options_read_int(observer_options, "precision_x", &precision_x) != 0) {
    if ((precision_x < 1) || (precision_x > 32))
      precision_x = 8;
  }

  precision_f = 15;
  if (coco_options_read_int(observer_options, "precision_f", &precision_f) != 0) {
    if ((precision_f < 1) || (precision_f > 32))
      precision_f = 15;
  }

  observer = coco_observer_allocate(path, observer_name, algorithm_name, algorithm_info,
      number_target_triggers, target_precision, number_evaluation_triggers, base_evaluation_triggers,
      precision_x, precision_f);

  coco_free_memory(path);
  coco_free_memory(result_folder);
  coco_free_memory(algorithm_name);
  coco_free_memory(algorithm_info);
  coco_free_memory(base_evaluation_triggers);

  /* Here each observer must have an entry - a call to a specific function that sets the additional_option_keys
   * and the following observer fields:
   * - logger_allocate_function
   * - logger_free_function
   * - data_free_function
   * - data */
  if (0 == strcmp(observer_name, "toy")) {
    observer_toy(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "bbob") || 0 == strcmp(observer_name, "bbob-largescale")) {
    observer_bbob(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "bbob-biobj")) {
    observer_biobj(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "bbob-biobj-ext")) {
    observer_biobj(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "bbob-largescale")) {
    observer_bbob(observer, observer_options, &additional_option_keys);
  } else {
    coco_warning("Unknown observer!");
    return NULL;
  }

  /* Check for redundant option keys */
  known_option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);
  coco_option_keys_add(&known_option_keys, additional_option_keys);
  given_option_keys = coco_option_keys(observer_options);

  if (given_option_keys) {
    redundant_option_keys = coco_option_keys_get_redundant(known_option_keys, given_option_keys);

    if ((redundant_option_keys != NULL) && (redundant_option_keys->count > 0)) {
      /* Warn the user that some of given options are being ignored and output the valid options */
      char *output_redundant = coco_option_keys_get_output_string(redundant_option_keys,
          "coco_observer(): Some keys in observer options were ignored:\n");
      char *output_valid = coco_option_keys_get_output_string(known_option_keys,
          "Valid keys for observer options are:\n");
      coco_warning("%s%s", output_redundant, output_valid);
      coco_free_memory(output_redundant);
      coco_free_memory(output_valid);
    }

    coco_option_keys_free(given_option_keys);
    coco_option_keys_free(redundant_option_keys);
  }
  coco_option_keys_free(known_option_keys);
  coco_option_keys_free(additional_option_keys);

  return observer;
}

/**
 * Wraps the observer's logger around the problem if the observer is not NULL and invokes the initialization
 * of this logger.
 *
 * @param problem The given COCO problem.
 * @param observer The COCO observer, whose logger will wrap the problem.
 *
 * @returns The observed problem in the form of a new COCO problem instance or the same problem if the
 * observer is NULL.
 */
coco_problem_t *coco_problem_add_observer(coco_problem_t *problem, coco_observer_t *observer) {

  if (problem == NULL)
	  return NULL;

  if ((observer == NULL) || (observer->is_active == 0)) {
    coco_warning("The problem will not be observed. %s",
        observer == NULL ? "(observer == NULL)" : "(observer not active)");
    return problem;
  }

  assert(observer->logger_allocate_function);
  return observer->logger_allocate_function(observer, problem);
}

/**
 * Frees the observer's logger and returns the inner problem.
 *
 * @param problem The observed COCO problem.
 * @param observer The COCO observer, whose logger was wrapping the problem.
 *
 * @returns The unobserved problem as a pointer to the inner problem or the same problem if the problem
 * was not observed.
 */
coco_problem_t *coco_problem_remove_observer(coco_problem_t *problem, coco_observer_t *observer) {

  coco_problem_t *problem_unobserved;
  char *prefix;

  if ((observer == NULL) || (observer->is_active == 0)) {
    coco_warning("The problem was not observed. %s",
        observer == NULL ? "(observer == NULL)" : "(observer not active)");
    return problem;
  }

  /* Check that we are removing the observer that is actually wrapping the problem.
   *
   * This is a hack - it assumes that the name of the problem is formatted as "observer_name(problem_name)".
   * While not elegant, it does the job and is better than nothing. */
  prefix = coco_remove_from_string(problem->problem_name, "(", "");
  if (strcmp(prefix, observer->observer_name) != 0) {
    coco_error("coco_problem_remove_observer(): trying to remove observer %s instead of %s",
        observer->observer_name, prefix);
  }
  coco_free_memory(prefix);

  /* Keep the inner problem and remove the logger data */
  problem_unobserved = coco_problem_transformed_get_inner_problem(problem);
  coco_problem_transformed_free_data(problem);
  problem = NULL;

  return problem_unobserved;
}
