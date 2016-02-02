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
 * The target values that trigger logging are 10**(exponent/number_of_targets) from DBL_MAX down to
 * precision and from -precision on with step -10**(exponent/number_of_targets) without any explicit
 * minimal value.
 */
typedef struct {

  int exponent;               /**< @brief Exponent used for computing the currently hit target. */
  double value;               /**< @brief Value of the currently hit target. */
  size_t number_of_targets;   /**< @brief The number of targets between 10**(i/n) and 10**((i+1)/n). */
  double precision;           /**< @brief Minimal precision of interest. */

} coco_observer_targets_t;

/**
 * @brief The type for triggers based on numbers of evaluations.
 *
 * The numbers of evaluations that trigger logging are 1 and every base_evaluation*dim*(10**exponent), where
 * dim is the dimensionality of the problem (number of variables) and exponent is an integer >= 0.
 */
typedef struct {

  size_t evaluation_trigger;  /**< @brief The next evaluation number that triggers logging. */
  size_t *base_evaluations;   /**< @brief The base evaluation numbers used to compute the actual evaluation
                                   numbers that trigger logging. */
  size_t base_count;          /**< @brief The number of base evaluations. */
  size_t base_index;          /**< @brief The next index of the base evaluations. */
  size_t exponent;            /**< @brief Exponent used for computing the next trigger. */

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
  targets->number_of_targets = number_of_targets;
  targets->precision = precision;

  return targets;
}

/**
 * @brief Computes and returns whether the given value should trigger logging while also updating the state
 * of the targets.
 *
 * @note Takes into account also the negative case and the almost-zero case (given_value smaller than the
 * minimal precision).
 */
static int coco_observer_targets_trigger(coco_observer_targets_t *targets, const double given_value) {

  int update_performed = 0;
  const double number_of_targets_double = (double) (long) targets->number_of_targets;
  int last_exponent = targets->exponent;
  double verified_value = given_value;
  int sign = 1;

  assert(targets != NULL);

  /* Handle the almost zero case */
  if (fabs(given_value) < targets->precision) {
    verified_value = targets->precision;
  }
  /* Handle the negative case larger than -targets->minimal_precision */
  else if (given_value < 0) {
    verified_value = -given_value;
    sign = -1;
  }

  /* If this is the first time the update was called, set the last_exponent to some value greater than the
   * current exponent */
  if (last_exponent == INT_MAX) {
    last_exponent = (int) (ceil(log10(verified_value) * number_of_targets_double)) + sign;
  }
  targets->exponent = (int) (ceil(log10(verified_value) * number_of_targets_double));

  if (targets->exponent != last_exponent) {
    targets->value = pow(10, (double) targets->exponent / number_of_targets_double) * sign;
    update_performed = 1;
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
 * @param base_evaluations Evaluation numbers formatted as a string, which are used as the base to compute
 * the actual triggers. For example, if base_evaluations = "1,2,5", the logger will be triggered by
 * evaluations 1, dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2, 10*dim*5, 100*dim*1, 100*dim*2, 100*dim*5, ...
 */
static coco_observer_evaluations_t *coco_observer_evaluations(const char *base_evaluations) {

  coco_observer_evaluations_t *evaluations = (coco_observer_evaluations_t *) coco_allocate_memory(
      sizeof(*evaluations));
  evaluations->evaluation_trigger = 1;
  evaluations->base_index = 0;
  evaluations->base_evaluations = coco_string_parse_ranges(base_evaluations, 1, 0, "base_evaluations",
      COCO_MAX_EVALS_TO_LOG);
  evaluations->base_count = coco_count_numbers(evaluations->base_evaluations, COCO_MAX_EVALS_TO_LOG,
      "base_evaluations");
  evaluations->exponent = 0;

  return evaluations;
}

/**
 * @brief Computes and returns whether the given evaluation number should trigger logging while also updating
 * the state of the evaluation trigger.
 */
static int coco_observer_evaluations_trigger(coco_observer_evaluations_t *evaluations,
                                             const size_t evaluation_number,
                                             const size_t dimension) {

  if (evaluation_number == evaluations->evaluation_trigger) {
    /* Compute the next trigger */
    if (evaluation_number != 1) {
      if (evaluations->base_index < evaluations->base_count - 1) {
        evaluations->base_index++;
      } else {
        evaluations->base_index = 0;
        evaluations->exponent++;
      }
    }
    evaluations->evaluation_trigger = (size_t) (pow(10, (double) evaluations->exponent)
        * (double) (long) dimension * (double) (long) evaluations->base_evaluations[evaluations->base_index]);
    return 1;
  }
  return 0;
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
 * @brief A set of numbers from which the evaluations that should always be logged are computed.
 *
 * For example, if logger_biobj_always_log[3] = {1, 2, 5}, the logger will always output evaluations
 * 1, dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2, 10*dim*5, 100*dim*1, 100*dim*2, 100*dim*5, ...
 */
static const size_t coco_observer_always_log[3] = {1, 2, 5};

/**
 * @brief Determines whether the evaluation should be logged.
 *
 * Returns true if the evaluation_number corresponds to a number that should always be logged and false
 * otherwise (computed from coco_observer_always_log). For example, if coco_observer_always_log = {1, 2, 5},
 * returns true for 1, dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2, 10*dim*5, 100*dim*1, 100*dim*2, ...
 */
static int coco_observer_evaluation_to_log(const size_t evaluation_number, const size_t dimension) {

  size_t i;
  double j = 0, factor = 10;
  size_t count = sizeof(coco_observer_always_log) / sizeof(size_t);

  if (evaluation_number == 1)
    return 1;

  while ((size_t) pow(factor, j) * dimension <= evaluation_number) {
    for (i = 0; i < count; i++) {
      if (evaluation_number == (size_t) pow(factor, j) * dimension * coco_observer_always_log[i])
        return 1;
    }
    j++;
  }

  return 0;
}

/**
 * @brief Allocates memory for a coco_observer_t instance.
 */
static coco_observer_t *coco_observer_allocate(const char *result_folder,
                                               const char *algorithm_name,
                                               const char *algorithm_info,
                                               const size_t number_of_targets,
                                               const double target_precision,
                                               const char *base_evaluations,
                                               const int precision_x,
                                               const int precision_f) {

  coco_observer_t *observer;
  observer = (coco_observer_t *) coco_allocate_memory(sizeof(*observer));
  /* Initialize fields to sane/safe defaults */
  observer->result_folder = coco_strdup(result_folder);
  observer->algorithm_name = coco_strdup(algorithm_name);
  observer->algorithm_info = coco_strdup(algorithm_info);
  observer->number_of_targets = number_of_targets;
  observer->target_precision = target_precision;
  observer->base_evaluations = coco_strdup(base_evaluations);
  observer->precision_x = precision_x;
  observer->precision_f = precision_f;
  observer->data = NULL;
  observer->data_free_function = NULL;
  observer->logger_initialize_function = NULL;
  observer->is_active = 1;
  return observer;
}

void coco_observer_free(coco_observer_t *observer) {

  if (observer != NULL) {
    observer->is_active = 0;
    if (observer->result_folder != NULL)
      coco_free_memory(observer->result_folder);
    if (observer->algorithm_name != NULL)
      coco_free_memory(observer->algorithm_name);
    if (observer->algorithm_info != NULL)
      coco_free_memory(observer->algorithm_info);

    if (observer->base_evaluations != NULL)
      coco_free_memory(observer->base_evaluations);

    if (observer->data != NULL) {
      if (observer->data_free_function != NULL) {
        observer->data_free_function(observer->data);
      }
      coco_free_memory(observer->data);
      observer->data = NULL;
    }

    observer->logger_initialize_function = NULL;
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
 * - "number_of_targets: VALUE" defines the number of targets between each 10**(i/n) and 10**((i+1)/n)
 * (equally spaced in the logarithmic scale). The default value is 10.
 * - "target_precision: VALUE" defines the precision used for targets (there are no targets for
 * abs(values) < target_precision). The default value is 1e-8.
 * - "evaluations_to_log: VALUES" defines the "base evaluations" used to compute the evaluations that are to
 * be logged independently on their target values. The numbers of evaluations that trigger logging are 1 and
 * every base_evaluation*dim*(10**exponent), where dim is the dimensionality of the problem (number of
 * variables) and exponent is an integer >= 0. For example, if base_evaluations = "1,2,5", the logger will
 * be triggered by evaluations 1, dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2, 10*dim*5, 100*dim*1, 100*dim*2,
 * 100*dim*5, ... The default value is "1,2,5".
 * - "precision_x: VALUE" defines the precision used when outputting variables and corresponds to the number
 * of digits to be printed after the decimal point. The default value is 8.
 * - precision_f: VALUE defines the precision used when outputting f values and corresponds to the number of
 * digits to be printed after the decimal point. The default value is 15.
 *
 * @return The constructed observer object or NULL if observer_name equals NULL, "" or "no_observer".
 */
coco_observer_t *coco_observer(const char *observer_name, const char *observer_options) {

  coco_observer_t *observer;
  char *path, *result_folder, *algorithm_name, *algorithm_info;
  const char *outer_folder_name = "exdata";
  int precision_x, precision_f;

  size_t number_of_targets;
  double target_precision;
  char *base_evaluations;

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

  number_of_targets = 10;
  if (coco_options_read_size_t(observer_options, "number_of_targets", &number_of_targets) != 0) {
    if (number_of_targets == 0)
      number_of_targets = 10;
  }

  target_precision = 1e-8;
  if (coco_options_read_double(observer_options, "target_precision", &target_precision) != 0) {
    if ((target_precision > 1) || (target_precision <= 0))
      target_precision = 1e-8;
  }

  base_evaluations = coco_allocate_string(COCO_PATH_MAX);
  if (coco_options_read_string(observer_options, "evaluations_to_log", base_evaluations) == 0) {
    strcpy(base_evaluations, "1,2,5");
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

  observer = coco_observer_allocate(path, algorithm_name, algorithm_info, number_of_targets, target_precision,
      base_evaluations, precision_x, precision_f);

  coco_free_memory(path);
  coco_free_memory(result_folder);
  coco_free_memory(algorithm_name);
  coco_free_memory(algorithm_info);
  coco_free_memory(base_evaluations);

  /* Here each observer must have an entry */
  if (0 == strcmp(observer_name, "toy")) {
    observer_toy(observer, observer_options);
  } else if (0 == strcmp(observer_name, "bbob")) {
    observer_bbob(observer, observer_options);
  } else if (0 == strcmp(observer_name, "bbob-biobj")) {
    observer_biobj(observer, observer_options);
  } else {
    coco_warning("Unknown observer!");
    return NULL;
  }

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

  if ((observer == NULL) || (observer->is_active == 0)) {
    coco_warning("The problem will not be observed. %s", observer == NULL ? "(observer == NULL)" : "(observer not active)");
    return problem;
  }

  return observer->logger_initialize_function(observer, problem);
}

