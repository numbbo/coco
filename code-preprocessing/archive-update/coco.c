
/************************************************************************
 * WARNING
 *
 * This file is an auto-generated amalgamation. Any changes made to this
 * file will be lost when it is regenerated!
 ************************************************************************/

#line 1 "code-experiments/src/coco_random.c"
/**
 * @file coco_random.c
 * @brief Definitions of functions regarding COCO random numbers.
 *
 * @note This file contains non-C89-standard types (such as uint32_t and uint64_t), which should
 * eventually be fixed.
 */

#include <math.h>

#line 1 "code-experiments/src/coco.h"
/**
 * @file coco.h
 * @brief All public COCO functions, constants and variables are defined in this file.
 *
 * It is the authoritative reference, if any function deviates from the documented behavior it is considered
 * a bug. See the function definitions for their detailed descriptions.
 */
 
#ifndef __COCO_H__
#define __COCO_H__

#include <stddef.h>

/* Definitions of some 32 and 64-bit types (used by the random number generator) */
#ifdef _MSC_VER
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

/* Include definition for NAN among other things */
#include <math.h>
#ifndef NAN
/** @brief To be used only if undefined by the included headers */
#define NAN 8.8888e88
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief COCO's own pi constant. Simplifies the case, when the value of pi changes.
 */
/**@{*/
static const double coco_pi = 3.14159265358979323846;
static const double coco_two_pi = 2.0 * 3.14159265358979323846;
/**@}*/

/***********************************************************************************************************/

/** @brief Logging level type. */
typedef enum {
  COCO_ERROR,     /**< @brief only error messages are output */
  COCO_WARNING,   /**< @brief error and warning messages are output */
  COCO_INFO,      /**< @brief error, warning and info messages are output */
  COCO_DEBUG      /**< @brief error, warning, info and debug messages are output */
} coco_log_level_type_e;

/***********************************************************************************************************/

/** @brief Structure containing a COCO problem. */
struct coco_problem_s;

/**
 * @brief The COCO problem type.
 *
 * See coco_problem_s for more information on its fields. */
typedef struct coco_problem_s coco_problem_t;

/** @brief Structure containing a COCO suite. */
struct coco_suite_s;

/**
 * @brief The COCO suite type.
 *
 * See coco_suite_s for more information on its fields. */
typedef struct coco_suite_s coco_suite_t;

/** @brief Structure containing a COCO observer. */
struct coco_observer_s;

/**
 * @brief The COCO observer type.
 *
 * See coco_observer_s for more information on its fields. */
typedef struct coco_observer_s coco_observer_t;

/** @brief Structure containing a COCO archive. */
struct coco_archive_s;

/**
 * @brief The COCO archive type.
 *
 * See coco_archive_s for more information on its fields. */
typedef struct coco_archive_s coco_archive_t;

/** @brief Structure containing a COCO random state. */
struct coco_random_state_s;

/**
 * @brief The COCO random state type.
 *
 * See coco_random_state_s for more information on its fields. */
typedef struct coco_random_state_s coco_random_state_t;

/***********************************************************************************************************/

/**
 * @name Methods regarding COCO suite
 */
/**@{*/

/**
 * @brief Constructs a COCO suite.
 */
coco_suite_t *coco_suite(const char *suite_name, const char *suite_instance, const char *suite_options);

/**
 * @brief Frees the given suite.
 */
void coco_suite_free(coco_suite_t *suite);

/**
 * @brief Returns the next (observed) problem of the suite or NULL if there is no next problem left.
 */
coco_problem_t *coco_suite_get_next_problem(coco_suite_t *suite, coco_observer_t *observer);

/**
 * @brief Returns the problem of the suite defined by problem_index.
 */
coco_problem_t *coco_suite_get_problem(coco_suite_t *suite, const size_t problem_index);

/**
 * @brief Returns the number of problems in the given suite.
 */
size_t coco_suite_get_number_of_problems(const coco_suite_t *suite);

/**
 * @brief Returns the function number in the suite in position function_idx (counting from 0).
 */
size_t coco_suite_get_function_from_function_index(const coco_suite_t *suite, const size_t function_idx);

/**
 * @brief Returns the dimension number in the suite in position dimension_idx (counting from 0).
 */
size_t coco_suite_get_dimension_from_dimension_index(const coco_suite_t *suite, const size_t dimension_idx);

/**
 * @brief Returns the instance number in the suite in position instance_idx (counting from 0).
 */
size_t coco_suite_get_instance_from_instance_index(const coco_suite_t *suite, const size_t instance_idx);
/**@}*/

/**
 * @name Encoding/decoding problem index
 *
 * General schema for encoding/decoding a problem index. Note that the index depends on the number of
 * instances a suite is defined with (it should be called a suite-instance-depending index...).
 * Also, while functions, instances and dimensions start from 1, function_idx, instance_idx and dimension_idx
 * as well as suite_dep_index start from 0!
 *
 * Showing an example with 2 dimensions (2, 3), 5 instances (6, 7, 8, 9, 10) and 2 functions (1, 2):
 *
   \verbatim
   index | instance | function | dimension
   ------+----------+----------+-----------
       0 |        6 |        1 |         2
       1 |        7 |        1 |         2
       2 |        8 |        1 |         2
       3 |        9 |        1 |         2
       4 |       10 |        1 |         2
       5 |        6 |        2 |         2
       6 |        7 |        2 |         2
       7 |        8 |        2 |         2
       8 |        9 |        2 |         2
       9 |       10 |        2 |         2
      10 |        6 |        1 |         3
      11 |        7 |        1 |         3
      12 |        8 |        1 |         3
      13 |        9 |        1 |         3
      14 |       10 |        1 |         3
      15 |        6 |        2 |         2
      16 |        7 |        2 |         3
      17 |        8 |        2 |         3
      18 |        9 |        2 |         3
      19 |       10 |        2 |         3

   index | instance_idx | function_idx | dimension_idx
   ------+--------------+--------------+---------------
       0 |            0 |            0 |             0
       1 |            1 |            0 |             0
       2 |            2 |            0 |             0
       3 |            3 |            0 |             0
       4 |            4 |            0 |             0
       5 |            0 |            1 |             0
       6 |            1 |            1 |             0
       7 |            2 |            1 |             0
       8 |            3 |            1 |             0
       9 |            4 |            1 |             0
      10 |            0 |            0 |             1
      11 |            1 |            0 |             1
      12 |            2 |            0 |             1
      13 |            3 |            0 |             1
      14 |            4 |            0 |             1
      15 |            0 |            1 |             1
      16 |            1 |            1 |             1
      17 |            2 |            1 |             1
      18 |            3 |            1 |             1
      19 |            4 |            1 |             1
   \endverbatim
 */
/**@{*/
/**
 * @brief Computes the index of the problem in the suite that corresponds to the given function, dimension
 * and instance indices.
 */
size_t coco_suite_encode_problem_index(const coco_suite_t *suite,
                                       const size_t function_idx,
                                       const size_t dimension_idx,
                                       const size_t instance_idx);

/**
 * @brief Computes the function, dimension and instance indexes of the problem with problem_index in the
 * given suite.
 */
void coco_suite_decode_problem_index(const coco_suite_t *suite,
                                     const size_t problem_index,
                                     size_t *function_idx,
                                     size_t *instance_idx,
                                     size_t *dimension_idx);
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding COCO observer
 */
/**@{*/
/**
 * @brief Constructs a COCO observer.
 */
coco_observer_t *coco_observer(const char *observer_name, const char *options);

/**
 * @brief Frees the given observer.
 */
void coco_observer_free(coco_observer_t *observer);

/**
 * @brief Adds an observer to the given problem.
 */
coco_problem_t *coco_problem_add_observer(coco_problem_t *problem, coco_observer_t *observer);

/**
 * @brief Removes an observer from the given problem.
 */
coco_problem_t *coco_problem_remove_observer(coco_problem_t *problem, coco_observer_t *observer);

/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding COCO problem
 */
/**@{*/
/**
 * @brief Evaluates the problem function in point x and save the result in y.
 */
void coco_evaluate_function(coco_problem_t *problem, const double *x, double *y);

/**
 * @brief Evaluates the problem constraints in point x and save the result in y.
 */
void coco_evaluate_constraint(coco_problem_t *problem, const double *x, double *y);

/**
 * @brief Recommends a solution as the current best guesses to the problem.
 */
void coco_recommend_solution(coco_problem_t *problem, const double *x);

/**
 * @brief Frees the given problem.
 */
void coco_problem_free(coco_problem_t *problem);

/**
 * @brief Returns the name of the problem.
 */
const char *coco_problem_get_name(const coco_problem_t *problem);

/**
 * @brief Returns the ID of the problem.
 */
const char *coco_problem_get_id(const coco_problem_t *problem);

/**
 * @brief Returns the number of variables i.e. the dimension of the problem.
 */
size_t coco_problem_get_dimension(const coco_problem_t *problem);

/**
 * @brief Returns the number of objectives of the problem.
 */
size_t coco_problem_get_number_of_objectives(const coco_problem_t *problem);

/**
 * @brief Returns the number of constraints of the problem.
 */
size_t coco_problem_get_number_of_constraints(const coco_problem_t *problem);

/**
 * @brief Returns the number of evaluations done on the problem.
 */
size_t coco_problem_get_evaluations(const coco_problem_t *problem);

/**
 * @brief Returns 1 if the final target was hit, 0 otherwise.
 */
int coco_problem_final_target_hit(const coco_problem_t *problem);

/**
 * @brief Returns the best observed value for the first objective.
 */
double coco_problem_get_best_observed_fvalue1(const coco_problem_t *problem);

/**
 * @brief Returns the target value for the first objective.
 */
double depreciated_coco_problem_get_final_target_fvalue1(const coco_problem_t *problem);

/**
 * @brief Returns a vector of size 'dimension' with lower bounds of the region of interest in
 * the decision space.
 */
const double *coco_problem_get_smallest_values_of_interest(const coco_problem_t *problem);

/**
 * @brief Returns a vector of size 'dimension' with upper bounds of the region of interest in
 * the decision space.
 */
const double *coco_problem_get_largest_values_of_interest(const coco_problem_t *problem);

/**
 * @brief Returns the problem_index of the problem in its current suite.
 */
size_t coco_problem_get_suite_dep_index(const coco_problem_t *problem);

/**
 * @brief Returns an initial solution, i.e. a feasible variable setting, to the problem.
 */
void coco_problem_get_initial_solution(const coco_problem_t *problem, double *initial_solution);
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding random numbers
 */
/**@{*/

/**
 * @brief Creates and returns a new random number state using the given seed.
 */
coco_random_state_t *coco_random_new(uint32_t seed);

/**
 * @brief Frees all memory associated with the random state.
 */
void coco_random_free(coco_random_state_t *state);

/**
 * @brief Returns one uniform [0, 1) random value from the random number generator associated with the given
 * state.
 */
double coco_random_uniform(coco_random_state_t *state);

/**
 * @brief Generates an approximately normal random number.
 */
double coco_random_normal(coco_random_state_t *state);
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods managing memory
 */
/**@{*/
/**
 * @brief Safe memory allocation that either succeeds or triggers a coco_error.
 */
void *coco_allocate_memory(const size_t size);

/**
 * @brief Safe memory allocation for a vector of doubles that either succeeds or triggers a coco_error.
 */
double *coco_allocate_vector(const size_t size);

/**
 * @brief Frees the allocated memory.
 */
void coco_free_memory(void *data);
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding COCO messages
 */
/**@{*/
/**
 * @brief Signals a fatal error.
 */
void coco_error(const char *message, ...);

/**
 * @brief Warns about error conditions.
 */
void coco_warning(const char *message, ...);

/**
 * @brief Outputs some information.
 */
void coco_info(const char *message, ...);

/**
 * @brief Prints only the given message without any prefix and new line.
 *
 * A function similar to coco_info but producing no additional text than
 * the given message.
 *
 * The output is only produced if coco_log_level >= COCO_INFO.
 */
void coco_info_partial(const char *message, ...);

/**
 * @brief Outputs detailed information usually used for debugging.
 */
void coco_debug(const char *message, ...);

/**
 * @brief Sets the COCO log level to the given value and returns the previous value of the log level.
 */
const char *coco_set_log_level(const char *level);
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding COCO archives (used when pre-processing MO data)
 */
/**@{*/

/**
 * @brief Constructs a COCO archive.
 */
coco_archive_t *coco_archive(const char *suite_name,
                             const size_t function,
                             const size_t dimension,
                             const size_t instance);
/**
 * @brief Adds a solution with objectives (f1, f2) to the archive if none of the existing solutions in the
 * archive dominates it. In this case, returns 1, otherwise the archive is not updated and the method
 * returns 0.
 */
int coco_archive_add_solution(coco_archive_t *archive, const double f1, const double f2, const char *text);

/**
 * @brief Returns the number of (non-dominated) solutions in the archive (if needed, they are first computed).
 */
size_t coco_archive_get_number_of_solutions(coco_archive_t *archive);

/**
 * @brief Returns the hypervolume of the archive (if needed, it is first computed).
 */
double coco_archive_get_hypervolume(coco_archive_t *archive);

/**
 * @brief Returns the text of the next (non-dominated) solution in the archive.
 */
const char *coco_archive_get_next_solution_text(coco_archive_t *archive);

/**
 * @brief Frees the archive.
 */
void coco_archive_free(coco_archive_t *archive);
/**@}*/

/***********************************************************************************************************/

/**
 * @name Other useful methods
 */
/**@{*/
/**
 * @brief Removes the given directory and all its contents.
 */
int coco_remove_directory(const char *path);

/**
 * @brief Formatted string duplication.
 */
char *coco_strdupf(const char *str, ...);
/**@}*/

/***********************************************************************************************************/

#ifdef __cplusplus
}
#endif
#endif
#line 12 "code-experiments/src/coco_random.c"

#define COCO_NORMAL_POLAR /* Use polar transformation method */

#define COCO_SHORT_LAG 273
#define COCO_LONG_LAG 607

/**
 * @brief A structure containing the state of the COCO random generator.
 */
struct coco_random_state_s {
  double x[COCO_LONG_LAG];
  size_t index;
};

/**
 * @brief A lagged Fibonacci random number generator.
 *
 * This generator is nice because it is reasonably small and directly generates double values. The chosen
 * lags (607 and 273) lead to a generator with a period in excess of 2^607-1.
 */
static void coco_random_generate(coco_random_state_t *state) {
  size_t i;
  for (i = 0; i < COCO_SHORT_LAG; ++i) {
    double t = state->x[i] + state->x[i + (COCO_LONG_LAG - COCO_SHORT_LAG)];
    if (t >= 1.0)
      t -= 1.0;
    state->x[i] = t;
  }
  for (i = COCO_SHORT_LAG; i < COCO_LONG_LAG; ++i) {
    double t = state->x[i] + state->x[i - COCO_SHORT_LAG];
    if (t >= 1.0)
      t -= 1.0;
    state->x[i] = t;
  }
  state->index = 0;
}

coco_random_state_t *coco_random_new(uint32_t seed) {
  coco_random_state_t *state = (coco_random_state_t *) coco_allocate_memory(sizeof(*state));
  size_t i;
  /* Expand seed to fill initial state array. */
  for (i = 0; i < COCO_LONG_LAG; ++i) {
    /* Uses uint64_t to silence the compiler ("shift count negative or too big, undefined behavior" warning) */
    state->x[i] = ((double) seed) / (double) (((uint64_t) 1UL << 32) - 1);
    /* Advance seed based on simple RNG from TAOCP */
    seed = (uint32_t) 1812433253UL * (seed ^ (seed >> 30)) + ((uint32_t) i + 1);
  }
  state->index = 0;
  return state;
}

void coco_random_free(coco_random_state_t *state) {
  coco_free_memory(state);
}

double coco_random_uniform(coco_random_state_t *state) {
  /* If we have consumed all random numbers in our archive, it is time to run the actual generator for one
   * iteration to refill the state with 'LONG_LAG' new values. */
  if (state->index >= COCO_LONG_LAG)
    coco_random_generate(state);
  return state->x[state->index++];
}

/**
 * Instead of using the (expensive) polar method, we may cheat and abuse the central limit theorem. The sum
 * of 12 uniform random values has mean 6, variance 1 and is approximately normal. Subtract 6 and you get
 * an approximately N(0, 1) random number.
 */
double coco_random_normal(coco_random_state_t *state) {
  double normal;
#ifdef COCO_NORMAL_POLAR
  const double u1 = coco_random_uniform(state);
  const double u2 = coco_random_uniform(state);
  normal = sqrt(-2 * log(u1)) * cos(2 * coco_pi * u2);
#else
  int i;
  normal = 0.0;
  for (i = 0; i < 12; ++i) {
    normal += coco_random_uniform(state);
  }
  normal -= 6.0;
#endif
  return normal;
}

/* Be hygienic (for amalgamation) and undef lags. */
#undef COCO_SHORT_LAG
#undef COCO_LONG_LAG
#line 1 "code-experiments/src/coco_suite.c"
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

#line 15 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/coco_internal.h"
/**
 * @file coco_internal.h
 * @brief Definitions of internal COCO structures and typedefs.
 *
 * These are used throughout the COCO code base but should not be used by any external code.
 */

#ifndef __COCO_INTERNAL__
#define __COCO_INTERNAL__

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************************************************************/
/**
 * @brief The data free function type.
 *
 * This is a template for functions that free the contents of data (used to free the contents of data
 * fields in coco_problem, coco_suite and coco_observer).
 */
typedef void (*coco_data_free_function_t)(void *data);

/**
 * @brief The problem free function type.
 *
 * This is a template for functions that free the problem structure.
 */
typedef void (*coco_problem_free_function_t)(coco_problem_t *problem);

/**
 * @brief The initial solution function type.
 *
 * This is a template for functions that return an initial solution of the problem.
 */
typedef void (*coco_initial_solution_function_t)(const coco_problem_t *problem, double *y);

/**
 * @brief The evaluate function type.
 *
 * This is a template for functions that perform an evaluation of the problem (to evaluate the problem
 * function, the problems constraints etc.).
 */
typedef void (*coco_evaluate_function_t)(coco_problem_t *problem, const double *x, double *y);

/**
 * @brief The recommend solutions function type.
 *
 * This is a template for functions that log a recommended solution.
 */
typedef void (*coco_recommend_function_t)(coco_problem_t *problem, const double *x);

/**
 * @brief The allocate logger function type.
 *
 * This is a template for functions that allocate a logger (wrap a logger around the given problem and return
 * the wrapped problem).
 */
typedef coco_problem_t *(*coco_logger_allocate_function_t)(coco_observer_t *observer,
                                                           coco_problem_t *problem);

/**
 * @brief The free logger function type.
 *
 * This is a template for functions that free a logger.
 */
typedef void (*coco_logger_free_function_t)(void *logger);


/**
 * @brief The transformed COCO problem data type.
 *
 * This is a type of a generic structure for a transformed ("outer") coco_problem. It makes possible the
 * wrapping of problems as layers of an onion. Initialized in the coco_problem_transformed_allocate function,
 * it makes the current ("outer") transformed problem a "derived problem class", which inherits from the
 * "inner" problem, the "base class".
 *
 * From the perspective of the inner problem:
 * - data holds the meta-information to administer the inheritance
 * - data->data holds the additional fields of the derived class (the outer problem)
 * - data->inner_problem points to the inner problem (now we have a linked list)
 */
typedef struct {
  coco_problem_t *inner_problem;                  /**< @brief Pointer to the inner problem */
  void *data;                                     /**< @brief Pointer to data, which enables further
                                                  wrapping of the problem */
  coco_data_free_function_t data_free_function;   /**< @brief Function to free the contents of data */
} coco_problem_transformed_data_t;

/**
 * @brief The stacked COCO problem data type.
 *
 * This is a type of a structure used when stacking two problems (especially useful for constructing
 * multi-objective problems).
 */
typedef struct {
  coco_problem_t *problem1; /**< @brief Pointer to the first problem (objective) */
  coco_problem_t *problem2; /**< @brief Pointer to the second problem (objective) */
} coco_problem_stacked_data_t;

/**
 * @brief The option keys data type.
 *
 * This is a type of a structure used to contain a set of known option keys (used by suites and observers).
 */
typedef struct {
  size_t count;  /**< @brief Number of option keys */
  char **keys;   /**< @brief Pointer to option keys */
} coco_option_keys_t;


/***********************************************************************************************************/

/**
 * @brief The COCO problem structure.
 *
 * This is one of the main structures in COCO. It contains information about a problem to be optimized. The
 * problems can be wrapped around each other (similar to the onion layers) by means of the data field and
 * the coco_problem_transformed_data_t structure creating some kind of "object inheritance". Even the logger
 * is considered as just another coco_problem instance wrapped around the original problem.
 */
struct coco_problem_s {

  coco_initial_solution_function_t initial_solution;  /**< @brief  The function for creating an initial solution. */
  coco_evaluate_function_t evaluate_function;         /**< @brief  The function for evaluating the problem. */
  coco_evaluate_function_t evaluate_constraint;       /**< @brief  The function for evaluating the constraints. */
  coco_recommend_function_t recommend_solution;       /**< @brief  The function for recommending a solution. */
  coco_problem_free_function_t problem_free_function; /**< @brief  The function for freeing this problem. */

  size_t number_of_variables;          /**< @brief Number of variables expected by the function, i.e.
                                       problem dimension */
  size_t number_of_objectives;         /**< @brief Number of objectives. */
  size_t number_of_constraints;        /**< @brief Number of constraints. */

  double *smallest_values_of_interest; /**< @brief The lower bounds of the ROI in the decision space. */
  double *largest_values_of_interest;  /**< @brief The upper bounds of the ROI in the decision space. */

  double *best_value;                  /**< @brief Optimal (smallest) function value */
  double *nadir_value;                 /**< @brief The nadir point (defined when number_of_objectives > 1) */
  double *best_parameter;              /**< @brief Optimal decision vector (defined only when unique) */

  char *problem_name;                  /**< @brief Problem name. */
  char *problem_id;                    /**< @brief Problem ID (unique in the containing suite) */
  char *problem_type;                  /**< @brief Problem type */

  size_t evaluations;                  /**< @brief Number of evaluations performed on the problem. */

  /* Convenience fields for output generation */

  double final_target_delta[1];        /**< @brief Final target delta. */
  double best_observed_fvalue[1];      /**< @brief The best observed value so far. */
  size_t best_observed_evaluation[1];  /**< @brief The evaluation when the best value so far was achieved. */

  /* Fields depending on the containing benchmark suite */

  size_t suite_dep_index;              /**< @brief Suite-depending problem index (starting from 0) */
  size_t suite_dep_function;           /**< @brief Suite-depending function */
  size_t suite_dep_instance;           /**< @brief Suite-depending instance */

  void *data;                          /**< @brief Pointer to a data instance @see coco_problem_transformed_data_t */
};

/**
 * @brief The COCO observer structure.
 *
 * An observer observes the whole benchmark process. It is independent of suites and problems. Each time a
 * new problem of the suite is being observed, the observer initializes a new logger (wraps the observed
 * problem with the corresponding logger).
 */
struct coco_observer_s {

  int is_active;             /**< @brief Whether the observer is active (the logger will log some output). */
  char *observer_name;       /**< @brief Name of the observer for identification purposes. */
  char *result_folder;       /**< @brief Name of the result folder. */
  char *algorithm_name;      /**< @brief Name of the algorithm to be used in logger output. */
  char *algorithm_info;      /**< @brief Additional information on the algorithm to be used in logger output. */
  size_t number_target_triggers;
                             /**< @brief The number of targets between each 10**i and 10**(i+1). */
  double target_precision;   /**< @brief The minimal precision used for targets. */
  size_t number_evaluation_triggers;
                             /**< @brief The number of triggers between each 10**i and 10**(i+1) evaluation number. */
  char *base_evaluation_triggers;
                             /**< @brief The "base evaluations" used to evaluations that trigger logging. */
  int precision_x;           /**< @brief Output precision for decision variables. */
  int precision_f;           /**< @brief Output precision for function values. */
  void *data;                /**< @brief Void pointer that can be used to point to data specific to an observer. */

  coco_data_free_function_t data_free_function;             /**< @brief  The function for freeing this observer. */
  coco_logger_allocate_function_t logger_allocate_function; /**< @brief  The function for allocating the logger. */
  coco_logger_free_function_t logger_free_function;         /**< @brief  The function for freeing the logger. */
};

/**
 * @brief The COCO suite structure.
 *
 * A suite is a collection of problems constructed by a Cartesian product of the suite's optimization
 * functions, dimensions and instances. The functions and dimensions are fixed for a suite with some name,
 * while the instances are defined dynamically. The suite can be filtered - only the chosen functions,
 * dimensions and instances will be taken into account when iterating through the suite.
 */
struct coco_suite_s {

  char *suite_name;                /**< @brief Name of the suite. */

  size_t number_of_dimensions;     /**< @brief Number of dimensions contained in the suite. */
  size_t *dimensions;              /**< @brief The dimensions contained in the suite. */

  size_t number_of_functions;      /**< @brief Number of functions contained in the suite. */
  size_t *functions;               /**< @brief The functions contained in the suite. */

  size_t number_of_instances;      /**< @brief Number of instances contained in the suite. */
  char *default_instances;         /**< @brief The instances contained in the suite by default. */
  size_t *instances;               /**< @brief The instances contained in the suite. */

  coco_problem_t *current_problem; /**< @brief Pointer to the currently tackled problem. */
  long current_dimension_idx;      /**< @brief The dimension index of the currently tackled problem. */
  long current_function_idx;       /**< @brief The function index of the currently tackled problem. */
  long current_instance_idx;       /**< @brief The instance index of the currently tackled problem. */

  void *data;                      /**< @brief Void pointer that can be used to point to data specific to a suite. */

  coco_data_free_function_t data_free_function; /**< @brief The function for freeing this suite. */

};

#ifdef __cplusplus
}
#endif
#endif

#line 16 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/coco_utilities.c"
/**
 * @file coco_utilities.c
 * @brief Definitions of miscellaneous functions used throughout the COCO framework.
 */

#line 1 "code-experiments/src/coco_platform.h"
/**
 * @file coco_platform.h
 * @brief Automatic platform-dependent configuration of the COCO framework.
 *
 * Some platforms and standard conforming compilers require extra defines or includes to provide some
 * functionality.
 *
 * Because most feature defines need to be set before the first system header is included and we do not
 * know when a system header is included for the first time in the amalgamation, all internal files
 * that need these definitions should include this file before any system headers.
 */

#ifndef __COCO_PLATFORM__ 
#define __COCO_PLATFORM__

#include <stddef.h>

/* Definitions of COCO_PATH_MAX, coco_path_separator, HAVE_GFA and HAVE_STAT heavily used by functions in
 * coco_utilities.c */
#if defined(_WIN32) || defined(_WIN64) || defined(__MINGW64__) || defined(__CYGWIN__)
#include <windows.h>
static const char *coco_path_separator = "\\";
#define COCO_PATH_MAX MAX_PATH
#define HAVE_GFA 1
#elif defined(__gnu_linux__)
#include <sys/stat.h>
#include <sys/types.h>
#include <linux/limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#elif defined(__APPLE__)
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syslimits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#elif defined(__FreeBSD__)
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#elif (defined(__sun) || defined(sun)) && (defined(__SVR4) || defined(__svr4__))
/* Solaris */
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#else
#error Unknown platform
#endif
#if !defined(COCO_PATH_MAX)
#error COCO_PATH_MAX undefined
#endif

/* Definitions needed for creating and removing directories */
/* Separately handle the special case of Microsoft Visual Studio 2008 with x86_64-w64-mingw32-gcc */
#if _MSC_VER
#include <direct.h>
#elif defined(__MINGW32__) || defined(__MINGW64__)
#include <dirent.h>
#else
#include <dirent.h>

#ifdef __cplusplus
extern "C" {
#endif

/* To silence the compiler (implicit-function-declaration warning). */
/** @cond */
int rmdir(const char *pathname);
int unlink(const char *file_name);
int mkdir(const char *pathname, mode_t mode);
/** @endcond */
#endif

/* Definition of the S_IRWXU constant needed to set file permissions */
#if defined(HAVE_GFA)
#define S_IRWXU 0700
#endif

/* To silence the Visual Studio compiler (C4996 warnings in the python build). */
#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

#ifdef __cplusplus
}
#endif

#endif
#line 7 "code-experiments/src/coco_utilities.c"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <ctype.h>

#line 16 "code-experiments/src/coco_utilities.c"
#line 17 "code-experiments/src/coco_utilities.c"
#line 1 "code-experiments/src/coco_string.c"
/**
 * @file coco_string.c
 * @brief Definitions of functions that manipulate strings.
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#line 11 "code-experiments/src/coco_string.c"

static size_t *coco_allocate_vector_size_t(const size_t number_of_elements);

/**
 * @brief Creates a duplicate copy of string and returns a pointer to it.
 *
 * The caller is responsible for freeing the allocated memory using coco_free_memory().
 */
static char *coco_strdup(const char *string) {
  size_t len;
  char *duplicate;
  if (string == NULL)
    return NULL;
  len = strlen(string);
  duplicate = (char *) coco_allocate_memory(len + 1);
  memcpy(duplicate, string, len + 1);
  return duplicate;
}

/**
 * @brief The length of the buffer used in the coco_vstrdupf function.
 *
 * @note This should be handled differently!
 */
#define COCO_VSTRDUPF_BUFLEN 444

/**
 * @brief Formatted string duplication, with va_list arguments.
 */
static char *coco_vstrdupf(const char *str, va_list args) {
  static char buf[COCO_VSTRDUPF_BUFLEN];
  long written;
  /* apparently args can only be used once, therefore
   * len = vsnprintf(NULL, 0, str, args) to find out the
   * length does not work. Therefore we use a buffer
   * which limits the max length. Longer strings should
   * never appear anyway, so this is rather a non-issue. */

#if 0
  written = vsnprintf(buf, COCO_VSTRDUPF_BUFLEN - 2, str, args);
  if (written < 0)
  coco_error("coco_vstrdupf(): vsnprintf failed on '%s'", str);
#else /* less safe alternative, if vsnprintf is not available */
  assert(strlen(str) < COCO_VSTRDUPF_BUFLEN / 2 - 2);
  if (strlen(str) >= COCO_VSTRDUPF_BUFLEN / 2 - 2)
    coco_error("coco_vstrdupf(): string is too long");
  written = vsprintf(buf, str, args);
  if (written < 0)
    coco_error("coco_vstrdupf(): vsprintf failed on '%s'", str);
#endif
  if (written > COCO_VSTRDUPF_BUFLEN - 3)
    coco_error("coco_vstrdupf(): A suspiciously long string is tried to being duplicated '%s'", buf);
  return coco_strdup(buf);
}

#undef COCO_VSTRDUPF_BUFLEN

/**
 * Optional arguments are used like in sprintf.
 */
char *coco_strdupf(const char *str, ...) {
  va_list args;
  char *s;

  va_start(args, str);
  s = coco_vstrdupf(str, args);
  va_end(args);
  return s;
}

/**
 * @brief Returns a concatenate copy of string1 + string2.
 *
 * The caller is responsible for freeing the allocated memory using coco_free_memory().
 */
static char *coco_strconcat(const char *s1, const char *s2) {
  size_t len1 = strlen(s1);
  size_t len2 = strlen(s2);
  char *s = (char *) coco_allocate_memory(len1 + len2 + 1);

  memcpy(s, s1, len1);
  memcpy(&s[len1], s2, len2 + 1);
  return s;
}

/**
 * @brief Returns the first index where seq occurs in base and -1 if it doesn't.
 *
 * @note If there is an equivalent standard C function, this can/should be removed.
 */
static long coco_strfind(const char *base, const char *seq) {
  const size_t strlen_seq = strlen(seq);
  const size_t last_first_idx = strlen(base) - strlen(seq);
  size_t i, j;

  if (strlen(base) < strlen(seq))
    return -1;

  for (i = 0; i <= last_first_idx; ++i) {
    if (base[i] == seq[0]) {
      for (j = 0; j < strlen_seq; ++j) {
        if (base[i + j] != seq[j])
          break;
      }
      if (j == strlen_seq) {
        if (i > 1e9)
          coco_error("coco_strfind(): strange values observed i=%lu, j=%lu, strlen(base)=%lu", i, j,
              strlen(base));
        return (long) i;
      }
    }
  }
  return -1;
}

/**
 * @brief Splits a string based on the given delimiter.
 *
 * Returns a pointer to the resulting substrings with NULL as the last one.
 * The caller is responsible for freeing the allocated memory using coco_free_memory().
 */
static char **coco_string_split(const char *string, const char delimiter) {

  char **result;
  char *str_copy, *ptr, *token;
  char str_delimiter[2];
  size_t i;
  size_t count = 1;

  str_copy = coco_strdup(string);

  /* Counts the parts between delimiters */
  ptr = str_copy;
  while (*ptr != '\0') {
    if (*ptr == delimiter) {
      count++;
    }
    ptr++;
  }
  /* Makes room for an empty string that will be appended at the end */
  count++;

  result = (char **) coco_allocate_memory(count * sizeof(char *));

  /* Iterates through tokens
   * NOTE: strtok() ignores multiple delimiters, therefore the final number of detected substrings might be
   * lower than the count. This is OK. */
  i = 0;
  /* A char* delimiter needs to be used, otherwise strtok() can surprise */
  str_delimiter[0] = delimiter;
  str_delimiter[1] = '\0';
  token = strtok(str_copy, str_delimiter);
  while (token)
  {
      assert(i < count);
      *(result + i++) = coco_strdup(token);
      token = strtok(NULL, str_delimiter);
  }
  *(result + i) = NULL;

  coco_free_memory(str_copy);

  return result;
}

/**
 * @brief Creates and returns a string with removed characters between from and to.
 *
 * If you wish to remove characters from the beginning of the string, set from to "".
 * If you wish to remove characters until the end of the string, set to to "".
 *
 * The caller is responsible for freeing the allocated memory using coco_free_memory().
 */
static char *coco_remove_from_string(const char *string, const char *from, const char *to) {

  char *result, *start, *stop;

  result = coco_strdup(string);

  if (strcmp(from, "") == 0) {
    /* Remove from the start */
    start = result;
  } else
    start = strstr(result, from);

  if (strcmp(to, "") == 0) {
    /* Remove until the end */
    stop = result + strlen(result);
  } else
    stop = strstr(result, to);

  if ((start == NULL) || (stop == NULL) || (stop < start)) {
    coco_error("coco_remove_from_string(): failed to remove characters between %s and %s from string %s",
        from, to, string);
    return NULL; /* Never reached */
  }

  memmove(start, stop, strlen(stop) + 1);

  return result;
}


/**
 * @brief Returns the numbers defined by the ranges.
 *
 * Reads ranges from a string of positive ranges separated by commas. For example: "-3,5-6,8-". Returns the
 * numbers that are defined by the ranges if min and max are used as their extremes. If the ranges with open
 * beginning/end are not allowed, use 0 as min/max. The returned string has an appended 0 to mark its end.
 * A maximum of max_count values is returned. If there is a problem with one of the ranges, the parsing stops
 * and the current result is returned. The memory of the returned object needs to be freed by the caller.
 */
static size_t *coco_string_parse_ranges(const char *string,
                                        const size_t min,
                                        const size_t max,
                                        const char *name,
                                        const size_t max_count) {

  char *ptr, *dash = NULL;
  char **ranges, **numbers;
  size_t i, j, count;
  size_t num[2];

  size_t *result;
  size_t i_result = 0;

  char *str = coco_strdup(string);

  /* Check for empty string */
  if ((str == NULL) || (strlen(str) == 0)) {
    coco_warning("coco_string_parse_ranges(): cannot parse empty ranges");
    return NULL;
  }

  ptr = str;
  /* Check for disallowed characters */
  while (*ptr != '\0') {
    if ((*ptr != '-') && (*ptr != ',') && !isdigit((unsigned char )*ptr)) {
      coco_warning("coco_string_parse_ranges(): problem parsing '%s' - cannot parse ranges with '%c'", str,
          *ptr);
      return NULL;
    } else
      ptr++;
  }

  /* Check for incorrect boundaries */
  if ((max > 0) && (min > max)) {
    coco_warning("coco_string_parse_ranges(): incorrect boundaries");
    return NULL;
  }

  result = coco_allocate_vector_size_t(max_count + 1);

  /* Split string to ranges w.r.t commas */
  ranges = coco_string_split(str, ',');
  coco_free_memory(str);

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
        coco_warning("coco_string_parse_ranges(): problem parsing '%s' - too many '-'s", string);
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
            coco_warning("coco_string_parse_ranges(): '%s' ranges cannot have an open ends; some ranges ignored", name);
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
              coco_warning("coco_string_parse_ranges(): '%s' ranges cannot have an open beginning; some ranges ignored", name);
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
              coco_warning("coco_string_parse_ranges(): '%s' ranges cannot have an open end; some ranges ignored", name);
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
        coco_warning("coco_string_parse_ranges(): '%s' ranges adjusted to be >= %lu", name, min);
      }
      if ((max > 0) && (num[1] > max)) {
        num[1] = max;
        coco_warning("coco_string_parse_ranges(): '%s' ranges adjusted to be <= %lu", name, max);
      }
      if (num[0] > num[1]) {
        coco_warning("coco_string_parse_ranges(): '%s' ranges not within boundaries; some ranges ignored", name);
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
        if (i_result > max_count - 1)
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
#line 18 "code-experiments/src/coco_utilities.c"

/***********************************************************************************************************/

/**
 * @brief Initializes the logging level to COCO_INFO.
 */
static coco_log_level_type_e coco_log_level = COCO_INFO;

/**
 * @param log_level Denotes the level of information given to the user through the standard output and
 * error streams. Can take on the values:
 * - "error" (only error messages are output),
 * - "warning" (only error and warning messages are output),
 * - "info" (only error, warning and info messages are output) and
 * - "debug" (all messages are output).
 * - "" does not set a new value
 * The default value is info.
 *
 * @return The previous coco_log_level value as an immutable string.
 */
const char *coco_set_log_level(const char *log_level) {

  coco_log_level_type_e previous_log_level = coco_log_level;

  if (strcmp(log_level, "error") == 0)
    coco_log_level = COCO_ERROR;
  else if (strcmp(log_level, "warning") == 0)
    coco_log_level = COCO_WARNING;
  else if (strcmp(log_level, "info") == 0)
    coco_log_level = COCO_INFO;
  else if (strcmp(log_level, "debug") == 0)
    coco_log_level = COCO_DEBUG;
  else if (strcmp(log_level, "") == 0) {
    /* Do nothing */
  } else {
    coco_warning("coco_set_log_level(): unknown level %s", log_level);
  }

  if (previous_log_level == COCO_ERROR)
    return "error";
  else if (previous_log_level == COCO_WARNING)
    return "warning";
  else if (previous_log_level == COCO_INFO)
    return "info";
  else if (previous_log_level == COCO_DEBUG)
    return "debug";
  else {
    coco_error("coco_set_log_level(): unknown previous log level");
    return "";
  }
}

/***********************************************************************************************************/

/**
 * @name Methods regarding file, directory and path manipulations
 */
/**@{*/
/**
 * @brief Creates a platform-dependent path from the given strings.
 *
 * @note The last argument must be NULL.
 * @note The first parameter must be able to accommodate path_max_length characters and the length
 * of the joined path must not exceed path_max_length characters.
 * @note Should work cross-platform.
 *
 * Usage examples:
 * - coco_join_path(base_path, 100, folder1, folder2, folder3, NULL) creates base_path/folder1/folder2/folder3
 * - coco_join_path(base_path, 100, folder1, file_name, NULL) creates base_path/folder1/file_name
 * @param path The base path; it's also where the joined path is stored to.
 * @param path_max_length The maximum length of the path.
 * @param ... Additional strings, must end with NULL
 */
static void coco_join_path(char *path, const size_t path_max_length, ...) {
  const size_t path_separator_length = strlen(coco_path_separator);
  va_list args;
  char *path_component;
  size_t path_length = strlen(path);

  va_start(args, path_max_length);
  while (NULL != (path_component = va_arg(args, char *))) {
    size_t component_length = strlen(path_component);
    if (path_length + path_separator_length + component_length >= path_max_length) {
      coco_error("coco_join_path() failed because the ${path} is too short.");
      return; /* never reached */
    }
    /* Both should be safe because of the above check. */
    if (strlen(path) > 0)
      strncat(path, coco_path_separator, path_max_length - strlen(path) - 1);
    strncat(path, path_component, path_max_length - strlen(path) - 1);
  }
  va_end(args);
}

/**
 * @brief Checks if the given directory exists.
 *
 * @note Should work cross-platform.
 *
 * @param path The given path.
 *
 * @return 1 if the path exists and corresponds to a directory and 0 otherwise.
 */
static int coco_directory_exists(const char *path) {
  int res;
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributes(path);
  res = (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#elif defined(HAVE_STAT)
  struct stat buf;
  res = (!stat(path, &buf) && S_ISDIR(buf.st_mode));
#else
#error Ooops
#endif
  return res;
}

/**
 * @brief Checks if the given file exists.
 *
 * @note Should work cross-platform.
 *
 * @param path The given path.
 *
 * @return 1 if the path exists and corresponds to a file and 0 otherwise.
 */
static int coco_file_exists(const char *path) {
  int res;
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributes(path);
  res = (dwAttrib != INVALID_FILE_ATTRIBUTES) && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY);
#elif defined(HAVE_STAT)
  struct stat buf;
  res = (!stat(path, &buf) && !S_ISDIR(buf.st_mode));
#else
#error Ooops
#endif
  return res;
}

/**
 * @brief Calls the right mkdir() method (depending on the platform).
 *
 * @param path The directory path.
 *
 * @return 0 on successful completion, and -1 on error.
 */
static int coco_mkdir(const char *path) {
#if _MSC_VER
  return _mkdir(path);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  return mkdir(path);
#else
  return mkdir(path, S_IRWXU);
#endif
}

/**
 * @brief Creates a directory with full privileges for the user.
 *
 * @note Should work cross-platform.
 *
 * @param path The directory path.
 */
static void coco_create_directory(const char *path) {
  char *tmp = NULL;
  char *p;
  size_t len = strlen(path);
  char path_sep = coco_path_separator[0];

  /* Nothing to do if the path exists. */
  if (coco_directory_exists(path))
    return;

  tmp = coco_strdup(path);
  /* Remove possible trailing slash */
  if (tmp[len - 1] == path_sep)
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++) {
    if (*p == path_sep) {
      *p = 0;
      if (!coco_directory_exists(tmp)) {
        if (0 != coco_mkdir(tmp))
          coco_error("coco_create_path(): failed creating %s", tmp);
      }
      *p = path_sep;
    }
  }
  if (0 != coco_mkdir(tmp))
    coco_error("coco_create_path(): failed creating %s", tmp);
  coco_free_memory(tmp);
  return;
}

/* Commented to silence the compiler (unused function warning) */
#if 0
/**
 * @brief Creates a unique file name from the given file_name.
 *
 * If the file_name does not yet exit, it is left as is, otherwise it is changed(!) by prepending a number
 * to it. If filename.ext already exists, 01-filename.ext will be tried. If this one exists as well,
 * 02-filename.ext will be tried, and so on. If 99-filename.ext exists as well, the function throws
 * an error.
 */
static void coco_create_unique_filename(char **file_name) {

  int counter = 1;
  char *new_file_name;

  /* Do not change the file_name if it does not yet exist */
  if (!coco_file_exists(*file_name)) {
    return;
  }

  while (counter < 99) {

    new_file_name = coco_strdupf("%02d-%s", counter, *file_name);

    if (!coco_file_exists(new_file_name)) {
      coco_free_memory(*file_name);
      *file_name = new_file_name;
      return;
    } else {
      counter++;
      coco_free_memory(new_file_name);
    }

  }

  coco_free_memory(new_file_name);
  coco_error("coco_create_unique_filename(): could not create a unique file name");
  return; /* Never reached */
}
#endif

/**
 * @brief Creates a unique directory from the given path.
 *
 * If the given path does not yet exit, it is left as is, otherwise it is changed(!) by appending a number
 * to it. If path already exists, path-01 will be tried. If this one exists as well, path-02 will be tried,
 * and so on. If path-99 exists as well, the function throws an error.
 */
static void coco_create_unique_directory(char **path) {

  int counter = 1;
  char *new_path;

  /* Create the path if it does not yet exist */
  if (!coco_directory_exists(*path)) {
    coco_create_directory(*path);
    return;
  }

  while (counter < 999) {

    new_path = coco_strdupf("%s-%03d", *path, counter);

    if (!coco_directory_exists(new_path)) {
      coco_free_memory(*path);
      *path = new_path;
      coco_create_directory(*path);
      return;
    } else {
      counter++;
      coco_free_memory(new_path);
    }

  }

  coco_error("coco_create_unique_path(): could not create a unique path with name %s", *path);
  return; /* Never reached */
}

/**
 * The method should work across different platforms/compilers.
 *
 * @path The path to the directory
 *
 * @return 0 on successful completion, and -1 on error.
 */
int coco_remove_directory(const char *path) {
#if _MSC_VER
  WIN32_FIND_DATA find_data_file;
  HANDLE find_handle = NULL;
  char *buf;
  int r = -1;
  int r2 = -1;

  buf = coco_strdupf("%s\\*.*", path);
  /* Nothing to do if the folder does not exist */
  if ((find_handle = FindFirstFile(buf, &find_data_file)) == INVALID_HANDLE_VALUE) {
    coco_free_memory(buf);
    return 0;
  }
  coco_free_memory(buf);

  do {
    r = 0;

    /* Skip the names "." and ".." as we don't want to recurse on them */
    if (strcmp(find_data_file.cFileName, ".") != 0 && strcmp(find_data_file.cFileName, "..") != 0) {
      /* Build the new path using the argument path the file/folder name we just found */
      buf = coco_strdupf("%s\\%s", path, find_data_file.cFileName);

      if (find_data_file.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        /* Buf is a directory, recurse on it */
        r2 = coco_remove_directory(buf);
      } else {
        /* Buf is a file, delete it */
        /* Careful, DeleteFile returns 0 if it fails and nonzero otherwise! */
        r2 = -(DeleteFile(buf) == 0);
      }

      coco_free_memory(buf);
    }

    r = r2;

  }while (FindNextFile(find_handle, &find_data_file)); /* Find the next file */

  FindClose(find_handle);

  if (!r) {
    /* Path is an empty directory, delete it */
    /* Careful, RemoveDirectory returns 0 if it fails and nonzero otherwise! */
    r = -(RemoveDirectory(path) == 0);
  }

  return r;
#else
  DIR *d = opendir(path);
  int r = -1;
  int r2 = -1;
  char *buf;

  /* Nothing to do if the folder does not exist */
  if (!coco_directory_exists(path))
    return 0;

  if (d) {
    struct dirent *p;

    r = 0;

    while (!r && (p = readdir(d))) {

      /* Skip the names "." and ".." as we don't want to recurse on them */
      if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
        continue;
      }

      buf = coco_strdupf("%s/%s", path, p->d_name);
      if (buf) {
        if (coco_directory_exists(buf)) {
          /* Buf is a directory, recurse on it */
          r2 = coco_remove_directory(buf);
        } else {
          /* Buf is a file, delete it */
          r2 = unlink(buf);
        }
      }
      coco_free_memory(buf);

      r = r2;
    }

    closedir(d);
  }

  if (!r) {
    /* Path is an empty directory, delete it */
    r = rmdir(path);
  }

  return r;
#endif
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding memory allocations
 */
/**@{*/
double *coco_allocate_vector(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(double);
  return (double *) coco_allocate_memory(block_size);
}

/**
 * @brief Safe memory allocation for a vector with size_t elements that either succeeds or triggers a
 * coco_error.
 */
static size_t *coco_allocate_vector_size_t(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(size_t);
  return (size_t *) coco_allocate_memory(block_size);
}

static char *coco_allocate_string(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(char);
  return (char *) coco_allocate_memory(block_size);
}

static double *coco_duplicate_vector(const double *src, const size_t number_of_elements) {
  size_t i;
  double *dst;

  assert(src != NULL);
  assert(number_of_elements > 0);

  dst = coco_allocate_vector(number_of_elements);
  for (i = 0; i < number_of_elements; ++i) {
    dst[i] = src[i];
  }
  return dst;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding string options
 */
/**@{*/

/**
 * @brief Allocates an option keys structure holding the given number of option keys.
 */
static coco_option_keys_t *coco_option_keys_allocate(const size_t count, const char **keys) {

  size_t i;
  coco_option_keys_t *option_keys;

  if ((count == 0) || (keys == NULL))
    return NULL;

  option_keys = (coco_option_keys_t *) coco_allocate_memory(sizeof(*option_keys));

  option_keys->keys = (char **) coco_allocate_memory(count * sizeof(char *));
  for (i = 0; i < count; i++) {
    assert(keys[i]);
    option_keys->keys[i] = coco_strdup(keys[i]);
  }
  option_keys->count = count;

  return option_keys;
}

/**
 * @brief Frees the given option keys structure.
 */
static void coco_option_keys_free(coco_option_keys_t *option_keys) {

  size_t i;

  if (option_keys) {
    for (i = 0; i < option_keys->count; i++) {
      coco_free_memory(option_keys->keys[i]);
    }
    coco_free_memory(option_keys->keys);
    coco_free_memory(option_keys);
  }
}

/**
 * @brief Returns redundant option keys (the ones present in given_option_keys but not in known_option_keys).
 */
static coco_option_keys_t *coco_option_keys_get_redundant(const coco_option_keys_t *known_option_keys,
                                                          const coco_option_keys_t *given_option_keys) {

  size_t i, j, count = 0;
  int found;
  char **redundant_keys;
  coco_option_keys_t *redundant_option_keys;

  assert(known_option_keys != NULL);
  assert(given_option_keys != NULL);

  /* Find the redundant keys */
  redundant_keys = (char **) coco_allocate_memory(given_option_keys->count * sizeof(char *));
  for (i = 0; i < given_option_keys->count; i++) {
    found = 0;
    for (j = 0; j < known_option_keys->count; j++) {
      if (strcmp(given_option_keys->keys[i], known_option_keys->keys[j]) == 0) {
        found = 1;
        break;
      }
    }
    if (!found) {
      redundant_keys[count++] = coco_strdup(given_option_keys->keys[i]);
    }
  }
  redundant_option_keys = coco_option_keys_allocate(count, (const char**) redundant_keys);

  /* Free memory */
  for (i = 0; i < count; i++) {
    coco_free_memory(redundant_keys[i]);
  }
  coco_free_memory(redundant_keys);

  return redundant_option_keys;
}

/**
 * @brief Adds additional option keys to the given basic option keys (changes the basic keys).
 */
static void coco_option_keys_add(coco_option_keys_t **basic_option_keys,
                                 const coco_option_keys_t *additional_option_keys) {

  size_t i, j;
  size_t new_count;
  char **new_keys;
  coco_option_keys_t *new_option_keys;

  assert(*basic_option_keys != NULL);
  if (additional_option_keys == NULL)
    return;

  /* Construct the union of both keys */
  new_count = (*basic_option_keys)->count + additional_option_keys->count;
  new_keys = (char **) coco_allocate_memory(new_count * sizeof(char *));
  for (i = 0; i < (*basic_option_keys)->count; i++) {
    new_keys[i] = coco_strdup((*basic_option_keys)->keys[i]);
  }
  for (j = 0; j < additional_option_keys->count; j++) {
    new_keys[(*basic_option_keys)->count + j] = coco_strdup(additional_option_keys->keys[j]);
  }
  new_option_keys = coco_option_keys_allocate(new_count, (const char**) new_keys);

  /* Free the old basic keys */
  coco_option_keys_free(*basic_option_keys);
  *basic_option_keys = new_option_keys;
  for (i = 0; i < new_count; i++) {
    coco_free_memory(new_keys[i]);
  }
  coco_free_memory(new_keys);
}

/**
 * @brief Creates an instance of option keys from the given string of options containing keys and values
 * separated by colons.
 *
 * @note Relies heavily on the "key: value" format and might fail if the number of colons doesn't match the
 * number of keys.
 */
static coco_option_keys_t *coco_option_keys(const char *option_string) {

  size_t i, j;
  char **keys;
  coco_option_keys_t *option_keys = NULL;
  char *string_to_parse, *key;

  /* Check for empty string and string without colons */
  if ((option_string == NULL) || (strlen(option_string) == 0)|| (strchr(option_string, ':') == NULL)) {
    return NULL;
  }

  /* Split the options w.r.t ':' */
  keys = coco_string_split(option_string, ':');

  if (keys) {
    /* Keys now contain something like this: "values_of_previous_key this_key" except for the first, which
     * contains only the key and the last, which contains only the previous values */
    for (i = 0; *(keys + i); i++) {
      string_to_parse = coco_strdup(*(keys + i));

      /* Remove any trailing spaces */
      for (j = strlen(string_to_parse) - 1; (j > 0) && isspace((unsigned char) string_to_parse[j]); j--) {
        string_to_parse[j] = '\0';
      }

      /* Disregard everything before the last space */
      key = strrchr(string_to_parse, ' ');
      if (key == NULL) {
        /* No spaces left, everything is the key */
        key = string_to_parse;
      } else {
        /* Move to the start of the key (one char after the space) */
        key++;
      }

      /* Put the key in keys */
      coco_free_memory(*(keys + i));
      *(keys + i) = coco_strdup(key);
      coco_free_memory(string_to_parse);

      /* Stop if this is the last substring that contains a key */
      if (*(keys + i + 1) == NULL) {
        coco_free_memory(*(keys + i + 1));
        break;
      }
    }

    option_keys = coco_option_keys_allocate(i, (const char**) keys);

    /* Free the keys */
    for (i = 0; *(keys + i); i++) {
      coco_free_memory(*(keys + i));
    }
    coco_free_memory(keys);
  }

  return option_keys;
}

/**
 * @brief Creates and returns a string containing the info_string and all keys from option_keys.
 *
 * Can be used to output information about the given option_keys.
 */
static char *coco_option_keys_get_output_string(const coco_option_keys_t *option_keys,
                                                const char *info_string) {
  size_t i;
  char *string, *new_string;

  if ((option_keys != NULL) && (option_keys->count > 0)) {

    string = coco_strdup(info_string);
    for (i = 0; i < option_keys->count; i++) {
      new_string = coco_strdupf("%s %s\n", string, option_keys->keys[i]);
      coco_free_memory(string);
      string = new_string;
    }
  }

  return string;
}

/**
 * @brief Parses options in the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - value needs to be a single string (no spaces allowed)
 *
 * @return The number of successful assignments.
 */
static int coco_options_read(const char *options, const char *name, const char *format, void *pointer) {

  long i1, i2;

  if ((!options) || (strlen(options) == 0))
    return 0;

  i1 = coco_strfind(options, name);
  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;

  /* Remove trailing whitespaces */
  while (isspace((unsigned char) options[i2]))
    i2++;

  if (i2 <= i1){
    return 0;
  }

  return sscanf(&options[i2], format, pointer);
}

/**
 * @brief Reads an integer from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be an integer
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_int(const char *options, const char *name, int *pointer) {
  return coco_options_read(options, name, " %i", pointer);
}

/**
 * @brief Reads a size_t from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be a size_t
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_size_t(const char *options, const char *name, size_t *pointer) {
  return coco_options_read(options, name, "%lu", pointer);
}

/**
 * @brief Reads a double value from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be a double
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_double(const char *options, const char *name, double *pointer) {
  return coco_options_read(options, name, "%lf", pointer);
}

/**
 * @brief Reads a string from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be a string - either a single word or multiple words
 * in double quotes
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_string(const char *options, const char *name, char *pointer) {

  long i1, i2;

  if ((!options) || (strlen(options) == 0))
    return 0;

  i1 = coco_strfind(options, name);
  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;

  /* Remove trailing white spaces */
  while (isspace((unsigned char) options[i2]))
    i2++;

  if (i2 <= i1){
    return 0;
  }

  if (options[i2] == '\"') {
    /* The value starts with a quote: read everything between two quotes into a string */
    return sscanf(&options[i2], "\"%[^\"]\"", pointer);
  } else
    return sscanf(&options[i2], "%s", pointer);
}

/**
 * @brief Reads (possibly delimited) values from options using the form "name1: value1,value2,value3 name2: value4",
 * i.e. reads all characters from the corresponding name up to the next whitespace or end of string.
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_values(const char *options, const char *name, char *pointer) {

  long i1, i2;
  int i;

  if ((!options) || (strlen(options) == 0))
    return 0;

  i1 = coco_strfind(options, name);
  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;

  /* Remove trailing white spaces */
  while (isspace((unsigned char) options[i2]))
    i2++;

  if (i2 <= i1) {
    return 0;
  }

  i = 0;
  while (!isspace((unsigned char) options[i2 + i]) && (options[i2 + i] != '\0')) {
    pointer[i] = options[i2 + i];
    i++;
  }
  pointer[i] = '\0';
  return i;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods implementing functions on double values not contained in C89 standard
 */
/**@{*/

/**
 * @brief Rounds the given double to the nearest integer.
 */
static double coco_double_round(const double number) {
  return floor(number + 0.5);
}

/**
 * @brief Returns the maximum of a and b.
 */
static double coco_double_max(const double a, const double b) {
  if (a >= b) {
    return a;
  } else {
    return b;
  }
}

/**
 * @brief Returns the minimum of a and b.
 */
static double coco_double_min(const double a, const double b) {
  if (a <= b) {
    return a;
  } else {
    return b;
  }
}

/**
 * @brief Performs a "safer" double to size_t conversion.
 */
static size_t coco_double_to_size_t(const double number) {
  return (size_t) coco_double_round(number);
}

/* Commented to silence the compiler (unused function warning) */
#if 0
/**
 * @brief  Returns 1 if |a - b| < accuracy and 0 otherwise.
 */
static int coco_double_almost_equal(const double a, const double b, const double accuracy) {
  return ((fabs(a - b) < accuracy) == 0);
}
#endif
/**@}*/

/***********************************************************************************************************/

/**
 * @name Miscellaneous methods
 */
/**@{*/

/**
 * @brief Returns the current time as a string.
 *
 * The caller is responsible for freeing the allocated memory using coco_free_memory().
 */
static char *coco_current_time_get_string(void) {
  time_t timer;
  char *time_string = coco_allocate_string(30);
  struct tm* tm_info;
  time(&timer);
  tm_info = localtime(&timer);
  assert(tm_info != NULL);
  strftime(time_string, 30, "%d.%m.%y %H:%M:%S", tm_info);
  return time_string;
}

/**
 * @brief Returns the number of positive numbers pointed to by numbers (the count stops when the first
 * 0 is encountered of max_count numbers have been read).
 *
 * If there are more than max_count numbers, a coco_error is raised. The name argument is used
 * only to provide more informative output in case of any problems.
 */
static size_t coco_count_numbers(const size_t *numbers, const size_t max_count, const char *name) {

  size_t count = 0;
  while ((count < max_count) && (numbers[count] != 0)) {
    count++;
  }
  if (count == max_count) {
    coco_error("coco_count_numbers(): over %lu numbers in %s", max_count, name);
    return 0; /* Never reached*/
  }

  return count;
}
/**@}*/

/***********************************************************************************************************/
#line 17 "code-experiments/src/coco_suite.c"

#line 1 "code-experiments/src/suite_bbob.c"
/**
 * @file suite_bbob.c
 * @brief Implementation of the bbob suite containing 24 noiseless single-objective functions in 6
 * dimensions.
 */

#line 8 "code-experiments/src/suite_bbob.c"

#line 1 "code-experiments/src/f_attractive_sector.c"
/**
 * @file f_attractive_sector.c
 * @brief Implementation of the attractive sector function and problem.
 */

#include <assert.h>
#include <math.h>

#line 10 "code-experiments/src/f_attractive_sector.c"
#line 1 "code-experiments/src/coco_problem.c"
/**
 * @file coco_problem.c
 * @brief Definitions of functions regarding COCO problems.
 */

#include <float.h>
#line 8 "code-experiments/src/coco_problem.c"
#line 9 "code-experiments/src/coco_problem.c"

#line 11 "code-experiments/src/coco_problem.c"

/***********************************************************************************************************/

/**
 * @name Methods regarding the basic COCO problem
 */
/**@{*/
/**
 * Evaluates the problem function, increases the number of evaluations and updates the best observed value
 * and the best observed evaluation number.
 *
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The objective vector that is the result of the evaluation (in single-objective problems only the
 * first vector item is being set).
 */
void coco_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of problem->evaluate(problem, x, y) */
  assert(problem != NULL);
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, x, y);
  problem->evaluations++; /* each derived class has its own counter, only the most outer will be visible */
#if 1
  /* A little bit of bookkeeping */
  if (y[0] < problem->best_observed_fvalue[0]) {
    problem->best_observed_fvalue[0] = y[0];
    problem->best_observed_evaluation[0] = problem->evaluations;
  }
#endif
}

/**
 * @note None of the problems implement this function yet!
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The vector of constraints that is the result of the evaluation.
 */
void coco_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of problem->evaluate(problem, x, y) */
  assert(problem != NULL);
  if (problem->evaluate_constraint == NULL) {
    coco_error("coco_evaluate_constraint(): No constraint function implemented for problem %s",
        problem->problem_id);
  }
  problem->evaluate_constraint(problem, x, y);
}

/**
 * Evaluates and logs the given solution (as the coco_evaluate_function), but does not return the evaluated
 * value.
 *
 * @note None of the observers implements this function yet!
 * @note x must point to a correctly sized allocated memory region.

 * @param problem The given COCO problem.
 * @param x The decision vector.
 */
void coco_recommend_solution(coco_problem_t *problem, const double *x) {
  assert(problem != NULL);
  if (problem->recommend_solution == NULL) {
    coco_error("coco_recommend_solutions(): No recommend solution function implemented for problem %s",
        problem->problem_id);
  }
  problem->recommend_solution(problem, x);
}

/***********************************************************************************************************/

/**
 * @brief Allocates a new coco_problem_t for the given number of variables, number of objectives and
 * number of constraints.
 */
static coco_problem_t *coco_problem_allocate(const size_t number_of_variables,
                                             const size_t number_of_objectives,
                                             const size_t number_of_constraints) {
  coco_problem_t *problem;
  problem = (coco_problem_t *) coco_allocate_memory(sizeof(*problem));
  /* Initialize fields to sane/safe defaults */
  problem->initial_solution = NULL;
  problem->evaluate_function = NULL;
  problem->evaluate_constraint = NULL;
  problem->recommend_solution = NULL;
  problem->problem_free_function = NULL;
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = number_of_objectives;
  problem->number_of_constraints = number_of_constraints;
  problem->smallest_values_of_interest = coco_allocate_vector(number_of_variables);
  problem->largest_values_of_interest = coco_allocate_vector(number_of_variables);
  problem->best_parameter = coco_allocate_vector(number_of_variables);
  problem->best_value = coco_allocate_vector(number_of_objectives);
  if (number_of_objectives > 1)
    problem->nadir_value = coco_allocate_vector(number_of_objectives);
  else
    problem->nadir_value = NULL;
  problem->problem_name = NULL;
  problem->problem_id = NULL;
  problem->problem_type = NULL;
  problem->evaluations = 0;
  problem->final_target_delta[0] = 1e-8; /* in case to be modified by the benchmark */
  problem->best_observed_fvalue[0] = DBL_MAX;
  problem->best_observed_evaluation[0] = 0;
  problem->suite_dep_index = 0;
  problem->suite_dep_function = 0;
  problem->suite_dep_instance = 0;
  problem->data = NULL;
  return problem;
}

/**
 * @brief Creates a duplicate of the 'other' problem for all fields except for data, which points to NULL.
 */
static coco_problem_t *coco_problem_duplicate(const coco_problem_t *other) {
  size_t i;
  coco_problem_t *problem;
  problem = coco_problem_allocate(other->number_of_variables, other->number_of_objectives,
      other->number_of_constraints);

  problem->initial_solution = other->initial_solution;
  problem->evaluate_function = other->evaluate_function;
  problem->evaluate_constraint = other->evaluate_constraint;
  problem->recommend_solution = other->recommend_solution;
  problem->problem_free_function = other->problem_free_function;

  for (i = 0; i < problem->number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = other->smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] = other->largest_values_of_interest[i];
    if (other->best_parameter)
      problem->best_parameter[i] = other->best_parameter[i];
  }

  if (other->best_value)
    for (i = 0; i < problem->number_of_objectives; ++i) {
      problem->best_value[i] = other->best_value[i];
    }

  if (other->nadir_value)
    for (i = 0; i < problem->number_of_objectives; ++i) {
      problem->nadir_value[i] = other->nadir_value[i];
    }

  problem->problem_name = coco_strdup(other->problem_name);
  problem->problem_id = coco_strdup(other->problem_id);
  problem->problem_type = coco_strdup(other->problem_type);

  problem->evaluations = other->evaluations;
  problem->final_target_delta[0] = other->final_target_delta[0];
  problem->best_observed_fvalue[0] = other->best_observed_fvalue[0];
  problem->best_observed_evaluation[0] = other->best_observed_evaluation[0];

  problem->suite_dep_index = other->suite_dep_index;
  problem->suite_dep_function = other->suite_dep_function;
  problem->suite_dep_instance = other->suite_dep_instance;

  problem->data = NULL;

  return problem;
}

/**
 * @brief Allocates a problem using scalar values for smallest_value_of_interest, largest_value_of_interest
 * and best_parameter.
 */
static coco_problem_t *coco_problem_allocate_from_scalars(const char *problem_name,
                                                          coco_evaluate_function_t evaluate_function,
                                                          coco_problem_free_function_t problem_free_function,
                                                          const size_t number_of_variables,
                                                          const double smallest_value_of_interest,
                                                          const double largest_value_of_interest,
                                                          const double best_parameter) {
  size_t i;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 1, 0);

  problem->problem_name = coco_strdup(problem_name);
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = evaluate_function;
  problem->problem_free_function = problem_free_function;

  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = smallest_value_of_interest;
    problem->largest_values_of_interest[i] = largest_value_of_interest;
    problem->best_parameter[i] = best_parameter;
  }
  return problem;
}

void coco_problem_free(coco_problem_t *problem) {
  assert(problem != NULL);
  if (problem->problem_free_function != NULL) {
    problem->problem_free_function(problem);
  } else {
    /* Best guess at freeing all relevant structures */
    if (problem->smallest_values_of_interest != NULL)
      coco_free_memory(problem->smallest_values_of_interest);
    if (problem->largest_values_of_interest != NULL)
      coco_free_memory(problem->largest_values_of_interest);
    if (problem->best_parameter != NULL)
      coco_free_memory(problem->best_parameter);
    if (problem->best_value != NULL)
      coco_free_memory(problem->best_value);
    if (problem->nadir_value != NULL)
      coco_free_memory(problem->nadir_value);
    if (problem->problem_name != NULL)
      coco_free_memory(problem->problem_name);
    if (problem->problem_id != NULL)
      coco_free_memory(problem->problem_id);
    if (problem->problem_type != NULL)
      coco_free_memory(problem->problem_type);
    if (problem->data != NULL)
      coco_free_memory(problem->data);
    problem->smallest_values_of_interest = NULL;
    problem->largest_values_of_interest = NULL;
    problem->best_parameter = NULL;
    problem->best_value = NULL;
    problem->nadir_value = NULL;
    problem->data = NULL;
    coco_free_memory(problem);
  }
}

/***********************************************************************************************************/

/**
 * @brief Checks whether the given string is in the right format to be a problem_id.
 *
 * No non-alphanumeric characters besides '-', '_' and '.' are allowed.
 */
static int coco_problem_id_is_fine(const char *id, ...) {
  va_list args;
  const int reject = 0;
  const int accept = 1;
  const char *cp;
  char *s;
  int result = accept;

  va_start(args, id);
  s = coco_vstrdupf(id, args);
  va_end(args);
  for (cp = s; *cp != '\0'; ++cp) {
    if (('A' <= *cp) && (*cp <= 'Z'))
      continue;
    if (('a' <= *cp) && (*cp <= 'z'))
      continue;
    if ((*cp == '_') || (*cp == '-'))
      continue;
    if (('0' <= *cp) && (*cp <= '9'))
      continue;
    result = reject;
  }
  coco_free_memory(s);
  return result;
}

/**
 * @brief Sets the problem_id using formatted printing (as in printf).
 *
 * Takes care of memory (de-)allocation and verifies that the problem_id is in the correct format.
 */
static void coco_problem_set_id(coco_problem_t *problem, const char *id, ...) {
  va_list args;

  va_start(args, id);
  if (problem->problem_id != NULL)
    coco_free_memory(problem->problem_id);
  problem->problem_id = coco_vstrdupf(id, args);
  va_end(args);
  if (!coco_problem_id_is_fine(problem->problem_id)) {
    coco_error("Problem id should only contain standard chars, not like '%s'", problem->problem_id);
  }
}

/**
 * @brief Sets the problem_name using formatted printing (as in printf).
 *
 * Takes care of memory (de-)allocation.
 */
static void coco_problem_set_name(coco_problem_t *problem, const char *name, ...) {
  va_list args;

  va_start(args, name);
  if (problem->problem_name != NULL)
    coco_free_memory(problem->problem_name);
  problem->problem_name = coco_vstrdupf(name, args);
  va_end(args);
}

/**
 * @brief Sets the problem_type using formatted printing (as in printf).
 *
 * Takes care of memory (de-)allocation.
 */
static void coco_problem_set_type(coco_problem_t *problem, const char *type, ...) {
  va_list args;

  va_start(args, type);
  if (problem->problem_type != NULL)
    coco_free_memory(problem->problem_type);
  problem->problem_type = coco_vstrdupf(type, args);
  va_end(args);
}

size_t coco_problem_get_evaluations(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->evaluations;
}

/**
 * @note Can be used to prevent unnessary burning of CPU time. 
 */
int coco_problem_final_target_hit(const coco_problem_t *problem) {
  assert(problem != NULL);
  if (coco_problem_get_number_of_objectives(problem) != 1 ||
      coco_problem_get_evaluations(problem) < 1) 
    return 0;
  if (problem->best_value == NULL ||
      problem->final_target_delta == NULL ||
      problem->best_observed_fvalue == NULL)
    return 0;
  return problem->best_observed_fvalue[0] <= problem->best_value[0] + problem->final_target_delta[0] ?
    1 : 0;
}
/**
 * @note Tentative...
 */
double coco_problem_get_best_observed_fvalue1(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->best_observed_fvalue[0];
}

/**
 * @note This function breaks the black-box property: the returned  value is not
 * meant to be used by the optimization algorithm.
 */
double coco_problem_get_final_target_fvalue1(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->best_value != NULL);
  assert(problem->final_target_delta != NULL);
  return problem->best_value[0] + problem->final_target_delta[0];
}

/**
 * @note Do not modify the returned string! If you free the problem, the returned pointer becomes invalid.
 * When in doubt, use coco_strdup() on the returned value.
 */
const char *coco_problem_get_name(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_name != NULL);
  return problem->problem_name;
}

/**
 * The ID is guaranteed to contain only characters in the set [a-z0-9_-]. It should therefore be safe to use
 * it to construct filenames or other identifiers.
 *
 * Each problem ID should be unique within each benchmark suite.
 *
 * @note Do not modify the returned string! If you free the problem, the returned pointer becomes invalid.
 * When in doubt, use coco_strdup() on the returned value.
 */
const char *coco_problem_get_id(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_id != NULL);
  return problem->problem_id;
}

const char *coco_problem_get_type(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->problem_type != NULL);
  return problem->problem_type;
}

size_t coco_problem_get_dimension(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->number_of_variables > 0);
  return problem->number_of_variables;
}

size_t coco_problem_get_number_of_objectives(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->number_of_objectives > 0);
  return problem->number_of_objectives;
}

size_t coco_problem_get_number_of_constraints(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->number_of_constraints;
}

const double *coco_problem_get_smallest_values_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->smallest_values_of_interest != NULL);
  return problem->smallest_values_of_interest;
}

const double *coco_problem_get_largest_values_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->largest_values_of_interest != NULL);
  return problem->largest_values_of_interest;
}

/**
 * If a special method for setting an initial solution to the problem does not exist, the center of the
 * problem's region of interest is the initial solution.
 * @param problem The given COCO problem.
 * @param initial_solution The pointer to the initial solution being set by this method.
 */
void coco_problem_get_initial_solution(const coco_problem_t *problem, double *initial_solution) {
  assert(problem != NULL);
  if (problem->initial_solution != NULL) {
    problem->initial_solution(problem, initial_solution);
  } else {
    size_t i;
    assert(problem->smallest_values_of_interest != NULL);
    assert(problem->largest_values_of_interest != NULL);
    for (i = 0; i < problem->number_of_variables; ++i)
      initial_solution[i] = problem->smallest_values_of_interest[i] + 0.5
          * (problem->largest_values_of_interest[i] - problem->smallest_values_of_interest[i]);
  }
}


size_t coco_problem_get_suite_dep_index(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->suite_dep_index;
}

size_t coco_problem_get_suite_dep_function(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_function > 0);
  return problem->suite_dep_function;
}

size_t coco_problem_get_suite_dep_instance(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_instance > 0);
  return problem->suite_dep_instance;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding the transformed COCO problem
 */
/**@{*/

/**
 * @brief Returns the data of the transformed problem.
 */
static void *coco_problem_transformed_get_data(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->data != NULL);
  assert(((coco_problem_transformed_data_t *) problem->data)->data != NULL);

  return ((coco_problem_transformed_data_t *) problem->data)->data;
}

/**
 * @brief Returns the inner problem of the transformed problem.
 */
static coco_problem_t *coco_problem_transformed_get_inner_problem(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->data != NULL);
  assert(((coco_problem_transformed_data_t *) problem->data)->inner_problem != NULL);

  return ((coco_problem_transformed_data_t *) problem->data)->inner_problem;
}

/**
 * @brief Calls the coco_evaluate_function function on the inner problem.
 */
static void coco_problem_transformed_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  coco_evaluate_function(data->inner_problem, x, y);
}

/**
 * @brief Calls the coco_evaluate_constraint function on the inner problem.
 */
static void coco_problem_transformed_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  coco_evaluate_constraint(data->inner_problem, x, y);
}

/**
 * @brief Calls the coco_recommend_solution function on the inner problem.
 */
static void coco_problem_transformed_recommend_solution(coco_problem_t *problem, const double *x) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  coco_recommend_solution(data->inner_problem, x);
}

/**
 * @brief Frees only the data of the transformed problem leaving the inner problem intact.
 *
 * @note If there is no other pointer to the inner problem, access to it will be lost.
 */
static void coco_problem_transformed_free_data(coco_problem_t *problem) {
  coco_problem_transformed_data_t *data;

  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;

  if (data->data != NULL) {
    if (data->data_free_function != NULL) {
      data->data_free_function(data->data);
      data->data_free_function = NULL;
    }
    coco_free_memory(data->data);
    data->data = NULL;
  }
  /* Let the generic free problem code deal with the rest of the fields. For this we clear the free_problem
   * function pointer and recall the generic function. */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Frees the transformed problem.
 */
static void coco_problem_transformed_free(coco_problem_t *problem) {
  coco_problem_transformed_data_t *data;

  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);
  if (data->inner_problem != NULL) {
    coco_problem_free(data->inner_problem);
    data->inner_problem = NULL;
  }
  coco_problem_transformed_free_data(problem);
}

/**
 * @brief Allocates a transformed problem that wraps the inner_problem.
 *
 * By default all methods will dispatch to the inner_problem. A prefix is prepended to the problem name
 * in order to reflect the transformation somewhere.
 */
static coco_problem_t *coco_problem_transformed_allocate(coco_problem_t *inner_problem,
                                                         void *user_data,
                                                         coco_data_free_function_t data_free_function,
                                                         const char *name_prefix) {
  coco_problem_transformed_data_t *problem;
  coco_problem_t *inner_copy;
  char *old_name = coco_strdup(inner_problem->problem_name);

  problem = (coco_problem_transformed_data_t *) coco_allocate_memory(sizeof(*problem));
  problem->inner_problem = inner_problem;
  problem->data = user_data;
  problem->data_free_function = data_free_function;

  inner_copy = coco_problem_duplicate(inner_problem);
  inner_copy->evaluate_function = coco_problem_transformed_evaluate_function;
  inner_copy->evaluate_constraint = coco_problem_transformed_evaluate_constraint;
  inner_copy->recommend_solution = coco_problem_transformed_recommend_solution;
  inner_copy->problem_free_function = coco_problem_transformed_free;
  inner_copy->data = problem;

  coco_problem_set_name(inner_copy, "%s(%s)", name_prefix, old_name);
  coco_free_memory(old_name);

  return inner_copy;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding the stacked COCO problem
 */
/**@{*/

/**
 * @brief Calls the coco_evaluate_function function on the underlying problems.
 */
static void coco_problem_stacked_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_stacked_data_t* data = (coco_problem_stacked_data_t *) problem->data;

  assert(
      coco_problem_get_number_of_objectives(problem)
          == coco_problem_get_number_of_objectives(data->problem1)
              + coco_problem_get_number_of_objectives(data->problem2));

  coco_evaluate_function(data->problem1, x, &y[0]);
  coco_evaluate_function(data->problem2, x, &y[coco_problem_get_number_of_objectives(data->problem1)]);
}

/**
 * @brief Calls the coco_evaluate_constraint function on the underlying problems.
 */
static void coco_problem_stacked_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_stacked_data_t* data = (coco_problem_stacked_data_t*) problem->data;

  assert(
      coco_problem_get_number_of_constraints(problem)
          == coco_problem_get_number_of_constraints(data->problem1)
              + coco_problem_get_number_of_constraints(data->problem2));

  if (coco_problem_get_number_of_constraints(data->problem1) > 0)
    coco_evaluate_constraint(data->problem1, x, y);
  if (coco_problem_get_number_of_constraints(data->problem2) > 0)
    coco_evaluate_constraint(data->problem2, x, &y[coco_problem_get_number_of_constraints(data->problem1)]);
}

/* TODO: Missing coco_problem_stacked_recommend_solution function! */

/**
 * @brief Frees the stacked problem.
 */
static void coco_problem_stacked_free(coco_problem_t *problem) {
  coco_problem_stacked_data_t *data;

  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_stacked_data_t*) problem->data;

  if (data->problem1 != NULL) {
    coco_problem_free(data->problem1);
    data->problem1 = NULL;
  }
  if (data->problem2 != NULL) {
    coco_problem_free(data->problem2);
    data->problem2 = NULL;
  }
  /* Let the generic free problem code deal with the rest of the fields. For this we clear the free_problem
   * function pointer and recall the generic function. */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Allocates a problem constructed by stacking two COCO problems.
 * 
 * This is particularly useful for generating multi-objective problems, e.g. a bi-objective problem from two
 * single-objective problems. The stacked problem must behave like a normal COCO problem accepting the same
 * input.
 *
 * @note Regions of interest in the decision space must either agree or at least one of them must be NULL.
 * @note Best parameter becomes somewhat meaningless, but the nadir value make sense now.
 */
static coco_problem_t *coco_problem_stacked_allocate(coco_problem_t *problem1, coco_problem_t *problem2) {

  const size_t number_of_variables = coco_problem_get_dimension(problem1);
  const size_t number_of_objectives = coco_problem_get_number_of_objectives(problem1)
      + coco_problem_get_number_of_objectives(problem2);
  const size_t number_of_constraints = coco_problem_get_number_of_constraints(problem1)
      + coco_problem_get_number_of_constraints(problem2);
  size_t i;
  char *s;
  const double *smallest, *largest;
  coco_problem_stacked_data_t *data;
  coco_problem_t *problem; /* the new coco problem */

  assert(coco_problem_get_dimension(problem1) == coco_problem_get_dimension(problem2));

  problem = coco_problem_allocate(number_of_variables, number_of_objectives, number_of_constraints);

  s = coco_strconcat(coco_problem_get_id(problem1), "__");
  problem->problem_id = coco_strconcat(s, coco_problem_get_id(problem2));
  coco_free_memory(s);
  s = coco_strconcat(coco_problem_get_name(problem1), " + ");
  problem->problem_name = coco_strconcat(s, coco_problem_get_name(problem2));
  coco_free_memory(s);

  problem->evaluate_function = coco_problem_stacked_evaluate_function;
  if (number_of_constraints > 0)
    problem->evaluate_constraint = coco_problem_stacked_evaluate_constraint;

  /* set/copy ROI boundaries */
  smallest = problem1->smallest_values_of_interest;
  if (smallest == NULL)
    smallest = problem2->smallest_values_of_interest;

  largest = problem1->largest_values_of_interest;
  if (largest == NULL)
    largest = problem2->largest_values_of_interest;

  for (i = 0; i < number_of_variables; ++i) {
    if (problem2->smallest_values_of_interest != NULL)
      assert(smallest[i] == problem2->smallest_values_of_interest[i]);
    if (problem2->largest_values_of_interest != NULL)
      assert(largest[i] == problem2->largest_values_of_interest[i]);

    if (smallest != NULL)
      problem->smallest_values_of_interest[i] = smallest[i];
    if (largest != NULL)
      problem->largest_values_of_interest[i] = largest[i];

    if (problem->best_parameter) /* logger_bbob doesn't work then anymore */
      coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
  }

  /* Compute the ideal and nadir values */
  assert(problem->best_value);
  assert(problem->nadir_value);
  problem->best_value[0] = problem1->best_value[0];
  problem->best_value[1] = problem2->best_value[0];
  coco_evaluate_function(problem1, problem2->best_parameter, &problem->nadir_value[0]);
  coco_evaluate_function(problem2, problem1->best_parameter, &problem->nadir_value[1]);

  /* setup data holder */
  data = (coco_problem_stacked_data_t *) coco_allocate_memory(sizeof(*data));
  data->problem1 = problem1;
  data->problem2 = problem2;

  problem->data = data;
  problem->problem_free_function = coco_problem_stacked_free;

  return problem;
}
/**@}*/

/***********************************************************************************************************/
#line 11 "code-experiments/src/f_attractive_sector.c"
#line 1 "code-experiments/src/suite_bbob_legacy_code.c"
/**
 * @file suite_bbob_legacy_code.c
 * @brief Legacy code from BBOB2009 required to replicate the 2009 functions.
 *
 * All of this code should only be used by the suite_bbob2009 functions to provide compatibility to the
 * legacy code. New test beds should strive to use the new COCO facilities for random number generation etc.
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#line 13 "code-experiments/src/suite_bbob_legacy_code.c"

/** @brief Maximal dimension used in BBOB2009. */
#define SUITE_BBOB2009_MAX_DIM 40

/** @brief Computes the minimum of the two values. */
static double bbob2009_fmin(double a, double b) {
  return (a < b) ? a : b;
}

/** @brief Computes the maximum of the two values. */
static double bbob2009_fmax(double a, double b) {
  return (a > b) ? a : b;
}

/** @brief Rounds the given value. */
static double bbob2009_round(double x) {
  return floor(x + 0.5);
}

/**
 * @brief Allocates a n by m matrix structured as an array of pointers to double arrays.
 */
static double **bbob2009_allocate_matrix(const size_t n, const size_t m) {
  double **matrix = NULL;
  size_t i;
  matrix = (double **) coco_allocate_memory(sizeof(double *) * n);
  for (i = 0; i < n; ++i) {
    matrix[i] = coco_allocate_vector(m);
  }
  return matrix;
}

/**
 * @brief Frees the matrix structured as an array of pointers to double arrays.
 */
static void bbob2009_free_matrix(double **matrix, const size_t n) {
  size_t i;
  for (i = 0; i < n; ++i) {
    if (matrix[i] != NULL) {
      coco_free_memory(matrix[i]);
      matrix[i] = NULL;
    }
  }
  coco_free_memory(matrix);
}

/**
 * @brief Generates N uniform random numbers using inseed as the seed and stores them in r.
 */
static void bbob2009_unif(double *r, size_t N, long inseed) {
  /* generates N uniform numbers with starting seed */
  long aktseed;
  long tmp;
  long rgrand[32];
  long aktrand;
  long i;

  if (inseed < 0)
    inseed = -inseed;
  if (inseed < 1)
    inseed = 1;
  aktseed = inseed;
  for (i = 39; i >= 0; i--) {
    tmp = (int) floor((double) aktseed / (double) 127773);
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    if (aktseed < 0)
      aktseed = aktseed + 2147483647;
    if (i < 32)
      rgrand[i] = aktseed;
  }
  aktrand = rgrand[0];
  for (i = 0; i < N; i++) {
    tmp = (int) floor((double) aktseed / (double) 127773);
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    if (aktseed < 0)
      aktseed = aktseed + 2147483647;
    tmp = (int) floor((double) aktrand / (double) 67108865);
    aktrand = rgrand[tmp];
    rgrand[tmp] = aktseed;
    r[i] = (double) aktrand / 2.147483647e9;
    if (r[i] == 0.) {
      r[i] = 1e-99;
    }
  }
  return;
}

/**
 * @brief Converts from packed matrix storage to an array of array of double representation.
 */
static double **bbob2009_reshape(double **B, double *vector, const size_t m, const size_t n) {
  size_t i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      B[i][j] = vector[j * m + i];
    }
  }
  return B;
}

/**
 * @brief Generates N Gaussian random numbers using the given seed and stores them in g.
 */
static void bbob2009_gauss(double *g, const size_t N, const long seed) {
  size_t i;
  double uniftmp[6000];
  assert(2 * N < 6000);
  bbob2009_unif(uniftmp, 2 * N, seed);

  for (i = 0; i < N; i++) {
    g[i] = sqrt(-2 * log(uniftmp[i])) * cos(2 * coco_pi * uniftmp[N + i]);
    if (g[i] == 0.)
      g[i] = 1e-99;
  }
  return;
}

/**
 * @brief Computes a DIM by DIM rotation matrix based on seed and stores it in B.
 */
static void bbob2009_compute_rotation(double **B, const long seed, const size_t DIM) {
  /* To ensure temporary data fits into gvec */
  double prod;
  double gvect[2000];
  long i, j, k; /* Loop over pairs of column vectors. */

  assert(DIM * DIM < 2000);

  bbob2009_gauss(gvect, DIM * DIM, seed);
  bbob2009_reshape(B, gvect, DIM, DIM);
  /*1st coordinate is row, 2nd is column.*/

  for (i = 0; i < DIM; i++) {
    for (j = 0; j < i; j++) {
      prod = 0;
      for (k = 0; k < DIM; k++)
        prod += B[k][i] * B[k][j];
      for (k = 0; k < DIM; k++)
        B[k][i] -= prod * B[k][j];
    }
    prod = 0;
    for (k = 0; k < DIM; k++)
      prod += B[k][i] * B[k][i];
    for (k = 0; k < DIM; k++)
      B[k][i] /= sqrt(prod);
  }
}

static void bbob2009_copy_rotation_matrix(double **rot, double *M, double *b, const size_t DIM) {
  size_t row, column;
  double *current_row;

  for (row = 0; row < DIM; ++row) {
    current_row = M + row * DIM;
    for (column = 0; column < DIM; ++column) {
      current_row[column] = rot[row][column];
    }
    b[row] = 0.0;
  }
}

/**
 * @brief Randomly computes the location of the global optimum.
 */
static void bbob2009_compute_xopt(double *xopt, const long seed, const size_t DIM) {
  long i;
  bbob2009_unif(xopt, DIM, seed);
  for (i = 0; i < DIM; i++) {
    xopt[i] = 8 * floor(1e4 * xopt[i]) / 1e4 - 4;
    if (xopt[i] == 0.0)
      xopt[i] = -1e-5;
  }
}

/**
 * @brief Randomly chooses the objective offset for the given function and instance.
 */
static double bbob2009_compute_fopt(const size_t function, const size_t instance) {
  long rseed, rrseed;
  double gval, gval2;

  if (function == 4)
    rseed = 3;
  else if (function == 18)
    rseed = 17;
  else if (function == 101 || function == 102 || function == 103 || function == 107
      || function == 108 || function == 109)
    rseed = 1;
  else if (function == 104 || function == 105 || function == 106 || function == 110
      || function == 111 || function == 112)
    rseed = 8;
  else if (function == 113 || function == 114 || function == 115)
    rseed = 7;
  else if (function == 116 || function == 117 || function == 118)
    rseed = 10;
  else if (function == 119 || function == 120 || function == 121)
    rseed = 14;
  else if (function == 122 || function == 123 || function == 124)
    rseed = 17;
  else if (function == 125 || function == 126 || function == 127)
    rseed = 19;
  else if (function == 128 || function == 129 || function == 130)
    rseed = 21;
  else
    rseed = (long) function;

  rrseed = rseed + (long) (10000 * instance);
  bbob2009_gauss(&gval, 1, rrseed);
  bbob2009_gauss(&gval2, 1, rrseed + 1);
  return bbob2009_fmin(1000., bbob2009_fmax(-1000., bbob2009_round(100. * 100. * gval / gval2) / 100.));
}
#line 12 "code-experiments/src/f_attractive_sector.c"
#line 1 "code-experiments/src/transform_obj_oscillate.c"
/**
 * @file transform_obj_oscillate.c
 * @brief Implementation of oscillating the objective value.
 */

#include <assert.h>
#include <math.h>

#line 10 "code-experiments/src/transform_obj_oscillate.c"
#line 11 "code-experiments/src/transform_obj_oscillate.c"

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_oscillate_evaluate(coco_problem_t *problem, const double *x, double *y) {
  static const double factor = 0.1;
  size_t i;
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  for (i = 0; i < problem->number_of_objectives; i++) {
      if (y[i] != 0) {
          double log_y;
          log_y = log(fabs(y[i])) / factor;
          if (y[i] > 0) {
              y[i] = pow(exp(log_y + 0.49 * (sin(log_y) + sin(0.79 * log_y))), factor);
          } else {
              y[i] = -pow(exp(log_y + 0.49 * (sin(0.55 * log_y) + sin(0.31 * log_y))), factor);
          }
      }
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_oscillate(coco_problem_t *inner_problem) {
  coco_problem_t *problem;
  problem = coco_problem_transformed_allocate(inner_problem, NULL, NULL, "transform_obj_oscillate");
  problem->evaluate_function = transform_obj_oscillate_evaluate;
  /* Compute best value */
  /* Maybe not the most efficient solution */
  transform_obj_oscillate_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 13 "code-experiments/src/f_attractive_sector.c"
#line 1 "code-experiments/src/transform_obj_power.c"
/**
 * @file transform_obj_power.c
 * @brief Implementation of raising the objective value to the power of a given exponent.
 */

#include <assert.h>
#include <math.h>

#line 10 "code-experiments/src/transform_obj_power.c"
#line 11 "code-experiments/src/transform_obj_power.c"

/**
 * @brief Data type for transform_obj_power.
 */
typedef struct {
  double exponent;
} transform_obj_power_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_power_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_power_data_t *data;
  size_t i;
  data = (transform_obj_power_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  for (i = 0; i < problem->number_of_objectives; i++) {
      y[i] = pow(y[i], data->exponent);
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_power(coco_problem_t *inner_problem, const double exponent) {
  transform_obj_power_data_t *data;
  coco_problem_t *problem;

  data = (transform_obj_power_data_t *) coco_allocate_memory(sizeof(*data));
  data->exponent = exponent;

  problem = coco_problem_transformed_allocate(inner_problem, data, NULL, "transform_obj_power");
  problem->evaluate_function = transform_obj_power_evaluate;
  /* Compute best value */
  transform_obj_power_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 14 "code-experiments/src/f_attractive_sector.c"
#line 1 "code-experiments/src/transform_obj_shift.c"
/**
 * @file transform_obj_shift.c
 * @brief Implementation of shifting the objective value by the given offset.
 */

#include <assert.h>

#line 9 "code-experiments/src/transform_obj_shift.c"
#line 10 "code-experiments/src/transform_obj_shift.c"

/**
 * @brief Data type for transform_obj_shift.
 */
typedef struct {
  double offset;
} transform_obj_shift_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_shift_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  size_t i;
  data = (transform_obj_shift_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  for (i = 0; i < problem->number_of_objectives; i++) {
      y[i] += data->offset;
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_shift(coco_problem_t *inner_problem, const double offset) {
  coco_problem_t *problem;
  transform_obj_shift_data_t *data;
  size_t i;
  data = (transform_obj_shift_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  problem = coco_problem_transformed_allocate(inner_problem, data, NULL, "transform_obj_shift");
  problem->evaluate_function = transform_obj_shift_evaluate;
  for (i = 0; i < problem->number_of_objectives; i++) {
      problem->best_value[0] += offset;
  }
  return problem;
}
#line 15 "code-experiments/src/f_attractive_sector.c"
#line 1 "code-experiments/src/transform_vars_affine.c"
/**
 * @file transform_vars_affine.c
 * @brief Implementation of performing an affine transformation on decision values.
 *
 * x |-> Mx + b <br>
 * The matrix M is stored in row-major format.
 */

#include <assert.h>

#line 12 "code-experiments/src/transform_vars_affine.c"
#line 13 "code-experiments/src/transform_vars_affine.c"

/**
 * @brief Data type for transform_vars_affine.
 */
typedef struct {
  double *M, *b, *x;
} transform_vars_affine_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_affine_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j;
  transform_vars_affine_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_affine_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    /* data->M has problem->number_of_variables columns and inner_problem->number_of_variables rows. */
    const double *current_row = data->M + i * problem->number_of_variables;
    data->x[i] = data->b[i];
    for (j = 0; j < problem->number_of_variables; ++j) {
      data->x[i] += x[j] * current_row[j];
    }
  }
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_affine_free(void *thing) {
  transform_vars_affine_data_t *data = (transform_vars_affine_data_t *) thing;
  coco_free_memory(data->M);
  coco_free_memory(data->b);
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_affine(coco_problem_t *inner_problem,
                                             const double *M,
                                             const double *b,
                                             const size_t number_of_variables) {
  /*
   * TODOs:
   * - Calculate new smallest/largest values of interest?
   * - Resize bounds vectors if input and output dimensions do not match
   */

  coco_problem_t *problem;
  transform_vars_affine_data_t *data;
  size_t entries_in_M;

  entries_in_M = inner_problem->number_of_variables * number_of_variables;
  data = (transform_vars_affine_data_t *) coco_allocate_memory(sizeof(*data));
  data->M = coco_duplicate_vector(M, entries_in_M);
  data->b = coco_duplicate_vector(b, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_affine_free, "transform_vars_affine");
  problem->evaluate_function = transform_vars_affine_evaluate;
  return problem;
}
#line 16 "code-experiments/src/f_attractive_sector.c"
#line 1 "code-experiments/src/transform_vars_shift.c"
/**
 * @file transform_vars_shift.c
 * @brief Implementation of shifting all decision values by an offset.
 */

#include <assert.h>

#line 9 "code-experiments/src/transform_vars_shift.c"
#line 10 "code-experiments/src/transform_vars_shift.c"

/**
 * @brief Data type for transform_vars_shift.
 */
typedef struct {
  double *offset;
  double *shifted_x;
  coco_problem_free_function_t old_free_problem;
} transform_vars_shift_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_shift_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_shift_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  coco_evaluate_function(inner_problem, data->shifted_x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_shift_free(void *thing) {
  transform_vars_shift_data_t *data = (transform_vars_shift_data_t *) thing;
  coco_free_memory(data->shifted_x);
  coco_free_memory(data->offset);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_shift(coco_problem_t *inner_problem,
                                            const double *offset,
                                            const int shift_bounds) {
  transform_vars_shift_data_t *data;
  coco_problem_t *problem;
  size_t i;
  if (shift_bounds)
    coco_error("shift_bounds not implemented.");

  data = (transform_vars_shift_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = coco_duplicate_vector(offset, inner_problem->number_of_variables);
  data->shifted_x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_shift_free, "transform_vars_shift");
  problem->evaluate_function = transform_vars_shift_evaluate;
  /* Compute best parameter */
  for (i = 0; i < problem->number_of_variables; i++) {
      problem->best_parameter[i] += data->offset[i];
  }
  return problem;
}
#line 17 "code-experiments/src/f_attractive_sector.c"

/**
 * @brief Data type for the attractive sector problem.
 */
typedef struct {
  double *xopt;
} f_attractive_sector_data_t;

/**
 * @brief Implements the attractive sector function without connections to any COCO structures.
 */
static double f_attractive_sector_raw(const double *x,
                                      const size_t number_of_variables,
                                      f_attractive_sector_data_t *data) {
  size_t i;
  double result;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    if (data->xopt[i] * x[i] > 0.0) {
      result += 100.0 * 100.0 * x[i] * x[i];
    } else {
      result += x[i] * x[i];
    }
  }
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_attractive_sector_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_attractive_sector_raw(x, problem->number_of_variables, (f_attractive_sector_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the attractive sector data object.
 */
static void f_attractive_sector_free(coco_problem_t *problem) {
  f_attractive_sector_data_t *data;
  data = (f_attractive_sector_data_t *) problem->data;
  coco_free_memory(data->xopt);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Allocates the basic attractive sector problem.
 */
static coco_problem_t *f_attractive_sector_allocate(const size_t number_of_variables, const double *xopt) {

  f_attractive_sector_data_t *data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("attractive sector function",
      f_attractive_sector_evaluate, f_attractive_sector_free, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "attractive_sector", number_of_variables);

  data = (f_attractive_sector_data_t *) coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, number_of_variables);
  problem->data = data;

  /* Compute best solution */
  f_attractive_sector_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB attractive sector problem.
 */
static coco_problem_t *f_attractive_sector_bbob_problem_allocate(const size_t function,
                                                                 const size_t dimension,
                                                                 const size_t instance,
                                                                 const long rseed,
                                                                 const char *problem_id_template,
                                                                 const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j, k;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  /* Compute affine transformation M from two rotation matrices */
  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  rot2 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(rot2, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      for (k = 0; k < dimension; ++k) {
        double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
      }
    }
  }
  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  problem = f_attractive_sector_allocate(dimension, xopt);
  problem = transform_obj_oscillate(problem);
  problem = transform_obj_power(problem, 0.9);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
#line 10 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_bent_cigar.c"
/**
 * @file f_bent_cigar.c
 * @brief Implementation of the bent cigar function and problem.
 */

#include <stdio.h>
#include <assert.h>

#line 10 "code-experiments/src/f_bent_cigar.c"
#line 11 "code-experiments/src/f_bent_cigar.c"
#line 12 "code-experiments/src/f_bent_cigar.c"
#line 13 "code-experiments/src/f_bent_cigar.c"
#line 14 "code-experiments/src/f_bent_cigar.c"
#line 1 "code-experiments/src/transform_vars_asymmetric.c"
/**
 * @file transform_vars_asymmetric.c
 * @brief Implementation of performing an asymmetric transformation on decision values.
 */

#include <math.h>
#include <assert.h>

#line 10 "code-experiments/src/transform_vars_asymmetric.c"
#line 11 "code-experiments/src/transform_vars_asymmetric.c"

/**
 * @brief Data type for transform_vars_asymmetric.
 */
typedef struct {
  double *x;
  double beta;
} transform_vars_asymmetric_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_asymmetric_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double exponent;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent = 1.0
          + (data->beta * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

static void transform_vars_asymmetric_free(void *thing) {
  transform_vars_asymmetric_data_t *data = (transform_vars_asymmetric_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_asymmetric(coco_problem_t *inner_problem, const double beta) {
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_asymmetric_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->beta = beta;
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_asymmetric_free, "transform_vars_asymmetric");
  problem->evaluate_function = transform_vars_asymmetric_evaluate;
  return problem;
}
#line 15 "code-experiments/src/f_bent_cigar.c"
#line 16 "code-experiments/src/f_bent_cigar.c"

/**
 * @brief Implements the bent cigar function without connections to any COCO structures.
 */
static double f_bent_cigar_raw(const double *x, const size_t number_of_variables) {

  static const double condition = 1.0e6;
  size_t i;
  double result;

  result = x[0] * x[0];
  for (i = 1; i < number_of_variables; ++i) {
    result += condition * x[i] * x[i];
  }
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_bent_cigar_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_bent_cigar_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic bent cigar problem.
 */
static coco_problem_t *f_bent_cigar_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("bent cigar function",
      f_bent_cigar_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "bent_cigar", number_of_variables);

  /* Compute best solution */
  f_bent_cigar_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB bent cigar problem.
 */
static coco_problem_t *f_bent_cigar_bbob_problem_allocate(const size_t function,
                                                          const size_t dimension,
                                                          const size_t instance,
                                                          const long rseed,
                                                          const char *problem_id_template,
                                                          const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed + 1000000, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_bent_cigar_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_asymmetric(problem, 0.5);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
#line 11 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_bueche_rastrigin.c"
/**
 * @file f_bueche_rastrigin.c
 * @brief Implementation of the Bueche-Rastrigin function and problem.
 */

#include <math.h>
#include <assert.h>

#line 10 "code-experiments/src/f_bueche_rastrigin.c"
#line 11 "code-experiments/src/f_bueche_rastrigin.c"
#line 12 "code-experiments/src/f_bueche_rastrigin.c"
#line 1 "code-experiments/src/transform_vars_brs.c"
/**
 * @file transform_vars_brs.c
 * @brief Implementation of the ominous 's_i scaling' of the BBOB Bueche-Rastrigin problem.
 */

#include <math.h>
#include <assert.h>

#line 10 "code-experiments/src/transform_vars_brs.c"
#line 11 "code-experiments/src/transform_vars_brs.c"

/**
 * @brief Data type for transform_vars_brs.
 */
typedef struct {
  double *x;
} transform_vars_brs_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_brs_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double factor;
  transform_vars_brs_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_brs_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    /* Function documentation says we should compute 10^(0.5 *
     * (i-1)/(D-1)). Instead we compute the equivalent
     * sqrt(10)^((i-1)/(D-1)) just like the legacy code.
     */
    factor = pow(sqrt(10.0), (double) (long) i / ((double) (long) problem->number_of_variables - 1.0));
    /* Documentation specifies odd indices and starts indexing
     * from 1, we use all even indices since C starts indexing
     * with 0.
     */
    if (x[i] > 0.0 && i % 2 == 0) {
      factor *= 10.0;
    }
    data->x[i] = factor * x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_brs_free(void *thing) {
  transform_vars_brs_data_t *data = (transform_vars_brs_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_brs(coco_problem_t *inner_problem) {
  transform_vars_brs_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_brs_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_brs_free, "transform_vars_brs");
  problem->evaluate_function = transform_vars_brs_evaluate;
  return problem;
}
#line 13 "code-experiments/src/f_bueche_rastrigin.c"
#line 1 "code-experiments/src/transform_vars_oscillate.c"
/**
 * @file transform_vars_oscillate.c
 * @brief Implementation of oscillating the decision values.
 */

#include <math.h>
#include <assert.h>

#line 10 "code-experiments/src/transform_vars_oscillate.c"
#line 11 "code-experiments/src/transform_vars_oscillate.c"

/**
 * @brief Data type for transform_vars_oscillate.
 */
typedef struct {
  double *oscillated_x;
} transform_vars_oscillate_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_oscillate_evaluate(coco_problem_t *problem, const double *x, double *y) {
  static const double alpha = 0.1;
  double tmp, base, *oscillated_x;
  size_t i;
  transform_vars_oscillate_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_oscillate_data_t *) coco_problem_transformed_get_data(problem);
  oscillated_x = data->oscillated_x; /* short cut to make code more readable */
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      tmp = log(x[i]) / alpha;
      base = exp(tmp + 0.49 * (sin(tmp) + sin(0.79 * tmp)));
      oscillated_x[i] = pow(base, alpha);
    } else if (x[i] < 0.0) {
      tmp = log(-x[i]) / alpha;
      base = exp(tmp + 0.49 * (sin(0.55 * tmp) + sin(0.31 * tmp)));
      oscillated_x[i] = -pow(base, alpha);
    } else {
      oscillated_x[i] = 0.0;
    }
  }
  coco_evaluate_function(inner_problem, oscillated_x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_oscillate_free(void *thing) {
  transform_vars_oscillate_data_t *data = (transform_vars_oscillate_data_t *) thing;
  coco_free_memory(data->oscillated_x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_oscillate(coco_problem_t *inner_problem) {
  transform_vars_oscillate_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_oscillate_data_t *) coco_allocate_memory(sizeof(*data));
  data->oscillated_x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_oscillate_free, "transform_vars_oscillate");
  problem->evaluate_function = transform_vars_oscillate_evaluate;
  return problem;
}
#line 14 "code-experiments/src/f_bueche_rastrigin.c"
#line 15 "code-experiments/src/f_bueche_rastrigin.c"
#line 16 "code-experiments/src/f_bueche_rastrigin.c"
#line 1 "code-experiments/src/transform_obj_penalize.c"
/**
 * @file transform_obj_penalize.c
 * @brief Implementation of adding a penalty to the objective value for solutions outside of the ROI in the
 * decision space.
 */

#include <assert.h>

#line 10 "code-experiments/src/transform_obj_penalize.c"
#line 11 "code-experiments/src/transform_obj_penalize.c"

/**
 * @brief Data type for transform_obj_penalize.
 */
typedef struct {
  double factor;
} transform_obj_penalize_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_obj_penalize_evaluate(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_penalize_data_t *data = (transform_obj_penalize_data_t *) coco_problem_transformed_get_data(problem);
  const double *lower_bounds = problem->smallest_values_of_interest;
  const double *upper_bounds = problem->largest_values_of_interest;
  double penalty = 0.0;
  size_t i;
  for (i = 0; i < problem->number_of_variables; ++i) {
    const double c1 = x[i] - upper_bounds[i];
    const double c2 = lower_bounds[i] - x[i];
    assert(lower_bounds[i] < upper_bounds[i]);
    if (c1 > 0.0) {
      penalty += c1 * c1;
    } else if (c2 > 0.0) {
      penalty += c2 * c2;
    }
  }
  assert(coco_problem_transformed_get_inner_problem(problem) != NULL);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  for (i = 0; i < problem->number_of_objectives; ++i) {
    y[i] += data->factor * penalty;
  }
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_penalize(coco_problem_t *inner_problem, const double factor) {
  coco_problem_t *problem;
  transform_obj_penalize_data_t *data;
  assert(inner_problem != NULL);

  data = (transform_obj_penalize_data_t *) coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  problem = coco_problem_transformed_allocate(inner_problem, data, NULL, "transform_obj_penalize");
  problem->evaluate_function = transform_obj_penalize_evaluate;
  /* No need to update the best value as the best parameter is feasible */
  return problem;
}
#line 17 "code-experiments/src/f_bueche_rastrigin.c"

/**
 * @brief Implements the Bueche-Rastrigin function without connections to any COCO structures.
 */
static double f_bueche_rastrigin_raw(const double *x, const size_t number_of_variables) {

  double tmp = 0., tmp2 = 0.;
  size_t i;
  double result;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    tmp += cos(2 * coco_pi * x[i]);
    tmp2 += x[i] * x[i];
  }
  result = 10.0 * ((double) (long) number_of_variables - tmp) + tmp2 + 0;
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_bueche_rastrigin_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_bueche_rastrigin_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Bueche-Rastrigin problem.
 */
static coco_problem_t *f_bueche_rastrigin_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Bueche-Rastrigin function",
      f_bueche_rastrigin_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "bueche-rastrigin", number_of_variables);

  /* Compute best solution */
  f_bueche_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Bueche-Rastrigin problem.
 */
static coco_problem_t *f_bueche_rastrigin_bbob_problem_allocate(const size_t function,
                                                                const size_t dimension,
                                                                const size_t instance,
                                                                const long rseed,
                                                                const char *problem_id_template,
                                                                const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  const double penalty_factor = 100.0;
  size_t i;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  /* OME: This step is in the legacy C code but _not_ in the function description. */
  for (i = 0; i < dimension; i += 2) {
    xopt[i] = fabs(xopt[i]);
  }

  problem = f_bueche_rastrigin_allocate(dimension);
  problem = transform_vars_brs(problem);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_obj_penalize(problem, penalty_factor);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}
#line 12 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_different_powers.c"
/**
 * @file f_different_powers.c
 * @brief Implementation of the different powers function and problem.
 */

#include <assert.h>
#include <math.h>

#line 10 "code-experiments/src/f_different_powers.c"
#line 11 "code-experiments/src/f_different_powers.c"
#line 12 "code-experiments/src/f_different_powers.c"
#line 13 "code-experiments/src/f_different_powers.c"
#line 14 "code-experiments/src/f_different_powers.c"
#line 15 "code-experiments/src/f_different_powers.c"

/**
 * @brief Implements the different powers function without connections to any COCO structures.
 */
static double f_different_powers_raw(const double *x, const size_t number_of_variables) {

  size_t i;
  double sum = 0.0;
  double result;

  for (i = 0; i < number_of_variables; ++i) {
    double exponent = 2.0 + (4.0 * (double) (long) i) / ((double) (long) number_of_variables - 1.0);
    sum += pow(fabs(x[i]), exponent);
  }
  result = sqrt(sum);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_different_powers_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_different_powers_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic different powers problem.
 */
static coco_problem_t *f_different_powers_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("different powers function",
      f_different_powers_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "different_powers", number_of_variables);

  /* Compute best solution */
  f_different_powers_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB different powers problem.
 */
static coco_problem_t *f_different_powers_bbob_problem_allocate(const size_t function,
                                                                const size_t dimension,
                                                                const size_t instance,
                                                                const long rseed,
                                                                const char *problem_id_template,
                                                                const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_different_powers_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
#line 13 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_discus.c"
/**
 * @file f_discus.c
 * @brief Implementation of the discus function and problem.
 */

#include <assert.h>

#line 9 "code-experiments/src/f_discus.c"
#line 10 "code-experiments/src/f_discus.c"
#line 11 "code-experiments/src/f_discus.c"
#line 12 "code-experiments/src/f_discus.c"
#line 13 "code-experiments/src/f_discus.c"
#line 14 "code-experiments/src/f_discus.c"
#line 15 "code-experiments/src/f_discus.c"

/**
 * @brief Implements the discus function without connections to any COCO structures.
 */
static double f_discus_raw(const double *x, const size_t number_of_variables) {

  static const double condition = 1.0e6;
  size_t i;
  double result;

  result = condition * x[0] * x[0];
  for (i = 1; i < number_of_variables; ++i) {
    result += x[i] * x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_discus_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_discus_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic discus problem.
 */
static coco_problem_t *f_discus_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("discus function",
      f_discus_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "discus", number_of_variables);

  /* Compute best solution */
  f_discus_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB discus problem.
 */
static coco_problem_t *f_discus_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const long rseed,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_discus_allocate(dimension);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
#line 14 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_ellipsoid.c"
/**
 * @file f_ellipsoid.c
 * @brief Implementation of the ellipsoid function and problem.
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>

#line 11 "code-experiments/src/f_ellipsoid.c"
#line 12 "code-experiments/src/f_ellipsoid.c"
#line 13 "code-experiments/src/f_ellipsoid.c"
#line 14 "code-experiments/src/f_ellipsoid.c"
#line 15 "code-experiments/src/f_ellipsoid.c"
#line 16 "code-experiments/src/f_ellipsoid.c"
#line 1 "code-experiments/src/transform_vars_permblockdiag.c"
/**
 * @file transform_vars_permblockdiag.c
 */

#include <assert.h>

#line 8 "code-experiments/src/transform_vars_permblockdiag.c"
#line 9 "code-experiments/src/transform_vars_permblockdiag.c"
#line 1 "code-experiments/src/large_scale_transformations.c"
#include <stdio.h>
#include <assert.h>
#line 4 "code-experiments/src/large_scale_transformations.c"

#line 6 "code-experiments/src/large_scale_transformations.c"
#line 7 "code-experiments/src/large_scale_transformations.c"

#include <time.h> /*tmp*/

/* TODO: Document this file in doxygen style! */

static double *ls_random_data;/* global variable used to generate the random permutations */

/**
 * ls_allocate_blockmatrix(n, m, bs):
 *
 * Allocate a ${n} by ${m} block matrix of nb_blocks block sizes block_sizes structured as an array of pointers
 * to double arrays.
 * each row constains only the block_sizes[i] possibly non-zero elements
 */
static double **ls_allocate_blockmatrix(const size_t n, const size_t* block_sizes, const size_t nb_blocks) {
  double **matrix = NULL;
  size_t current_blocksize;
  size_t next_bs_change;
  size_t idx_blocksize;
  size_t i;
  size_t sum_block_sizes;
  
  sum_block_sizes = 0;
  for (i = 0; i < nb_blocks; i++){
    sum_block_sizes += block_sizes[i];
  }
  assert(sum_block_sizes == n);
  
  matrix = (double **) coco_allocate_memory(sizeof(double *) * n);
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];
  
  for (i = 0; i < n; ++i) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    matrix[i] = coco_allocate_vector(current_blocksize);
    
  }
  return matrix;
}


/*
 * frees a block diagonal matrix (same as a matrix but in case of change, easier to update separatly from free_matrix)
 */
static void ls_free_block_matrix(double **matrix, const size_t n) {
  size_t i;
  for (i = 0; i < n; ++i) {
    if (matrix[i] != NULL) {
      coco_free_memory(matrix[i]);
      matrix[i] = NULL;
    }
  }
  coco_free_memory(matrix);
}



/**
 * ls_compute_blockrotation(B, seed, DIM):
 *
 * Compute a ${DIM}x${DIM} block-diagonal matrix based on ${seed} and block_sizes and stores it in ${B}.
 * B is a 2D vector with DIM lines and each line has blocksize(line) elements (the zeros are not stored)
 */
static void ls_compute_blockrotation(double **B, long seed, size_t n, size_t *block_sizes, size_t nb_blocks) {
  double prod;
  /*double *gvect;*/
  double **current_block;
  size_t i, j, k; /* Loop over pairs of column vectors. */
  size_t idx_block, current_blocksize,cumsum_prev_block_sizes, sum_block_sizes;
  size_t nb_entries;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);
  
  nb_entries = 0;
  sum_block_sizes = 0;
  for (i = 0; i < nb_blocks; i++){
    sum_block_sizes += block_sizes[i];
    nb_entries += block_sizes[i] * block_sizes[i];
  }
  assert(sum_block_sizes == n);
  
  cumsum_prev_block_sizes = 0;/* shift in rows to account for the previous blocks */
  for (idx_block = 0; idx_block < nb_blocks; idx_block++) {
    current_blocksize = block_sizes[idx_block];
    current_block = bbob2009_allocate_matrix(current_blocksize, current_blocksize);
    for (i = 0; i < current_blocksize; i++) {
      for (j = 0; j < current_blocksize; j++) {
        current_block[i][j] = coco_random_normal(rng);
      }
    }
    
    for (i = 0; i < current_blocksize; i++) {
      for (j = 0; j < i; j++) {
        prod = 0;
        for (k = 0; k < current_blocksize; k++){
          prod += current_block[k][i] * current_block[k][j];
        }
        for (k = 0; k < current_blocksize; k++){
          current_block[k][i] -= prod * current_block[k][j];
        }
      }
      prod = 0;
      for (k = 0; k < current_blocksize; k++){
        prod += current_block[k][i] * current_block[k][i];
      }
      for (k = 0; k < current_blocksize; k++){
        current_block[k][i] /= sqrt(prod);
      }
    }
    
    /* now fill the block matrix*/
    for (i = 0 ; i < current_blocksize; i++) {
      for (j = 0; j < current_blocksize; j++) {
        B[i + cumsum_prev_block_sizes][j]=current_block[i][j];
      }
    }
    
    cumsum_prev_block_sizes+=current_blocksize;
    /*current_gvect_pos += current_blocksize * current_blocksize;*/
    ls_free_block_matrix(current_block, current_blocksize);
  }
  /*coco_free_memory(gvect);*/
  coco_random_free(rng);
}

/*
 * makes a copy of a block_matrix
 */
static double **ls_copy_block_matrix(const double *const *B, const size_t dimension, const size_t *block_sizes, const size_t nb_blocks) {
  double **dest;
  size_t i, j, idx_blocksize, current_blocksize, next_bs_change;
  
  dest = ls_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  idx_blocksize = 0;
  current_blocksize = block_sizes[idx_blocksize];
  next_bs_change = block_sizes[idx_blocksize];
  assert(nb_blocks != 0); /*tmp*//*to silence warning*/
  for (i = 0; i < dimension; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    for (j = 0; j < current_blocksize; j++) {
      dest[i][j] = B[i][j];
    }
  }
  return dest;
}

/**
 * Comparison function used for sorting.
 * In our case, it serves as a random permutation generator
 */
static int f_compare_doubles_for_random_permutation(const void *a, const void *b) {
  double temp = ls_random_data[*(const size_t *) a] - ls_random_data[*(const size_t *) b];
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}

/*
 * generates a random, uniformly sampled, permutation and puts it in P
 */
static void ls_compute_random_permutation(size_t *P, long seed, size_t n) {
  long i;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);
  ls_random_data = coco_allocate_vector(n);
  for (i = 0; i < n; i++){
    P[i] = (size_t) i;
    ls_random_data[i] = coco_random_uniform(rng);
  }
  qsort(P, n, sizeof(size_t), f_compare_doubles_for_random_permutation);
  coco_random_free(rng);
}


/*
 * returns a uniformly distributed integer between lower_bound and upper_bound using seed.
 */
long ls_rand_int(long lower_bound, long upper_bound, coco_random_state_t *rng){
  long range;
  range = upper_bound - lower_bound + 1;
  return ((long)(coco_random_uniform(rng) * (double) range)) + lower_bound;
}



/*
 * generates a random permutation resulting from nb_swaps truncated uniform swaps of range swap_range
 * missing paramteters: dynamic_not_static pool, seems empirically irrelevant
 * for now so dynamic is implemented (simple since no need for tracking indices
 * if swap_range is the largest possible size_t value ( (size_t) -1 ), a random uniform permutation is generated
 */
static void ls_compute_truncated_uniform_swap_permutation(size_t *P, long seed, size_t n, size_t nb_swaps, size_t swap_range) {
  long i, idx_swap;
  size_t lower_bound, upper_bound, first_swap_var, second_swap_var, tmp;
  size_t *idx_order;
  coco_random_state_t *rng = coco_random_new((uint32_t) seed);

  ls_random_data = coco_allocate_vector(n);
  idx_order = coco_allocate_vector_size_t(n);
  for (i = 0; i < n; i++){
    P[i] = (size_t) i;
    idx_order[i] = (size_t) i;
    ls_random_data[i] = coco_random_uniform(rng);
  }
  
  if (swap_range > 0) {
    /*sort the random data in random_data and arange idx_order accordingly*/
    /*did not use ls_compute_random_permutation to only use the seed once*/
    qsort(idx_order, n, sizeof(size_t), f_compare_doubles_for_random_permutation);
    for (idx_swap = 0; idx_swap < nb_swaps; idx_swap++) {
      first_swap_var = idx_order[idx_swap];
      if (first_swap_var < swap_range) {
        lower_bound = 0;
      }
      else{
        lower_bound = first_swap_var - swap_range;
      }
      if (first_swap_var + swap_range > n - 1) {
        upper_bound = n - 1;
      }
      else{
        upper_bound = first_swap_var + swap_range;
      }

      second_swap_var = (size_t) ls_rand_int((long) lower_bound, (long) upper_bound, rng);
      while (first_swap_var == second_swap_var) {
        second_swap_var = (size_t) ls_rand_int((long) lower_bound, (long) upper_bound, rng);
      }
      /* swap*/
      tmp = P[first_swap_var];
      P[first_swap_var] = P[second_swap_var];
      P[second_swap_var] = tmp;
    }
  } else {
    if ( swap_range == (size_t) -1) {
      /* generate random permutation instead */
      ls_compute_random_permutation(P, seed, n);
    }
    
  }
  coco_random_free(rng);
}



/*
 * duplicates a size_t vector
 */
size_t *coco_duplicate_size_t_vector(const size_t *src, const size_t number_of_elements) {
  size_t i;
  size_t *dst;
  
  assert(src != NULL);
  assert(number_of_elements > 0);
  
  dst = coco_allocate_vector_size_t(number_of_elements);
  for (i = 0; i < number_of_elements; ++i) {
    dst[i] = src[i];
  }
  return dst;
}


/*
 * returns the list of block_sizes and sets nb_blocks to its correct value
 * TODO: update with chosen parameter setting
 */
size_t *ls_get_block_sizes(size_t *nb_blocks, size_t dimension){
  size_t *block_sizes;
  size_t block_size;
  int i;
  
  block_size = coco_double_to_size_t(bbob2009_fmin((double)dimension / 4, 100));
  *nb_blocks = dimension / block_size + ((dimension % block_size) > 0);
  block_sizes = coco_allocate_vector_size_t(*nb_blocks);
  for (i = 0; i < *nb_blocks - 1; i++) {
    block_sizes[i] = block_size;
  }
  block_sizes[*nb_blocks - 1] = dimension - (*nb_blocks - 1) * block_size; /*add rest*/
  return block_sizes;
}


/*
 * return the swap_range corresponding to the problem
 * TODO: update with chosen parameter setting
 */
size_t ls_get_swap_range(size_t dimension){
  return dimension / 3;
}


/*
 * return the number of swaps corresponding to the problem
 * TODO: update with chosen parameter setting
 */
size_t ls_get_nb_swaps(size_t dimension){
  return dimension;
}




#line 10 "code-experiments/src/transform_vars_permblockdiag.c"

/**
 * @brief Data type for transform_vars_permblockdiag.
 */
typedef struct {
  double **B;
  double *x;
  size_t *P1; /*permutation matrices, P1 for the columns of B and P2 for its rows*/
  size_t *P2;
  size_t *block_sizes;
  size_t nb_blocks;
  size_t *block_size_map; /* maps rows to blocksizes, keep until better way is found */
  size_t *first_non_zero_map; /* maps a row to the index of its first non zero element */
} transform_vars_permblockdiag_t;

static void transform_vars_permblockdiag_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j, current_blocksize, first_non_zero_ind;
  transform_vars_permblockdiag_t *data;
  coco_problem_t *inner_problem;
  
  data = (transform_vars_permblockdiag_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  
  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    current_blocksize = data->block_size_map[data->P2[i]];/*the block_size is that of the permuted line*/
    first_non_zero_ind = data->first_non_zero_map[data->P2[i]];
    data->x[i] = 0;
    /*compute data->x[i] = < B[P2[i]] , x[P1] >  */
    for (j = first_non_zero_ind; j < first_non_zero_ind + current_blocksize; ++j) {/*blocksize[P2[i]]*/
      data->x[i] += data->B[data->P2[i]][j - first_non_zero_ind] * x[data->P1[j]];/*all B lines start at 0*/
    }
    if (data->x[i] > 100 || data->x[i] < -100 || 1) {
    }
    
  }
  
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

static void transform_vars_permblockdiag_free(void *thing) {
  transform_vars_permblockdiag_t *data = (transform_vars_permblockdiag_t *) thing;
  coco_free_memory(data->B);
  coco_free_memory(data->P1);
  coco_free_memory(data->P2);
  coco_free_memory(data->block_sizes);
  coco_free_memory(data->x);
  coco_free_memory(data->block_size_map);
}

/*
 * Apply a double permuted orthogonal block-diagonal transfromation matrix to the search space
 *
 *
 * The matrix M is stored in row-major format.
 */
static coco_problem_t *transform_vars_permblockdiag(coco_problem_t *inner_problem,
                                                    const double * const *B,
                                                    const size_t *P1,
                                                    const size_t *P2,
                                                    const size_t number_of_variables,
                                                    const size_t *block_sizes,
                                                    const size_t nb_blocks) {
  coco_problem_t *problem;
  transform_vars_permblockdiag_t *data;
  size_t entries_in_M, idx_blocksize, next_bs_change, current_blocksize;
  int i;
  entries_in_M = 0;
  assert(number_of_variables > 0);/*tmp*/
  for (i = 0; i < nb_blocks; i++) {
    entries_in_M += block_sizes[i] * block_sizes[i];
  }
  data = (transform_vars_permblockdiag_t *) coco_allocate_memory(sizeof(*data));
  data->B = ls_copy_block_matrix(B, number_of_variables, block_sizes, nb_blocks);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->P1 = coco_duplicate_size_t_vector(P1, inner_problem->number_of_variables);
  data->P2 = coco_duplicate_size_t_vector(P2, inner_problem->number_of_variables);
  data->block_sizes = coco_duplicate_size_t_vector(block_sizes, nb_blocks);
  data->nb_blocks = nb_blocks;
  data->block_size_map = coco_allocate_vector_size_t(number_of_variables);
  data->first_non_zero_map = coco_allocate_vector_size_t(number_of_variables);
  
  idx_blocksize = 0;
  next_bs_change = block_sizes[idx_blocksize];
  for (i = 0; i < number_of_variables; i++) {
    if (i >= next_bs_change) {
      idx_blocksize++;
      next_bs_change += block_sizes[idx_blocksize];
    }
    current_blocksize=block_sizes[idx_blocksize];
    data->block_size_map[i] = current_blocksize;
    data->first_non_zero_map[i] = next_bs_change - current_blocksize;/* next_bs_change serves also as a cumsum for blocksizes*/
  }
  
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_permblockdiag_free, "transform_vars_permblockdiag");
  problem->evaluate_function = transform_vars_permblockdiag_evaluate;
  return problem;
}


#line 17 "code-experiments/src/f_ellipsoid.c"

/**
 * @brief Implements the ellipsoid function without connections to any COCO structures.
 */
static double f_ellipsoid_raw(const double *x, const size_t number_of_variables) {

  static const double condition = 1.0e6;
  size_t i = 0;
  double result;

  result = x[i] * x[i];
  for (i = 1; i < number_of_variables; ++i) {
    const double exponent = 1.0 * (double) (long) i / ((double) (long) number_of_variables - 1.0);
    result += pow(condition, exponent) * x[i] * x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_ellipsoid_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_ellipsoid_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("ellipsoid function",
      f_ellipsoid_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "ellipsoid", number_of_variables);

  /* Compute best solution */
  f_ellipsoid_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  problem = f_ellipsoid_allocate(dimension);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the BBOB rotated ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_rotated_bbob_problem_allocate(const size_t function,
                                                                 const size_t dimension,
                                                                 const size_t instance,
                                                                 const long rseed,
                                                                 const char *problem_id_template,
                                                                 const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  bbob2009_free_matrix(rot1, dimension);

  problem = f_ellipsoid_allocate(dimension);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}

static coco_problem_t *f_ellipsoid_permblockdiag_bbob_problem_allocate(const size_t function,
                                                                       const size_t dimension,
                                                                       const size_t instance,
                                                                       const long rseed,
                                                                       const char *problem_id_template,
                                                                       const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  double **B;
  const double *const *B_copy;
  size_t *P1 = coco_allocate_vector_size_t(dimension);
  size_t *P2 = coco_allocate_vector_size_t(dimension);
  size_t *block_sizes;
  size_t nb_blocks;
  size_t swap_range;
  size_t nb_swaps;
  
  block_sizes = ls_get_block_sizes(&nb_blocks, dimension);
  swap_range = ls_get_swap_range(dimension);
  nb_swaps = ls_get_nb_swaps(dimension);

  /*printf("f:%zu  n:%zu  i:%zu  bs:[%zu,...,%zu,%zu]  sR:%zu\n", function, dimension, instance, block_sizes[0], block_sizes[0],block_sizes[nb_blocks-1], swap_range);*/
  
  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  
  B = ls_allocate_blockmatrix(dimension, block_sizes, nb_blocks);
  B_copy = (const double *const *)B;/*TODO: silences the warning, not sure if it prevents the modification of B at all levels*/

  ls_compute_blockrotation(B, rseed + 1000000, dimension, block_sizes, nb_blocks);
  ls_compute_truncated_uniform_swap_permutation(P1, rseed + 2000000, dimension, nb_swaps, swap_range);
  ls_compute_truncated_uniform_swap_permutation(P2, rseed + 3000000, dimension, nb_swaps, swap_range);

  
  problem = f_ellipsoid_allocate(dimension);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_permblockdiag(problem, B_copy, P1, P2, dimension, block_sizes, nb_blocks);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "large_scale_block_rotated");/*TODO: no large scale prefix*/

  ls_free_block_matrix(B, dimension);
  coco_free_memory(P1);
  coco_free_memory(P2);
  coco_free_memory(block_sizes);
  
  return problem;
}




#line 15 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_gallagher.c"
/**
 * @file f_gallagher.c
 * @brief Implementation of the Gallagher function and problem.
 */

#include <assert.h>
#include <math.h>

#line 10 "code-experiments/src/f_gallagher.c"
#line 11 "code-experiments/src/f_gallagher.c"
#line 12 "code-experiments/src/f_gallagher.c"
#line 13 "code-experiments/src/f_gallagher.c"
#line 14 "code-experiments/src/f_gallagher.c"

/**
 * @brief A random permutation type for the Gallagher problem.
 *
 * Needed to create a random permutation that is compatible with the old code.
 */
typedef struct {
  double value;
  size_t index;
} f_gallagher_permutation_t;

/**
 * @brief Data type for the Gallagher problem.
 */
typedef struct {
  long rseed;
  double *xopt;
  double **rotation, **x_local, **arr_scales;
  size_t number_of_peaks;
  double *peak_values;
  coco_problem_free_function_t old_free_problem;
} f_gallagher_data_t;

/**
 * Comparison function used for sorting.
 */
static int f_gallagher_compare_doubles(const void *a, const void *b) {
  double temp = (*(f_gallagher_permutation_t *) a).value - (*(f_gallagher_permutation_t *) b).value;
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}

/**
 * @brief Implements the Gallagher function without connections to any COCO structures.
 */
static double f_gallagher_raw(const double *x, const size_t number_of_variables, f_gallagher_data_t *data) {
  size_t i, j; /* Loop over dim */
  double *tmx;
  double a = 0.1;
  double tmp2, f = 0., f_add, tmp, f_pen = 0., f_true = 0.;
  double fac;
  double result;

  fac = -0.5 / (double) number_of_variables;

  /* Boundary handling */
  for (i = 0; i < number_of_variables; ++i) {
    tmp = fabs(x[i]) - 5.;
    if (tmp > 0.) {
      f_pen += tmp * tmp;
    }
  }
  f_add = f_pen;
  /* Transformation in search space */
  /* TODO: this should rather be done in f_gallagher */
  tmx = coco_allocate_vector(number_of_variables);
  for (i = 0; i < number_of_variables; i++) {
    tmx[i] = 0;
    for (j = 0; j < number_of_variables; ++j) {
      tmx[i] += data->rotation[i][j] * x[j];
    }
  }
  /* Computation core*/
  for (i = 0; i < data->number_of_peaks; ++i) {
    tmp2 = 0.;
    for (j = 0; j < number_of_variables; ++j) {
      tmp = (tmx[j] - data->x_local[j][i]);
      tmp2 += data->arr_scales[i][j] * tmp * tmp;
    }
    tmp2 = data->peak_values[i] * exp(fac * tmp2);
    f = coco_double_max(f, tmp2);
  }

  f = 10. - f;
  if (f > 0) {
    f_true = log(f) / a;
    f_true = pow(exp(f_true + 0.49 * (sin(f_true) + sin(0.79 * f_true))), a);
  } else if (f < 0) {
    f_true = log(-f) / a;
    f_true = -pow(exp(f_true + 0.49 * (sin(0.55 * f_true) + sin(0.31 * f_true))), a);
  } else
    f_true = f;

  f_true *= f_true;
  f_true += f_add;
  result = f_true;
  coco_free_memory(tmx);
  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_gallagher_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_gallagher_raw(x, problem->number_of_variables, (f_gallagher_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the Gallagher data object.
 */
static void f_gallagher_free(coco_problem_t *problem) {
  f_gallagher_data_t *data;
  data = (f_gallagher_data_t *) problem->data;
  coco_free_memory(data->xopt);
  coco_free_memory(data->peak_values);
  bbob2009_free_matrix(data->rotation, problem->number_of_variables);
  bbob2009_free_matrix(data->x_local, problem->number_of_variables);
  bbob2009_free_matrix(data->arr_scales, data->number_of_peaks);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB Gallagher problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_gallagher_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const size_t number_of_peaks,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {

  f_gallagher_data_t *data;
  /* problem_name and best_parameter will be overwritten below */
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Gallagher function",
      f_gallagher_evaluate, f_gallagher_free, dimension, -5.0, 5.0, 0.0);

  const size_t peaks_21 = 21;
  const size_t peaks_101 = 101;

  double fopt;
  size_t i, j, k;
  double maxcondition = 1000.;
  /* maxcondition1 satisfies the old code and the doc but seems wrong in that it is, with very high
   * probability, not the largest condition level!!! */
  double maxcondition1 = 1000.;
  double *arrCondition;
  double fitvalues[2] = { 1.1, 9.1 };
  /* Parameters for generating local optima. In the old code, they are different in f21 and f22 */
  double b, c;
  /* Random permutation */
  f_gallagher_permutation_t *rperm;
  double *random_numbers;

  data = (f_gallagher_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->number_of_peaks = number_of_peaks;
  data->xopt = coco_allocate_vector(dimension);
  data->rotation = bbob2009_allocate_matrix(dimension, dimension);
  data->x_local = bbob2009_allocate_matrix(dimension, number_of_peaks);
  data->arr_scales = bbob2009_allocate_matrix(number_of_peaks, dimension);

  if (number_of_peaks == peaks_101) {
    maxcondition1 = sqrt(maxcondition1);
    b = 10.;
    c = 5.;
  } else if (number_of_peaks == peaks_21) {
    b = 9.8;
    c = 4.9;
  } else {
    coco_error("f_gallagher_bbob_problem_allocate(): '%lu' is a non-supported number of peaks",
        number_of_peaks);
  }
  data->rseed = rseed;
  bbob2009_compute_rotation(data->rotation, rseed, dimension);

  /* Initialize all the data of the inner problem */
  random_numbers = coco_allocate_vector(number_of_peaks * dimension); /* This is large enough for all cases below */
  bbob2009_unif(random_numbers, number_of_peaks - 1, data->rseed);
  rperm = (f_gallagher_permutation_t *) coco_allocate_memory(sizeof(*rperm) * (number_of_peaks - 1));
  for (i = 0; i < number_of_peaks - 1; ++i) {
    rperm[i].value = random_numbers[i];
    rperm[i].index = i;
  }
  qsort(rperm, number_of_peaks - 1, sizeof(*rperm), f_gallagher_compare_doubles);

  /* Random permutation */
  arrCondition = coco_allocate_vector(number_of_peaks);
  arrCondition[0] = maxcondition1;
  data->peak_values = coco_allocate_vector(number_of_peaks);
  data->peak_values[0] = 10;
  for (i = 1; i < number_of_peaks; ++i) {
    arrCondition[i] = pow(maxcondition, (double) (rperm[i - 1].index) / ((double) (number_of_peaks - 2)));
    data->peak_values[i] = (double) (i - 1) / (double) (number_of_peaks - 2) * (fitvalues[1] - fitvalues[0])
        + fitvalues[0];
  }
  coco_free_memory(rperm);

  rperm = (f_gallagher_permutation_t *) coco_allocate_memory(sizeof(*rperm) * dimension);
  for (i = 0; i < number_of_peaks; ++i) {
    bbob2009_unif(random_numbers, dimension, data->rseed + (long) (1000 * i));
    for (j = 0; j < dimension; ++j) {
      rperm[j].value = random_numbers[j];
      rperm[j].index = j;
    }
    qsort(rperm, dimension, sizeof(*rperm), f_gallagher_compare_doubles);
    for (j = 0; j < dimension; ++j) {
      data->arr_scales[i][j] = pow(arrCondition[i],
          ((double) rperm[j].index) / ((double) (dimension - 1)) - 0.5);
    }
  }
  coco_free_memory(rperm);

  bbob2009_unif(random_numbers, dimension * number_of_peaks, data->rseed);
  for (i = 0; i < dimension; ++i) {
    data->xopt[i] = 0.8 * (b * random_numbers[i] - c);
    problem->best_parameter[i] = 0.8 * (b * random_numbers[i] - c);
    for (j = 0; j < number_of_peaks; ++j) {
      data->x_local[i][j] = 0.;
      for (k = 0; k < dimension; ++k) {
        data->x_local[i][j] += data->rotation[i][k] * (b * random_numbers[j * dimension + k] - c);
      }
      if (j == 0) {
        data->x_local[i][j] *= 0.8;
      }
    }
  }
  coco_free_memory(arrCondition);
  coco_free_memory(random_numbers);

  problem->data = data;

  /* Compute best solution */
  f_gallagher_evaluate(problem, problem->best_parameter, problem->best_value);

  fopt = bbob2009_compute_fopt(function, instance);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  return problem;
}

#line 16 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_griewank_rosenbrock.c"
/**
 * @file f_griewank_rosenbrock.c
 * @brief Implementation of the Griewank-Rosenbrock function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 11 "code-experiments/src/f_griewank_rosenbrock.c"
#line 12 "code-experiments/src/f_griewank_rosenbrock.c"
#line 13 "code-experiments/src/f_griewank_rosenbrock.c"
#line 14 "code-experiments/src/f_griewank_rosenbrock.c"
#line 15 "code-experiments/src/f_griewank_rosenbrock.c"
#line 16 "code-experiments/src/f_griewank_rosenbrock.c"

/**
 * @brief Implements the Griewank-Rosenbrock function without connections to any COCO structures.
 */
static double f_griewank_rosenbrock_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double tmp = 0;
  double result;

  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables - 1; ++i) {
    const double c1 = x[i] * x[i] - x[i + 1];
    const double c2 = 1.0 - x[i];
    tmp = 100.0 * c1 * c1 + c2 * c2;
    result += tmp / 4000. - cos(tmp);
  }
  result = 10. + 10. * result / (double) (number_of_variables - 1);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_griewank_rosenbrock_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_griewank_rosenbrock_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Griewank-Rosenbrock problem.
 */
static coco_problem_t *f_griewank_rosenbrock_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Griewank Rosenbrock function",
      f_griewank_rosenbrock_evaluate, NULL, number_of_variables, -5.0, 5.0, 1);
  coco_problem_set_id(problem, "%s_d%02lu", "griewank_rosenbrock", number_of_variables);

  /* Compute best solution */
  f_griewank_rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Griewank-Rosenbrock problem.
 */
static coco_problem_t *f_griewank_rosenbrock_bbob_problem_allocate(const size_t function,
                                                                   const size_t dimension,
                                                                   const size_t instance,
                                                                   const long rseed,
                                                                   const char *problem_id_template,
                                                                   const char *problem_name_template) {
  double fopt;
  coco_problem_t *problem = NULL;
  size_t i, j;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *shift = coco_allocate_vector(dimension);
  double scales, **rot1;

  fopt = bbob2009_compute_fopt(function, instance);
  for (i = 0; i < dimension; ++i) {
    shift[i] = -0.5;
  }

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed, dimension);
  scales = coco_double_max(1., sqrt((double) dimension) / 8.);
  for (i = 0; i < dimension; ++i) {
    for (j = 0; j < dimension; ++j) {
      rot1[i][j] *= scales;
    }
  }

  problem = f_griewank_rosenbrock_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_shift(problem, shift, 0);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = transform_vars_affine(problem, M, b, dimension);

  bbob2009_free_matrix(rot1, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "4-multi-modal");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(shift);
  return problem;
}
#line 17 "code-experiments/src/suite_bbob.c"
#line 18 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_katsuura.c"
/**
 * @file f_katsuura.c
 * @brief Implementation of the Katsuura function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 11 "code-experiments/src/f_katsuura.c"
#line 12 "code-experiments/src/f_katsuura.c"
#line 13 "code-experiments/src/f_katsuura.c"
#line 14 "code-experiments/src/f_katsuura.c"
#line 15 "code-experiments/src/f_katsuura.c"
#line 16 "code-experiments/src/f_katsuura.c"
#line 17 "code-experiments/src/f_katsuura.c"
#line 18 "code-experiments/src/f_katsuura.c"

/**
 * @brief Implements the Katsuura function without connections to any COCO structures.
 */
static double f_katsuura_raw(const double *x, const size_t number_of_variables) {

  size_t i, j;
  double tmp, tmp2;
  double result;

  /* Computation core */
  result = 1.0;
  for (i = 0; i < number_of_variables; ++i) {
    tmp = 0;
    for (j = 1; j < 33; ++j) {
      tmp2 = pow(2., (double) j);
      tmp += fabs(tmp2 * x[i] - coco_double_round(tmp2 * x[i])) / tmp2;
    }
    tmp = 1.0 + ((double) (long) i + 1) * tmp;
    result *= tmp;
  }
  result = 10. / ((double) number_of_variables) / ((double) number_of_variables)
      * (-1. + pow(result, 10. / pow((double) number_of_variables, 1.2)));

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_katsuura_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_katsuura_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Katsuura problem.
 */
static coco_problem_t *f_katsuura_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Katsuura function",
      f_katsuura_evaluate, NULL, number_of_variables, -5.0, 5.0, 1);
  coco_problem_set_id(problem, "%s_d%02lu", "katsuura", number_of_variables);

  /* Compute best solution */
  f_katsuura_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Katsuura problem.
 */
static coco_problem_t *f_katsuura_bbob_problem_allocate(const size_t function,
                                                        const size_t dimension,
                                                        const size_t instance,
                                                        const long rseed,
                                                        const char *problem_id_template,
                                                        const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j, k;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  const double penalty_factor = 1.0;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  rot2 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(rot2, rseed, dimension);

  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      for (k = 0; k < dimension; ++k) {
        double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(sqrt(100), exponent) * rot2[k][j];
      }
    }
  }

  problem = f_katsuura_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_penalize(problem, penalty_factor);

  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
#line 19 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_linear_slope.c"
/**
 * @file f_linear_slope.c
 * @brief Implementation of the linear slope function and problem.
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>

#line 11 "code-experiments/src/f_linear_slope.c"
#line 12 "code-experiments/src/f_linear_slope.c"
#line 13 "code-experiments/src/f_linear_slope.c"
#line 14 "code-experiments/src/f_linear_slope.c"

/**
 * @brief Implements the linear slope function without connections to any COCO structures.
 */
static double f_linear_slope_raw(const double *x,
                                 const size_t number_of_variables,
                                 const double *best_parameter) {

  static const double alpha = 100.0;
  size_t i;
  double result = 0.0;

  for (i = 0; i < number_of_variables; ++i) {
    double base, exponent, si;

    base = sqrt(alpha);
    exponent = (double) (long) i / ((double) (long) number_of_variables - 1);
    if (best_parameter[i] > 0.0) {
      si = pow(base, exponent);
    } else {
      si = -pow(base, exponent);
    }
    result += 5.0 * fabs(si) - si * x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_linear_slope_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_linear_slope_raw(x, problem->number_of_variables, problem->best_parameter);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic linear slope problem.
 */
static coco_problem_t *f_linear_slope_allocate(const size_t number_of_variables, const double *best_parameter) {

  size_t i;
  /* best_parameter will be overwritten below */
  coco_problem_t *problem = coco_problem_allocate_from_scalars("linear slope function",
      f_linear_slope_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "linear_slope", number_of_variables);

  /* Compute best solution */
  for (i = 0; i < number_of_variables; ++i) {
    if (best_parameter[i] < 0.0) {
      problem->best_parameter[i] = problem->smallest_values_of_interest[i];
    } else {
      problem->best_parameter[i] = problem->largest_values_of_interest[i];
    }
  }
  f_linear_slope_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB linear slope problem.
 */
static coco_problem_t *f_linear_slope_bbob_problem_allocate(const size_t function,
                                                            const size_t dimension,
                                                            const size_t instance,
                                                            const long rseed,
                                                            const char *problem_id_template,
                                                            const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  problem = f_linear_slope_allocate(dimension, xopt);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}
#line 20 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_lunacek_bi_rastrigin.c"
/**
 * @file f_lunacek_bi_rastrigin.c
 * @brief Implementation of the Lunacek bi-Rastrigin function and problem.
 */

#include <assert.h>
#include <math.h>

#line 10 "code-experiments/src/f_lunacek_bi_rastrigin.c"
#line 11 "code-experiments/src/f_lunacek_bi_rastrigin.c"
#line 12 "code-experiments/src/f_lunacek_bi_rastrigin.c"
#line 13 "code-experiments/src/f_lunacek_bi_rastrigin.c"

/**
 * @brief Data type for the Lunacek bi-Rastrigin problem.
 */
typedef struct {
  double *x_hat, *z;
  double *xopt, fopt;
  double **rot1, **rot2;
  long rseed;
  coco_problem_free_function_t old_free_problem;
} f_lunacek_bi_rastrigin_data_t;

/**
 * @brief Implements the Lunacek bi-Rastrigin function without connections to any COCO structures.
 */
static double f_lunacek_bi_rastrigin_raw(const double *x,
                                         const size_t number_of_variables,
                                         f_lunacek_bi_rastrigin_data_t *data) {
  double result;
  static const double condition = 100.;
  size_t i, j;
  double penalty = 0.0;
  static const double mu0 = 2.5;
  static const double d = 1.;
  const double s = 1. - 0.5 / (sqrt((double) (number_of_variables + 20)) - 4.1);
  const double mu1 = -sqrt((mu0 * mu0 - d) / s);
  double *tmpvect, sum1 = 0., sum2 = 0., sum3 = 0.;

  assert(number_of_variables > 1);

  for (i = 0; i < number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* x_hat */
  for (i = 0; i < number_of_variables; ++i) {
    data->x_hat[i] = 2. * x[i];
    if (data->xopt[i] < 0.) {
      data->x_hat[i] *= -1.;
    }
  }

  tmpvect = coco_allocate_vector(number_of_variables);
  /* affine transformation */
  for (i = 0; i < number_of_variables; ++i) {
    double c1;
    tmpvect[i] = 0.0;
    c1 = pow(sqrt(condition), ((double) i) / (double) (number_of_variables - 1));
    for (j = 0; j < number_of_variables; ++j) {
      tmpvect[i] += c1 * data->rot2[i][j] * (data->x_hat[j] - mu0);
    }
  }
  for (i = 0; i < number_of_variables; ++i) {
    data->z[i] = 0;
    for (j = 0; j < number_of_variables; ++j) {
      data->z[i] += data->rot1[i][j] * tmpvect[j];
    }
  }
  /* Computation core */
  for (i = 0; i < number_of_variables; ++i) {
    sum1 += (data->x_hat[i] - mu0) * (data->x_hat[i] - mu0);
    sum2 += (data->x_hat[i] - mu1) * (data->x_hat[i] - mu1);
    sum3 += cos(2 * coco_pi * data->z[i]);
  }
  result = coco_double_min(sum1, d * (double) number_of_variables + s * sum2)
      + 10. * ((double) number_of_variables - sum3) + 1e4 * penalty;
  coco_free_memory(tmpvect);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_lunacek_bi_rastrigin_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_lunacek_bi_rastrigin_raw(x, problem->number_of_variables, (f_lunacek_bi_rastrigin_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the Lunacek bi-Rastrigin data object.
 */
static void f_lunacek_bi_rastrigin_free(coco_problem_t *problem) {
  f_lunacek_bi_rastrigin_data_t *data;
  data = (f_lunacek_bi_rastrigin_data_t *) problem->data;
  coco_free_memory(data->x_hat);
  coco_free_memory(data->z);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, problem->number_of_variables);
  bbob2009_free_matrix(data->rot2, problem->number_of_variables);

  /* Let the generic free problem code deal with all of the
   * coco_problem_t fields.
   */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB Lunacek bi-Rastrigin problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_lunacek_bi_rastrigin_bbob_problem_allocate(const size_t function,
                                                                    const size_t dimension,
                                                                    const size_t instance,
                                                                    const long rseed,
                                                                    const char *problem_id_template,
                                                                    const char *problem_name_template) {

  f_lunacek_bi_rastrigin_data_t *data;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Lunacek's bi-Rastrigin function",
      f_lunacek_bi_rastrigin_evaluate, f_lunacek_bi_rastrigin_free, dimension, -5.0, 5.0, 0.0);

  const double mu0 = 2.5;

  double fopt, *tmpvect;
  size_t i;

  data = (f_lunacek_bi_rastrigin_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x_hat = coco_allocate_vector(dimension);
  data->z = coco_allocate_vector(dimension);
  data->xopt = coco_allocate_vector(dimension);
  data->rot1 = bbob2009_allocate_matrix(dimension, dimension);
  data->rot2 = bbob2009_allocate_matrix(dimension, dimension);
  data->rseed = rseed;

  data->fopt = bbob2009_compute_fopt(24, instance);
  bbob2009_compute_xopt(data->xopt, rseed, dimension);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(data->rot2, rseed, dimension);

  problem->data = data;

  /* Compute best solution */
  tmpvect = coco_allocate_vector(dimension);
  bbob2009_gauss(tmpvect, dimension, rseed);
  for (i = 0; i < dimension; ++i) {
    data->xopt[i] = 0.5 * mu0;
    if (tmpvect[i] < 0.0) {
      data->xopt[i] *= -1.0;
    }
    problem->best_parameter[i] = data->xopt[i];
  }
  coco_free_memory(tmpvect);
  f_lunacek_bi_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);

  fopt = bbob2009_compute_fopt(function, instance);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  return problem;
}
#line 21 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_rastrigin.c"
/**
 * @file f_rastrigin.c
 * @brief Implementation of the Rastrigin function and problem.
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>

#line 11 "code-experiments/src/f_rastrigin.c"
#line 12 "code-experiments/src/f_rastrigin.c"
#line 13 "code-experiments/src/f_rastrigin.c"
#line 1 "code-experiments/src/transform_vars_conditioning.c"
/**
 * @file transform_vars_conditioning.c
 * @brief Implementation of conditioning decision values.
 */

#include <math.h>
#include <assert.h>

#line 10 "code-experiments/src/transform_vars_conditioning.c"
#line 11 "code-experiments/src/transform_vars_conditioning.c"

/**
 * @brief Data type for transform_vars_conditioning.
 */
typedef struct {
  double *x;
  double alpha;
} transform_vars_conditioning_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_conditioning_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_conditioning_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_conditioning_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    /* OME: We could precalculate the scaling coefficients if we
     * really wanted to.
     */
    data->x[i] = pow(data->alpha, 0.5 * (double) (long) i / ((double) (long) problem->number_of_variables - 1.0))
        * x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

static void transform_vars_conditioning_free(void *thing) {
  transform_vars_conditioning_data_t *data = (transform_vars_conditioning_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_conditioning(coco_problem_t *inner_problem, const double alpha) {
  transform_vars_conditioning_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_conditioning_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->alpha = alpha;
  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_conditioning_free, "transform_vars_conditioning");
  problem->evaluate_function = transform_vars_conditioning_evaluate;
  return problem;
}
#line 14 "code-experiments/src/f_rastrigin.c"
#line 15 "code-experiments/src/f_rastrigin.c"
#line 16 "code-experiments/src/f_rastrigin.c"
#line 17 "code-experiments/src/f_rastrigin.c"
#line 18 "code-experiments/src/f_rastrigin.c"
#line 19 "code-experiments/src/f_rastrigin.c"

/**
 * @brief Implements the Rastrigin function without connections to any COCO structures.
 */
static double f_rastrigin_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double sum1 = 0.0, sum2 = 0.0;

  for (i = 0; i < number_of_variables; ++i) {
    sum1 += cos(coco_two_pi * x[i]);
    sum2 += x[i] * x[i];
  }
  result = 10.0 * ((double) (long) number_of_variables - sum1) + sum2;

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_rastrigin_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_rastrigin_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Rastrigin problem.
 */
static coco_problem_t *f_rastrigin_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Rastrigin function",
      f_rastrigin_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "rastrigin", number_of_variables);

  /* Compute best solution */
  f_rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Rastrigin problem.
 */
static coco_problem_t *f_rastrigin_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  problem = f_rastrigin_allocate(dimension);
  problem = transform_vars_conditioning(problem, 10.0);
  problem = transform_vars_asymmetric(problem, 0.2);
  problem = transform_vars_oscillate(problem);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the BBOB rotated Rastrigin problem.
 */
static coco_problem_t *f_rastrigin_rotated_bbob_problem_allocate(const size_t function,
                                                                 const size_t dimension,
                                                                 const size_t instance,
                                                                 const long rseed,
                                                                 const char *problem_id_template,
                                                                 const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j, k;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  rot2 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(rot2, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      for (k = 0; k < dimension; ++k) {
        double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
      }
    }
  }

  problem = f_rastrigin_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_asymmetric(problem, 0.2);
  problem = transform_vars_oscillate(problem);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "4-multi-modal");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}

#line 22 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_rosenbrock.c"
/**
 * @file f_rosenbrock.c
 * @brief Implementation of the Rosenbrock function and problem.
 */

#include <assert.h>

#line 9 "code-experiments/src/f_rosenbrock.c"
#line 10 "code-experiments/src/f_rosenbrock.c"
#line 11 "code-experiments/src/f_rosenbrock.c"
#line 12 "code-experiments/src/f_rosenbrock.c"
#line 1 "code-experiments/src/transform_vars_scale.c"
/**
 * @file transform_vars_scale.c
 * @brief Implementation of scaling decision values by a given factor.
 */

#include <assert.h>

#line 9 "code-experiments/src/transform_vars_scale.c"
#line 10 "code-experiments/src/transform_vars_scale.c"

/**
 * @brief Data type for transform_vars_scale.
 */
typedef struct {
  double factor;
  double *x;
} transform_vars_scale_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_scale_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_scale_data_t *data;
  coco_problem_t *inner_problem;
  data = (transform_vars_scale_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  do {
    const double factor = data->factor;

    for (i = 0; i < problem->number_of_variables; ++i) {
      data->x[i] = factor * x[i];
    }
    coco_evaluate_function(inner_problem, data->x, y);
    assert(y[0] + 1e-13 >= problem->best_value[0]);
  } while (0);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_scale_free(void *thing) {
  transform_vars_scale_data_t *data = (transform_vars_scale_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_scale(coco_problem_t *inner_problem, const double factor) {
  transform_vars_scale_data_t *data;
  coco_problem_t *problem;

  data = (transform_vars_scale_data_t *) coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_scale_free, "transform_vars_scale");
  problem->evaluate_function = transform_vars_scale_evaluate;
  return problem;
}
#line 13 "code-experiments/src/f_rosenbrock.c"
#line 14 "code-experiments/src/f_rosenbrock.c"
#line 15 "code-experiments/src/f_rosenbrock.c"

/**
 * @brief Implements the Rosenbrock function without connections to any COCO structures.
 */
static double f_rosenbrock_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double s1 = 0.0, s2 = 0.0, tmp;

  assert(number_of_variables > 1);

  for (i = 0; i < number_of_variables - 1; ++i) {
    tmp = (x[i] * x[i] - x[i + 1]);
    s1 += tmp * tmp;
    tmp = (x[i] - 1.0);
    s2 += tmp * tmp;
  }
  result = 100.0 * s1 + s2;

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_rosenbrock_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_rosenbrock_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Rosenbrock problem.
 */
static coco_problem_t *f_rosenbrock_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Rosenbrock function",
      f_rosenbrock_evaluate, NULL, number_of_variables, -5.0, 5.0, 1.0);
  coco_problem_set_id(problem, "%s_d%02lu", "rosenbrock", number_of_variables);

  /* Compute best solution */
  f_rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Rosenbrock problem.
 */
static coco_problem_t *f_rosenbrock_bbob_problem_allocate(const size_t function,
                                                          const size_t dimension,
                                                          const size_t instance,
                                                          const long rseed,
                                                          const char *problem_id_template,
                                                          const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i;
  double *minus_one, factor;

  minus_one = coco_allocate_vector(dimension);
  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    minus_one[i] = -1.0;
    xopt[i] *= 0.75;
  }
  fopt = bbob2009_compute_fopt(function, instance);
  factor = coco_double_max(1.0, sqrt((double) dimension) / 8.0);

  problem = f_rosenbrock_allocate(dimension);
  problem = transform_vars_shift(problem, minus_one, 0);
  problem = transform_vars_scale(problem, factor);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(minus_one);
  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the BBOB rotated Rosenbrock problem.
 */
static coco_problem_t *f_rosenbrock_rotated_bbob_problem_allocate(const size_t function,
                                                                  const size_t dimension,
                                                                  const size_t instance,
                                                                  const long rseed,
                                                                  const char *problem_id_template,
                                                                  const char *problem_name_template) {

  double fopt;
  coco_problem_t *problem = NULL;
  size_t row, column;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, factor;

  fopt = bbob2009_compute_fopt(function, instance);
  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed, dimension);

  factor = coco_double_max(1.0, sqrt((double) dimension) / 8.0);
  /* Compute affine transformation */
  for (row = 0; row < dimension; ++row) {
    current_row = M + row * dimension;
    for (column = 0; column < dimension; ++column) {
      current_row[column] = rot1[row][column];
      if (row == column)
        current_row[column] *= factor;
    }
    b[row] = 0.5;
  }
  bbob2009_free_matrix(rot1, dimension);

  problem = f_rosenbrock_allocate(dimension);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  coco_free_memory(M);
  coco_free_memory(b);
  return problem;
}
#line 23 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_schaffers.c"
/**
 * @file f_schaffers.c
 * @brief Implementation of the Schaffer's F7 function and problem, transformations not implemented for the
 * moment.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 12 "code-experiments/src/f_schaffers.c"
#line 13 "code-experiments/src/f_schaffers.c"
#line 14 "code-experiments/src/f_schaffers.c"
#line 15 "code-experiments/src/f_schaffers.c"
#line 16 "code-experiments/src/f_schaffers.c"
#line 17 "code-experiments/src/f_schaffers.c"
#line 18 "code-experiments/src/f_schaffers.c"
#line 19 "code-experiments/src/f_schaffers.c"

/**
 * @brief Implements the Schaffer's F7 function without connections to any COCO structures.
 */
static double f_schaffers_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;

  assert(number_of_variables > 1);

  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables - 1; ++i) {
    const double tmp = x[i] * x[i] + x[i + 1] * x[i + 1];
    result += pow(tmp, 0.25) * (1.0 + pow(sin(50.0 * pow(tmp, 0.1)), 2.0));
  }
  result = pow(result / ((double) (long) number_of_variables - 1.0), 2.0);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_schaffers_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_schaffers_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Schaffer's F7 problem.
 */
static coco_problem_t *f_schaffers_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Schaffer's function",
      f_schaffers_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "schaffers", number_of_variables);

  /* Compute best solution */
  f_schaffers_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Schaffer's F7 problem.
 */
static coco_problem_t *f_schaffers_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const double conditioning,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  const double penalty_factor = 10.0;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  rot2 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(rot2, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      double exponent = 1.0 * (int) i / ((double) (long) dimension - 1.0);
      current_row[j] = rot2[i][j] * pow(sqrt(conditioning), exponent);
    }
  }

  problem = f_schaffers_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_asymmetric(problem, 0.5);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_penalize(problem, penalty_factor);

  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "4-multi-modal");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
#line 24 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_schwefel.c"
/**
 * @file f_schwefel.c
 * @brief Implementation of the Schwefel function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 11 "code-experiments/src/f_schwefel.c"
#line 12 "code-experiments/src/f_schwefel.c"
#line 13 "code-experiments/src/f_schwefel.c"
#line 14 "code-experiments/src/f_schwefel.c"
#line 15 "code-experiments/src/f_schwefel.c"
#line 16 "code-experiments/src/f_schwefel.c"
#line 17 "code-experiments/src/f_schwefel.c"
#line 1 "code-experiments/src/transform_vars_z_hat.c"
/**
 * @file transform_vars_z_hat.c
 * @brief Implementation of the z^hat transformation of decision values for the BBOB Schwefel problem.
 */

#include <assert.h>

#line 9 "code-experiments/src/transform_vars_z_hat.c"
#line 10 "code-experiments/src/transform_vars_z_hat.c"

/**
 * @brief Data type for transform_vars_z_hat.
 */
typedef struct {
  double *xopt;
  double *z;
  coco_problem_free_function_t old_free_problem;
} transform_vars_z_hat_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_z_hat_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_z_hat_data_t *data;
  coco_problem_t *inner_problem;

  data = (transform_vars_z_hat_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  data->z[0] = x[0];

  for (i = 1; i < problem->number_of_variables; ++i) {
    data->z[i] = x[i] + 0.25 * (x[i - 1] - 2.0 * fabs(data->xopt[i - 1]));
  }
  coco_evaluate_function(inner_problem, data->z, y);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_z_hat_free(void *thing) {
  transform_vars_z_hat_data_t *data = (transform_vars_z_hat_data_t *) thing;
  coco_free_memory(data->xopt);
  coco_free_memory(data->z);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_z_hat(coco_problem_t *inner_problem, const double *xopt) {
  transform_vars_z_hat_data_t *data;
  coco_problem_t *problem;

  data = (transform_vars_z_hat_data_t *) coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, inner_problem->number_of_variables);
  data->z = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_z_hat_free, "transform_vars_z_hat");
  problem->evaluate_function = transform_vars_z_hat_evaluate;
  return problem;
}
#line 18 "code-experiments/src/f_schwefel.c"
#line 1 "code-experiments/src/transform_vars_x_hat.c"
/**
 * @file transform_vars_x_hat.c
 * @brief Implementation of multiplying the decision values by the vector 1+-.
 */

#include <assert.h>

#line 9 "code-experiments/src/transform_vars_x_hat.c"
#line 10 "code-experiments/src/transform_vars_x_hat.c"
#line 11 "code-experiments/src/transform_vars_x_hat.c"

/**
 * @brief Data type for transform_vars_x_hat.
 */
typedef struct {
  long seed;
  double *x;
  coco_problem_free_function_t old_free_problem;
} transform_vars_x_hat_data_t;

/**
 * @brief Evaluates the transformation.
 */
static void transform_vars_x_hat_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_x_hat_data_t *data;
  coco_problem_t *inner_problem;
  data = (transform_vars_x_hat_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  do {
    bbob2009_unif(data->x, problem->number_of_variables, data->seed);

    for (i = 0; i < problem->number_of_variables; ++i) {
      if (data->x[i] - 0.5 < 0.0) {
        data->x[i] = -x[i];
      } else {
        data->x[i] = x[i];
      }
    }
    coco_evaluate_function(inner_problem, data->x, y);
    assert(y[0] + 1e-13 >= problem->best_value[0]);
  } while (0);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_x_hat_free(void *thing) {
  transform_vars_x_hat_data_t *data = (transform_vars_x_hat_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_x_hat(coco_problem_t *inner_problem, const long seed) {
  transform_vars_x_hat_data_t *data;
  coco_problem_t *problem;
  size_t i;

  data = (transform_vars_x_hat_data_t *) coco_allocate_memory(sizeof(*data));
  data->seed = seed;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_x_hat_free, "transform_vars_x_hat");
  problem->evaluate_function = transform_vars_x_hat_evaluate;
  /* Dirty way of setting the best parameter of the transformed f_schwefel... */
  bbob2009_unif(data->x, problem->number_of_variables, data->seed);
  for (i = 0; i < problem->number_of_variables; ++i) {
      if (data->x[i] - 0.5 < 0.0) {
          problem->best_parameter[i] = -0.5 * 4.2096874633;
      } else {
          problem->best_parameter[i] = 0.5 * 4.2096874633;
      }
  }
  return problem;
}
#line 19 "code-experiments/src/f_schwefel.c"

/**
 * @brief Implements the Schwefel function without connections to any COCO structures.
 */
static double f_schwefel_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double penalty, sum;

  /* Boundary handling*/
  penalty = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    const double tmp = fabs(x[i]) - 500.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* Computation core */
  sum = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    sum += x[i] * sin(sqrt(fabs(x[i])));
  }
  result = 0.01 * (penalty + 418.9828872724339 - sum / (double) number_of_variables);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_schwefel_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_schwefel_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Schwefel problem.
 */
static coco_problem_t *f_schwefel_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Schwefel function",
      f_schwefel_evaluate, NULL, number_of_variables, -5.0, 5.0, 420.96874633);
  coco_problem_set_id(problem, "%s_d%02lu", "schwefel", number_of_variables);

  /* Compute best solution: best_parameter[i] = 200 * fabs(xopt[i]) */
  f_schwefel_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Schwefel problem.
 */
static coco_problem_t *f_schwefel_bbob_problem_allocate(const size_t function,
                                                        const size_t dimension,
                                                        const size_t instance,
                                                        const long rseed,
                                                        const char *problem_id_template,
                                                        const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j;

  const double condition = 10.;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row;

  double *tmp1 = coco_allocate_vector(dimension);
  double *tmp2 = coco_allocate_vector(dimension);

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_unif(tmp1, dimension, rseed);
  for (i = 0; i < dimension; ++i) {
    xopt[i] = 0.5 * 4.2096874637;
    if (tmp1[i] - 0.5 < 0) {
      xopt[i] *= -1;
    }
  }

  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      if (i == j) {
        double exponent = 1.0 * (int) i / ((double) (long) dimension - 1);
        current_row[j] = pow(sqrt(condition), exponent);
      }
    }
  }

  for (i = 0; i < dimension; ++i) {
    tmp1[i] = -2 * fabs(xopt[i]);
    tmp2[i] = 2 * fabs(xopt[i]);
  }

  problem = f_schwefel_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_scale(problem, 100);
  problem = transform_vars_shift(problem, tmp1, 0);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, tmp2, 0);
  problem = transform_vars_z_hat(problem, xopt);
  problem = transform_vars_scale(problem, 2);
  problem = transform_vars_x_hat(problem, rseed);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(tmp1);
  coco_free_memory(tmp2);
  coco_free_memory(xopt);
  return problem;
}
#line 25 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_sharp_ridge.c"
/**
 * @file f_sharp_ridge.c
 * @brief Implementation of the sharp ridge function and problem.
 */

#include <assert.h>
#include <math.h>

#line 10 "code-experiments/src/f_sharp_ridge.c"
#line 11 "code-experiments/src/f_sharp_ridge.c"
#line 12 "code-experiments/src/f_sharp_ridge.c"
#line 13 "code-experiments/src/f_sharp_ridge.c"
#line 14 "code-experiments/src/f_sharp_ridge.c"
#line 15 "code-experiments/src/f_sharp_ridge.c"

/**
 * @brief Implements the sharp ridge function without connections to any COCO structures.
 */
static double f_sharp_ridge_raw(const double *x, const size_t number_of_variables) {

  static const double alpha = 100.0;
  size_t i = 0;
  double result;

  assert(number_of_variables > 1);

  result = 0.0;
  for (i = 1; i < number_of_variables; ++i) {
    result += x[i] * x[i];
  }
  result = alpha * sqrt(result) + x[0] * x[0];

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_sharp_ridge_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_sharp_ridge_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic sharp ridge problem.
 */
static coco_problem_t *f_sharp_ridge_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("sharp ridge function",
      f_sharp_ridge_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "sharp_ridge", number_of_variables);

  /* Compute best solution */
  f_sharp_ridge_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB sharp ridge problem.
 */
static coco_problem_t *f_sharp_ridge_bbob_problem_allocate(const size_t function,
                                                           const size_t dimension,
                                                           const size_t instance,
                                                           const long rseed,
                                                           const char *problem_id_template,
                                                           const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j, k;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  rot2 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(rot2, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      for (k = 0; k < dimension; ++k) {
        double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
      }
    }
  }
  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);
  problem = f_sharp_ridge_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "3-ill-conditioned");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}
#line 26 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_sphere.c"
/**
 * @file f_sphere.c
 * @brief Implementation of the sphere function and problem.
 */

#include <stdio.h>
#include <assert.h>

#line 10 "code-experiments/src/f_sphere.c"
#line 11 "code-experiments/src/f_sphere.c"
#line 12 "code-experiments/src/f_sphere.c"
#line 13 "code-experiments/src/f_sphere.c"
#line 14 "code-experiments/src/f_sphere.c"

/**
 * @brief Implements the sphere function without connections to any COCO structures.
 */
static double f_sphere_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    result += x[i] * x[i];
  }

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_sphere_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_sphere_raw(x, problem->number_of_variables);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic sphere problem.
 */
static coco_problem_t *f_sphere_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("sphere function",
      f_sphere_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  coco_problem_set_id(problem, "%s_d%02lu", "sphere", number_of_variables);

  /* Compute best solution */
  f_sphere_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

/**
 * @brief Creates the BBOB sphere problem.
 */
static coco_problem_t *f_sphere_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const long rseed,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  problem = f_sphere_allocate(dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

#line 27 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_step_ellipsoid.c"
/**
 * @file f_step_ellipsoid.c
 * @brief Implementation of the step ellipsoid function and problem.
 *
 * The BBOB step ellipsoid function intertwines the variable and objective transformations in such a way
 * that it is hard to devise a composition of generic transformations to implement it. In the end one would
 * have to implement several custom transformations which would be used solely by this problem. Therefore
 * we opt to implement it as a monolithic function instead.
 *
 * TODO: It would be nice to have a generic step ellipsoid function to complement this one.
 */
#include <assert.h>

#line 15 "code-experiments/src/f_step_ellipsoid.c"
#line 16 "code-experiments/src/f_step_ellipsoid.c"
#line 17 "code-experiments/src/f_step_ellipsoid.c"
#line 18 "code-experiments/src/f_step_ellipsoid.c"

/**
 * @brief Data type for the step ellipsoid problem.
 */
typedef struct {
  double *x, *xx;
  double *xopt, fopt;
  double **rot1, **rot2;
} f_step_ellipsoid_data_t;

/**
 * @brief Implements the step ellipsoid function without connections to any COCO structures.
 */
static double f_step_ellipsoid_raw(const double *x, const size_t number_of_variables, f_step_ellipsoid_data_t *data) {

  static const double condition = 100;
  static const double alpha = 10.0;
  size_t i, j;
  double penalty = 0.0, x1;
  double result;

  assert(number_of_variables > 1);

  for (i = 0; i < number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  for (i = 0; i < number_of_variables; ++i) {
    double c1;
    data->x[i] = 0.0;
    c1 = sqrt(pow(condition / 10., (double) i / (double) (number_of_variables - 1)));
    for (j = 0; j < number_of_variables; ++j) {
      data->x[i] += c1 * data->rot2[i][j] * (x[j] - data->xopt[j]);
    }
  }
  x1 = data->x[0];

  for (i = 0; i < number_of_variables; ++i) {
    if (fabs(data->x[i]) > 0.5)
      data->x[i] = coco_double_round(data->x[i]);
    else
      data->x[i] = coco_double_round(alpha * data->x[i]) / alpha;
  }

  for (i = 0; i < number_of_variables; ++i) {
    data->xx[i] = 0.0;
    for (j = 0; j < number_of_variables; ++j) {
      data->xx[i] += data->rot1[i][j] * data->x[j];
    }
  }

  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    double exponent;
    exponent = (double) (long) i / ((double) (long) number_of_variables - 1.0);
    result += pow(condition, exponent) * data->xx[i] * data->xx[i];
    ;
  }
  result = 0.1 * coco_double_max(fabs(x1) * 1.0e-4, result) + penalty + data->fopt;

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_step_ellipsoid_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_step_ellipsoid_raw(x, problem->number_of_variables, (f_step_ellipsoid_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Frees the step ellipsoid data object.
 */
static void f_step_ellipsoid_free(coco_problem_t *problem) {
  f_step_ellipsoid_data_t *data;
  data = (f_step_ellipsoid_data_t *) problem->data;
  coco_free_memory(data->x);
  coco_free_memory(data->xx);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, problem->number_of_variables);
  bbob2009_free_matrix(data->rot2, problem->number_of_variables);
  /* Let the generic free problem code deal with all of the coco_problem_t fields */
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Creates the BBOB step ellipsoid problem.
 *
 * @note There is no separate basic allocate function.
 */
static coco_problem_t *f_step_ellipsoid_bbob_problem_allocate(const size_t function,
                                                              const size_t dimension,
                                                              const size_t instance,
                                                              const long rseed,
                                                              const char *problem_id_template,
                                                              const char *problem_name_template) {

  f_step_ellipsoid_data_t *data;
  size_t i;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("step ellipsoid function",
      f_step_ellipsoid_evaluate, f_step_ellipsoid_free, dimension, -5.0, 5.0, 0);

  data = (f_step_ellipsoid_data_t *) coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x = coco_allocate_vector(dimension);
  data->xx = coco_allocate_vector(dimension);
  data->xopt = coco_allocate_vector(dimension);
  data->rot1 = bbob2009_allocate_matrix(dimension, dimension);
  data->rot2 = bbob2009_allocate_matrix(dimension, dimension);

  data->fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(data->xopt, rseed, dimension);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(data->rot2, rseed, dimension);

  problem->data = data;
  
  /* Compute best solution
   *
   * OME: Dirty hack for now because I did not want to invert the
   * transformations to find the best_parameter :/
   */
  for (i = 0; i < problem->number_of_variables; i++) {
      problem->best_parameter[i] = data->xopt[i];
  }
  problem->best_value[0] = data->fopt;

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "2-moderate");

  return problem;
}
#line 28 "code-experiments/src/suite_bbob.c"
#line 1 "code-experiments/src/f_weierstrass.c"
/**
 * @file f_weierstrass.c
 * @brief Implementation of the Weierstrass function and problem.
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 11 "code-experiments/src/f_weierstrass.c"
#line 12 "code-experiments/src/f_weierstrass.c"
#line 13 "code-experiments/src/f_weierstrass.c"
#line 14 "code-experiments/src/f_weierstrass.c"
#line 15 "code-experiments/src/f_weierstrass.c"
#line 16 "code-experiments/src/f_weierstrass.c"
#line 17 "code-experiments/src/f_weierstrass.c"
#line 18 "code-experiments/src/f_weierstrass.c"

/** @brief Number of summands in the Weierstrass problem. */
#define F_WEIERSTRASS_SUMMANDS 12

/**
 * @brief Data type for the Weierstrass problem.
 */
typedef struct {
  double f0;
  double ak[F_WEIERSTRASS_SUMMANDS];
  double bk[F_WEIERSTRASS_SUMMANDS];
} f_weierstrass_data_t;

/**
 * @brief Implements the Weierstrass function without connections to any COCO structures.
 */
static double f_weierstrass_raw(const double *x, const size_t number_of_variables, f_weierstrass_data_t *data) {

  size_t i, j;
  double result;

  result = 0.0;
  for (i = 0; i < number_of_variables; ++i) {
    for (j = 0; j < F_WEIERSTRASS_SUMMANDS; ++j) {
      result += cos(2 * coco_pi * (x[i] + 0.5) * data->bk[j]) * data->ak[j];
    }
  }
  result = 10.0 * pow(result / (double) (long) number_of_variables - data->f0, 3.0);

  return result;
}

/**
 * @brief Uses the raw function to evaluate the COCO problem.
 */
static void f_weierstrass_evaluate(coco_problem_t *problem, const double *x, double *y) {
  assert(problem->number_of_objectives == 1);
  y[0] = f_weierstrass_raw(x, problem->number_of_variables, (f_weierstrass_data_t *) problem->data);
  assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Allocates the basic Weierstrass problem.
 */
static coco_problem_t *f_weierstrass_allocate(const size_t number_of_variables) {

  f_weierstrass_data_t *data;
  size_t i;
  double *non_unique_best_value;
  coco_problem_t *problem = coco_problem_allocate_from_scalars("Weierstrass function",
      f_weierstrass_evaluate, NULL, number_of_variables, -5.0, 5.0, NAN);
  coco_problem_set_id(problem, "%s_d%02lu", "weierstrass", number_of_variables);

  data = (f_weierstrass_data_t *) coco_allocate_memory(sizeof(*data));
  data->f0 = 0.0;
  for (i = 0; i < F_WEIERSTRASS_SUMMANDS; ++i) {
    data->ak[i] = pow(0.5, (double) i);
    data->bk[i] = pow(3., (double) i);
    data->f0 += data->ak[i] * cos(2 * coco_pi * data->bk[i] * 0.5);
  }
  problem->data = data;

  /* Compute best solution */
  non_unique_best_value = coco_allocate_vector(number_of_variables);
  for (i = 0; i < number_of_variables; i++)
    non_unique_best_value[i] = 0.0;
  f_weierstrass_evaluate(problem, non_unique_best_value, problem->best_value);
  coco_free_memory(non_unique_best_value);
  return problem;
}

/**
 * @brief Creates the BBOB Weierstrass problem.
 */
static coco_problem_t *f_weierstrass_bbob_problem_allocate(const size_t function,
                                                           const size_t dimension,
                                                           const size_t instance,
                                                           const long rseed,
                                                           const char *problem_id_template,
                                                           const char *problem_name_template) {
  double *xopt, fopt;
  coco_problem_t *problem = NULL;
  size_t i, j, k;
  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double *current_row, **rot1, **rot2;

  const double condition = 100.0;
  const double penalty_factor = 10.0 / (double) dimension;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  rot1 = bbob2009_allocate_matrix(dimension, dimension);
  rot2 = bbob2009_allocate_matrix(dimension, dimension);
  bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);
  bbob2009_compute_rotation(rot2, rseed, dimension);
  for (i = 0; i < dimension; ++i) {
    b[i] = 0.0;
    current_row = M + i * dimension;
    for (j = 0; j < dimension; ++j) {
      current_row[j] = 0.0;
      for (k = 0; k < dimension; ++k) {
        const double base = 1.0 / sqrt(condition);
        const double exponent = 1.0 * (int) k / ((double) (long) dimension - 1.0);
        current_row[j] += rot1[i][k] * pow(base, exponent) * rot2[k][j];
      }
    }
  }

  problem = f_weierstrass_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_oscillate(problem);
  bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
  problem = transform_vars_affine(problem, M, b, dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_penalize(problem, penalty_factor);

  bbob2009_free_matrix(rot1, dimension);
  bbob2009_free_matrix(rot2, dimension);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "4-multi-modal");

  coco_free_memory(M);
  coco_free_memory(b);
  coco_free_memory(xopt);
  return problem;
}

#undef F_WEIERSTRASS_SUMMANDS
#line 29 "code-experiments/src/suite_bbob.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob suite.
 */
static coco_suite_t *suite_bbob_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("bbob", 24, 6, dimensions, "year: 2016");

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob suite.
 */
static const char *suite_bbob_get_instances_by_year(const int year) {

  if (year == 2009) {
    return "1-5,1-5,1-5";
  }
  else if (year == 2010) {
    return "1-15";
  }
  else if (year == 2012) {
    return "1-5,21-30";
  }
  else if (year == 2013) {
    return "1-5,31-40";
  }
  else if (year == 2015) {
    return "1-5,41-50";
  }
  else if (year == 2016) {
    return "1-5,51-60";
  }
  else {
    coco_error("suite_bbob_get_instances_by_year(): year %d not defined for suite_bbob", year);
    return NULL;
  }
}

/**
 * @brief Creates and returns a BBOB problem without needing the actual bbob suite.
 *
 * Useful for other suites as well (see for example suite_biobj.c).
 */
static coco_problem_t *coco_get_bbob_problem(const size_t function,
                                             const size_t dimension,
                                             const size_t instance) {
  coco_problem_t *problem = NULL;

  const char *problem_id_template = "bbob_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "BBOB suite problem f%lu instance %lu in %luD";

  const long rseed = (long) (function + 10000 * instance);
  const long rseed_3 = (long) (3 + 10000 * instance);
  const long rseed_17 = (long) (17 + 10000 * instance);

  if (function == 1) {
    problem = f_sphere_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 2) {
    problem = f_ellipsoid_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 3) {
    problem = f_rastrigin_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 4) {
    problem = f_bueche_rastrigin_bbob_problem_allocate(function, dimension, instance, rseed_3,
        problem_id_template, problem_name_template);
  } else if (function == 5) {
    problem = f_linear_slope_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 6) {
    problem = f_attractive_sector_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 7) {
    problem = f_step_ellipsoid_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 8) {
    problem = f_rosenbrock_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 9) {
    problem = f_rosenbrock_rotated_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 10) {
    problem = f_ellipsoid_rotated_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 11) {
    problem = f_discus_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 12) {
    problem = f_bent_cigar_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 13) {
    problem = f_sharp_ridge_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 14) {
    problem = f_different_powers_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 15) {
    problem = f_rastrigin_rotated_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 16) {
    problem = f_weierstrass_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 17) {
    problem = f_schaffers_bbob_problem_allocate(function, dimension, instance, rseed, 10,
        problem_id_template, problem_name_template);
  } else if (function == 18) {
    problem = f_schaffers_bbob_problem_allocate(function, dimension, instance, rseed_17, 1000,
        problem_id_template, problem_name_template);
  } else if (function == 19) {
    problem = f_griewank_rosenbrock_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 20) {
    problem = f_schwefel_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 21) {
    problem = f_gallagher_bbob_problem_allocate(function, dimension, instance, rseed, 101,
        problem_id_template, problem_name_template);
  } else if (function == 22) {
    problem = f_gallagher_bbob_problem_allocate(function, dimension, instance, rseed, 21,
        problem_id_template, problem_name_template);
  } else if (function == 23) {
    problem = f_katsuura_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else if (function == 24) {
    problem = f_lunacek_bi_rastrigin_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else {
    coco_error("get_bbob_problem(): cannot retrieve problem f%lu instance %lu in %luD", function, instance, dimension);
    return NULL; /* Never reached */
  }

  return problem;
}

/**
 * @brief Returns the problem from the bbob suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_bbob_get_problem(coco_suite_t *suite,
                                              const size_t function_idx,
                                              const size_t dimension_idx,
                                              const size_t instance_idx) {

  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = coco_get_bbob_problem(function, dimension, instance);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
#line 19 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/suite_biobj.c"
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

#line 16 "code-experiments/src/suite_biobj.c"
#line 1 "code-experiments/src/mo_utilities.c"
/**
 * @file mo_utilities.c
 * @brief Definitions of miscellaneous functions used for multi-objective problems.
 */

#include <stdlib.h>
#include <stdio.h>
#line 9 "code-experiments/src/mo_utilities.c"

/**
 * @brief Checks the dominance relation in the unconstrained minimization case between objectives1 and
 * objectives2.
 *
 * @return
 *  1 if objectives1 dominates objectives2 <br>
 *  0 if objectives1 and objectives2 are non-dominated <br>
 * -1 if objectives2 dominates objectives1 <br>
 * -2 if objectives1 is identical to objectives2
 */
static int mo_get_dominance(const double *objectives1, const double *objectives2, const size_t num_obj) {
  /* TODO: Should we care about comparison precision? */
  size_t i;

  int flag1 = 0;
  int flag2 = 0;

  for (i = 0; i < num_obj; i++) {
    if (objectives1[i] < objectives2[i]) {
      flag1 = 1;
    } else if (objectives1[i] > objectives2[i]) {
      flag2 = 1;
    }
  }

  if (flag1 && !flag2) {
    return 1;
  } else if (!flag1 && flag2) {
    return -1;
  } else if (flag1 && flag2) {
    return 0;
  } else { /* (!flag1 && !flag2) */
    return -2;
  }
}

/**
 * @brief Computes and returns the Euclidean norm of two dim-dimensional points first and second.
 */
static double mo_get_norm(const double *first, const double *second, const size_t dim) {

  size_t i;
  double norm = 0;

  for (i = 0; i < dim; i++) {
    norm += pow(first[i] - second[i], 2);
  }

  return sqrt(norm);
}

/**
 * @brief Computes and returns the minimal normalized distance from the point y to the ROI.
 *
 * @note Assumes the point is dominated by the ideal point and dimension equals 2.
 */
static double mo_get_distance_to_ROI(const double *y,
                                     const double *ideal,
                                     const double *nadir,
                                     const size_t dimension) {

  double distance = 0;

  assert(dimension == 2);
  assert(mo_get_dominance(ideal, y, 2) == 1);

  /* y is weakly dominated by the nadir point */
  if (mo_get_dominance(y, nadir, 2) <= -1) {
    distance = mo_get_norm(y, nadir, 2);
  }
  else if (y[0] < nadir[0])
    distance = y[1] - nadir[1];
  else if (y[1] < nadir[1])
    distance = y[0] - nadir[0];
  else {
    coco_error("mo_get_distance_to_ROI(): unexpected exception");
    return 0; /* Never reached */
  }

  return distance / ((nadir[1] - ideal[1]) * (nadir[0] - ideal[0]));

}

/**
 * @brief Computes and returns whether the solution y is within the ROI.
 *
 * @note Assumes the solution is dominated by the ideal point and dimension equals 2.
 */
static int mo_solution_is_within_ROI(const double *y,
                                     const double *ideal,
                                     const double *nadir,
                                     const size_t dimension) {

  assert(dimension == 2);
  assert(mo_get_dominance(ideal, y, 2) == 1);

  return (mo_get_dominance(nadir, y, 2) < 0);
}
#line 17 "code-experiments/src/suite_biobj.c"
#line 18 "code-experiments/src/suite_biobj.c"
#line 1 "code-experiments/src/suite_biobj_best_values_hyp.c"
/**
 * @file suite_biobj_best_values_hyp.c
 * @brief Contains the best known hypervolume values for the bbob-biobj suite problems.
 */

/** @brief The best known hypervolume values for the bbob-biobj suite problems. */
static const char *suite_biobj_best_values_hyp[] = {
  "bbob-biobj_f01_i01_d02 0.833332381880295",
  "bbob-biobj_f02_i01_d02 0.995822556045719",
  "bbob-biobj_f03_i01_d02 0.811763839749560",
  "bbob-biobj_f04_i01_d02 0.965338541757230",
  "bbob-biobj_f05_i01_d02 0.754127772584089",
  "bbob-biobj_f06_i01_d02 0.667254078045086",
  "bbob-biobj_f07_i01_d02 0.936972394688744",
  "bbob-biobj_f08_i01_d02 0.903849266969214",
  "bbob-biobj_f09_i01_d02 0.925657479493310",
  "bbob-biobj_f10_i01_d02 0.922987538734603",
  "bbob-biobj_f11_i01_d02 0.823972801661334",
  "bbob-biobj_f12_i01_d02 0.981018923089192",
  "bbob-biobj_f13_i01_d02 0.944887984976649",
  "bbob-biobj_f14_i01_d02 0.912471738968398",
  "bbob-biobj_f15_i01_d02 0.978306336956131",
  "bbob-biobj_f16_i01_d02 0.981136699536161",
  "bbob-biobj_f17_i01_d02 0.979958695543183",
  "bbob-biobj_f18_i01_d02 0.969204862736960",
  "bbob-biobj_f19_i01_d02 0.865811586473494",
  "bbob-biobj_f20_i01_d02 0.995717988090093",
  "bbob-biobj_f21_i01_d02 0.999736610851221",
  "bbob-biobj_f22_i01_d02 0.700915089341030",
  "bbob-biobj_f23_i01_d02 0.992533949247653",
  "bbob-biobj_f24_i01_d02 0.886262886736722",
  "bbob-biobj_f25_i01_d02 0.890542921944345",
  "bbob-biobj_f26_i01_d02 0.978921933860648",
  "bbob-biobj_f27_i01_d02 0.903502204113926",
  "bbob-biobj_f28_i01_d02 0.977401734681390",
  "bbob-biobj_f29_i01_d02 0.972456361472672",
  "bbob-biobj_f30_i01_d02 0.976212389263759",
  "bbob-biobj_f31_i01_d02 0.955984084027268",
  "bbob-biobj_f32_i01_d02 0.920330760313077",
  "bbob-biobj_f33_i01_d02 0.997888832413846",
  "bbob-biobj_f34_i01_d02 0.929758803965651",
  "bbob-biobj_f35_i01_d02 0.928163014468774",
  "bbob-biobj_f36_i01_d02 0.945913606633571",
  "bbob-biobj_f37_i01_d02 0.887325683129611",
  "bbob-biobj_f38_i01_d02 0.877156589643738",
  "bbob-biobj_f39_i01_d02 0.904223376562383",
  "bbob-biobj_f40_i01_d02 0.800161239789517",
  "bbob-biobj_f41_i01_d02 0.822032599539178",
  "bbob-biobj_f42_i01_d02 0.948464760953435",
  "bbob-biobj_f43_i01_d02 0.806235779393273",
  "bbob-biobj_f44_i01_d02 0.990000539848556",
  "bbob-biobj_f45_i01_d02 0.951720470099642",
  "bbob-biobj_f46_i01_d02 0.761065142390955",
  "bbob-biobj_f47_i01_d02 0.712271512083957",
  "bbob-biobj_f48_i01_d02 0.848025761787477",
  "bbob-biobj_f49_i01_d02 0.926653565305302",
  "bbob-biobj_f50_i01_d02 0.908152950514152",
  "bbob-biobj_f51_i01_d02 0.940860536161742",
  "bbob-biobj_f52_i01_d02 0.947011510956342",
  "bbob-biobj_f53_i01_d02 0.997875678381243",
  "bbob-biobj_f54_i01_d02 0.943563862014704",
  "bbob-biobj_f55_i01_d02 0.994911895481398",
  "bbob-biobj_f01_i02_d02 0.833332702706814",
  "bbob-biobj_f02_i02_d02 0.917892764681189",
  "bbob-biobj_f03_i02_d02 0.870718411906487",
  "bbob-biobj_f04_i02_d02 0.970390620254440",
  "bbob-biobj_f05_i02_d02 0.954989278665609",
  "bbob-biobj_f06_i02_d02 0.901470226591165",
  "bbob-biobj_f07_i02_d02 0.906340754327532",
  "bbob-biobj_f08_i02_d02 0.784765068625940",
  "bbob-biobj_f09_i02_d02 0.977793675518883",
  "bbob-biobj_f10_i02_d02 0.889244296995994",
  "bbob-biobj_f11_i02_d02 0.834474630498268",
  "bbob-biobj_f12_i02_d02 0.999911393192066",
  "bbob-biobj_f13_i02_d02 0.999759633691082",
  "bbob-biobj_f14_i02_d02 0.997754487039552",
  "bbob-biobj_f15_i02_d02 0.941436752524279",
  "bbob-biobj_f16_i02_d02 0.965606133139248",
  "bbob-biobj_f17_i02_d02 0.942990084985095",
  "bbob-biobj_f18_i02_d02 0.953409527167994",
  "bbob-biobj_f19_i02_d02 0.920899374306612",
  "bbob-biobj_f20_i02_d02 0.910959902066631",
  "bbob-biobj_f21_i02_d02 0.985471374619735",
  "bbob-biobj_f22_i02_d02 0.999055909737319",
  "bbob-biobj_f23_i02_d02 0.996445249237841",
  "bbob-biobj_f24_i02_d02 0.988139628791233",
  "bbob-biobj_f25_i02_d02 0.944370780971908",
  "bbob-biobj_f26_i02_d02 0.994494596966963",
  "bbob-biobj_f27_i02_d02 0.951958965619902",
  "bbob-biobj_f28_i02_d02 0.998930188872317",
  "bbob-biobj_f29_i02_d02 0.999165519331051",
  "bbob-biobj_f30_i02_d02 0.996222765792260",
  "bbob-biobj_f31_i02_d02 0.963697231127155",
  "bbob-biobj_f32_i02_d02 0.675234304274444",
  "bbob-biobj_f33_i02_d02 0.997492967539081",
  "bbob-biobj_f34_i02_d02 0.996825921713365",
  "bbob-biobj_f35_i02_d02 0.502441558908274",
  "bbob-biobj_f36_i02_d02 0.982338855577553",
  "bbob-biobj_f37_i02_d02 0.737269275089530",
  "bbob-biobj_f38_i02_d02 0.906613078300759",
  "bbob-biobj_f39_i02_d02 0.983637059586349",
  "bbob-biobj_f40_i02_d02 0.814422881685882",
  "bbob-biobj_f41_i02_d02 0.914731705960483",
  "bbob-biobj_f42_i02_d02 0.938889923994633",
  "bbob-biobj_f43_i02_d02 0.928981096739791",
  "bbob-biobj_f44_i02_d02 0.996682440185835",
  "bbob-biobj_f45_i02_d02 0.797327487716362",
  "bbob-biobj_f46_i02_d02 0.848797232382807",
  "bbob-biobj_f47_i02_d02 0.939177828232952",
  "bbob-biobj_f48_i02_d02 0.994793668269219",
  "bbob-biobj_f49_i02_d02 0.961467340366137",
  "bbob-biobj_f50_i02_d02 0.951055944905910",
  "bbob-biobj_f51_i02_d02 0.909955737710893",
  "bbob-biobj_f52_i02_d02 0.786879893098394",
  "bbob-biobj_f53_i02_d02 0.975730723054336",
  "bbob-biobj_f54_i02_d02 0.929274289867332",
  "bbob-biobj_f55_i02_d02 0.936711237564454",
  "bbob-biobj_f01_i03_d02 0.833332308645262",
  "bbob-biobj_f02_i03_d02 0.990979154749639",
  "bbob-biobj_f03_i03_d02 0.843026921526248",
  "bbob-biobj_f04_i03_d02 0.971131575360400",
  "bbob-biobj_f05_i03_d02 0.684482240828589",
  "bbob-biobj_f06_i03_d02 0.884299586864210",
  "bbob-biobj_f07_i03_d02 0.886134176223403",
  "bbob-biobj_f08_i03_d02 0.748604736034215",
  "bbob-biobj_f09_i03_d02 0.968705721628269",
  "bbob-biobj_f10_i03_d02 0.921417910401632",
  "bbob-biobj_f11_i03_d02 0.817436851656416",
  "bbob-biobj_f12_i03_d02 0.999915738925275",
  "bbob-biobj_f13_i03_d02 0.999951135495008",
  "bbob-biobj_f14_i03_d02 0.966432413236932",
  "bbob-biobj_f15_i03_d02 0.998672050309677",
  "bbob-biobj_f16_i03_d02 0.959401148905943",
  "bbob-biobj_f17_i03_d02 0.941726836105394",
  "bbob-biobj_f18_i03_d02 0.999403466523437",
  "bbob-biobj_f19_i03_d02 0.904545238945873",
  "bbob-biobj_f20_i03_d02 0.985777662447601",
  "bbob-biobj_f21_i03_d02 0.973889754548564",
  "bbob-biobj_f22_i03_d02 0.678522876973610",
  "bbob-biobj_f23_i03_d02 0.923156435473200",
  "bbob-biobj_f24_i03_d02 0.952811571802282",
  "bbob-biobj_f25_i03_d02 0.978559979410386",
  "bbob-biobj_f26_i03_d02 0.999888275904127",
  "bbob-biobj_f27_i03_d02 0.957759309603809",
  "bbob-biobj_f28_i03_d02 0.999666277471747",
  "bbob-biobj_f29_i03_d02 0.993280308173123",
  "bbob-biobj_f30_i03_d02 0.941596011487025",
  "bbob-biobj_f31_i03_d02 0.981722020457142",
  "bbob-biobj_f32_i03_d02 0.921767706667579",
  "bbob-biobj_f33_i03_d02 0.999823325123336",
  "bbob-biobj_f34_i03_d02 0.990964370887801",
  "bbob-biobj_f35_i03_d02 0.875494760634630",
  "bbob-biobj_f36_i03_d02 0.808774314903373",
  "bbob-biobj_f37_i03_d02 0.802436053394483",
  "bbob-biobj_f38_i03_d02 0.700820133315453",
  "bbob-biobj_f39_i03_d02 0.979200981846001",
  "bbob-biobj_f40_i03_d02 0.841353965905939",
  "bbob-biobj_f41_i03_d02 0.853587361521924",
  "bbob-biobj_f42_i03_d02 0.813087181156719",
  "bbob-biobj_f43_i03_d02 0.703325481833674",
  "bbob-biobj_f44_i03_d02 0.982516592468149",
  "bbob-biobj_f45_i03_d02 0.729949985493878",
  "bbob-biobj_f46_i03_d02 0.924046813817190",
  "bbob-biobj_f47_i03_d02 0.739793088506663",
  "bbob-biobj_f48_i03_d02 0.974655292653474",
  "bbob-biobj_f49_i03_d02 0.848123276637367",
  "bbob-biobj_f50_i03_d02 0.921885271065775",
  "bbob-biobj_f51_i03_d02 0.941593174399642",
  "bbob-biobj_f52_i03_d02 0.931104781087782",
  "bbob-biobj_f53_i03_d02 0.975730722493676",
  "bbob-biobj_f54_i03_d02 0.991716129205650",
  "bbob-biobj_f55_i03_d02 0.979743447263764",
  "bbob-biobj_f01_i04_d02 0.833332466832419",
  "bbob-biobj_f02_i04_d02 0.956280219817568",
  "bbob-biobj_f03_i04_d02 0.816336889930302",
  "bbob-biobj_f04_i04_d02 0.977012410891707",
  "bbob-biobj_f05_i04_d02 0.878628972558252",
  "bbob-biobj_f06_i04_d02 0.945804093632039",
  "bbob-biobj_f07_i04_d02 0.870759571247920",
  "bbob-biobj_f08_i04_d02 0.743267143878666",
  "bbob-biobj_f09_i04_d02 0.948342390905787",
  "bbob-biobj_f10_i04_d02 0.942089275587398",
  "bbob-biobj_f11_i04_d02 0.883087513516298",
  "bbob-biobj_f12_i04_d02 0.891857684789194",
  "bbob-biobj_f13_i04_d02 0.999963191340162",
  "bbob-biobj_f14_i04_d02 0.982899445074872",
  "bbob-biobj_f15_i04_d02 0.825078940847438",
  "bbob-biobj_f16_i04_d02 0.995345563275184",
  "bbob-biobj_f17_i04_d02 0.766767349502570",
  "bbob-biobj_f18_i04_d02 0.990014933357924",
  "bbob-biobj_f19_i04_d02 0.999884820902939",
  "bbob-biobj_f20_i04_d02 0.890179810267597",
  "bbob-biobj_f21_i04_d02 0.999788748780840",
  "bbob-biobj_f22_i04_d02 0.846364918537386",
  "bbob-biobj_f23_i04_d02 0.963999189380296",
  "bbob-biobj_f24_i04_d02 0.953732210965317",
  "bbob-biobj_f25_i04_d02 0.827380282336984",
  "bbob-biobj_f26_i04_d02 0.929918973100548",
  "bbob-biobj_f27_i04_d02 0.960857685351265",
  "bbob-biobj_f28_i04_d02 0.984721335437252",
  "bbob-biobj_f29_i04_d02 0.966908552820882",
  "bbob-biobj_f30_i04_d02 0.983029995579055",
  "bbob-biobj_f31_i04_d02 0.952958503821472",
  "bbob-biobj_f32_i04_d02 0.944003922120117",
  "bbob-biobj_f33_i04_d02 0.959892839230054",
  "bbob-biobj_f34_i04_d02 0.967251356146663",
  "bbob-biobj_f35_i04_d02 0.818249295839983",
  "bbob-biobj_f36_i04_d02 0.913201795196784",
  "bbob-biobj_f37_i04_d02 0.827629878786050",
  "bbob-biobj_f38_i04_d02 0.629303680384380",
  "bbob-biobj_f39_i04_d02 0.978307286629431",
  "bbob-biobj_f40_i04_d02 0.545131536551849",
  "bbob-biobj_f41_i04_d02 0.512372984066831",
  "bbob-biobj_f42_i04_d02 0.882605825029119",
  "bbob-biobj_f43_i04_d02 0.955511245826293",
  "bbob-biobj_f44_i04_d02 0.975372264066639",
  "bbob-biobj_f45_i04_d02 0.962996534388227",
  "bbob-biobj_f46_i04_d02 0.934266785105635",
  "bbob-biobj_f47_i04_d02 0.779551089811421",
  "bbob-biobj_f48_i04_d02 0.991892533182284",
  "bbob-biobj_f49_i04_d02 0.934365738264289",
  "bbob-biobj_f50_i04_d02 0.908169127554731",
  "bbob-biobj_f51_i04_d02 0.986038923713432",
  "bbob-biobj_f52_i04_d02 0.719116574845850",
  "bbob-biobj_f53_i04_d02 0.997875679038962",
  "bbob-biobj_f54_i04_d02 0.984075278318847",
  "bbob-biobj_f55_i04_d02 0.931559626376912",
  "bbob-biobj_f01_i05_d02 0.833332473672737",
  "bbob-biobj_f02_i05_d02 0.960749285196603",
  "bbob-biobj_f03_i05_d02 0.854018790706943",
  "bbob-biobj_f04_i05_d02 0.924874180331373",
  "bbob-biobj_f05_i05_d02 0.926274334896450",
  "bbob-biobj_f06_i05_d02 0.942603142355389",
  "bbob-biobj_f07_i05_d02 0.911523131921032",
  "bbob-biobj_f08_i05_d02 0.865136507485212",
  "bbob-biobj_f09_i05_d02 0.860780267730441",
  "bbob-biobj_f10_i05_d02 0.940003461277463",
  "bbob-biobj_f11_i05_d02 0.849343550099736",
  "bbob-biobj_f12_i05_d02 0.908759812361773",
  "bbob-biobj_f13_i05_d02 0.999818303887064",
  "bbob-biobj_f14_i05_d02 0.987027886030896",
  "bbob-biobj_f15_i05_d02 0.988337532350678",
  "bbob-biobj_f16_i05_d02 0.999043228362523",
  "bbob-biobj_f17_i05_d02 0.940686723549207",
  "bbob-biobj_f18_i05_d02 0.999935347950664",
  "bbob-biobj_f19_i05_d02 0.997258703820410",
  "bbob-biobj_f20_i05_d02 0.980549181813676",
  "bbob-biobj_f21_i05_d02 0.890724777349068",
  "bbob-biobj_f22_i05_d02 0.856038284580849",
  "bbob-biobj_f23_i05_d02 0.738908820827753",
  "bbob-biobj_f24_i05_d02 0.823876901281695",
  "bbob-biobj_f25_i05_d02 0.903625066204998",
  "bbob-biobj_f26_i05_d02 0.732358949009342",
  "bbob-biobj_f27_i05_d02 0.930274708362895",
  "bbob-biobj_f28_i05_d02 0.999901255187674",
  "bbob-biobj_f29_i05_d02 0.988529001485094",
  "bbob-biobj_f30_i05_d02 0.995608407690678",
  "bbob-biobj_f31_i05_d02 0.989988681723107",
  "bbob-biobj_f32_i05_d02 0.844964257151837",
  "bbob-biobj_f33_i05_d02 0.966110375002355",
  "bbob-biobj_f34_i05_d02 0.956107363130014",
  "bbob-biobj_f35_i05_d02 0.572775132157389",
  "bbob-biobj_f36_i05_d02 0.771461371110819",
  "bbob-biobj_f37_i05_d02 0.833850863453652",
  "bbob-biobj_f38_i05_d02 0.802625408060758",
  "bbob-biobj_f39_i05_d02 0.986275105475074",
  "bbob-biobj_f40_i05_d02 0.710069754586310",
  "bbob-biobj_f41_i05_d02 0.821557876863414",
  "bbob-biobj_f42_i05_d02 0.980207488827826",
  "bbob-biobj_f43_i05_d02 0.527808619658566",
  "bbob-biobj_f44_i05_d02 0.992233423072499",
  "bbob-biobj_f45_i05_d02 0.924715061037750",
  "bbob-biobj_f46_i05_d02 0.925925504479710",
  "bbob-biobj_f47_i05_d02 0.944459621622800",
  "bbob-biobj_f48_i05_d02 0.988654636530694",
  "bbob-biobj_f49_i05_d02 0.928529326530270",
  "bbob-biobj_f50_i05_d02 0.808869729563639",
  "bbob-biobj_f51_i05_d02 0.971128837321376",
  "bbob-biobj_f52_i05_d02 0.781598659109070",
  "bbob-biobj_f53_i05_d02 0.706100990975307",
  "bbob-biobj_f54_i05_d02 0.977343736135633",
  "bbob-biobj_f55_i05_d02 0.890941429551648",
  "bbob-biobj_f01_i01_d03 0.833330625661050",
  "bbob-biobj_f02_i01_d03 0.879310044577270",
  "bbob-biobj_f03_i01_d03 0.974987097859968",
  "bbob-biobj_f04_i01_d03 0.968687164946734",
  "bbob-biobj_f05_i01_d03 0.728470633560049",
  "bbob-biobj_f06_i01_d03 0.954291127409624",
  "bbob-biobj_f07_i01_d03 0.937571153021485",
  "bbob-biobj_f08_i01_d03 0.911798594849045",
  "bbob-biobj_f09_i01_d03 0.904194957863023",
  "bbob-biobj_f10_i01_d03 0.927567677878328",
  "bbob-biobj_f11_i01_d03 0.878620361843933",
  "bbob-biobj_f12_i01_d03 0.996585174463940",
  "bbob-biobj_f13_i01_d03 0.999495770222436",
  "bbob-biobj_f14_i01_d03 0.832549445682511",
  "bbob-biobj_f15_i01_d03 0.928761731458004",
  "bbob-biobj_f16_i01_d03 0.999116510660995",
  "bbob-biobj_f17_i01_d03 0.899961432050402",
  "bbob-biobj_f18_i01_d03 0.998688618941126",
  "bbob-biobj_f19_i01_d03 0.973016856746679",
  "bbob-biobj_f20_i01_d03 0.813226512698083",
  "bbob-biobj_f21_i01_d03 0.911539415650202",
  "bbob-biobj_f22_i01_d03 0.694480232049214",
  "bbob-biobj_f23_i01_d03 0.872234566808190",
  "bbob-biobj_f24_i01_d03 0.987113954053245",
  "bbob-biobj_f25_i01_d03 0.996715641165654",
  "bbob-biobj_f26_i01_d03 0.999860002469473",
  "bbob-biobj_f27_i01_d03 0.987126435674334",
  "bbob-biobj_f28_i01_d03 0.998639588439490",
  "bbob-biobj_f29_i01_d03 0.866365245335982",
  "bbob-biobj_f30_i01_d03 0.795597023991393",
  "bbob-biobj_f31_i01_d03 0.988038407614155",
  "bbob-biobj_f32_i01_d03 0.915330390200906",
  "bbob-biobj_f33_i01_d03 0.997624560661621",
  "bbob-biobj_f34_i01_d03 0.914923757473936",
  "bbob-biobj_f35_i01_d03 0.534050189516324",
  "bbob-biobj_f36_i01_d03 0.700104510378408",
  "bbob-biobj_f37_i01_d03 0.807646369904650",
  "bbob-biobj_f38_i01_d03 0.866616427546108",
  "bbob-biobj_f39_i01_d03 0.853571429088739",
  "bbob-biobj_f40_i01_d03 0.914339495288001",
  "bbob-biobj_f41_i01_d03 0.885735346283810",
  "bbob-biobj_f42_i01_d03 0.964392943937267",
  "bbob-biobj_f43_i01_d03 0.961022421747782",
  "bbob-biobj_f44_i01_d03 0.989550386270271",
  "bbob-biobj_f45_i01_d03 0.828185757495603",
  "bbob-biobj_f46_i01_d03 0.903454081049543",
  "bbob-biobj_f47_i01_d03 0.868929027146708",
  "bbob-biobj_f48_i01_d03 0.973369990523646",
  "bbob-biobj_f49_i01_d03 0.957627069123927",
  "bbob-biobj_f50_i01_d03 0.969141599163299",
  "bbob-biobj_f51_i01_d03 0.868050177774589",
  "bbob-biobj_f52_i01_d03 0.887645413730364",
  "bbob-biobj_f53_i01_d03 0.999784713871949",
  "bbob-biobj_f54_i01_d03 0.975179043634054",
  "bbob-biobj_f55_i01_d03 0.969815953388893",
  "bbob-biobj_f01_i02_d03 0.833330862027532",
  "bbob-biobj_f02_i02_d03 0.981135795583470",
  "bbob-biobj_f03_i02_d03 0.845123943144692",
  "bbob-biobj_f04_i02_d03 0.955013920344669",
  "bbob-biobj_f05_i02_d03 0.688039742765706",
  "bbob-biobj_f06_i02_d03 0.863890745758219",
  "bbob-biobj_f07_i02_d03 0.923759709009026",
  "bbob-biobj_f08_i02_d03 0.881999475736607",
  "bbob-biobj_f09_i02_d03 0.992207196354342",
  "bbob-biobj_f10_i02_d03 0.883785944698432",
  "bbob-biobj_f11_i02_d03 0.833333045070521",
  "bbob-biobj_f12_i02_d03 0.999935181038573",
  "bbob-biobj_f13_i02_d03 0.998700044654841",
  "bbob-biobj_f14_i02_d03 0.996012898774857",
  "bbob-biobj_f15_i02_d03 0.954421200561461",
  "bbob-biobj_f16_i02_d03 0.934170171805547",
  "bbob-biobj_f17_i02_d03 0.931413069408532",
  "bbob-biobj_f18_i02_d03 0.993777465831754",
  "bbob-biobj_f19_i02_d03 0.986110860218861",
  "bbob-biobj_f20_i02_d03 0.974016935549722",
  "bbob-biobj_f21_i02_d03 0.980783963376774",
  "bbob-biobj_f22_i02_d03 0.742598839023062",
  "bbob-biobj_f23_i02_d03 0.862093818070480",
  "bbob-biobj_f24_i02_d03 0.906306212968833",
  "bbob-biobj_f25_i02_d03 0.961641429125833",
  "bbob-biobj_f26_i02_d03 0.988829159179452",
  "bbob-biobj_f27_i02_d03 0.953825801036066",
  "bbob-biobj_f28_i02_d03 0.993519338993658",
  "bbob-biobj_f29_i02_d03 0.939190481177387",
  "bbob-biobj_f30_i02_d03 0.931860098796110",
  "bbob-biobj_f31_i02_d03 0.923773258244386",
  "bbob-biobj_f32_i02_d03 0.922150697719170",
  "bbob-biobj_f33_i02_d03 0.998822645972432",
  "bbob-biobj_f34_i02_d03 0.914729556179324",
  "bbob-biobj_f35_i02_d03 0.788702400676018",
  "bbob-biobj_f36_i02_d03 0.737149966770445",
  "bbob-biobj_f37_i02_d03 0.835003569058186",
  "bbob-biobj_f38_i02_d03 0.793052225590621",
  "bbob-biobj_f39_i02_d03 0.963320808065134",
  "bbob-biobj_f40_i02_d03 0.684383656263477",
  "bbob-biobj_f41_i02_d03 0.923050849072400",
  "bbob-biobj_f42_i02_d03 0.896772979402947",
  "bbob-biobj_f43_i02_d03 0.900486501147008",
  "bbob-biobj_f44_i02_d03 0.744914585896674",
  "bbob-biobj_f45_i02_d03 0.976528182782067",
  "bbob-biobj_f46_i02_d03 0.962406648963739",
  "bbob-biobj_f47_i02_d03 0.954698871412203",
  "bbob-biobj_f48_i02_d03 0.870171996025677",
  "bbob-biobj_f49_i02_d03 0.946678900765912",
  "bbob-biobj_f50_i02_d03 0.967313830200992",
  "bbob-biobj_f51_i02_d03 0.990170952029660",
  "bbob-biobj_f52_i02_d03 0.975826617684440",
  "bbob-biobj_f53_i02_d03 0.981828773918581",
  "bbob-biobj_f54_i02_d03 0.971505406466657",
  "bbob-biobj_f55_i02_d03 0.932686521297934",
  "bbob-biobj_f01_i03_d03 0.833330453018851",
  "bbob-biobj_f02_i03_d03 0.952600924477390",
  "bbob-biobj_f03_i03_d03 0.860124995330607",
  "bbob-biobj_f04_i03_d03 0.910479626757205",
  "bbob-biobj_f05_i03_d03 0.802883672795523",
  "bbob-biobj_f06_i03_d03 0.833636257813634",
  "bbob-biobj_f07_i03_d03 0.921397877927388",
  "bbob-biobj_f08_i03_d03 0.850945391164016",
  "bbob-biobj_f09_i03_d03 0.986085287643183",
  "bbob-biobj_f10_i03_d03 0.901132994991743",
  "bbob-biobj_f11_i03_d03 0.827369814756058",
  "bbob-biobj_f12_i03_d03 0.999886884014938",
  "bbob-biobj_f13_i03_d03 0.962145745196926",
  "bbob-biobj_f14_i03_d03 0.983419659230451",
  "bbob-biobj_f15_i03_d03 0.994036317066106",
  "bbob-biobj_f16_i03_d03 0.943744207185075",
  "bbob-biobj_f17_i03_d03 0.956408713997142",
  "bbob-biobj_f18_i03_d03 0.999304353336504",
  "bbob-biobj_f19_i03_d03 0.957586878136096",
  "bbob-biobj_f20_i03_d03 0.999442770417740",
  "bbob-biobj_f21_i03_d03 0.968953448582413",
  "bbob-biobj_f22_i03_d03 0.951233969408275",
  "bbob-biobj_f23_i03_d03 0.880341977880396",
  "bbob-biobj_f24_i03_d03 0.855065188105058",
  "bbob-biobj_f25_i03_d03 0.957694706824249",
  "bbob-biobj_f26_i03_d03 0.996485237016538",
  "bbob-biobj_f27_i03_d03 0.964059233915336",
  "bbob-biobj_f28_i03_d03 0.977206316465709",
  "bbob-biobj_f29_i03_d03 0.980540427492229",
  "bbob-biobj_f30_i03_d03 0.913952638696125",
  "bbob-biobj_f31_i03_d03 0.984775968884612",
  "bbob-biobj_f32_i03_d03 0.968195228796908",
  "bbob-biobj_f33_i03_d03 0.935833285400742",
  "bbob-biobj_f34_i03_d03 0.962893742806279",
  "bbob-biobj_f35_i03_d03 0.573977375862587",
  "bbob-biobj_f36_i03_d03 0.842941404105936",
  "bbob-biobj_f37_i03_d03 0.779815835700207",
  "bbob-biobj_f38_i03_d03 0.753306133299475",
  "bbob-biobj_f39_i03_d03 0.842273286569522",
  "bbob-biobj_f40_i03_d03 0.938506336709179",
  "bbob-biobj_f41_i03_d03 0.914426171797939",
  "bbob-biobj_f42_i03_d03 0.936775797328375",
  "bbob-biobj_f43_i03_d03 0.738226600212524",
  "bbob-biobj_f44_i03_d03 0.988829928149258",
  "bbob-biobj_f45_i03_d03 0.953105203861944",
  "bbob-biobj_f46_i03_d03 0.899712681623091",
  "bbob-biobj_f47_i03_d03 0.961190084929696",
  "bbob-biobj_f48_i03_d03 0.980824613790234",
  "bbob-biobj_f49_i03_d03 0.942235584921603",
  "bbob-biobj_f50_i03_d03 0.912990448629017",
  "bbob-biobj_f51_i03_d03 0.933222623382586",
  "bbob-biobj_f52_i03_d03 0.944891086251895",
  "bbob-biobj_f53_i03_d03 0.981828806073236",
  "bbob-biobj_f54_i03_d03 0.991021288687016",
  "bbob-biobj_f55_i03_d03 0.960030175857650",
  "bbob-biobj_f01_i04_d03 0.833330439399264",
  "bbob-biobj_f02_i04_d03 0.889456786293680",
  "bbob-biobj_f03_i04_d03 0.965016673576464",
  "bbob-biobj_f04_i04_d03 0.994699591408625",
  "bbob-biobj_f05_i04_d03 0.744993050883400",
  "bbob-biobj_f06_i04_d03 0.921369955953119",
  "bbob-biobj_f07_i04_d03 0.933979869632330",
  "bbob-biobj_f08_i04_d03 0.667005234415415",
  "bbob-biobj_f09_i04_d03 0.940843066710342",
  "bbob-biobj_f10_i04_d03 0.932701717371814",
  "bbob-biobj_f11_i04_d03 0.841523496261840",
  "bbob-biobj_f12_i04_d03 0.934335915105271",
  "bbob-biobj_f13_i04_d03 0.999978298631445",
  "bbob-biobj_f14_i04_d03 0.999112786189168",
  "bbob-biobj_f15_i04_d03 0.975014444393992",
  "bbob-biobj_f16_i04_d03 0.997825549172083",
  "bbob-biobj_f17_i04_d03 0.860455977643319",
  "bbob-biobj_f18_i04_d03 0.998331130655686",
  "bbob-biobj_f19_i04_d03 0.996380900904860",
  "bbob-biobj_f20_i04_d03 0.925391239228724",
  "bbob-biobj_f21_i04_d03 0.954073899348508",
  "bbob-biobj_f22_i04_d03 0.803804038308167",
  "bbob-biobj_f23_i04_d03 0.925755832451665",
  "bbob-biobj_f24_i04_d03 0.931779667951526",
  "bbob-biobj_f25_i04_d03 0.985238565236331",
  "bbob-biobj_f26_i04_d03 0.996886142025746",
  "bbob-biobj_f27_i04_d03 0.952867261004457",
  "bbob-biobj_f28_i04_d03 0.992073880591932",
  "bbob-biobj_f29_i04_d03 0.973837872435705",
  "bbob-biobj_f30_i04_d03 0.977327596327004",
  "bbob-biobj_f31_i04_d03 0.994016359061764",
  "bbob-biobj_f32_i04_d03 0.906090113753270",
  "bbob-biobj_f33_i04_d03 0.989064144200488",
  "bbob-biobj_f34_i04_d03 0.985996576175221",
  "bbob-biobj_f35_i04_d03 0.945462624909156",
  "bbob-biobj_f36_i04_d03 0.970230594996878",
  "bbob-biobj_f37_i04_d03 0.927470551875993",
  "bbob-biobj_f38_i04_d03 0.886831959453645",
  "bbob-biobj_f39_i04_d03 0.956662451361173",
  "bbob-biobj_f40_i04_d03 0.820235647557919",
  "bbob-biobj_f41_i04_d03 0.695234921931116",
  "bbob-biobj_f42_i04_d03 0.956285579577800",
  "bbob-biobj_f43_i04_d03 0.898274387524015",
  "bbob-biobj_f44_i04_d03 0.927812988201398",
  "bbob-biobj_f45_i04_d03 0.954353857297788",
  "bbob-biobj_f46_i04_d03 0.892786780071618",
  "bbob-biobj_f47_i04_d03 0.939953605978330",
  "bbob-biobj_f48_i04_d03 0.977080994868941",
  "bbob-biobj_f49_i04_d03 0.927009769272452",
  "bbob-biobj_f50_i04_d03 0.826467322858010",
  "bbob-biobj_f51_i04_d03 0.947516570775813",
  "bbob-biobj_f52_i04_d03 0.963896486103395",
  "bbob-biobj_f53_i04_d03 0.999784713869230",
  "bbob-biobj_f54_i04_d03 0.985246589920415",
  "bbob-biobj_f55_i04_d03 0.978790992231310",
  "bbob-biobj_f01_i05_d03 0.833330487059910",
  "bbob-biobj_f02_i05_d03 0.924022323888533",
  "bbob-biobj_f03_i05_d03 0.879234393038782",
  "bbob-biobj_f04_i05_d03 0.923160620965396",
  "bbob-biobj_f05_i05_d03 0.701514053994569",
  "bbob-biobj_f06_i05_d03 0.899135381658772",
  "bbob-biobj_f07_i05_d03 0.887626238519752",
  "bbob-biobj_f08_i05_d03 0.893995568811727",
  "bbob-biobj_f09_i05_d03 0.939788089015001",
  "bbob-biobj_f10_i05_d03 0.934362569190694",
  "bbob-biobj_f11_i05_d03 0.775580552202392",
  "bbob-biobj_f12_i05_d03 0.934982056573171",
  "bbob-biobj_f13_i05_d03 0.999920715812960",
  "bbob-biobj_f14_i05_d03 0.986701882411767",
  "bbob-biobj_f15_i05_d03 0.932552037707103",
  "bbob-biobj_f16_i05_d03 0.953342473193128",
  "bbob-biobj_f17_i05_d03 0.855772678108889",
  "bbob-biobj_f18_i05_d03 0.975931585763652",
  "bbob-biobj_f19_i05_d03 0.959361637514777",
  "bbob-biobj_f20_i05_d03 0.963674532282437",
  "bbob-biobj_f21_i05_d03 0.999573344448168",
  "bbob-biobj_f22_i05_d03 0.929852433616299",
  "bbob-biobj_f23_i05_d03 0.840964634500044",
  "bbob-biobj_f24_i05_d03 0.892822563372256",
  "bbob-biobj_f25_i05_d03 0.987241521970443",
  "bbob-biobj_f26_i05_d03 0.919257061669218",
  "bbob-biobj_f27_i05_d03 0.960952524777076",
  "bbob-biobj_f28_i05_d03 0.990126902830565",
  "bbob-biobj_f29_i05_d03 0.949153924567331",
  "bbob-biobj_f30_i05_d03 0.938513305197803",
  "bbob-biobj_f31_i05_d03 0.975602048556156",
  "bbob-biobj_f32_i05_d03 0.983672743739564",
  "bbob-biobj_f33_i05_d03 0.998861207154187",
  "bbob-biobj_f34_i05_d03 0.990692064296993",
  "bbob-biobj_f35_i05_d03 0.773373144669810",
  "bbob-biobj_f36_i05_d03 0.852779821331026",
  "bbob-biobj_f37_i05_d03 0.841495012859572",
  "bbob-biobj_f38_i05_d03 0.930554242816143",
  "bbob-biobj_f39_i05_d03 0.958678871287242",
  "bbob-biobj_f40_i05_d03 0.761268354646857",
  "bbob-biobj_f41_i05_d03 0.822869743753209",
  "bbob-biobj_f42_i05_d03 0.933371877335820",
  "bbob-biobj_f43_i05_d03 0.967868455163294",
  "bbob-biobj_f44_i05_d03 0.972160421709319",
  "bbob-biobj_f45_i05_d03 0.942410710847329",
  "bbob-biobj_f46_i05_d03 0.846189881536941",
  "bbob-biobj_f47_i05_d03 0.910086304698320",
  "bbob-biobj_f48_i05_d03 0.980478419308161",
  "bbob-biobj_f49_i05_d03 0.899347714075874",
  "bbob-biobj_f50_i05_d03 0.990885002779570",
  "bbob-biobj_f51_i05_d03 0.983375775680358",
  "bbob-biobj_f52_i05_d03 0.970414164343752",
  "bbob-biobj_f53_i05_d03 0.998471495709904",
  "bbob-biobj_f54_i05_d03 0.956276743155901",
  "bbob-biobj_f55_i05_d03 0.890296523345115",
  "bbob-biobj_f01_i01_d05 0.833323613048522",
  "bbob-biobj_f02_i01_d05 0.953382480572129",
  "bbob-biobj_f03_i01_d05 0.846748978060505",
  "bbob-biobj_f04_i01_d05 0.943986240958572",
  "bbob-biobj_f05_i01_d05 0.732271882563166",
  "bbob-biobj_f06_i01_d05 0.846008217982165",
  "bbob-biobj_f07_i01_d05 0.860198720717520",
  "bbob-biobj_f08_i01_d05 0.942636159464674",
  "bbob-biobj_f09_i01_d05 0.932177848166126",
  "bbob-biobj_f10_i01_d05 0.867906439608168",
  "bbob-biobj_f11_i01_d05 0.812582904862866",
  "bbob-biobj_f12_i01_d05 0.999992978765985",
  "bbob-biobj_f13_i01_d05 0.999802985541884",
  "bbob-biobj_f14_i01_d05 0.935767091759654",
  "bbob-biobj_f15_i01_d05 0.948210966449382",
  "bbob-biobj_f16_i01_d05 0.991904920470480",
  "bbob-biobj_f17_i01_d05 0.980121146965135",
  "bbob-biobj_f18_i01_d05 0.998492797896518",
  "bbob-biobj_f19_i01_d05 0.992521209197766",
  "bbob-biobj_f20_i01_d05 0.967868337013301",
  "bbob-biobj_f21_i01_d05 0.912539986187987",
  "bbob-biobj_f22_i01_d05 0.986971314681578",
  "bbob-biobj_f23_i01_d05 0.980181428165705",
  "bbob-biobj_f24_i01_d05 0.869765596613810",
  "bbob-biobj_f25_i01_d05 0.992992939913715",
  "bbob-biobj_f26_i01_d05 0.949032729323168",
  "bbob-biobj_f27_i01_d05 0.992203653046644",
  "bbob-biobj_f28_i01_d05 0.995557132172934",
  "bbob-biobj_f29_i01_d05 0.870409620405391",
  "bbob-biobj_f30_i01_d05 0.940209025665832",
  "bbob-biobj_f31_i01_d05 0.973300514645855",
  "bbob-biobj_f32_i01_d05 0.972535028583623",
  "bbob-biobj_f33_i01_d05 0.998341534043992",
  "bbob-biobj_f34_i01_d05 0.981336357786654",
  "bbob-biobj_f35_i01_d05 0.635731327611580",
  "bbob-biobj_f36_i01_d05 0.885930265615739",
  "bbob-biobj_f37_i01_d05 0.835092213812330",
  "bbob-biobj_f38_i01_d05 0.915463977226424",
  "bbob-biobj_f39_i01_d05 0.891171288243724",
  "bbob-biobj_f40_i01_d05 0.903887041271556",
  "bbob-biobj_f41_i01_d05 0.975606933105833",
  "bbob-biobj_f42_i01_d05 0.935336117367536",
  "bbob-biobj_f43_i01_d05 0.898037239151051",
  "bbob-biobj_f44_i01_d05 0.993557077920995",
  "bbob-biobj_f45_i01_d05 0.917823354092003",
  "bbob-biobj_f46_i01_d05 0.947148190193365",
  "bbob-biobj_f47_i01_d05 0.941993431399186",
  "bbob-biobj_f48_i01_d05 0.970790086010117",
  "bbob-biobj_f49_i01_d05 0.956450784209943",
  "bbob-biobj_f50_i01_d05 0.971029074709937",
  "bbob-biobj_f51_i01_d05 0.986037375175202",
  "bbob-biobj_f52_i01_d05 0.961482111814171",
  "bbob-biobj_f53_i01_d05 0.999101042830160",
  "bbob-biobj_f54_i01_d05 0.986735369472236",
  "bbob-biobj_f55_i01_d05 0.966512656733354",
  "bbob-biobj_f01_i02_d05 0.833323678644197",
  "bbob-biobj_f02_i02_d05 0.966471970749557",
  "bbob-biobj_f03_i02_d05 0.961673575952914",
  "bbob-biobj_f04_i02_d05 0.963227369233591",
  "bbob-biobj_f05_i02_d05 0.714726002035663",
  "bbob-biobj_f06_i02_d05 0.867717106453388",
  "bbob-biobj_f07_i02_d05 0.893270563392809",
  "bbob-biobj_f08_i02_d05 0.909304697813100",
  "bbob-biobj_f09_i02_d05 0.961852667556734",
  "bbob-biobj_f10_i02_d05 0.898112374939464",
  "bbob-biobj_f11_i02_d05 0.813661370076743",
  "bbob-biobj_f12_i02_d05 0.999910204628200",
  "bbob-biobj_f13_i02_d05 0.998548417006938",
  "bbob-biobj_f14_i02_d05 0.842152409419129",
  "bbob-biobj_f15_i02_d05 0.979712501508148",
  "bbob-biobj_f16_i02_d05 0.954676931265197",
  "bbob-biobj_f17_i02_d05 0.971536148799575",
  "bbob-biobj_f18_i02_d05 0.991742303696388",
  "bbob-biobj_f19_i02_d05 0.989653695603756",
  "bbob-biobj_f20_i02_d05 0.813995737299184",
  "bbob-biobj_f21_i02_d05 0.998160272065372",
  "bbob-biobj_f22_i02_d05 0.764257613844081",
  "bbob-biobj_f23_i02_d05 0.919714796662331",
  "bbob-biobj_f24_i02_d05 0.947901589829288",
  "bbob-biobj_f25_i02_d05 0.983777178858051",
  "bbob-biobj_f26_i02_d05 0.979221336328691",
  "bbob-biobj_f27_i02_d05 0.932656508740310",
  "bbob-biobj_f28_i02_d05 0.991805254328833",
  "bbob-biobj_f29_i02_d05 0.882783597714326",
  "bbob-biobj_f30_i02_d05 0.982389954779447",
  "bbob-biobj_f31_i02_d05 0.977009713031631",
  "bbob-biobj_f32_i02_d05 0.938355889064927",
  "bbob-biobj_f33_i02_d05 0.989647286586073",
  "bbob-biobj_f34_i02_d05 0.991099202459270",
  "bbob-biobj_f35_i02_d05 0.581799542679685",
  "bbob-biobj_f36_i02_d05 0.930155890009152",
  "bbob-biobj_f37_i02_d05 0.929629661624318",
  "bbob-biobj_f38_i02_d05 0.872919173674776",
  "bbob-biobj_f39_i02_d05 0.979050846839239",
  "bbob-biobj_f40_i02_d05 0.823839715183383",
  "bbob-biobj_f41_i02_d05 0.924063808544279",
  "bbob-biobj_f42_i02_d05 0.918085094763491",
  "bbob-biobj_f43_i02_d05 0.953483956754120",
  "bbob-biobj_f44_i02_d05 0.951280016927477",
  "bbob-biobj_f45_i02_d05 0.892693151302912",
  "bbob-biobj_f46_i02_d05 0.902611800665956",
  "bbob-biobj_f47_i02_d05 0.930094310793038",
  "bbob-biobj_f48_i02_d05 0.991207696805778",
  "bbob-biobj_f49_i02_d05 0.955513219623643",
  "bbob-biobj_f50_i02_d05 0.954380000682352",
  "bbob-biobj_f51_i02_d05 0.980566922057903",
  "bbob-biobj_f52_i02_d05 0.978948285713597",
  "bbob-biobj_f53_i02_d05 0.998939985445457",
  "bbob-biobj_f54_i02_d05 0.988596134781331",
  "bbob-biobj_f55_i02_d05 0.982278785271738",
  "bbob-biobj_f01_i03_d05 0.833323412201801",
  "bbob-biobj_f02_i03_d05 0.950361831175043",
  "bbob-biobj_f03_i03_d05 0.836863196221714",
  "bbob-biobj_f04_i03_d05 0.937875769562535",
  "bbob-biobj_f05_i03_d05 0.699885573619157",
  "bbob-biobj_f06_i03_d05 0.848586604888048",
  "bbob-biobj_f07_i03_d05 0.868172809095841",
  "bbob-biobj_f08_i03_d05 0.805160900475253",
  "bbob-biobj_f09_i03_d05 0.930449808633550",
  "bbob-biobj_f10_i03_d05 0.890596373492179",
  "bbob-biobj_f11_i03_d05 0.841063082375848",
  "bbob-biobj_f12_i03_d05 0.947356672555347",
  "bbob-biobj_f13_i03_d05 0.999817326476568",
  "bbob-biobj_f14_i03_d05 0.792210323343157",
  "bbob-biobj_f15_i03_d05 0.980683450638681",
  "bbob-biobj_f16_i03_d05 0.990026143743998",
  "bbob-biobj_f17_i03_d05 0.967728389540592",
  "bbob-biobj_f18_i03_d05 0.946581624974437",
  "bbob-biobj_f19_i03_d05 0.982448542162378",
  "bbob-biobj_f20_i03_d05 0.892400286535832",
  "bbob-biobj_f21_i03_d05 0.929902738511761",
  "bbob-biobj_f22_i03_d05 0.735414605775649",
  "bbob-biobj_f23_i03_d05 0.894398402347281",
  "bbob-biobj_f24_i03_d05 0.935398598296377",
  "bbob-biobj_f25_i03_d05 0.969545388922887",
  "bbob-biobj_f26_i03_d05 0.984009930418544",
  "bbob-biobj_f27_i03_d05 0.978490094776665",
  "bbob-biobj_f28_i03_d05 0.993681730562182",
  "bbob-biobj_f29_i03_d05 0.830278849915943",
  "bbob-biobj_f30_i03_d05 0.914680090254004",
  "bbob-biobj_f31_i03_d05 0.975956139468783",
  "bbob-biobj_f32_i03_d05 0.983060370685761",
  "bbob-biobj_f33_i03_d05 0.999015972826592",
  "bbob-biobj_f34_i03_d05 0.968962634460452",
  "bbob-biobj_f35_i03_d05 0.768217022723346",
  "bbob-biobj_f36_i03_d05 0.627299549868112",
  "bbob-biobj_f37_i03_d05 0.841049270484574",
  "bbob-biobj_f38_i03_d05 0.890858931496801",
  "bbob-biobj_f39_i03_d05 0.903021027298148",
  "bbob-biobj_f40_i03_d05 0.809171122082573",
  "bbob-biobj_f41_i03_d05 0.974027388952628",
  "bbob-biobj_f42_i03_d05 0.860961686699998",
  "bbob-biobj_f43_i03_d05 0.988523450616811",
  "bbob-biobj_f44_i03_d05 0.958337980323592",
  "bbob-biobj_f45_i03_d05 0.953228105854077",
  "bbob-biobj_f46_i03_d05 0.892809921198084",
  "bbob-biobj_f47_i03_d05 0.976633663786186",
  "bbob-biobj_f48_i03_d05 0.992966678801973",
  "bbob-biobj_f49_i03_d05 0.964436030451722",
  "bbob-biobj_f50_i03_d05 0.961250256199450",
  "bbob-biobj_f51_i03_d05 0.972182902233710",
  "bbob-biobj_f52_i03_d05 0.981022713204623",
  "bbob-biobj_f53_i03_d05 0.997632016077083",
  "bbob-biobj_f54_i03_d05 0.990016829409416",
  "bbob-biobj_f55_i03_d05 0.964540771222069",
  "bbob-biobj_f01_i04_d05 0.833323604552418",
  "bbob-biobj_f02_i04_d05 0.881361047681442",
  "bbob-biobj_f03_i04_d05 0.832969879000999",
  "bbob-biobj_f04_i04_d05 0.944466760345521",
  "bbob-biobj_f05_i04_d05 0.776079683001309",
  "bbob-biobj_f06_i04_d05 0.935940507310882",
  "bbob-biobj_f07_i04_d05 0.870192086993759",
  "bbob-biobj_f08_i04_d05 0.927620206885280",
  "bbob-biobj_f09_i04_d05 0.950693931608951",
  "bbob-biobj_f10_i04_d05 0.946968847494744",
  "bbob-biobj_f11_i04_d05 0.886487621972585",
  "bbob-biobj_f12_i04_d05 0.973685651248799",
  "bbob-biobj_f13_i04_d05 0.999810911991238",
  "bbob-biobj_f14_i04_d05 0.854022178778476",
  "bbob-biobj_f15_i04_d05 0.944759123319520",
  "bbob-biobj_f16_i04_d05 0.984582374899148",
  "bbob-biobj_f17_i04_d05 0.995587732952418",
  "bbob-biobj_f18_i04_d05 0.973768217428830",
  "bbob-biobj_f19_i04_d05 0.992031326175875",
  "bbob-biobj_f20_i04_d05 0.999111453421520",
  "bbob-biobj_f21_i04_d05 0.928301112745217",
  "bbob-biobj_f22_i04_d05 0.834631475894313",
  "bbob-biobj_f23_i04_d05 0.887969873232309",
  "bbob-biobj_f24_i04_d05 0.880840406920242",
  "bbob-biobj_f25_i04_d05 0.992528890753554",
  "bbob-biobj_f26_i04_d05 0.965391874605957",
  "bbob-biobj_f27_i04_d05 0.981448534698392",
  "bbob-biobj_f28_i04_d05 0.997392629399003",
  "bbob-biobj_f29_i04_d05 0.899704016430431",
  "bbob-biobj_f30_i04_d05 0.979631070337090",
  "bbob-biobj_f31_i04_d05 0.977972027972164",
  "bbob-biobj_f32_i04_d05 0.987959694560786",
  "bbob-biobj_f33_i04_d05 0.999672662969389",
  "bbob-biobj_f34_i04_d05 0.990433078244488",
  "bbob-biobj_f35_i04_d05 0.735374749381652",
  "bbob-biobj_f36_i04_d05 0.914705891979480",
  "bbob-biobj_f37_i04_d05 0.767371858941308",
  "bbob-biobj_f38_i04_d05 0.807350027097187",
  "bbob-biobj_f39_i04_d05 0.968068726151961",
  "bbob-biobj_f40_i04_d05 0.914038069032498",
  "bbob-biobj_f41_i04_d05 0.889981332187799",
  "bbob-biobj_f42_i04_d05 0.929451911522505",
  "bbob-biobj_f43_i04_d05 0.945402755603790",
  "bbob-biobj_f44_i04_d05 0.965363681112066",
  "bbob-biobj_f45_i04_d05 0.946643284078182",
  "bbob-biobj_f46_i04_d05 0.909674476002128",
  "bbob-biobj_f47_i04_d05 0.922198297806564",
  "bbob-biobj_f48_i04_d05 0.979017031441479",
  "bbob-biobj_f49_i04_d05 0.968252711990607",
  "bbob-biobj_f50_i04_d05 0.972701814467758",
  "bbob-biobj_f51_i04_d05 0.983435426314444",
  "bbob-biobj_f52_i04_d05 0.990680814430356",
  "bbob-biobj_f53_i04_d05 0.999956638376682",
  "bbob-biobj_f54_i04_d05 0.992621223363064",
  "bbob-biobj_f55_i04_d05 0.987526196728242",
  "bbob-biobj_f01_i05_d05 0.833323980042135",
  "bbob-biobj_f02_i05_d05 0.861588901038615",
  "bbob-biobj_f03_i05_d05 0.959274685295916",
  "bbob-biobj_f04_i05_d05 0.942088111842157",
  "bbob-biobj_f05_i05_d05 0.737157063871410",
  "bbob-biobj_f06_i05_d05 0.930563028723551",
  "bbob-biobj_f07_i05_d05 0.911636778860361",
  "bbob-biobj_f08_i05_d05 0.917781387881829",
  "bbob-biobj_f09_i05_d05 0.968559812719704",
  "bbob-biobj_f10_i05_d05 0.934944731954434",
  "bbob-biobj_f11_i05_d05 0.834554829532043",
  "bbob-biobj_f12_i05_d05 0.999480702962406",
  "bbob-biobj_f13_i05_d05 0.999369898449103",
  "bbob-biobj_f14_i05_d05 0.966157234627486",
  "bbob-biobj_f15_i05_d05 0.867414820839768",
  "bbob-biobj_f16_i05_d05 0.995250994802709",
  "bbob-biobj_f17_i05_d05 0.989412571977172",
  "bbob-biobj_f18_i05_d05 0.941220367027244",
  "bbob-biobj_f19_i05_d05 0.993758171916552",
  "bbob-biobj_f20_i05_d05 0.946993713194924",
  "bbob-biobj_f21_i05_d05 0.997869805143790",
  "bbob-biobj_f22_i05_d05 0.892864250824833",
  "bbob-biobj_f23_i05_d05 0.891337009773493",
  "bbob-biobj_f24_i05_d05 0.958073366124566",
  "bbob-biobj_f25_i05_d05 0.938499543222742",
  "bbob-biobj_f26_i05_d05 0.999565002960340",
  "bbob-biobj_f27_i05_d05 0.975425664056032",
  "bbob-biobj_f28_i05_d05 0.987355817298832",
  "bbob-biobj_f29_i05_d05 0.927425993744280",
  "bbob-biobj_f30_i05_d05 0.933169511358876",
  "bbob-biobj_f31_i05_d05 0.985496065614287",
  "bbob-biobj_f32_i05_d05 0.982435899136349",
  "bbob-biobj_f33_i05_d05 0.999853010933812",
  "bbob-biobj_f34_i05_d05 0.991915429264175",
  "bbob-biobj_f35_i05_d05 0.761468769860206",
  "bbob-biobj_f36_i05_d05 0.954687157341047",
  "bbob-biobj_f37_i05_d05 0.837648548374559",
  "bbob-biobj_f38_i05_d05 0.906706204321810",
  "bbob-biobj_f39_i05_d05 0.921014246577477",
  "bbob-biobj_f40_i05_d05 0.946421955464566",
  "bbob-biobj_f41_i05_d05 0.876463711014165",
  "bbob-biobj_f42_i05_d05 0.970370080999607",
  "bbob-biobj_f43_i05_d05 0.902429181207842",
  "bbob-biobj_f44_i05_d05 0.966251217785535",
  "bbob-biobj_f45_i05_d05 0.909303851254534",
  "bbob-biobj_f46_i05_d05 0.942517226712857",
  "bbob-biobj_f47_i05_d05 0.870061627456340",
  "bbob-biobj_f48_i05_d05 0.973954640318775",
  "bbob-biobj_f49_i05_d05 0.951109201752516",
  "bbob-biobj_f50_i05_d05 0.994266661801892",
  "bbob-biobj_f51_i05_d05 0.994438896814828",
  "bbob-biobj_f52_i05_d05 0.985991910384035",
  "bbob-biobj_f53_i05_d05 0.998401667793023",
  "bbob-biobj_f54_i05_d05 0.991382280345378",
  "bbob-biobj_f55_i05_d05 0.971967974360685",
  "bbob-biobj_f01_i01_d10 0.833292978406429",
  "bbob-biobj_f02_i01_d10 0.978186157890750",
  "bbob-biobj_f03_i01_d10 0.916881240370743",
  "bbob-biobj_f04_i01_d10 0.944821826480296",
  "bbob-biobj_f05_i01_d10 0.714020831794021",
  "bbob-biobj_f06_i01_d10 0.936989707669913",
  "bbob-biobj_f07_i01_d10 0.892671295146030",
  "bbob-biobj_f08_i01_d10 0.980887156696771",
  "bbob-biobj_f09_i01_d10 0.940787123769946",
  "bbob-biobj_f10_i01_d10 0.879640686792226",
  "bbob-biobj_f11_i01_d10 0.836568513611919",
  "bbob-biobj_f12_i01_d10 0.997429679126502",
  "bbob-biobj_f13_i01_d10 0.999276391665501",
  "bbob-biobj_f14_i01_d10 0.855942383415148",
  "bbob-biobj_f15_i01_d10 0.964995546568669",
  "bbob-biobj_f16_i01_d10 0.983714350137259",
  "bbob-biobj_f17_i01_d10 0.989748917262919",
  "bbob-biobj_f18_i01_d10 0.992629101085457",
  "bbob-biobj_f19_i01_d10 0.992011817987032",
  "bbob-biobj_f20_i01_d10 0.999902649949533",
  "bbob-biobj_f21_i01_d10 0.993976563634552",
  "bbob-biobj_f22_i01_d10 0.837772481127108",
  "bbob-biobj_f23_i01_d10 0.991303780174126",
  "bbob-biobj_f24_i01_d10 0.951302434936020",
  "bbob-biobj_f25_i01_d10 0.985022819774543",
  "bbob-biobj_f26_i01_d10 0.999623620340383",
  "bbob-biobj_f27_i01_d10 0.981630264752229",
  "bbob-biobj_f28_i01_d10 0.994065399842423",
  "bbob-biobj_f29_i01_d10 0.893969124629912",
  "bbob-biobj_f30_i01_d10 0.926766592526958",
  "bbob-biobj_f31_i01_d10 0.954335155663069",
  "bbob-biobj_f32_i01_d10 0.947204575153772",
  "bbob-biobj_f33_i01_d10 0.995511594131477",
  "bbob-biobj_f34_i01_d10 0.966121457771338",
  "bbob-biobj_f35_i01_d10 0.580711713148923",
  "bbob-biobj_f36_i01_d10 0.781206395704261",
  "bbob-biobj_f37_i01_d10 0.781596154216846",
  "bbob-biobj_f38_i01_d10 0.839658644425447",
  "bbob-biobj_f39_i01_d10 0.980090073789881",
  "bbob-biobj_f40_i01_d10 0.728253186329877",
  "bbob-biobj_f41_i01_d10 0.976581999493699",
  "bbob-biobj_f42_i01_d10 0.941402475493333",
  "bbob-biobj_f43_i01_d10 0.992948107011226",
  "bbob-biobj_f44_i01_d10 0.977876816504011",
  "bbob-biobj_f45_i01_d10 0.962462125568001",
  "bbob-biobj_f46_i01_d10 0.912013396057950",
  "bbob-biobj_f47_i01_d10 0.952444770559979",
  "bbob-biobj_f48_i01_d10 0.971871224059603",
  "bbob-biobj_f49_i01_d10 0.923586013533762",
  "bbob-biobj_f50_i01_d10 0.933926263240110",
  "bbob-biobj_f51_i01_d10 0.978991639829062",
  "bbob-biobj_f52_i01_d10 0.871725659507174",
  "bbob-biobj_f53_i01_d10 0.997770035036584",
  "bbob-biobj_f54_i01_d10 0.971012057853709",
  "bbob-biobj_f55_i01_d10 0.911747267012899",
  "bbob-biobj_f01_i02_d10 0.833290582871414",
  "bbob-biobj_f02_i02_d10 0.954015387344154",
  "bbob-biobj_f03_i02_d10 0.980158545829574",
  "bbob-biobj_f04_i02_d10 0.954662390636814",
  "bbob-biobj_f05_i02_d10 0.730694945666825",
  "bbob-biobj_f06_i02_d10 0.875212175676682",
  "bbob-biobj_f07_i02_d10 0.893835936342750",
  "bbob-biobj_f08_i02_d10 0.914384234660696",
  "bbob-biobj_f09_i02_d10 0.975986289864238",
  "bbob-biobj_f10_i02_d10 0.798453979906590",
  "bbob-biobj_f11_i02_d10 0.829484211464355",
  "bbob-biobj_f12_i02_d10 0.999751278208595",
  "bbob-biobj_f13_i02_d10 0.999204837951072",
  "bbob-biobj_f14_i02_d10 0.880190575748954",
  "bbob-biobj_f15_i02_d10 0.992132257104426",
  "bbob-biobj_f16_i02_d10 0.953795427645086",
  "bbob-biobj_f17_i02_d10 0.981080411732161",
  "bbob-biobj_f18_i02_d10 0.962760412503552",
  "bbob-biobj_f19_i02_d10 0.998068771692893",
  "bbob-biobj_f20_i02_d10 0.999674665199401",
  "bbob-biobj_f21_i02_d10 0.994603489196482",
  "bbob-biobj_f22_i02_d10 0.728688051429797",
  "bbob-biobj_f23_i02_d10 0.957095980950643",
  "bbob-biobj_f24_i02_d10 0.894438074642480",
  "bbob-biobj_f25_i02_d10 0.971411549291309",
  "bbob-biobj_f26_i02_d10 0.999634961765278",
  "bbob-biobj_f27_i02_d10 0.903581186079658",
  "bbob-biobj_f28_i02_d10 0.992207380799539",
  "bbob-biobj_f29_i02_d10 0.804885884975236",
  "bbob-biobj_f30_i02_d10 0.979256713902900",
  "bbob-biobj_f31_i02_d10 0.937099221837444",
  "bbob-biobj_f32_i02_d10 0.962293389331358",
  "bbob-biobj_f33_i02_d10 0.995928694733017",
  "bbob-biobj_f34_i02_d10 0.959843707994441",
  "bbob-biobj_f35_i02_d10 0.592123826021201",
  "bbob-biobj_f36_i02_d10 0.793724479284668",
  "bbob-biobj_f37_i02_d10 0.806039400985827",
  "bbob-biobj_f38_i02_d10 0.885571675810017",
  "bbob-biobj_f39_i02_d10 0.936962624051114",
  "bbob-biobj_f40_i02_d10 0.744507529762664",
  "bbob-biobj_f41_i02_d10 0.931044605526823",
  "bbob-biobj_f42_i02_d10 0.928064521463463",
  "bbob-biobj_f43_i02_d10 0.942162067513955",
  "bbob-biobj_f44_i02_d10 0.975027661488822",
  "bbob-biobj_f45_i02_d10 0.937565228444463",
  "bbob-biobj_f46_i02_d10 0.897891387257671",
  "bbob-biobj_f47_i02_d10 0.899311341134821",
  "bbob-biobj_f48_i02_d10 0.986529674800281",
  "bbob-biobj_f49_i02_d10 0.847194810018122",
  "bbob-biobj_f50_i02_d10 0.957430436852076",
  "bbob-biobj_f51_i02_d10 0.990809788249675",
  "bbob-biobj_f52_i02_d10 0.908040542561868",
  "bbob-biobj_f53_i02_d10 0.995011570971999",
  "bbob-biobj_f54_i02_d10 0.979013067298796",
  "bbob-biobj_f55_i02_d10 0.933494200367506",
  "bbob-biobj_f01_i03_d10 0.833290074261168",
  "bbob-biobj_f02_i03_d10 0.890659367706656",
  "bbob-biobj_f03_i03_d10 0.985495762424353",
  "bbob-biobj_f04_i03_d10 0.951441054303583",
  "bbob-biobj_f05_i03_d10 0.683889567823860",
  "bbob-biobj_f06_i03_d10 0.895911962613326",
  "bbob-biobj_f07_i03_d10 0.889141490151494",
  "bbob-biobj_f08_i03_d10 0.929528209845753",
  "bbob-biobj_f09_i03_d10 0.955035805359493",
  "bbob-biobj_f10_i03_d10 0.710801768774196",
  "bbob-biobj_f11_i03_d10 0.821919374566912",
  "bbob-biobj_f12_i03_d10 0.999904509420673",
  "bbob-biobj_f13_i03_d10 0.999218783176809",
  "bbob-biobj_f14_i03_d10 0.931139097610614",
  "bbob-biobj_f15_i03_d10 0.991358691358509",
  "bbob-biobj_f16_i03_d10 0.967220788577883",
  "bbob-biobj_f17_i03_d10 0.989693685552891",
  "bbob-biobj_f18_i03_d10 0.992473381710274",
  "bbob-biobj_f19_i03_d10 0.991727239977203",
  "bbob-biobj_f20_i03_d10 0.972677430070006",
  "bbob-biobj_f21_i03_d10 0.998433693554811",
  "bbob-biobj_f22_i03_d10 0.862392606638333",
  "bbob-biobj_f23_i03_d10 0.985383623808623",
  "bbob-biobj_f24_i03_d10 0.908859896675063",
  "bbob-biobj_f25_i03_d10 0.987502886787558",
  "bbob-biobj_f26_i03_d10 0.997282016963610",
  "bbob-biobj_f27_i03_d10 0.991361868145687",
  "bbob-biobj_f28_i03_d10 0.994065468345770",
  "bbob-biobj_f29_i03_d10 0.851027090910249",
  "bbob-biobj_f30_i03_d10 0.986307083425666",
  "bbob-biobj_f31_i03_d10 0.952876582353792",
  "bbob-biobj_f32_i03_d10 0.960252193878653",
  "bbob-biobj_f33_i03_d10 0.998451520498762",
  "bbob-biobj_f34_i03_d10 0.983079617313669",
  "bbob-biobj_f35_i03_d10 0.606593739472515",
  "bbob-biobj_f36_i03_d10 0.756162001692599",
  "bbob-biobj_f37_i03_d10 0.832532461249713",
  "bbob-biobj_f38_i03_d10 0.874391675806419",
  "bbob-biobj_f39_i03_d10 0.857769726279999",
  "bbob-biobj_f40_i03_d10 0.729629427484062",
  "bbob-biobj_f41_i03_d10 0.871973857380220",
  "bbob-biobj_f42_i03_d10 0.906855287051747",
  "bbob-biobj_f43_i03_d10 0.981933844465956",
  "bbob-biobj_f44_i03_d10 0.978358085126126",
  "bbob-biobj_f45_i03_d10 0.939019471197658",
  "bbob-biobj_f46_i03_d10 0.887418745213056",
  "bbob-biobj_f47_i03_d10 0.945131223885536",
  "bbob-biobj_f48_i03_d10 0.979584767225720",
  "bbob-biobj_f49_i03_d10 0.926518854653493",
  "bbob-biobj_f50_i03_d10 0.933296224836959",
  "bbob-biobj_f51_i03_d10 0.968183663247016",
  "bbob-biobj_f52_i03_d10 0.927273659877501",
  "bbob-biobj_f53_i03_d10 0.999758254041145",
  "bbob-biobj_f54_i03_d10 0.971479343841167",
  "bbob-biobj_f55_i03_d10 0.953336701115373",
  "bbob-biobj_f01_i04_d10 0.833290616484510",
  "bbob-biobj_f02_i04_d10 0.972586493712609",
  "bbob-biobj_f03_i04_d10 0.908641915848349",
  "bbob-biobj_f04_i04_d10 0.936521549540696",
  "bbob-biobj_f05_i04_d10 0.716260273341175",
  "bbob-biobj_f06_i04_d10 0.930039833002329",
  "bbob-biobj_f07_i04_d10 0.883309054692379",
  "bbob-biobj_f08_i04_d10 0.950092630117365",
  "bbob-biobj_f09_i04_d10 0.944313359892786",
  "bbob-biobj_f10_i04_d10 0.904274355825781",
  "bbob-biobj_f11_i04_d10 0.834665692135462",
  "bbob-biobj_f12_i04_d10 0.999925818132752",
  "bbob-biobj_f13_i04_d10 0.991555249595398",
  "bbob-biobj_f14_i04_d10 0.871642182353333",
  "bbob-biobj_f15_i04_d10 0.994602668464917",
  "bbob-biobj_f16_i04_d10 0.984001143853939",
  "bbob-biobj_f17_i04_d10 0.992358720621488",
  "bbob-biobj_f18_i04_d10 0.950020765554779",
  "bbob-biobj_f19_i04_d10 0.992321911751036",
  "bbob-biobj_f20_i04_d10 0.999930879613361",
  "bbob-biobj_f21_i04_d10 0.904309427391170",
  "bbob-biobj_f22_i04_d10 0.842157199476190",
  "bbob-biobj_f23_i04_d10 0.980009144094798",
  "bbob-biobj_f24_i04_d10 0.967444562272865",
  "bbob-biobj_f25_i04_d10 0.971825471108975",
  "bbob-biobj_f26_i04_d10 0.999901964296866",
  "bbob-biobj_f27_i04_d10 0.987018409937990",
  "bbob-biobj_f28_i04_d10 0.992784318025797",
  "bbob-biobj_f29_i04_d10 0.894877823153367",
  "bbob-biobj_f30_i04_d10 0.980895333912404",
  "bbob-biobj_f31_i04_d10 0.968424501416157",
  "bbob-biobj_f32_i04_d10 0.986592717525944",
  "bbob-biobj_f33_i04_d10 0.996076390480589",
  "bbob-biobj_f34_i04_d10 0.985600748857182",
  "bbob-biobj_f35_i04_d10 0.582430180978704",
  "bbob-biobj_f36_i04_d10 0.850505488775800",
  "bbob-biobj_f37_i04_d10 0.782186014741844",
  "bbob-biobj_f38_i04_d10 0.883814171115460",
  "bbob-biobj_f39_i04_d10 0.905978834540773",
  "bbob-biobj_f40_i04_d10 0.707815492885900",
  "bbob-biobj_f41_i04_d10 0.951214535801685",
  "bbob-biobj_f42_i04_d10 0.868210535403942",
  "bbob-biobj_f43_i04_d10 0.961476365589940",
  "bbob-biobj_f44_i04_d10 0.987127077215091",
  "bbob-biobj_f45_i04_d10 0.932558858551399",
  "bbob-biobj_f46_i04_d10 0.902412857071501",
  "bbob-biobj_f47_i04_d10 0.951407500128080",
  "bbob-biobj_f48_i04_d10 0.970244582385456",
  "bbob-biobj_f49_i04_d10 0.878070391585995",
  "bbob-biobj_f50_i04_d10 0.978954738647618",
  "bbob-biobj_f51_i04_d10 0.997434032890661",
  "bbob-biobj_f52_i04_d10 0.991665441681003",
  "bbob-biobj_f53_i04_d10 0.998086298388451",
  "bbob-biobj_f54_i04_d10 0.987229116292269",
  "bbob-biobj_f55_i04_d10 0.853189259275938",
  "bbob-biobj_f01_i05_d10 0.833292291687670",
  "bbob-biobj_f02_i05_d10 0.933906138198159",
  "bbob-biobj_f03_i05_d10 0.881566802414681",
  "bbob-biobj_f04_i05_d10 0.941617512589267",
  "bbob-biobj_f05_i05_d10 0.749448355510482",
  "bbob-biobj_f06_i05_d10 0.743322727228610",
  "bbob-biobj_f07_i05_d10 0.867122678653645",
  "bbob-biobj_f08_i05_d10 0.928738765599749",
  "bbob-biobj_f09_i05_d10 0.940721644129061",
  "bbob-biobj_f10_i05_d10 0.842803695957583",
  "bbob-biobj_f11_i05_d10 0.840890202958803",
  "bbob-biobj_f12_i05_d10 0.999713183960265",
  "bbob-biobj_f13_i05_d10 0.994187469334294",
  "bbob-biobj_f14_i05_d10 0.940413629063480",
  "bbob-biobj_f15_i05_d10 0.985317312964736",
  "bbob-biobj_f16_i05_d10 0.973755090012530",
  "bbob-biobj_f17_i05_d10 0.987526807338334",
  "bbob-biobj_f18_i05_d10 0.954457659398138",
  "bbob-biobj_f19_i05_d10 0.992080633898391",
  "bbob-biobj_f20_i05_d10 0.999846078037702",
  "bbob-biobj_f21_i05_d10 0.958073478645331",
  "bbob-biobj_f22_i05_d10 0.819446456617006",
  "bbob-biobj_f23_i05_d10 0.954447620118574",
  "bbob-biobj_f24_i05_d10 0.938514787676160",
  "bbob-biobj_f25_i05_d10 0.956803883288751",
  "bbob-biobj_f26_i05_d10 0.998146168797799",
  "bbob-biobj_f27_i05_d10 0.981690114394249",
  "bbob-biobj_f28_i05_d10 0.993719824098620",
  "bbob-biobj_f29_i05_d10 0.900580483041280",
  "bbob-biobj_f30_i05_d10 0.978381478368368",
  "bbob-biobj_f31_i05_d10 0.970546900414730",
  "bbob-biobj_f32_i05_d10 0.992912108242810",
  "bbob-biobj_f33_i05_d10 0.999693756289780",
  "bbob-biobj_f34_i05_d10 0.986747418676788",
  "bbob-biobj_f35_i05_d10 0.597642642362544",
  "bbob-biobj_f36_i05_d10 0.808684862939724",
  "bbob-biobj_f37_i05_d10 0.754869791041963",
  "bbob-biobj_f38_i05_d10 0.872711761339408",
  "bbob-biobj_f39_i05_d10 0.884061916865357",
  "bbob-biobj_f40_i05_d10 0.738254830143031",
  "bbob-biobj_f41_i05_d10 0.966114871816312",
  "bbob-biobj_f42_i05_d10 0.912900396135625",
  "bbob-biobj_f43_i05_d10 0.968627057192036",
  "bbob-biobj_f44_i05_d10 0.972820224675321",
  "bbob-biobj_f45_i05_d10 0.875458356931092",
  "bbob-biobj_f46_i05_d10 0.895569952154866",
  "bbob-biobj_f47_i05_d10 0.974872587495967",
  "bbob-biobj_f48_i05_d10 0.979607662302305",
  "bbob-biobj_f49_i05_d10 0.874128159982844",
  "bbob-biobj_f50_i05_d10 0.973107044380098",
  "bbob-biobj_f51_i05_d10 0.964617015167318",
  "bbob-biobj_f52_i05_d10 0.879809048011156",
  "bbob-biobj_f53_i05_d10 0.999964504148671",
  "bbob-biobj_f54_i05_d10 0.963565055756231",
  "bbob-biobj_f55_i05_d10 0.931471452678062",
  "bbob-biobj_f01_i01_d20 0.833185904412730",
  "bbob-biobj_f02_i01_d20 0.951879760885421",
  "bbob-biobj_f03_i01_d20 0.887034106660898",
  "bbob-biobj_f04_i01_d20 0.935828732838047",
  "bbob-biobj_f05_i01_d20 0.694320591154180",
  "bbob-biobj_f06_i01_d20 0.930982077320492",
  "bbob-biobj_f07_i01_d20 0.941094512447098",
  "bbob-biobj_f08_i01_d20 0.964979887551948",
  "bbob-biobj_f09_i01_d20 0.960292556025729",
  "bbob-biobj_f10_i01_d20 0.840913568791116",
  "bbob-biobj_f11_i01_d20 0.836353784774267",
  "bbob-biobj_f12_i01_d20 0.999353367517152",
  "bbob-biobj_f13_i01_d20 0.949968158767985",
  "bbob-biobj_f14_i01_d20 0.838028753234985",
  "bbob-biobj_f15_i01_d20 0.988152204597621",
  "bbob-biobj_f16_i01_d20 0.967418396951644",
  "bbob-biobj_f17_i01_d20 0.961976829573151",
  "bbob-biobj_f18_i01_d20 0.947902479133759",
  "bbob-biobj_f19_i01_d20 0.985654207450615",
  "bbob-biobj_f20_i01_d20 0.999754625794669",
  "bbob-biobj_f21_i01_d20 0.981392978940988",
  "bbob-biobj_f22_i01_d20 0.771073216843813",
  "bbob-biobj_f23_i01_d20 0.941082365614892",
  "bbob-biobj_f24_i01_d20 0.938837784563285",
  "bbob-biobj_f25_i01_d20 0.973606632317517",
  "bbob-biobj_f26_i01_d20 0.999902167548546",
  "bbob-biobj_f27_i01_d20 0.977435672212356",
  "bbob-biobj_f28_i01_d20 0.992419760470962",
  "bbob-biobj_f29_i01_d20 0.808558091384653",
  "bbob-biobj_f30_i01_d20 0.961962695453807",
  "bbob-biobj_f31_i01_d20 0.952918435698674",
  "bbob-biobj_f32_i01_d20 0.974201660284137",
  "bbob-biobj_f33_i01_d20 0.990220567975616",
  "bbob-biobj_f34_i01_d20 0.949944119709586",
  "bbob-biobj_f35_i01_d20 0.554742901893168",
  "bbob-biobj_f36_i01_d20 0.804503390556641",
  "bbob-biobj_f37_i01_d20 0.839647691394615",
  "bbob-biobj_f38_i01_d20 0.852470443769851",
  "bbob-biobj_f39_i01_d20 0.866767035706089",
  "bbob-biobj_f40_i01_d20 0.603655395940413",
  "bbob-biobj_f41_i01_d20 0.971476433309333",
  "bbob-biobj_f42_i01_d20 0.957800101909049",
  "bbob-biobj_f43_i01_d20 0.933698802979667",
  "bbob-biobj_f44_i01_d20 0.989473622757610",
  "bbob-biobj_f45_i01_d20 0.916261834183411",
  "bbob-biobj_f46_i01_d20 0.930681466224983",
  "bbob-biobj_f47_i01_d20 0.945093837018368",
  "bbob-biobj_f48_i01_d20 0.967793252199761",
  "bbob-biobj_f49_i01_d20 0.882218965514988",
  "bbob-biobj_f50_i01_d20 0.961919064060329",
  "bbob-biobj_f51_i01_d20 0.985273744129146",
  "bbob-biobj_f52_i01_d20 0.945425704680598",
  "bbob-biobj_f53_i01_d20 0.993616976712955",
  "bbob-biobj_f54_i01_d20 0.975704363282836",
  "bbob-biobj_f55_i01_d20 0.589593665487798",
  "bbob-biobj_f01_i02_d20 0.833207552060403",
  "bbob-biobj_f02_i02_d20 0.980904237253779",
  "bbob-biobj_f03_i02_d20 0.950022939724837",
  "bbob-biobj_f04_i02_d20 0.941710836096996",
  "bbob-biobj_f05_i02_d20 0.689040055642120",
  "bbob-biobj_f06_i02_d20 0.910480566964683",
  "bbob-biobj_f07_i02_d20 0.896241865239925",
  "bbob-biobj_f08_i02_d20 0.900729077728150",
  "bbob-biobj_f09_i02_d20 0.962586221986243",
  "bbob-biobj_f10_i02_d20 0.812023622265132",
  "bbob-biobj_f11_i02_d20 0.835438688258357",
  "bbob-biobj_f12_i02_d20 0.999563671412677",
  "bbob-biobj_f13_i02_d20 0.970013320088166",
  "bbob-biobj_f14_i02_d20 0.908925932399497",
  "bbob-biobj_f15_i02_d20 0.957692036968514",
  "bbob-biobj_f16_i02_d20 0.968556195438330",
  "bbob-biobj_f17_i02_d20 0.958204217188874",
  "bbob-biobj_f18_i02_d20 0.950121118486904",
  "bbob-biobj_f19_i02_d20 0.971648034605764",
  "bbob-biobj_f20_i02_d20 0.999628088279715",
  "bbob-biobj_f21_i02_d20 0.993485919542800",
  "bbob-biobj_f22_i02_d20 0.756555470428662",
  "bbob-biobj_f23_i02_d20 0.916152727277069",
  "bbob-biobj_f24_i02_d20 0.985351328166353",
  "bbob-biobj_f25_i02_d20 0.927089439963058",
  "bbob-biobj_f26_i02_d20 0.996319681989621",
  "bbob-biobj_f27_i02_d20 0.966076293144647",
  "bbob-biobj_f28_i02_d20 0.989904675571837",
  "bbob-biobj_f29_i02_d20 0.858914192126769",
  "bbob-biobj_f30_i02_d20 0.948003767444235",
  "bbob-biobj_f31_i02_d20 0.954964697786911",
  "bbob-biobj_f32_i02_d20 0.952825148680034",
  "bbob-biobj_f33_i02_d20 0.995145856008500",
  "bbob-biobj_f34_i02_d20 0.945734288886854",
  "bbob-biobj_f35_i02_d20 0.562626515534495",
  "bbob-biobj_f36_i02_d20 0.853002962519736",
  "bbob-biobj_f37_i02_d20 0.791206082149838",
  "bbob-biobj_f38_i02_d20 0.831416002232897",
  "bbob-biobj_f39_i02_d20 0.925983417616201",
  "bbob-biobj_f40_i02_d20 0.642477793559315",
  "bbob-biobj_f41_i02_d20 0.981567395886612",
  "bbob-biobj_f42_i02_d20 0.923183103750550",
  "bbob-biobj_f43_i02_d20 0.949407597000968",
  "bbob-biobj_f44_i02_d20 0.984808860864267",
  "bbob-biobj_f45_i02_d20 0.852098050671580",
  "bbob-biobj_f46_i02_d20 0.958146492122573",
  "bbob-biobj_f47_i02_d20 0.945678805745313",
  "bbob-biobj_f48_i02_d20 0.967533523071136",
  "bbob-biobj_f49_i02_d20 0.837247494706153",
  "bbob-biobj_f50_i02_d20 0.959931559263060",
  "bbob-biobj_f51_i02_d20 0.992534422266171",
  "bbob-biobj_f52_i02_d20 0.950858291006400",
  "bbob-biobj_f53_i02_d20 0.987969509582036",
  "bbob-biobj_f54_i02_d20 0.978225145117985",
  "bbob-biobj_f55_i02_d20 0.699993045686106",
  "bbob-biobj_f01_i03_d20 0.833207521400968",
  "bbob-biobj_f02_i03_d20 0.971885666260855",
  "bbob-biobj_f03_i03_d20 0.867590832620030",
  "bbob-biobj_f04_i03_d20 0.931434213203625",
  "bbob-biobj_f05_i03_d20 0.696985444884746",
  "bbob-biobj_f06_i03_d20 0.932749941659490",
  "bbob-biobj_f07_i03_d20 0.892063400048704",
  "bbob-biobj_f08_i03_d20 0.942965524283337",
  "bbob-biobj_f09_i03_d20 0.970314871167049",
  "bbob-biobj_f10_i03_d20 0.798213543037889",
  "bbob-biobj_f11_i03_d20 0.835169240533775",
  "bbob-biobj_f12_i03_d20 0.999764598785675",
  "bbob-biobj_f13_i03_d20 0.945554751631722",
  "bbob-biobj_f14_i03_d20 0.918198725867560",
  "bbob-biobj_f15_i03_d20 0.982795143126842",
  "bbob-biobj_f16_i03_d20 0.982441628031722",
  "bbob-biobj_f17_i03_d20 0.982008215732887",
  "bbob-biobj_f18_i03_d20 0.960648393061043",
  "bbob-biobj_f19_i03_d20 0.991084701753494",
  "bbob-biobj_f20_i03_d20 0.999797571885116",
  "bbob-biobj_f21_i03_d20 0.993042259812321",
  "bbob-biobj_f22_i03_d20 0.863037194905385",
  "bbob-biobj_f23_i03_d20 0.969015262636722",
  "bbob-biobj_f24_i03_d20 0.974158431697382",
  "bbob-biobj_f25_i03_d20 0.980237145217646",
  "bbob-biobj_f26_i03_d20 0.999941845924102",
  "bbob-biobj_f27_i03_d20 0.966326579324806",
  "bbob-biobj_f28_i03_d20 0.993689625845595",
  "bbob-biobj_f29_i03_d20 0.830492615308720",
  "bbob-biobj_f30_i03_d20 0.967954312930621",
  "bbob-biobj_f31_i03_d20 0.965341381122931",
  "bbob-biobj_f32_i03_d20 0.976568072290851",
  "bbob-biobj_f33_i03_d20 0.998359564750281",
  "bbob-biobj_f34_i03_d20 0.940411316149631",
  "bbob-biobj_f35_i03_d20 0.607203080701355",
  "bbob-biobj_f36_i03_d20 0.787245980941896",
  "bbob-biobj_f37_i03_d20 0.803796059074190",
  "bbob-biobj_f38_i03_d20 0.823215690310603",
  "bbob-biobj_f39_i03_d20 0.867987216605950",
  "bbob-biobj_f40_i03_d20 0.526915574313714",
  "bbob-biobj_f41_i03_d20 0.940969566962786",
  "bbob-biobj_f42_i03_d20 0.934841434007581",
  "bbob-biobj_f43_i03_d20 0.969352413799934",
  "bbob-biobj_f44_i03_d20 0.977777943741785",
  "bbob-biobj_f45_i03_d20 0.945073198593724",
  "bbob-biobj_f46_i03_d20 0.909377080263067",
  "bbob-biobj_f47_i03_d20 0.934536748258848",
  "bbob-biobj_f48_i03_d20 0.968046040742238",
  "bbob-biobj_f49_i03_d20 0.866707518347003",
  "bbob-biobj_f50_i03_d20 0.952394555329134",
  "bbob-biobj_f51_i03_d20 0.979882710827667",
  "bbob-biobj_f52_i03_d20 0.884818897545810",
  "bbob-biobj_f53_i03_d20 0.998224919757182",
  "bbob-biobj_f54_i03_d20 0.985256057145942",
  "bbob-biobj_f55_i03_d20 0.627087123225424",
  "bbob-biobj_f01_i04_d20 0.833192956226617",
  "bbob-biobj_f02_i04_d20 0.977340500860808",
  "bbob-biobj_f03_i04_d20 0.932742245214901",
  "bbob-biobj_f04_i04_d20 0.942489452391897",
  "bbob-biobj_f05_i04_d20 0.705079145373040",
  "bbob-biobj_f06_i04_d20 0.900973814111433",
  "bbob-biobj_f07_i04_d20 0.890270640994631",
  "bbob-biobj_f08_i04_d20 0.952939543470624",
  "bbob-biobj_f09_i04_d20 0.961704687457553",
  "bbob-biobj_f10_i04_d20 0.799982069242448",
  "bbob-biobj_f11_i04_d20 0.838271802147750",
  "bbob-biobj_f12_i04_d20 0.999893480350420",
  "bbob-biobj_f13_i04_d20 0.997752775119298",
  "bbob-biobj_f14_i04_d20 0.901904902084317",
  "bbob-biobj_f15_i04_d20 0.973572777182183",
  "bbob-biobj_f16_i04_d20 0.970719482640897",
  "bbob-biobj_f17_i04_d20 0.991471773388835",
  "bbob-biobj_f18_i04_d20 0.933574764370316",
  "bbob-biobj_f19_i04_d20 0.976602774560082",
  "bbob-biobj_f20_i04_d20 0.999612927554968",
  "bbob-biobj_f21_i04_d20 0.996013897009277",
  "bbob-biobj_f22_i04_d20 0.915550717900235",
  "bbob-biobj_f23_i04_d20 0.983076179674251",
  "bbob-biobj_f24_i04_d20 0.946754908478271",
  "bbob-biobj_f25_i04_d20 0.989725294009121",
  "bbob-biobj_f26_i04_d20 0.999909932046990",
  "bbob-biobj_f27_i04_d20 0.978836095719902",
  "bbob-biobj_f28_i04_d20 0.994424823780871",
  "bbob-biobj_f29_i04_d20 0.865072550285352",
  "bbob-biobj_f30_i04_d20 0.980946148952653",
  "bbob-biobj_f31_i04_d20 0.972561422620203",
  "bbob-biobj_f32_i04_d20 0.987310542589914",
  "bbob-biobj_f33_i04_d20 0.996908039867623",
  "bbob-biobj_f34_i04_d20 0.941896717675520",
  "bbob-biobj_f35_i04_d20 0.561293360823185",
  "bbob-biobj_f36_i04_d20 0.765144559919577",
  "bbob-biobj_f37_i04_d20 0.832238517568998",
  "bbob-biobj_f38_i04_d20 0.895786547029985",
  "bbob-biobj_f39_i04_d20 0.876651090885301",
  "bbob-biobj_f40_i04_d20 0.748024984489584",
  "bbob-biobj_f41_i04_d20 0.913016279845638",
  "bbob-biobj_f42_i04_d20 0.946069738747989",
  "bbob-biobj_f43_i04_d20 0.961067346749245",
  "bbob-biobj_f44_i04_d20 0.984380170569094",
  "bbob-biobj_f45_i04_d20 0.929099360296714",
  "bbob-biobj_f46_i04_d20 0.929183556098456",
  "bbob-biobj_f47_i04_d20 0.937896856037121",
  "bbob-biobj_f48_i04_d20 0.971950258196312",
  "bbob-biobj_f49_i04_d20 0.864994789361628",
  "bbob-biobj_f50_i04_d20 0.983291149779151",
  "bbob-biobj_f51_i04_d20 0.984992310658653",
  "bbob-biobj_f52_i04_d20 0.918989825158548",
  "bbob-biobj_f53_i04_d20 0.998920489662993",
  "bbob-biobj_f54_i04_d20 0.980047132463936",
  "bbob-biobj_f55_i04_d20 0.672975982289736",
  "bbob-biobj_f01_i05_d20 0.833193870767885",
  "bbob-biobj_f02_i05_d20 0.962907610238820",
  "bbob-biobj_f03_i05_d20 0.875546362923703",
  "bbob-biobj_f04_i05_d20 0.948719015894941",
  "bbob-biobj_f05_i05_d20 0.695336775493654",
  "bbob-biobj_f06_i05_d20 0.918011871071976",
  "bbob-biobj_f07_i05_d20 0.882566753824316",
  "bbob-biobj_f08_i05_d20 0.899000868137602",
  "bbob-biobj_f09_i05_d20 0.963712318231473",
  "bbob-biobj_f10_i05_d20 0.778425108988740",
  "bbob-biobj_f11_i05_d20 0.841599694854327",
  "bbob-biobj_f12_i05_d20 0.999092422297150",
  "bbob-biobj_f13_i05_d20 0.998296287709533",
  "bbob-biobj_f14_i05_d20 0.885441381715590",
  "bbob-biobj_f15_i05_d20 0.988065459065354",
  "bbob-biobj_f16_i05_d20 0.972334277129038",
  "bbob-biobj_f17_i05_d20 0.997205869239149",
  "bbob-biobj_f18_i05_d20 0.975433208837123",
  "bbob-biobj_f19_i05_d20 0.984394589632036",
  "bbob-biobj_f20_i05_d20 0.958353867297278",
  "bbob-biobj_f21_i05_d20 0.982048525244220",
  "bbob-biobj_f22_i05_d20 0.789275134960876",
  "bbob-biobj_f23_i05_d20 0.958464846939110",
  "bbob-biobj_f24_i05_d20 0.920973148314878",
  "bbob-biobj_f25_i05_d20 0.952640635479999",
  "bbob-biobj_f26_i05_d20 0.994301440957288",
  "bbob-biobj_f27_i05_d20 0.926053468064325",
  "bbob-biobj_f28_i05_d20 0.995255418032090",
  "bbob-biobj_f29_i05_d20 0.835821580662857",
  "bbob-biobj_f30_i05_d20 0.986514821846012",
  "bbob-biobj_f31_i05_d20 0.956051937738176",
  "bbob-biobj_f32_i05_d20 0.986798393586952",
  "bbob-biobj_f33_i05_d20 0.994306994720448",
  "bbob-biobj_f34_i05_d20 0.946864128557638",
  "bbob-biobj_f35_i05_d20 0.540572168829766",
  "bbob-biobj_f36_i05_d20 0.865754635393962",
  "bbob-biobj_f37_i05_d20 0.755137216311259",
  "bbob-biobj_f38_i05_d20 0.873197184881281",
  "bbob-biobj_f39_i05_d20 0.868176430670188",
  "bbob-biobj_f40_i05_d20 0.618872595399021",
  "bbob-biobj_f41_i05_d20 0.924477043807865",
  "bbob-biobj_f42_i05_d20 0.947227551485381",
  "bbob-biobj_f43_i05_d20 0.976336336221268",
  "bbob-biobj_f44_i05_d20 0.974125388255757",
  "bbob-biobj_f45_i05_d20 0.913161181130696",
  "bbob-biobj_f46_i05_d20 0.893095136109460",
  "bbob-biobj_f47_i05_d20 0.907278089365908",
  "bbob-biobj_f48_i05_d20 0.951741879255749",
  "bbob-biobj_f49_i05_d20 0.838461412725769",
  "bbob-biobj_f50_i05_d20 0.977959065747409",
  "bbob-biobj_f51_i05_d20 0.980069732709530",
  "bbob-biobj_f52_i05_d20 0.935549795477864",
  "bbob-biobj_f53_i05_d20 0.995339867406872",
  "bbob-biobj_f54_i05_d20 0.984327935423081",
  "bbob-biobj_f55_i05_d20 0.831000840653062",
  "bbob-biobj_f01_i01_d40 1",
  "bbob-biobj_f02_i01_d40 1",
  "bbob-biobj_f03_i01_d40 1",
  "bbob-biobj_f04_i01_d40 1",
  "bbob-biobj_f05_i01_d40 1",
  "bbob-biobj_f06_i01_d40 1",
  "bbob-biobj_f07_i01_d40 1",
  "bbob-biobj_f08_i01_d40 1",
  "bbob-biobj_f09_i01_d40 1",
  "bbob-biobj_f10_i01_d40 1",
  "bbob-biobj_f11_i01_d40 1",
  "bbob-biobj_f12_i01_d40 1",
  "bbob-biobj_f13_i01_d40 1",
  "bbob-biobj_f14_i01_d40 1",
  "bbob-biobj_f15_i01_d40 1",
  "bbob-biobj_f16_i01_d40 1",
  "bbob-biobj_f17_i01_d40 1",
  "bbob-biobj_f18_i01_d40 1",
  "bbob-biobj_f19_i01_d40 1",
  "bbob-biobj_f20_i01_d40 1",
  "bbob-biobj_f21_i01_d40 1",
  "bbob-biobj_f22_i01_d40 1",
  "bbob-biobj_f23_i01_d40 1",
  "bbob-biobj_f24_i01_d40 1",
  "bbob-biobj_f25_i01_d40 1",
  "bbob-biobj_f26_i01_d40 1",
  "bbob-biobj_f27_i01_d40 1",
  "bbob-biobj_f28_i01_d40 1",
  "bbob-biobj_f29_i01_d40 1",
  "bbob-biobj_f30_i01_d40 1",
  "bbob-biobj_f31_i01_d40 1",
  "bbob-biobj_f32_i01_d40 1",
  "bbob-biobj_f33_i01_d40 1",
  "bbob-biobj_f34_i01_d40 1",
  "bbob-biobj_f35_i01_d40 1",
  "bbob-biobj_f36_i01_d40 1",
  "bbob-biobj_f37_i01_d40 1",
  "bbob-biobj_f38_i01_d40 1",
  "bbob-biobj_f39_i01_d40 1",
  "bbob-biobj_f40_i01_d40 1",
  "bbob-biobj_f41_i01_d40 1",
  "bbob-biobj_f42_i01_d40 1",
  "bbob-biobj_f43_i01_d40 1",
  "bbob-biobj_f44_i01_d40 1",
  "bbob-biobj_f45_i01_d40 1",
  "bbob-biobj_f46_i01_d40 1",
  "bbob-biobj_f47_i01_d40 1",
  "bbob-biobj_f48_i01_d40 1",
  "bbob-biobj_f49_i01_d40 1",
  "bbob-biobj_f50_i01_d40 1",
  "bbob-biobj_f51_i01_d40 1",
  "bbob-biobj_f52_i01_d40 1",
  "bbob-biobj_f53_i01_d40 1",
  "bbob-biobj_f54_i01_d40 1",
  "bbob-biobj_f55_i01_d40 1",
  "bbob-biobj_f01_i02_d40 1",
  "bbob-biobj_f02_i02_d40 1",
  "bbob-biobj_f03_i02_d40 1",
  "bbob-biobj_f04_i02_d40 1",
  "bbob-biobj_f05_i02_d40 1",
  "bbob-biobj_f06_i02_d40 1",
  "bbob-biobj_f07_i02_d40 1",
  "bbob-biobj_f08_i02_d40 1",
  "bbob-biobj_f09_i02_d40 1",
  "bbob-biobj_f10_i02_d40 1",
  "bbob-biobj_f11_i02_d40 1",
  "bbob-biobj_f12_i02_d40 1",
  "bbob-biobj_f13_i02_d40 1",
  "bbob-biobj_f14_i02_d40 1",
  "bbob-biobj_f15_i02_d40 1",
  "bbob-biobj_f16_i02_d40 1",
  "bbob-biobj_f17_i02_d40 1",
  "bbob-biobj_f18_i02_d40 1",
  "bbob-biobj_f19_i02_d40 1",
  "bbob-biobj_f20_i02_d40 1",
  "bbob-biobj_f21_i02_d40 1",
  "bbob-biobj_f22_i02_d40 1",
  "bbob-biobj_f23_i02_d40 1",
  "bbob-biobj_f24_i02_d40 1",
  "bbob-biobj_f25_i02_d40 1",
  "bbob-biobj_f26_i02_d40 1",
  "bbob-biobj_f27_i02_d40 1",
  "bbob-biobj_f28_i02_d40 1",
  "bbob-biobj_f29_i02_d40 1",
  "bbob-biobj_f30_i02_d40 1",
  "bbob-biobj_f31_i02_d40 1",
  "bbob-biobj_f32_i02_d40 1",
  "bbob-biobj_f33_i02_d40 1",
  "bbob-biobj_f34_i02_d40 1",
  "bbob-biobj_f35_i02_d40 1",
  "bbob-biobj_f36_i02_d40 1",
  "bbob-biobj_f37_i02_d40 1",
  "bbob-biobj_f38_i02_d40 1",
  "bbob-biobj_f39_i02_d40 1",
  "bbob-biobj_f40_i02_d40 1",
  "bbob-biobj_f41_i02_d40 1",
  "bbob-biobj_f42_i02_d40 1",
  "bbob-biobj_f43_i02_d40 1",
  "bbob-biobj_f44_i02_d40 1",
  "bbob-biobj_f45_i02_d40 1",
  "bbob-biobj_f46_i02_d40 1",
  "bbob-biobj_f47_i02_d40 1",
  "bbob-biobj_f48_i02_d40 1",
  "bbob-biobj_f49_i02_d40 1",
  "bbob-biobj_f50_i02_d40 1",
  "bbob-biobj_f51_i02_d40 1",
  "bbob-biobj_f52_i02_d40 1",
  "bbob-biobj_f53_i02_d40 1",
  "bbob-biobj_f54_i02_d40 1",
  "bbob-biobj_f55_i02_d40 1",
  "bbob-biobj_f01_i03_d40 1",
  "bbob-biobj_f02_i03_d40 1",
  "bbob-biobj_f03_i03_d40 1",
  "bbob-biobj_f04_i03_d40 1",
  "bbob-biobj_f05_i03_d40 1",
  "bbob-biobj_f06_i03_d40 1",
  "bbob-biobj_f07_i03_d40 1",
  "bbob-biobj_f08_i03_d40 1",
  "bbob-biobj_f09_i03_d40 1",
  "bbob-biobj_f10_i03_d40 1",
  "bbob-biobj_f11_i03_d40 1",
  "bbob-biobj_f12_i03_d40 1",
  "bbob-biobj_f13_i03_d40 1",
  "bbob-biobj_f14_i03_d40 1",
  "bbob-biobj_f15_i03_d40 1",
  "bbob-biobj_f16_i03_d40 1",
  "bbob-biobj_f17_i03_d40 1",
  "bbob-biobj_f18_i03_d40 1",
  "bbob-biobj_f19_i03_d40 1",
  "bbob-biobj_f20_i03_d40 1",
  "bbob-biobj_f21_i03_d40 1",
  "bbob-biobj_f22_i03_d40 1",
  "bbob-biobj_f23_i03_d40 1",
  "bbob-biobj_f24_i03_d40 1",
  "bbob-biobj_f25_i03_d40 1",
  "bbob-biobj_f26_i03_d40 1",
  "bbob-biobj_f27_i03_d40 1",
  "bbob-biobj_f28_i03_d40 1",
  "bbob-biobj_f29_i03_d40 1",
  "bbob-biobj_f30_i03_d40 1",
  "bbob-biobj_f31_i03_d40 1",
  "bbob-biobj_f32_i03_d40 1",
  "bbob-biobj_f33_i03_d40 1",
  "bbob-biobj_f34_i03_d40 1",
  "bbob-biobj_f35_i03_d40 1",
  "bbob-biobj_f36_i03_d40 1",
  "bbob-biobj_f37_i03_d40 1",
  "bbob-biobj_f38_i03_d40 1",
  "bbob-biobj_f39_i03_d40 1",
  "bbob-biobj_f40_i03_d40 1",
  "bbob-biobj_f41_i03_d40 1",
  "bbob-biobj_f42_i03_d40 1",
  "bbob-biobj_f43_i03_d40 1",
  "bbob-biobj_f44_i03_d40 1",
  "bbob-biobj_f45_i03_d40 1",
  "bbob-biobj_f46_i03_d40 1",
  "bbob-biobj_f47_i03_d40 1",
  "bbob-biobj_f48_i03_d40 1",
  "bbob-biobj_f49_i03_d40 1",
  "bbob-biobj_f50_i03_d40 1",
  "bbob-biobj_f51_i03_d40 1",
  "bbob-biobj_f52_i03_d40 1",
  "bbob-biobj_f53_i03_d40 1",
  "bbob-biobj_f54_i03_d40 1",
  "bbob-biobj_f55_i03_d40 1",
  "bbob-biobj_f01_i04_d40 1",
  "bbob-biobj_f02_i04_d40 1",
  "bbob-biobj_f03_i04_d40 1",
  "bbob-biobj_f04_i04_d40 1",
  "bbob-biobj_f05_i04_d40 1",
  "bbob-biobj_f06_i04_d40 1",
  "bbob-biobj_f07_i04_d40 1",
  "bbob-biobj_f08_i04_d40 1",
  "bbob-biobj_f09_i04_d40 1",
  "bbob-biobj_f10_i04_d40 1",
  "bbob-biobj_f11_i04_d40 1",
  "bbob-biobj_f12_i04_d40 1",
  "bbob-biobj_f13_i04_d40 1",
  "bbob-biobj_f14_i04_d40 1",
  "bbob-biobj_f15_i04_d40 1",
  "bbob-biobj_f16_i04_d40 1",
  "bbob-biobj_f17_i04_d40 1",
  "bbob-biobj_f18_i04_d40 1",
  "bbob-biobj_f19_i04_d40 1",
  "bbob-biobj_f20_i04_d40 1",
  "bbob-biobj_f21_i04_d40 1",
  "bbob-biobj_f22_i04_d40 1",
  "bbob-biobj_f23_i04_d40 1",
  "bbob-biobj_f24_i04_d40 1",
  "bbob-biobj_f25_i04_d40 1",
  "bbob-biobj_f26_i04_d40 1",
  "bbob-biobj_f27_i04_d40 1",
  "bbob-biobj_f28_i04_d40 1",
  "bbob-biobj_f29_i04_d40 1",
  "bbob-biobj_f30_i04_d40 1",
  "bbob-biobj_f31_i04_d40 1",
  "bbob-biobj_f32_i04_d40 1",
  "bbob-biobj_f33_i04_d40 1",
  "bbob-biobj_f34_i04_d40 1",
  "bbob-biobj_f35_i04_d40 1",
  "bbob-biobj_f36_i04_d40 1",
  "bbob-biobj_f37_i04_d40 1",
  "bbob-biobj_f38_i04_d40 1",
  "bbob-biobj_f39_i04_d40 1",
  "bbob-biobj_f40_i04_d40 1",
  "bbob-biobj_f41_i04_d40 1",
  "bbob-biobj_f42_i04_d40 1",
  "bbob-biobj_f43_i04_d40 1",
  "bbob-biobj_f44_i04_d40 1",
  "bbob-biobj_f45_i04_d40 1",
  "bbob-biobj_f46_i04_d40 1",
  "bbob-biobj_f47_i04_d40 1",
  "bbob-biobj_f48_i04_d40 1",
  "bbob-biobj_f49_i04_d40 1",
  "bbob-biobj_f50_i04_d40 1",
  "bbob-biobj_f51_i04_d40 1",
  "bbob-biobj_f52_i04_d40 1",
  "bbob-biobj_f53_i04_d40 1",
  "bbob-biobj_f54_i04_d40 1",
  "bbob-biobj_f55_i04_d40 1",
  "bbob-biobj_f01_i05_d40 1",
  "bbob-biobj_f02_i05_d40 1",
  "bbob-biobj_f03_i05_d40 1",
  "bbob-biobj_f04_i05_d40 1",
  "bbob-biobj_f05_i05_d40 1",
  "bbob-biobj_f06_i05_d40 1",
  "bbob-biobj_f07_i05_d40 1",
  "bbob-biobj_f08_i05_d40 1",
  "bbob-biobj_f09_i05_d40 1",
  "bbob-biobj_f10_i05_d40 1",
  "bbob-biobj_f11_i05_d40 1",
  "bbob-biobj_f12_i05_d40 1",
  "bbob-biobj_f13_i05_d40 1",
  "bbob-biobj_f14_i05_d40 1",
  "bbob-biobj_f15_i05_d40 1",
  "bbob-biobj_f16_i05_d40 1",
  "bbob-biobj_f17_i05_d40 1",
  "bbob-biobj_f18_i05_d40 1",
  "bbob-biobj_f19_i05_d40 1",
  "bbob-biobj_f20_i05_d40 1",
  "bbob-biobj_f21_i05_d40 1",
  "bbob-biobj_f22_i05_d40 1",
  "bbob-biobj_f23_i05_d40 1",
  "bbob-biobj_f24_i05_d40 1",
  "bbob-biobj_f25_i05_d40 1",
  "bbob-biobj_f26_i05_d40 1",
  "bbob-biobj_f27_i05_d40 1",
  "bbob-biobj_f28_i05_d40 1",
  "bbob-biobj_f29_i05_d40 1",
  "bbob-biobj_f30_i05_d40 1",
  "bbob-biobj_f31_i05_d40 1",
  "bbob-biobj_f32_i05_d40 1",
  "bbob-biobj_f33_i05_d40 1",
  "bbob-biobj_f34_i05_d40 1",
  "bbob-biobj_f35_i05_d40 1",
  "bbob-biobj_f36_i05_d40 1",
  "bbob-biobj_f37_i05_d40 1",
  "bbob-biobj_f38_i05_d40 1",
  "bbob-biobj_f39_i05_d40 1",
  "bbob-biobj_f40_i05_d40 1",
  "bbob-biobj_f41_i05_d40 1",
  "bbob-biobj_f42_i05_d40 1",
  "bbob-biobj_f43_i05_d40 1",
  "bbob-biobj_f44_i05_d40 1",
  "bbob-biobj_f45_i05_d40 1",
  "bbob-biobj_f46_i05_d40 1",
  "bbob-biobj_f47_i05_d40 1",
  "bbob-biobj_f48_i05_d40 1",
  "bbob-biobj_f49_i05_d40 1",
  "bbob-biobj_f50_i05_d40 1",
  "bbob-biobj_f51_i05_d40 1",
  "bbob-biobj_f52_i05_d40 1",
  "bbob-biobj_f53_i05_d40 1",
  "bbob-biobj_f54_i05_d40 1",
  "bbob-biobj_f55_i05_d40 1",
  "bbob-biobj_f01_i06_d02 1",
  "bbob-biobj_f02_i06_d02 1",
  "bbob-biobj_f03_i06_d02 1",
  "bbob-biobj_f04_i06_d02 1",
  "bbob-biobj_f05_i06_d02 1",
  "bbob-biobj_f06_i06_d02 1",
  "bbob-biobj_f07_i06_d02 1",
  "bbob-biobj_f08_i06_d02 1",
  "bbob-biobj_f09_i06_d02 1",
  "bbob-biobj_f10_i06_d02 1",
  "bbob-biobj_f11_i06_d02 1",
  "bbob-biobj_f12_i06_d02 1",
  "bbob-biobj_f13_i06_d02 1",
  "bbob-biobj_f14_i06_d02 1",
  "bbob-biobj_f15_i06_d02 1",
  "bbob-biobj_f16_i06_d02 1",
  "bbob-biobj_f17_i06_d02 1",
  "bbob-biobj_f18_i06_d02 1",
  "bbob-biobj_f19_i06_d02 1",
  "bbob-biobj_f20_i06_d02 1",
  "bbob-biobj_f21_i06_d02 1",
  "bbob-biobj_f22_i06_d02 1",
  "bbob-biobj_f23_i06_d02 1",
  "bbob-biobj_f24_i06_d02 1",
  "bbob-biobj_f25_i06_d02 1",
  "bbob-biobj_f26_i06_d02 1",
  "bbob-biobj_f27_i06_d02 1",
  "bbob-biobj_f28_i06_d02 1",
  "bbob-biobj_f29_i06_d02 1",
  "bbob-biobj_f30_i06_d02 1",
  "bbob-biobj_f31_i06_d02 1",
  "bbob-biobj_f32_i06_d02 1",
  "bbob-biobj_f33_i06_d02 1",
  "bbob-biobj_f34_i06_d02 1",
  "bbob-biobj_f35_i06_d02 1",
  "bbob-biobj_f36_i06_d02 1",
  "bbob-biobj_f37_i06_d02 1",
  "bbob-biobj_f38_i06_d02 1",
  "bbob-biobj_f39_i06_d02 1",
  "bbob-biobj_f40_i06_d02 1",
  "bbob-biobj_f41_i06_d02 1",
  "bbob-biobj_f42_i06_d02 1",
  "bbob-biobj_f43_i06_d02 1",
  "bbob-biobj_f44_i06_d02 1",
  "bbob-biobj_f45_i06_d02 1",
  "bbob-biobj_f46_i06_d02 1",
  "bbob-biobj_f47_i06_d02 1",
  "bbob-biobj_f48_i06_d02 1",
  "bbob-biobj_f49_i06_d02 1",
  "bbob-biobj_f50_i06_d02 1",
  "bbob-biobj_f51_i06_d02 1",
  "bbob-biobj_f52_i06_d02 1",
  "bbob-biobj_f53_i06_d02 1",
  "bbob-biobj_f54_i06_d02 1",
  "bbob-biobj_f55_i06_d02 1",
  "bbob-biobj_f01_i07_d02 1",
  "bbob-biobj_f02_i07_d02 1",
  "bbob-biobj_f03_i07_d02 1",
  "bbob-biobj_f04_i07_d02 1",
  "bbob-biobj_f05_i07_d02 1",
  "bbob-biobj_f06_i07_d02 1",
  "bbob-biobj_f07_i07_d02 1",
  "bbob-biobj_f08_i07_d02 1",
  "bbob-biobj_f09_i07_d02 1",
  "bbob-biobj_f10_i07_d02 1",
  "bbob-biobj_f11_i07_d02 1",
  "bbob-biobj_f12_i07_d02 1",
  "bbob-biobj_f13_i07_d02 1",
  "bbob-biobj_f14_i07_d02 1",
  "bbob-biobj_f15_i07_d02 1",
  "bbob-biobj_f16_i07_d02 1",
  "bbob-biobj_f17_i07_d02 1",
  "bbob-biobj_f18_i07_d02 1",
  "bbob-biobj_f19_i07_d02 1",
  "bbob-biobj_f20_i07_d02 1",
  "bbob-biobj_f21_i07_d02 1",
  "bbob-biobj_f22_i07_d02 1",
  "bbob-biobj_f23_i07_d02 1",
  "bbob-biobj_f24_i07_d02 1",
  "bbob-biobj_f25_i07_d02 1",
  "bbob-biobj_f26_i07_d02 1",
  "bbob-biobj_f27_i07_d02 1",
  "bbob-biobj_f28_i07_d02 1",
  "bbob-biobj_f29_i07_d02 1",
  "bbob-biobj_f30_i07_d02 1",
  "bbob-biobj_f31_i07_d02 1",
  "bbob-biobj_f32_i07_d02 1",
  "bbob-biobj_f33_i07_d02 1",
  "bbob-biobj_f34_i07_d02 1",
  "bbob-biobj_f35_i07_d02 1",
  "bbob-biobj_f36_i07_d02 1",
  "bbob-biobj_f37_i07_d02 1",
  "bbob-biobj_f38_i07_d02 1",
  "bbob-biobj_f39_i07_d02 1",
  "bbob-biobj_f40_i07_d02 1",
  "bbob-biobj_f41_i07_d02 1",
  "bbob-biobj_f42_i07_d02 1",
  "bbob-biobj_f43_i07_d02 1",
  "bbob-biobj_f44_i07_d02 1",
  "bbob-biobj_f45_i07_d02 1",
  "bbob-biobj_f46_i07_d02 1",
  "bbob-biobj_f47_i07_d02 1",
  "bbob-biobj_f48_i07_d02 1",
  "bbob-biobj_f49_i07_d02 1",
  "bbob-biobj_f50_i07_d02 1",
  "bbob-biobj_f51_i07_d02 1",
  "bbob-biobj_f52_i07_d02 1",
  "bbob-biobj_f53_i07_d02 1",
  "bbob-biobj_f54_i07_d02 1",
  "bbob-biobj_f55_i07_d02 1",
  "bbob-biobj_f01_i08_d02 1",
  "bbob-biobj_f02_i08_d02 1",
  "bbob-biobj_f03_i08_d02 1",
  "bbob-biobj_f04_i08_d02 1",
  "bbob-biobj_f05_i08_d02 1",
  "bbob-biobj_f06_i08_d02 1",
  "bbob-biobj_f07_i08_d02 1",
  "bbob-biobj_f08_i08_d02 1",
  "bbob-biobj_f09_i08_d02 1",
  "bbob-biobj_f10_i08_d02 1",
  "bbob-biobj_f11_i08_d02 1",
  "bbob-biobj_f12_i08_d02 1",
  "bbob-biobj_f13_i08_d02 1",
  "bbob-biobj_f14_i08_d02 1",
  "bbob-biobj_f15_i08_d02 1",
  "bbob-biobj_f16_i08_d02 1",
  "bbob-biobj_f17_i08_d02 1",
  "bbob-biobj_f18_i08_d02 1",
  "bbob-biobj_f19_i08_d02 1",
  "bbob-biobj_f20_i08_d02 1",
  "bbob-biobj_f21_i08_d02 1",
  "bbob-biobj_f22_i08_d02 1",
  "bbob-biobj_f23_i08_d02 1",
  "bbob-biobj_f24_i08_d02 1",
  "bbob-biobj_f25_i08_d02 1",
  "bbob-biobj_f26_i08_d02 1",
  "bbob-biobj_f27_i08_d02 1",
  "bbob-biobj_f28_i08_d02 1",
  "bbob-biobj_f29_i08_d02 1",
  "bbob-biobj_f30_i08_d02 1",
  "bbob-biobj_f31_i08_d02 1",
  "bbob-biobj_f32_i08_d02 1",
  "bbob-biobj_f33_i08_d02 1",
  "bbob-biobj_f34_i08_d02 1",
  "bbob-biobj_f35_i08_d02 1",
  "bbob-biobj_f36_i08_d02 1",
  "bbob-biobj_f37_i08_d02 1",
  "bbob-biobj_f38_i08_d02 1",
  "bbob-biobj_f39_i08_d02 1",
  "bbob-biobj_f40_i08_d02 1",
  "bbob-biobj_f41_i08_d02 1",
  "bbob-biobj_f42_i08_d02 1",
  "bbob-biobj_f43_i08_d02 1",
  "bbob-biobj_f44_i08_d02 1",
  "bbob-biobj_f45_i08_d02 1",
  "bbob-biobj_f46_i08_d02 1",
  "bbob-biobj_f47_i08_d02 1",
  "bbob-biobj_f48_i08_d02 1",
  "bbob-biobj_f49_i08_d02 1",
  "bbob-biobj_f50_i08_d02 1",
  "bbob-biobj_f51_i08_d02 1",
  "bbob-biobj_f52_i08_d02 1",
  "bbob-biobj_f53_i08_d02 1",
  "bbob-biobj_f54_i08_d02 1",
  "bbob-biobj_f55_i08_d02 1",
  "bbob-biobj_f01_i09_d02 1",
  "bbob-biobj_f02_i09_d02 1",
  "bbob-biobj_f03_i09_d02 1",
  "bbob-biobj_f04_i09_d02 1",
  "bbob-biobj_f05_i09_d02 1",
  "bbob-biobj_f06_i09_d02 1",
  "bbob-biobj_f07_i09_d02 1",
  "bbob-biobj_f08_i09_d02 1",
  "bbob-biobj_f09_i09_d02 1",
  "bbob-biobj_f10_i09_d02 1",
  "bbob-biobj_f11_i09_d02 1",
  "bbob-biobj_f12_i09_d02 1",
  "bbob-biobj_f13_i09_d02 1",
  "bbob-biobj_f14_i09_d02 1",
  "bbob-biobj_f15_i09_d02 1",
  "bbob-biobj_f16_i09_d02 1",
  "bbob-biobj_f17_i09_d02 1",
  "bbob-biobj_f18_i09_d02 1",
  "bbob-biobj_f19_i09_d02 1",
  "bbob-biobj_f20_i09_d02 1",
  "bbob-biobj_f21_i09_d02 1",
  "bbob-biobj_f22_i09_d02 1",
  "bbob-biobj_f23_i09_d02 1",
  "bbob-biobj_f24_i09_d02 1",
  "bbob-biobj_f25_i09_d02 1",
  "bbob-biobj_f26_i09_d02 1",
  "bbob-biobj_f27_i09_d02 1",
  "bbob-biobj_f28_i09_d02 1",
  "bbob-biobj_f29_i09_d02 1",
  "bbob-biobj_f30_i09_d02 1",
  "bbob-biobj_f31_i09_d02 1",
  "bbob-biobj_f32_i09_d02 1",
  "bbob-biobj_f33_i09_d02 1",
  "bbob-biobj_f34_i09_d02 1",
  "bbob-biobj_f35_i09_d02 1",
  "bbob-biobj_f36_i09_d02 1",
  "bbob-biobj_f37_i09_d02 1",
  "bbob-biobj_f38_i09_d02 1",
  "bbob-biobj_f39_i09_d02 1",
  "bbob-biobj_f40_i09_d02 1",
  "bbob-biobj_f41_i09_d02 1",
  "bbob-biobj_f42_i09_d02 1",
  "bbob-biobj_f43_i09_d02 1",
  "bbob-biobj_f44_i09_d02 1",
  "bbob-biobj_f45_i09_d02 1",
  "bbob-biobj_f46_i09_d02 1",
  "bbob-biobj_f47_i09_d02 1",
  "bbob-biobj_f48_i09_d02 1",
  "bbob-biobj_f49_i09_d02 1",
  "bbob-biobj_f50_i09_d02 1",
  "bbob-biobj_f51_i09_d02 1",
  "bbob-biobj_f52_i09_d02 1",
  "bbob-biobj_f53_i09_d02 1",
  "bbob-biobj_f54_i09_d02 1",
  "bbob-biobj_f55_i09_d02 1",
  "bbob-biobj_f01_i10_d02 1",
  "bbob-biobj_f02_i10_d02 1",
  "bbob-biobj_f03_i10_d02 1",
  "bbob-biobj_f04_i10_d02 1",
  "bbob-biobj_f05_i10_d02 1",
  "bbob-biobj_f06_i10_d02 1",
  "bbob-biobj_f07_i10_d02 1",
  "bbob-biobj_f08_i10_d02 1",
  "bbob-biobj_f09_i10_d02 1",
  "bbob-biobj_f10_i10_d02 1",
  "bbob-biobj_f11_i10_d02 1",
  "bbob-biobj_f12_i10_d02 1",
  "bbob-biobj_f13_i10_d02 1",
  "bbob-biobj_f14_i10_d02 1",
  "bbob-biobj_f15_i10_d02 1",
  "bbob-biobj_f16_i10_d02 1",
  "bbob-biobj_f17_i10_d02 1",
  "bbob-biobj_f18_i10_d02 1",
  "bbob-biobj_f19_i10_d02 1",
  "bbob-biobj_f20_i10_d02 1",
  "bbob-biobj_f21_i10_d02 1",
  "bbob-biobj_f22_i10_d02 1",
  "bbob-biobj_f23_i10_d02 1",
  "bbob-biobj_f24_i10_d02 1",
  "bbob-biobj_f25_i10_d02 1",
  "bbob-biobj_f26_i10_d02 1",
  "bbob-biobj_f27_i10_d02 1",
  "bbob-biobj_f28_i10_d02 1",
  "bbob-biobj_f29_i10_d02 1",
  "bbob-biobj_f30_i10_d02 1",
  "bbob-biobj_f31_i10_d02 1",
  "bbob-biobj_f32_i10_d02 1",
  "bbob-biobj_f33_i10_d02 1",
  "bbob-biobj_f34_i10_d02 1",
  "bbob-biobj_f35_i10_d02 1",
  "bbob-biobj_f36_i10_d02 1",
  "bbob-biobj_f37_i10_d02 1",
  "bbob-biobj_f38_i10_d02 1",
  "bbob-biobj_f39_i10_d02 1",
  "bbob-biobj_f40_i10_d02 1",
  "bbob-biobj_f41_i10_d02 1",
  "bbob-biobj_f42_i10_d02 1",
  "bbob-biobj_f43_i10_d02 1",
  "bbob-biobj_f44_i10_d02 1",
  "bbob-biobj_f45_i10_d02 1",
  "bbob-biobj_f46_i10_d02 1",
  "bbob-biobj_f47_i10_d02 1",
  "bbob-biobj_f48_i10_d02 1",
  "bbob-biobj_f49_i10_d02 1",
  "bbob-biobj_f50_i10_d02 1",
  "bbob-biobj_f51_i10_d02 1",
  "bbob-biobj_f52_i10_d02 1",
  "bbob-biobj_f53_i10_d02 1",
  "bbob-biobj_f54_i10_d02 1",
  "bbob-biobj_f55_i10_d02 1",
  "bbob-biobj_f01_i06_d03 1",
  "bbob-biobj_f02_i06_d03 1",
  "bbob-biobj_f03_i06_d03 1",
  "bbob-biobj_f04_i06_d03 1",
  "bbob-biobj_f05_i06_d03 1",
  "bbob-biobj_f06_i06_d03 1",
  "bbob-biobj_f07_i06_d03 1",
  "bbob-biobj_f08_i06_d03 1",
  "bbob-biobj_f09_i06_d03 1",
  "bbob-biobj_f10_i06_d03 1",
  "bbob-biobj_f11_i06_d03 1",
  "bbob-biobj_f12_i06_d03 1",
  "bbob-biobj_f13_i06_d03 1",
  "bbob-biobj_f14_i06_d03 1",
  "bbob-biobj_f15_i06_d03 1",
  "bbob-biobj_f16_i06_d03 1",
  "bbob-biobj_f17_i06_d03 1",
  "bbob-biobj_f18_i06_d03 1",
  "bbob-biobj_f19_i06_d03 1",
  "bbob-biobj_f20_i06_d03 1",
  "bbob-biobj_f21_i06_d03 1",
  "bbob-biobj_f22_i06_d03 1",
  "bbob-biobj_f23_i06_d03 1",
  "bbob-biobj_f24_i06_d03 1",
  "bbob-biobj_f25_i06_d03 1",
  "bbob-biobj_f26_i06_d03 1",
  "bbob-biobj_f27_i06_d03 1",
  "bbob-biobj_f28_i06_d03 1",
  "bbob-biobj_f29_i06_d03 1",
  "bbob-biobj_f30_i06_d03 1",
  "bbob-biobj_f31_i06_d03 1",
  "bbob-biobj_f32_i06_d03 1",
  "bbob-biobj_f33_i06_d03 1",
  "bbob-biobj_f34_i06_d03 1",
  "bbob-biobj_f35_i06_d03 1",
  "bbob-biobj_f36_i06_d03 1",
  "bbob-biobj_f37_i06_d03 1",
  "bbob-biobj_f38_i06_d03 1",
  "bbob-biobj_f39_i06_d03 1",
  "bbob-biobj_f40_i06_d03 1",
  "bbob-biobj_f41_i06_d03 1",
  "bbob-biobj_f42_i06_d03 1",
  "bbob-biobj_f43_i06_d03 1",
  "bbob-biobj_f44_i06_d03 1",
  "bbob-biobj_f45_i06_d03 1",
  "bbob-biobj_f46_i06_d03 1",
  "bbob-biobj_f47_i06_d03 1",
  "bbob-biobj_f48_i06_d03 1",
  "bbob-biobj_f49_i06_d03 1",
  "bbob-biobj_f50_i06_d03 1",
  "bbob-biobj_f51_i06_d03 1",
  "bbob-biobj_f52_i06_d03 1",
  "bbob-biobj_f53_i06_d03 1",
  "bbob-biobj_f54_i06_d03 1",
  "bbob-biobj_f55_i06_d03 1",
  "bbob-biobj_f01_i07_d03 1",
  "bbob-biobj_f02_i07_d03 1",
  "bbob-biobj_f03_i07_d03 1",
  "bbob-biobj_f04_i07_d03 1",
  "bbob-biobj_f05_i07_d03 1",
  "bbob-biobj_f06_i07_d03 1",
  "bbob-biobj_f07_i07_d03 1",
  "bbob-biobj_f08_i07_d03 1",
  "bbob-biobj_f09_i07_d03 1",
  "bbob-biobj_f10_i07_d03 1",
  "bbob-biobj_f11_i07_d03 1",
  "bbob-biobj_f12_i07_d03 1",
  "bbob-biobj_f13_i07_d03 1",
  "bbob-biobj_f14_i07_d03 1",
  "bbob-biobj_f15_i07_d03 1",
  "bbob-biobj_f16_i07_d03 1",
  "bbob-biobj_f17_i07_d03 1",
  "bbob-biobj_f18_i07_d03 1",
  "bbob-biobj_f19_i07_d03 1",
  "bbob-biobj_f20_i07_d03 1",
  "bbob-biobj_f21_i07_d03 1",
  "bbob-biobj_f22_i07_d03 1",
  "bbob-biobj_f23_i07_d03 1",
  "bbob-biobj_f24_i07_d03 1",
  "bbob-biobj_f25_i07_d03 1",
  "bbob-biobj_f26_i07_d03 1",
  "bbob-biobj_f27_i07_d03 1",
  "bbob-biobj_f28_i07_d03 1",
  "bbob-biobj_f29_i07_d03 1",
  "bbob-biobj_f30_i07_d03 1",
  "bbob-biobj_f31_i07_d03 1",
  "bbob-biobj_f32_i07_d03 1",
  "bbob-biobj_f33_i07_d03 1",
  "bbob-biobj_f34_i07_d03 1",
  "bbob-biobj_f35_i07_d03 1",
  "bbob-biobj_f36_i07_d03 1",
  "bbob-biobj_f37_i07_d03 1",
  "bbob-biobj_f38_i07_d03 1",
  "bbob-biobj_f39_i07_d03 1",
  "bbob-biobj_f40_i07_d03 1",
  "bbob-biobj_f41_i07_d03 1",
  "bbob-biobj_f42_i07_d03 1",
  "bbob-biobj_f43_i07_d03 1",
  "bbob-biobj_f44_i07_d03 1",
  "bbob-biobj_f45_i07_d03 1",
  "bbob-biobj_f46_i07_d03 1",
  "bbob-biobj_f47_i07_d03 1",
  "bbob-biobj_f48_i07_d03 1",
  "bbob-biobj_f49_i07_d03 1",
  "bbob-biobj_f50_i07_d03 1",
  "bbob-biobj_f51_i07_d03 1",
  "bbob-biobj_f52_i07_d03 1",
  "bbob-biobj_f53_i07_d03 1",
  "bbob-biobj_f54_i07_d03 1",
  "bbob-biobj_f55_i07_d03 1",
  "bbob-biobj_f01_i08_d03 1",
  "bbob-biobj_f02_i08_d03 1",
  "bbob-biobj_f03_i08_d03 1",
  "bbob-biobj_f04_i08_d03 1",
  "bbob-biobj_f05_i08_d03 1",
  "bbob-biobj_f06_i08_d03 1",
  "bbob-biobj_f07_i08_d03 1",
  "bbob-biobj_f08_i08_d03 1",
  "bbob-biobj_f09_i08_d03 1",
  "bbob-biobj_f10_i08_d03 1",
  "bbob-biobj_f11_i08_d03 1",
  "bbob-biobj_f12_i08_d03 1",
  "bbob-biobj_f13_i08_d03 1",
  "bbob-biobj_f14_i08_d03 1",
  "bbob-biobj_f15_i08_d03 1",
  "bbob-biobj_f16_i08_d03 1",
  "bbob-biobj_f17_i08_d03 1",
  "bbob-biobj_f18_i08_d03 1",
  "bbob-biobj_f19_i08_d03 1",
  "bbob-biobj_f20_i08_d03 1",
  "bbob-biobj_f21_i08_d03 1",
  "bbob-biobj_f22_i08_d03 1",
  "bbob-biobj_f23_i08_d03 1",
  "bbob-biobj_f24_i08_d03 1",
  "bbob-biobj_f25_i08_d03 1",
  "bbob-biobj_f26_i08_d03 1",
  "bbob-biobj_f27_i08_d03 1",
  "bbob-biobj_f28_i08_d03 1",
  "bbob-biobj_f29_i08_d03 1",
  "bbob-biobj_f30_i08_d03 1",
  "bbob-biobj_f31_i08_d03 1",
  "bbob-biobj_f32_i08_d03 1",
  "bbob-biobj_f33_i08_d03 1",
  "bbob-biobj_f34_i08_d03 1",
  "bbob-biobj_f35_i08_d03 1",
  "bbob-biobj_f36_i08_d03 1",
  "bbob-biobj_f37_i08_d03 1",
  "bbob-biobj_f38_i08_d03 1",
  "bbob-biobj_f39_i08_d03 1",
  "bbob-biobj_f40_i08_d03 1",
  "bbob-biobj_f41_i08_d03 1",
  "bbob-biobj_f42_i08_d03 1",
  "bbob-biobj_f43_i08_d03 1",
  "bbob-biobj_f44_i08_d03 1",
  "bbob-biobj_f45_i08_d03 1",
  "bbob-biobj_f46_i08_d03 1",
  "bbob-biobj_f47_i08_d03 1",
  "bbob-biobj_f48_i08_d03 1",
  "bbob-biobj_f49_i08_d03 1",
  "bbob-biobj_f50_i08_d03 1",
  "bbob-biobj_f51_i08_d03 1",
  "bbob-biobj_f52_i08_d03 1",
  "bbob-biobj_f53_i08_d03 1",
  "bbob-biobj_f54_i08_d03 1",
  "bbob-biobj_f55_i08_d03 1",
  "bbob-biobj_f01_i09_d03 1",
  "bbob-biobj_f02_i09_d03 1",
  "bbob-biobj_f03_i09_d03 1",
  "bbob-biobj_f04_i09_d03 1",
  "bbob-biobj_f05_i09_d03 1",
  "bbob-biobj_f06_i09_d03 1",
  "bbob-biobj_f07_i09_d03 1",
  "bbob-biobj_f08_i09_d03 1",
  "bbob-biobj_f09_i09_d03 1",
  "bbob-biobj_f10_i09_d03 1",
  "bbob-biobj_f11_i09_d03 1",
  "bbob-biobj_f12_i09_d03 1",
  "bbob-biobj_f13_i09_d03 1",
  "bbob-biobj_f14_i09_d03 1",
  "bbob-biobj_f15_i09_d03 1",
  "bbob-biobj_f16_i09_d03 1",
  "bbob-biobj_f17_i09_d03 1",
  "bbob-biobj_f18_i09_d03 1",
  "bbob-biobj_f19_i09_d03 1",
  "bbob-biobj_f20_i09_d03 1",
  "bbob-biobj_f21_i09_d03 1",
  "bbob-biobj_f22_i09_d03 1",
  "bbob-biobj_f23_i09_d03 1",
  "bbob-biobj_f24_i09_d03 1",
  "bbob-biobj_f25_i09_d03 1",
  "bbob-biobj_f26_i09_d03 1",
  "bbob-biobj_f27_i09_d03 1",
  "bbob-biobj_f28_i09_d03 1",
  "bbob-biobj_f29_i09_d03 1",
  "bbob-biobj_f30_i09_d03 1",
  "bbob-biobj_f31_i09_d03 1",
  "bbob-biobj_f32_i09_d03 1",
  "bbob-biobj_f33_i09_d03 1",
  "bbob-biobj_f34_i09_d03 1",
  "bbob-biobj_f35_i09_d03 1",
  "bbob-biobj_f36_i09_d03 1",
  "bbob-biobj_f37_i09_d03 1",
  "bbob-biobj_f38_i09_d03 1",
  "bbob-biobj_f39_i09_d03 1",
  "bbob-biobj_f40_i09_d03 1",
  "bbob-biobj_f41_i09_d03 1",
  "bbob-biobj_f42_i09_d03 1",
  "bbob-biobj_f43_i09_d03 1",
  "bbob-biobj_f44_i09_d03 1",
  "bbob-biobj_f45_i09_d03 1",
  "bbob-biobj_f46_i09_d03 1",
  "bbob-biobj_f47_i09_d03 1",
  "bbob-biobj_f48_i09_d03 1",
  "bbob-biobj_f49_i09_d03 1",
  "bbob-biobj_f50_i09_d03 1",
  "bbob-biobj_f51_i09_d03 1",
  "bbob-biobj_f52_i09_d03 1",
  "bbob-biobj_f53_i09_d03 1",
  "bbob-biobj_f54_i09_d03 1",
  "bbob-biobj_f55_i09_d03 1",
  "bbob-biobj_f01_i10_d03 1",
  "bbob-biobj_f02_i10_d03 1",
  "bbob-biobj_f03_i10_d03 1",
  "bbob-biobj_f04_i10_d03 1",
  "bbob-biobj_f05_i10_d03 1",
  "bbob-biobj_f06_i10_d03 1",
  "bbob-biobj_f07_i10_d03 1",
  "bbob-biobj_f08_i10_d03 1",
  "bbob-biobj_f09_i10_d03 1",
  "bbob-biobj_f10_i10_d03 1",
  "bbob-biobj_f11_i10_d03 1",
  "bbob-biobj_f12_i10_d03 1",
  "bbob-biobj_f13_i10_d03 1",
  "bbob-biobj_f14_i10_d03 1",
  "bbob-biobj_f15_i10_d03 1",
  "bbob-biobj_f16_i10_d03 1",
  "bbob-biobj_f17_i10_d03 1",
  "bbob-biobj_f18_i10_d03 1",
  "bbob-biobj_f19_i10_d03 1",
  "bbob-biobj_f20_i10_d03 1",
  "bbob-biobj_f21_i10_d03 1",
  "bbob-biobj_f22_i10_d03 1",
  "bbob-biobj_f23_i10_d03 1",
  "bbob-biobj_f24_i10_d03 1",
  "bbob-biobj_f25_i10_d03 1",
  "bbob-biobj_f26_i10_d03 1",
  "bbob-biobj_f27_i10_d03 1",
  "bbob-biobj_f28_i10_d03 1",
  "bbob-biobj_f29_i10_d03 1",
  "bbob-biobj_f30_i10_d03 1",
  "bbob-biobj_f31_i10_d03 1",
  "bbob-biobj_f32_i10_d03 1",
  "bbob-biobj_f33_i10_d03 1",
  "bbob-biobj_f34_i10_d03 1",
  "bbob-biobj_f35_i10_d03 1",
  "bbob-biobj_f36_i10_d03 1",
  "bbob-biobj_f37_i10_d03 1",
  "bbob-biobj_f38_i10_d03 1",
  "bbob-biobj_f39_i10_d03 1",
  "bbob-biobj_f40_i10_d03 1",
  "bbob-biobj_f41_i10_d03 1",
  "bbob-biobj_f42_i10_d03 1",
  "bbob-biobj_f43_i10_d03 1",
  "bbob-biobj_f44_i10_d03 1",
  "bbob-biobj_f45_i10_d03 1",
  "bbob-biobj_f46_i10_d03 1",
  "bbob-biobj_f47_i10_d03 1",
  "bbob-biobj_f48_i10_d03 1",
  "bbob-biobj_f49_i10_d03 1",
  "bbob-biobj_f50_i10_d03 1",
  "bbob-biobj_f51_i10_d03 1",
  "bbob-biobj_f52_i10_d03 1",
  "bbob-biobj_f53_i10_d03 1",
  "bbob-biobj_f54_i10_d03 1",
  "bbob-biobj_f55_i10_d03 1",
  "bbob-biobj_f01_i06_d05 1",
  "bbob-biobj_f02_i06_d05 1",
  "bbob-biobj_f03_i06_d05 1",
  "bbob-biobj_f04_i06_d05 1",
  "bbob-biobj_f05_i06_d05 1",
  "bbob-biobj_f06_i06_d05 1",
  "bbob-biobj_f07_i06_d05 1",
  "bbob-biobj_f08_i06_d05 1",
  "bbob-biobj_f09_i06_d05 1",
  "bbob-biobj_f10_i06_d05 1",
  "bbob-biobj_f11_i06_d05 1",
  "bbob-biobj_f12_i06_d05 1",
  "bbob-biobj_f13_i06_d05 1",
  "bbob-biobj_f14_i06_d05 1",
  "bbob-biobj_f15_i06_d05 1",
  "bbob-biobj_f16_i06_d05 1",
  "bbob-biobj_f17_i06_d05 1",
  "bbob-biobj_f18_i06_d05 1",
  "bbob-biobj_f19_i06_d05 1",
  "bbob-biobj_f20_i06_d05 1",
  "bbob-biobj_f21_i06_d05 1",
  "bbob-biobj_f22_i06_d05 1",
  "bbob-biobj_f23_i06_d05 1",
  "bbob-biobj_f24_i06_d05 1",
  "bbob-biobj_f25_i06_d05 1",
  "bbob-biobj_f26_i06_d05 1",
  "bbob-biobj_f27_i06_d05 1",
  "bbob-biobj_f28_i06_d05 1",
  "bbob-biobj_f29_i06_d05 1",
  "bbob-biobj_f30_i06_d05 1",
  "bbob-biobj_f31_i06_d05 1",
  "bbob-biobj_f32_i06_d05 1",
  "bbob-biobj_f33_i06_d05 1",
  "bbob-biobj_f34_i06_d05 1",
  "bbob-biobj_f35_i06_d05 1",
  "bbob-biobj_f36_i06_d05 1",
  "bbob-biobj_f37_i06_d05 1",
  "bbob-biobj_f38_i06_d05 1",
  "bbob-biobj_f39_i06_d05 1",
  "bbob-biobj_f40_i06_d05 1",
  "bbob-biobj_f41_i06_d05 1",
  "bbob-biobj_f42_i06_d05 1",
  "bbob-biobj_f43_i06_d05 1",
  "bbob-biobj_f44_i06_d05 1",
  "bbob-biobj_f45_i06_d05 1",
  "bbob-biobj_f46_i06_d05 1",
  "bbob-biobj_f47_i06_d05 1",
  "bbob-biobj_f48_i06_d05 1",
  "bbob-biobj_f49_i06_d05 1",
  "bbob-biobj_f50_i06_d05 1",
  "bbob-biobj_f51_i06_d05 1",
  "bbob-biobj_f52_i06_d05 1",
  "bbob-biobj_f53_i06_d05 1",
  "bbob-biobj_f54_i06_d05 1",
  "bbob-biobj_f55_i06_d05 1",
  "bbob-biobj_f01_i07_d05 1",
  "bbob-biobj_f02_i07_d05 1",
  "bbob-biobj_f03_i07_d05 1",
  "bbob-biobj_f04_i07_d05 1",
  "bbob-biobj_f05_i07_d05 1",
  "bbob-biobj_f06_i07_d05 1",
  "bbob-biobj_f07_i07_d05 1",
  "bbob-biobj_f08_i07_d05 1",
  "bbob-biobj_f09_i07_d05 1",
  "bbob-biobj_f10_i07_d05 1",
  "bbob-biobj_f11_i07_d05 1",
  "bbob-biobj_f12_i07_d05 1",
  "bbob-biobj_f13_i07_d05 1",
  "bbob-biobj_f14_i07_d05 1",
  "bbob-biobj_f15_i07_d05 1",
  "bbob-biobj_f16_i07_d05 1",
  "bbob-biobj_f17_i07_d05 1",
  "bbob-biobj_f18_i07_d05 1",
  "bbob-biobj_f19_i07_d05 1",
  "bbob-biobj_f20_i07_d05 1",
  "bbob-biobj_f21_i07_d05 1",
  "bbob-biobj_f22_i07_d05 1",
  "bbob-biobj_f23_i07_d05 1",
  "bbob-biobj_f24_i07_d05 1",
  "bbob-biobj_f25_i07_d05 1",
  "bbob-biobj_f26_i07_d05 1",
  "bbob-biobj_f27_i07_d05 1",
  "bbob-biobj_f28_i07_d05 1",
  "bbob-biobj_f29_i07_d05 1",
  "bbob-biobj_f30_i07_d05 1",
  "bbob-biobj_f31_i07_d05 1",
  "bbob-biobj_f32_i07_d05 1",
  "bbob-biobj_f33_i07_d05 1",
  "bbob-biobj_f34_i07_d05 1",
  "bbob-biobj_f35_i07_d05 1",
  "bbob-biobj_f36_i07_d05 1",
  "bbob-biobj_f37_i07_d05 1",
  "bbob-biobj_f38_i07_d05 1",
  "bbob-biobj_f39_i07_d05 1",
  "bbob-biobj_f40_i07_d05 1",
  "bbob-biobj_f41_i07_d05 1",
  "bbob-biobj_f42_i07_d05 1",
  "bbob-biobj_f43_i07_d05 1",
  "bbob-biobj_f44_i07_d05 1",
  "bbob-biobj_f45_i07_d05 1",
  "bbob-biobj_f46_i07_d05 1",
  "bbob-biobj_f47_i07_d05 1",
  "bbob-biobj_f48_i07_d05 1",
  "bbob-biobj_f49_i07_d05 1",
  "bbob-biobj_f50_i07_d05 1",
  "bbob-biobj_f51_i07_d05 1",
  "bbob-biobj_f52_i07_d05 1",
  "bbob-biobj_f53_i07_d05 1",
  "bbob-biobj_f54_i07_d05 1",
  "bbob-biobj_f55_i07_d05 1",
  "bbob-biobj_f01_i08_d05 1",
  "bbob-biobj_f02_i08_d05 1",
  "bbob-biobj_f03_i08_d05 1",
  "bbob-biobj_f04_i08_d05 1",
  "bbob-biobj_f05_i08_d05 1",
  "bbob-biobj_f06_i08_d05 1",
  "bbob-biobj_f07_i08_d05 1",
  "bbob-biobj_f08_i08_d05 1",
  "bbob-biobj_f09_i08_d05 1",
  "bbob-biobj_f10_i08_d05 1",
  "bbob-biobj_f11_i08_d05 1",
  "bbob-biobj_f12_i08_d05 1",
  "bbob-biobj_f13_i08_d05 1",
  "bbob-biobj_f14_i08_d05 1",
  "bbob-biobj_f15_i08_d05 1",
  "bbob-biobj_f16_i08_d05 1",
  "bbob-biobj_f17_i08_d05 1",
  "bbob-biobj_f18_i08_d05 1",
  "bbob-biobj_f19_i08_d05 1",
  "bbob-biobj_f20_i08_d05 1",
  "bbob-biobj_f21_i08_d05 1",
  "bbob-biobj_f22_i08_d05 1",
  "bbob-biobj_f23_i08_d05 1",
  "bbob-biobj_f24_i08_d05 1",
  "bbob-biobj_f25_i08_d05 1",
  "bbob-biobj_f26_i08_d05 1",
  "bbob-biobj_f27_i08_d05 1",
  "bbob-biobj_f28_i08_d05 1",
  "bbob-biobj_f29_i08_d05 1",
  "bbob-biobj_f30_i08_d05 1",
  "bbob-biobj_f31_i08_d05 1",
  "bbob-biobj_f32_i08_d05 1",
  "bbob-biobj_f33_i08_d05 1",
  "bbob-biobj_f34_i08_d05 1",
  "bbob-biobj_f35_i08_d05 1",
  "bbob-biobj_f36_i08_d05 1",
  "bbob-biobj_f37_i08_d05 1",
  "bbob-biobj_f38_i08_d05 1",
  "bbob-biobj_f39_i08_d05 1",
  "bbob-biobj_f40_i08_d05 1",
  "bbob-biobj_f41_i08_d05 1",
  "bbob-biobj_f42_i08_d05 1",
  "bbob-biobj_f43_i08_d05 1",
  "bbob-biobj_f44_i08_d05 1",
  "bbob-biobj_f45_i08_d05 1",
  "bbob-biobj_f46_i08_d05 1",
  "bbob-biobj_f47_i08_d05 1",
  "bbob-biobj_f48_i08_d05 1",
  "bbob-biobj_f49_i08_d05 1",
  "bbob-biobj_f50_i08_d05 1",
  "bbob-biobj_f51_i08_d05 1",
  "bbob-biobj_f52_i08_d05 1",
  "bbob-biobj_f53_i08_d05 1",
  "bbob-biobj_f54_i08_d05 1",
  "bbob-biobj_f55_i08_d05 1",
  "bbob-biobj_f01_i09_d05 1",
  "bbob-biobj_f02_i09_d05 1",
  "bbob-biobj_f03_i09_d05 1",
  "bbob-biobj_f04_i09_d05 1",
  "bbob-biobj_f05_i09_d05 1",
  "bbob-biobj_f06_i09_d05 1",
  "bbob-biobj_f07_i09_d05 1",
  "bbob-biobj_f08_i09_d05 1",
  "bbob-biobj_f09_i09_d05 1",
  "bbob-biobj_f10_i09_d05 1",
  "bbob-biobj_f11_i09_d05 1",
  "bbob-biobj_f12_i09_d05 1",
  "bbob-biobj_f13_i09_d05 1",
  "bbob-biobj_f14_i09_d05 1",
  "bbob-biobj_f15_i09_d05 1",
  "bbob-biobj_f16_i09_d05 1",
  "bbob-biobj_f17_i09_d05 1",
  "bbob-biobj_f18_i09_d05 1",
  "bbob-biobj_f19_i09_d05 1",
  "bbob-biobj_f20_i09_d05 1",
  "bbob-biobj_f21_i09_d05 1",
  "bbob-biobj_f22_i09_d05 1",
  "bbob-biobj_f23_i09_d05 1",
  "bbob-biobj_f24_i09_d05 1",
  "bbob-biobj_f25_i09_d05 1",
  "bbob-biobj_f26_i09_d05 1",
  "bbob-biobj_f27_i09_d05 1",
  "bbob-biobj_f28_i09_d05 1",
  "bbob-biobj_f29_i09_d05 1",
  "bbob-biobj_f30_i09_d05 1",
  "bbob-biobj_f31_i09_d05 1",
  "bbob-biobj_f32_i09_d05 1",
  "bbob-biobj_f33_i09_d05 1",
  "bbob-biobj_f34_i09_d05 1",
  "bbob-biobj_f35_i09_d05 1",
  "bbob-biobj_f36_i09_d05 1",
  "bbob-biobj_f37_i09_d05 1",
  "bbob-biobj_f38_i09_d05 1",
  "bbob-biobj_f39_i09_d05 1",
  "bbob-biobj_f40_i09_d05 1",
  "bbob-biobj_f41_i09_d05 1",
  "bbob-biobj_f42_i09_d05 1",
  "bbob-biobj_f43_i09_d05 1",
  "bbob-biobj_f44_i09_d05 1",
  "bbob-biobj_f45_i09_d05 1",
  "bbob-biobj_f46_i09_d05 1",
  "bbob-biobj_f47_i09_d05 1",
  "bbob-biobj_f48_i09_d05 1",
  "bbob-biobj_f49_i09_d05 1",
  "bbob-biobj_f50_i09_d05 1",
  "bbob-biobj_f51_i09_d05 1",
  "bbob-biobj_f52_i09_d05 1",
  "bbob-biobj_f53_i09_d05 1",
  "bbob-biobj_f54_i09_d05 1",
  "bbob-biobj_f55_i09_d05 1",
  "bbob-biobj_f01_i10_d05 1",
  "bbob-biobj_f02_i10_d05 1",
  "bbob-biobj_f03_i10_d05 1",
  "bbob-biobj_f04_i10_d05 1",
  "bbob-biobj_f05_i10_d05 1",
  "bbob-biobj_f06_i10_d05 1",
  "bbob-biobj_f07_i10_d05 1",
  "bbob-biobj_f08_i10_d05 1",
  "bbob-biobj_f09_i10_d05 1",
  "bbob-biobj_f10_i10_d05 1",
  "bbob-biobj_f11_i10_d05 1",
  "bbob-biobj_f12_i10_d05 1",
  "bbob-biobj_f13_i10_d05 1",
  "bbob-biobj_f14_i10_d05 1",
  "bbob-biobj_f15_i10_d05 1",
  "bbob-biobj_f16_i10_d05 1",
  "bbob-biobj_f17_i10_d05 1",
  "bbob-biobj_f18_i10_d05 1",
  "bbob-biobj_f19_i10_d05 1",
  "bbob-biobj_f20_i10_d05 1",
  "bbob-biobj_f21_i10_d05 1",
  "bbob-biobj_f22_i10_d05 1",
  "bbob-biobj_f23_i10_d05 1",
  "bbob-biobj_f24_i10_d05 1",
  "bbob-biobj_f25_i10_d05 1",
  "bbob-biobj_f26_i10_d05 1",
  "bbob-biobj_f27_i10_d05 1",
  "bbob-biobj_f28_i10_d05 1",
  "bbob-biobj_f29_i10_d05 1",
  "bbob-biobj_f30_i10_d05 1",
  "bbob-biobj_f31_i10_d05 1",
  "bbob-biobj_f32_i10_d05 1",
  "bbob-biobj_f33_i10_d05 1",
  "bbob-biobj_f34_i10_d05 1",
  "bbob-biobj_f35_i10_d05 1",
  "bbob-biobj_f36_i10_d05 1",
  "bbob-biobj_f37_i10_d05 1",
  "bbob-biobj_f38_i10_d05 1",
  "bbob-biobj_f39_i10_d05 1",
  "bbob-biobj_f40_i10_d05 1",
  "bbob-biobj_f41_i10_d05 1",
  "bbob-biobj_f42_i10_d05 1",
  "bbob-biobj_f43_i10_d05 1",
  "bbob-biobj_f44_i10_d05 1",
  "bbob-biobj_f45_i10_d05 1",
  "bbob-biobj_f46_i10_d05 1",
  "bbob-biobj_f47_i10_d05 1",
  "bbob-biobj_f48_i10_d05 1",
  "bbob-biobj_f49_i10_d05 1",
  "bbob-biobj_f50_i10_d05 1",
  "bbob-biobj_f51_i10_d05 1",
  "bbob-biobj_f52_i10_d05 1",
  "bbob-biobj_f53_i10_d05 1",
  "bbob-biobj_f54_i10_d05 1",
  "bbob-biobj_f55_i10_d05 1",
  "bbob-biobj_f01_i06_d10 1",
  "bbob-biobj_f02_i06_d10 1",
  "bbob-biobj_f03_i06_d10 1",
  "bbob-biobj_f04_i06_d10 1",
  "bbob-biobj_f05_i06_d10 1",
  "bbob-biobj_f06_i06_d10 1",
  "bbob-biobj_f07_i06_d10 1",
  "bbob-biobj_f08_i06_d10 1",
  "bbob-biobj_f09_i06_d10 1",
  "bbob-biobj_f10_i06_d10 1",
  "bbob-biobj_f11_i06_d10 1",
  "bbob-biobj_f12_i06_d10 1",
  "bbob-biobj_f13_i06_d10 1",
  "bbob-biobj_f14_i06_d10 1",
  "bbob-biobj_f15_i06_d10 1",
  "bbob-biobj_f16_i06_d10 1",
  "bbob-biobj_f17_i06_d10 1",
  "bbob-biobj_f18_i06_d10 1",
  "bbob-biobj_f19_i06_d10 1",
  "bbob-biobj_f20_i06_d10 1",
  "bbob-biobj_f21_i06_d10 1",
  "bbob-biobj_f22_i06_d10 1",
  "bbob-biobj_f23_i06_d10 1",
  "bbob-biobj_f24_i06_d10 1",
  "bbob-biobj_f25_i06_d10 1",
  "bbob-biobj_f26_i06_d10 1",
  "bbob-biobj_f27_i06_d10 1",
  "bbob-biobj_f28_i06_d10 1",
  "bbob-biobj_f29_i06_d10 1",
  "bbob-biobj_f30_i06_d10 1",
  "bbob-biobj_f31_i06_d10 1",
  "bbob-biobj_f32_i06_d10 1",
  "bbob-biobj_f33_i06_d10 1",
  "bbob-biobj_f34_i06_d10 1",
  "bbob-biobj_f35_i06_d10 1",
  "bbob-biobj_f36_i06_d10 1",
  "bbob-biobj_f37_i06_d10 1",
  "bbob-biobj_f38_i06_d10 1",
  "bbob-biobj_f39_i06_d10 1",
  "bbob-biobj_f40_i06_d10 1",
  "bbob-biobj_f41_i06_d10 1",
  "bbob-biobj_f42_i06_d10 1",
  "bbob-biobj_f43_i06_d10 1",
  "bbob-biobj_f44_i06_d10 1",
  "bbob-biobj_f45_i06_d10 1",
  "bbob-biobj_f46_i06_d10 1",
  "bbob-biobj_f47_i06_d10 1",
  "bbob-biobj_f48_i06_d10 1",
  "bbob-biobj_f49_i06_d10 1",
  "bbob-biobj_f50_i06_d10 1",
  "bbob-biobj_f51_i06_d10 1",
  "bbob-biobj_f52_i06_d10 1",
  "bbob-biobj_f53_i06_d10 1",
  "bbob-biobj_f54_i06_d10 1",
  "bbob-biobj_f55_i06_d10 1",
  "bbob-biobj_f01_i07_d10 1",
  "bbob-biobj_f02_i07_d10 1",
  "bbob-biobj_f03_i07_d10 1",
  "bbob-biobj_f04_i07_d10 1",
  "bbob-biobj_f05_i07_d10 1",
  "bbob-biobj_f06_i07_d10 1",
  "bbob-biobj_f07_i07_d10 1",
  "bbob-biobj_f08_i07_d10 1",
  "bbob-biobj_f09_i07_d10 1",
  "bbob-biobj_f10_i07_d10 1",
  "bbob-biobj_f11_i07_d10 1",
  "bbob-biobj_f12_i07_d10 1",
  "bbob-biobj_f13_i07_d10 1",
  "bbob-biobj_f14_i07_d10 1",
  "bbob-biobj_f15_i07_d10 1",
  "bbob-biobj_f16_i07_d10 1",
  "bbob-biobj_f17_i07_d10 1",
  "bbob-biobj_f18_i07_d10 1",
  "bbob-biobj_f19_i07_d10 1",
  "bbob-biobj_f20_i07_d10 1",
  "bbob-biobj_f21_i07_d10 1",
  "bbob-biobj_f22_i07_d10 1",
  "bbob-biobj_f23_i07_d10 1",
  "bbob-biobj_f24_i07_d10 1",
  "bbob-biobj_f25_i07_d10 1",
  "bbob-biobj_f26_i07_d10 1",
  "bbob-biobj_f27_i07_d10 1",
  "bbob-biobj_f28_i07_d10 1",
  "bbob-biobj_f29_i07_d10 1",
  "bbob-biobj_f30_i07_d10 1",
  "bbob-biobj_f31_i07_d10 1",
  "bbob-biobj_f32_i07_d10 1",
  "bbob-biobj_f33_i07_d10 1",
  "bbob-biobj_f34_i07_d10 1",
  "bbob-biobj_f35_i07_d10 1",
  "bbob-biobj_f36_i07_d10 1",
  "bbob-biobj_f37_i07_d10 1",
  "bbob-biobj_f38_i07_d10 1",
  "bbob-biobj_f39_i07_d10 1",
  "bbob-biobj_f40_i07_d10 1",
  "bbob-biobj_f41_i07_d10 1",
  "bbob-biobj_f42_i07_d10 1",
  "bbob-biobj_f43_i07_d10 1",
  "bbob-biobj_f44_i07_d10 1",
  "bbob-biobj_f45_i07_d10 1",
  "bbob-biobj_f46_i07_d10 1",
  "bbob-biobj_f47_i07_d10 1",
  "bbob-biobj_f48_i07_d10 1",
  "bbob-biobj_f49_i07_d10 1",
  "bbob-biobj_f50_i07_d10 1",
  "bbob-biobj_f51_i07_d10 1",
  "bbob-biobj_f52_i07_d10 1",
  "bbob-biobj_f53_i07_d10 1",
  "bbob-biobj_f54_i07_d10 1",
  "bbob-biobj_f55_i07_d10 1",
  "bbob-biobj_f01_i08_d10 1",
  "bbob-biobj_f02_i08_d10 1",
  "bbob-biobj_f03_i08_d10 1",
  "bbob-biobj_f04_i08_d10 1",
  "bbob-biobj_f05_i08_d10 1",
  "bbob-biobj_f06_i08_d10 1",
  "bbob-biobj_f07_i08_d10 1",
  "bbob-biobj_f08_i08_d10 1",
  "bbob-biobj_f09_i08_d10 1",
  "bbob-biobj_f10_i08_d10 1",
  "bbob-biobj_f11_i08_d10 1",
  "bbob-biobj_f12_i08_d10 1",
  "bbob-biobj_f13_i08_d10 1",
  "bbob-biobj_f14_i08_d10 1",
  "bbob-biobj_f15_i08_d10 1",
  "bbob-biobj_f16_i08_d10 1",
  "bbob-biobj_f17_i08_d10 1",
  "bbob-biobj_f18_i08_d10 1",
  "bbob-biobj_f19_i08_d10 1",
  "bbob-biobj_f20_i08_d10 1",
  "bbob-biobj_f21_i08_d10 1",
  "bbob-biobj_f22_i08_d10 1",
  "bbob-biobj_f23_i08_d10 1",
  "bbob-biobj_f24_i08_d10 1",
  "bbob-biobj_f25_i08_d10 1",
  "bbob-biobj_f26_i08_d10 1",
  "bbob-biobj_f27_i08_d10 1",
  "bbob-biobj_f28_i08_d10 1",
  "bbob-biobj_f29_i08_d10 1",
  "bbob-biobj_f30_i08_d10 1",
  "bbob-biobj_f31_i08_d10 1",
  "bbob-biobj_f32_i08_d10 1",
  "bbob-biobj_f33_i08_d10 1",
  "bbob-biobj_f34_i08_d10 1",
  "bbob-biobj_f35_i08_d10 1",
  "bbob-biobj_f36_i08_d10 1",
  "bbob-biobj_f37_i08_d10 1",
  "bbob-biobj_f38_i08_d10 1",
  "bbob-biobj_f39_i08_d10 1",
  "bbob-biobj_f40_i08_d10 1",
  "bbob-biobj_f41_i08_d10 1",
  "bbob-biobj_f42_i08_d10 1",
  "bbob-biobj_f43_i08_d10 1",
  "bbob-biobj_f44_i08_d10 1",
  "bbob-biobj_f45_i08_d10 1",
  "bbob-biobj_f46_i08_d10 1",
  "bbob-biobj_f47_i08_d10 1",
  "bbob-biobj_f48_i08_d10 1",
  "bbob-biobj_f49_i08_d10 1",
  "bbob-biobj_f50_i08_d10 1",
  "bbob-biobj_f51_i08_d10 1",
  "bbob-biobj_f52_i08_d10 1",
  "bbob-biobj_f53_i08_d10 1",
  "bbob-biobj_f54_i08_d10 1",
  "bbob-biobj_f55_i08_d10 1",
  "bbob-biobj_f01_i09_d10 1",
  "bbob-biobj_f02_i09_d10 1",
  "bbob-biobj_f03_i09_d10 1",
  "bbob-biobj_f04_i09_d10 1",
  "bbob-biobj_f05_i09_d10 1",
  "bbob-biobj_f06_i09_d10 1",
  "bbob-biobj_f07_i09_d10 1",
  "bbob-biobj_f08_i09_d10 1",
  "bbob-biobj_f09_i09_d10 1",
  "bbob-biobj_f10_i09_d10 1",
  "bbob-biobj_f11_i09_d10 1",
  "bbob-biobj_f12_i09_d10 1",
  "bbob-biobj_f13_i09_d10 1",
  "bbob-biobj_f14_i09_d10 1",
  "bbob-biobj_f15_i09_d10 1",
  "bbob-biobj_f16_i09_d10 1",
  "bbob-biobj_f17_i09_d10 1",
  "bbob-biobj_f18_i09_d10 1",
  "bbob-biobj_f19_i09_d10 1",
  "bbob-biobj_f20_i09_d10 1",
  "bbob-biobj_f21_i09_d10 1",
  "bbob-biobj_f22_i09_d10 1",
  "bbob-biobj_f23_i09_d10 1",
  "bbob-biobj_f24_i09_d10 1",
  "bbob-biobj_f25_i09_d10 1",
  "bbob-biobj_f26_i09_d10 1",
  "bbob-biobj_f27_i09_d10 1",
  "bbob-biobj_f28_i09_d10 1",
  "bbob-biobj_f29_i09_d10 1",
  "bbob-biobj_f30_i09_d10 1",
  "bbob-biobj_f31_i09_d10 1",
  "bbob-biobj_f32_i09_d10 1",
  "bbob-biobj_f33_i09_d10 1",
  "bbob-biobj_f34_i09_d10 1",
  "bbob-biobj_f35_i09_d10 1",
  "bbob-biobj_f36_i09_d10 1",
  "bbob-biobj_f37_i09_d10 1",
  "bbob-biobj_f38_i09_d10 1",
  "bbob-biobj_f39_i09_d10 1",
  "bbob-biobj_f40_i09_d10 1",
  "bbob-biobj_f41_i09_d10 1",
  "bbob-biobj_f42_i09_d10 1",
  "bbob-biobj_f43_i09_d10 1",
  "bbob-biobj_f44_i09_d10 1",
  "bbob-biobj_f45_i09_d10 1",
  "bbob-biobj_f46_i09_d10 1",
  "bbob-biobj_f47_i09_d10 1",
  "bbob-biobj_f48_i09_d10 1",
  "bbob-biobj_f49_i09_d10 1",
  "bbob-biobj_f50_i09_d10 1",
  "bbob-biobj_f51_i09_d10 1",
  "bbob-biobj_f52_i09_d10 1",
  "bbob-biobj_f53_i09_d10 1",
  "bbob-biobj_f54_i09_d10 1",
  "bbob-biobj_f55_i09_d10 1",
  "bbob-biobj_f01_i10_d10 1",
  "bbob-biobj_f02_i10_d10 1",
  "bbob-biobj_f03_i10_d10 1",
  "bbob-biobj_f04_i10_d10 1",
  "bbob-biobj_f05_i10_d10 1",
  "bbob-biobj_f06_i10_d10 1",
  "bbob-biobj_f07_i10_d10 1",
  "bbob-biobj_f08_i10_d10 1",
  "bbob-biobj_f09_i10_d10 1",
  "bbob-biobj_f10_i10_d10 1",
  "bbob-biobj_f11_i10_d10 1",
  "bbob-biobj_f12_i10_d10 1",
  "bbob-biobj_f13_i10_d10 1",
  "bbob-biobj_f14_i10_d10 1",
  "bbob-biobj_f15_i10_d10 1",
  "bbob-biobj_f16_i10_d10 1",
  "bbob-biobj_f17_i10_d10 1",
  "bbob-biobj_f18_i10_d10 1",
  "bbob-biobj_f19_i10_d10 1",
  "bbob-biobj_f20_i10_d10 1",
  "bbob-biobj_f21_i10_d10 1",
  "bbob-biobj_f22_i10_d10 1",
  "bbob-biobj_f23_i10_d10 1",
  "bbob-biobj_f24_i10_d10 1",
  "bbob-biobj_f25_i10_d10 1",
  "bbob-biobj_f26_i10_d10 1",
  "bbob-biobj_f27_i10_d10 1",
  "bbob-biobj_f28_i10_d10 1",
  "bbob-biobj_f29_i10_d10 1",
  "bbob-biobj_f30_i10_d10 1",
  "bbob-biobj_f31_i10_d10 1",
  "bbob-biobj_f32_i10_d10 1",
  "bbob-biobj_f33_i10_d10 1",
  "bbob-biobj_f34_i10_d10 1",
  "bbob-biobj_f35_i10_d10 1",
  "bbob-biobj_f36_i10_d10 1",
  "bbob-biobj_f37_i10_d10 1",
  "bbob-biobj_f38_i10_d10 1",
  "bbob-biobj_f39_i10_d10 1",
  "bbob-biobj_f40_i10_d10 1",
  "bbob-biobj_f41_i10_d10 1",
  "bbob-biobj_f42_i10_d10 1",
  "bbob-biobj_f43_i10_d10 1",
  "bbob-biobj_f44_i10_d10 1",
  "bbob-biobj_f45_i10_d10 1",
  "bbob-biobj_f46_i10_d10 1",
  "bbob-biobj_f47_i10_d10 1",
  "bbob-biobj_f48_i10_d10 1",
  "bbob-biobj_f49_i10_d10 1",
  "bbob-biobj_f50_i10_d10 1",
  "bbob-biobj_f51_i10_d10 1",
  "bbob-biobj_f52_i10_d10 1",
  "bbob-biobj_f53_i10_d10 1",
  "bbob-biobj_f54_i10_d10 1",
  "bbob-biobj_f55_i10_d10 1",
  "bbob-biobj_f01_i06_d20 1",
  "bbob-biobj_f02_i06_d20 1",
  "bbob-biobj_f03_i06_d20 1",
  "bbob-biobj_f04_i06_d20 1",
  "bbob-biobj_f05_i06_d20 1",
  "bbob-biobj_f06_i06_d20 1",
  "bbob-biobj_f07_i06_d20 1",
  "bbob-biobj_f08_i06_d20 1",
  "bbob-biobj_f09_i06_d20 1",
  "bbob-biobj_f10_i06_d20 1",
  "bbob-biobj_f11_i06_d20 1",
  "bbob-biobj_f12_i06_d20 1",
  "bbob-biobj_f13_i06_d20 1",
  "bbob-biobj_f14_i06_d20 1",
  "bbob-biobj_f15_i06_d20 1",
  "bbob-biobj_f16_i06_d20 1",
  "bbob-biobj_f17_i06_d20 1",
  "bbob-biobj_f18_i06_d20 1",
  "bbob-biobj_f19_i06_d20 1",
  "bbob-biobj_f20_i06_d20 1",
  "bbob-biobj_f21_i06_d20 1",
  "bbob-biobj_f22_i06_d20 1",
  "bbob-biobj_f23_i06_d20 1",
  "bbob-biobj_f24_i06_d20 1",
  "bbob-biobj_f25_i06_d20 1",
  "bbob-biobj_f26_i06_d20 1",
  "bbob-biobj_f27_i06_d20 1",
  "bbob-biobj_f28_i06_d20 1",
  "bbob-biobj_f29_i06_d20 1",
  "bbob-biobj_f30_i06_d20 1",
  "bbob-biobj_f31_i06_d20 1",
  "bbob-biobj_f32_i06_d20 1",
  "bbob-biobj_f33_i06_d20 1",
  "bbob-biobj_f34_i06_d20 1",
  "bbob-biobj_f35_i06_d20 1",
  "bbob-biobj_f36_i06_d20 1",
  "bbob-biobj_f37_i06_d20 1",
  "bbob-biobj_f38_i06_d20 1",
  "bbob-biobj_f39_i06_d20 1",
  "bbob-biobj_f40_i06_d20 1",
  "bbob-biobj_f41_i06_d20 1",
  "bbob-biobj_f42_i06_d20 1",
  "bbob-biobj_f43_i06_d20 1",
  "bbob-biobj_f44_i06_d20 1",
  "bbob-biobj_f45_i06_d20 1",
  "bbob-biobj_f46_i06_d20 1",
  "bbob-biobj_f47_i06_d20 1",
  "bbob-biobj_f48_i06_d20 1",
  "bbob-biobj_f49_i06_d20 1",
  "bbob-biobj_f50_i06_d20 1",
  "bbob-biobj_f51_i06_d20 1",
  "bbob-biobj_f52_i06_d20 1",
  "bbob-biobj_f53_i06_d20 1",
  "bbob-biobj_f54_i06_d20 1",
  "bbob-biobj_f55_i06_d20 1",
  "bbob-biobj_f01_i07_d20 1",
  "bbob-biobj_f02_i07_d20 1",
  "bbob-biobj_f03_i07_d20 1",
  "bbob-biobj_f04_i07_d20 1",
  "bbob-biobj_f05_i07_d20 1",
  "bbob-biobj_f06_i07_d20 1",
  "bbob-biobj_f07_i07_d20 1",
  "bbob-biobj_f08_i07_d20 1",
  "bbob-biobj_f09_i07_d20 1",
  "bbob-biobj_f10_i07_d20 1",
  "bbob-biobj_f11_i07_d20 1",
  "bbob-biobj_f12_i07_d20 1",
  "bbob-biobj_f13_i07_d20 1",
  "bbob-biobj_f14_i07_d20 1",
  "bbob-biobj_f15_i07_d20 1",
  "bbob-biobj_f16_i07_d20 1",
  "bbob-biobj_f17_i07_d20 1",
  "bbob-biobj_f18_i07_d20 1",
  "bbob-biobj_f19_i07_d20 1",
  "bbob-biobj_f20_i07_d20 1",
  "bbob-biobj_f21_i07_d20 1",
  "bbob-biobj_f22_i07_d20 1",
  "bbob-biobj_f23_i07_d20 1",
  "bbob-biobj_f24_i07_d20 1",
  "bbob-biobj_f25_i07_d20 1",
  "bbob-biobj_f26_i07_d20 1",
  "bbob-biobj_f27_i07_d20 1",
  "bbob-biobj_f28_i07_d20 1",
  "bbob-biobj_f29_i07_d20 1",
  "bbob-biobj_f30_i07_d20 1",
  "bbob-biobj_f31_i07_d20 1",
  "bbob-biobj_f32_i07_d20 1",
  "bbob-biobj_f33_i07_d20 1",
  "bbob-biobj_f34_i07_d20 1",
  "bbob-biobj_f35_i07_d20 1",
  "bbob-biobj_f36_i07_d20 1",
  "bbob-biobj_f37_i07_d20 1",
  "bbob-biobj_f38_i07_d20 1",
  "bbob-biobj_f39_i07_d20 1",
  "bbob-biobj_f40_i07_d20 1",
  "bbob-biobj_f41_i07_d20 1",
  "bbob-biobj_f42_i07_d20 1",
  "bbob-biobj_f43_i07_d20 1",
  "bbob-biobj_f44_i07_d20 1",
  "bbob-biobj_f45_i07_d20 1",
  "bbob-biobj_f46_i07_d20 1",
  "bbob-biobj_f47_i07_d20 1",
  "bbob-biobj_f48_i07_d20 1",
  "bbob-biobj_f49_i07_d20 1",
  "bbob-biobj_f50_i07_d20 1",
  "bbob-biobj_f51_i07_d20 1",
  "bbob-biobj_f52_i07_d20 1",
  "bbob-biobj_f53_i07_d20 1",
  "bbob-biobj_f54_i07_d20 1",
  "bbob-biobj_f55_i07_d20 1",
  "bbob-biobj_f01_i08_d20 1",
  "bbob-biobj_f02_i08_d20 1",
  "bbob-biobj_f03_i08_d20 1",
  "bbob-biobj_f04_i08_d20 1",
  "bbob-biobj_f05_i08_d20 1",
  "bbob-biobj_f06_i08_d20 1",
  "bbob-biobj_f07_i08_d20 1",
  "bbob-biobj_f08_i08_d20 1",
  "bbob-biobj_f09_i08_d20 1",
  "bbob-biobj_f10_i08_d20 1",
  "bbob-biobj_f11_i08_d20 1",
  "bbob-biobj_f12_i08_d20 1",
  "bbob-biobj_f13_i08_d20 1",
  "bbob-biobj_f14_i08_d20 1",
  "bbob-biobj_f15_i08_d20 1",
  "bbob-biobj_f16_i08_d20 1",
  "bbob-biobj_f17_i08_d20 1",
  "bbob-biobj_f18_i08_d20 1",
  "bbob-biobj_f19_i08_d20 1",
  "bbob-biobj_f20_i08_d20 1",
  "bbob-biobj_f21_i08_d20 1",
  "bbob-biobj_f22_i08_d20 1",
  "bbob-biobj_f23_i08_d20 1",
  "bbob-biobj_f24_i08_d20 1",
  "bbob-biobj_f25_i08_d20 1",
  "bbob-biobj_f26_i08_d20 1",
  "bbob-biobj_f27_i08_d20 1",
  "bbob-biobj_f28_i08_d20 1",
  "bbob-biobj_f29_i08_d20 1",
  "bbob-biobj_f30_i08_d20 1",
  "bbob-biobj_f31_i08_d20 1",
  "bbob-biobj_f32_i08_d20 1",
  "bbob-biobj_f33_i08_d20 1",
  "bbob-biobj_f34_i08_d20 1",
  "bbob-biobj_f35_i08_d20 1",
  "bbob-biobj_f36_i08_d20 1",
  "bbob-biobj_f37_i08_d20 1",
  "bbob-biobj_f38_i08_d20 1",
  "bbob-biobj_f39_i08_d20 1",
  "bbob-biobj_f40_i08_d20 1",
  "bbob-biobj_f41_i08_d20 1",
  "bbob-biobj_f42_i08_d20 1",
  "bbob-biobj_f43_i08_d20 1",
  "bbob-biobj_f44_i08_d20 1",
  "bbob-biobj_f45_i08_d20 1",
  "bbob-biobj_f46_i08_d20 1",
  "bbob-biobj_f47_i08_d20 1",
  "bbob-biobj_f48_i08_d20 1",
  "bbob-biobj_f49_i08_d20 1",
  "bbob-biobj_f50_i08_d20 1",
  "bbob-biobj_f51_i08_d20 1",
  "bbob-biobj_f52_i08_d20 1",
  "bbob-biobj_f53_i08_d20 1",
  "bbob-biobj_f54_i08_d20 1",
  "bbob-biobj_f55_i08_d20 1",
  "bbob-biobj_f01_i09_d20 1",
  "bbob-biobj_f02_i09_d20 1",
  "bbob-biobj_f03_i09_d20 1",
  "bbob-biobj_f04_i09_d20 1",
  "bbob-biobj_f05_i09_d20 1",
  "bbob-biobj_f06_i09_d20 1",
  "bbob-biobj_f07_i09_d20 1",
  "bbob-biobj_f08_i09_d20 1",
  "bbob-biobj_f09_i09_d20 1",
  "bbob-biobj_f10_i09_d20 1",
  "bbob-biobj_f11_i09_d20 1",
  "bbob-biobj_f12_i09_d20 1",
  "bbob-biobj_f13_i09_d20 1",
  "bbob-biobj_f14_i09_d20 1",
  "bbob-biobj_f15_i09_d20 1",
  "bbob-biobj_f16_i09_d20 1",
  "bbob-biobj_f17_i09_d20 1",
  "bbob-biobj_f18_i09_d20 1",
  "bbob-biobj_f19_i09_d20 1",
  "bbob-biobj_f20_i09_d20 1",
  "bbob-biobj_f21_i09_d20 1",
  "bbob-biobj_f22_i09_d20 1",
  "bbob-biobj_f23_i09_d20 1",
  "bbob-biobj_f24_i09_d20 1",
  "bbob-biobj_f25_i09_d20 1",
  "bbob-biobj_f26_i09_d20 1",
  "bbob-biobj_f27_i09_d20 1",
  "bbob-biobj_f28_i09_d20 1",
  "bbob-biobj_f29_i09_d20 1",
  "bbob-biobj_f30_i09_d20 1",
  "bbob-biobj_f31_i09_d20 1",
  "bbob-biobj_f32_i09_d20 1",
  "bbob-biobj_f33_i09_d20 1",
  "bbob-biobj_f34_i09_d20 1",
  "bbob-biobj_f35_i09_d20 1",
  "bbob-biobj_f36_i09_d20 1",
  "bbob-biobj_f37_i09_d20 1",
  "bbob-biobj_f38_i09_d20 1",
  "bbob-biobj_f39_i09_d20 1",
  "bbob-biobj_f40_i09_d20 1",
  "bbob-biobj_f41_i09_d20 1",
  "bbob-biobj_f42_i09_d20 1",
  "bbob-biobj_f43_i09_d20 1",
  "bbob-biobj_f44_i09_d20 1",
  "bbob-biobj_f45_i09_d20 1",
  "bbob-biobj_f46_i09_d20 1",
  "bbob-biobj_f47_i09_d20 1",
  "bbob-biobj_f48_i09_d20 1",
  "bbob-biobj_f49_i09_d20 1",
  "bbob-biobj_f50_i09_d20 1",
  "bbob-biobj_f51_i09_d20 1",
  "bbob-biobj_f52_i09_d20 1",
  "bbob-biobj_f53_i09_d20 1",
  "bbob-biobj_f54_i09_d20 1",
  "bbob-biobj_f55_i09_d20 1",
  "bbob-biobj_f01_i10_d20 1",
  "bbob-biobj_f02_i10_d20 1",
  "bbob-biobj_f03_i10_d20 1",
  "bbob-biobj_f04_i10_d20 1",
  "bbob-biobj_f05_i10_d20 1",
  "bbob-biobj_f06_i10_d20 1",
  "bbob-biobj_f07_i10_d20 1",
  "bbob-biobj_f08_i10_d20 1",
  "bbob-biobj_f09_i10_d20 1",
  "bbob-biobj_f10_i10_d20 1",
  "bbob-biobj_f11_i10_d20 1",
  "bbob-biobj_f12_i10_d20 1",
  "bbob-biobj_f13_i10_d20 1",
  "bbob-biobj_f14_i10_d20 1",
  "bbob-biobj_f15_i10_d20 1",
  "bbob-biobj_f16_i10_d20 1",
  "bbob-biobj_f17_i10_d20 1",
  "bbob-biobj_f18_i10_d20 1",
  "bbob-biobj_f19_i10_d20 1",
  "bbob-biobj_f20_i10_d20 1",
  "bbob-biobj_f21_i10_d20 1",
  "bbob-biobj_f22_i10_d20 1",
  "bbob-biobj_f23_i10_d20 1",
  "bbob-biobj_f24_i10_d20 1",
  "bbob-biobj_f25_i10_d20 1",
  "bbob-biobj_f26_i10_d20 1",
  "bbob-biobj_f27_i10_d20 1",
  "bbob-biobj_f28_i10_d20 1",
  "bbob-biobj_f29_i10_d20 1",
  "bbob-biobj_f30_i10_d20 1",
  "bbob-biobj_f31_i10_d20 1",
  "bbob-biobj_f32_i10_d20 1",
  "bbob-biobj_f33_i10_d20 1",
  "bbob-biobj_f34_i10_d20 1",
  "bbob-biobj_f35_i10_d20 1",
  "bbob-biobj_f36_i10_d20 1",
  "bbob-biobj_f37_i10_d20 1",
  "bbob-biobj_f38_i10_d20 1",
  "bbob-biobj_f39_i10_d20 1",
  "bbob-biobj_f40_i10_d20 1",
  "bbob-biobj_f41_i10_d20 1",
  "bbob-biobj_f42_i10_d20 1",
  "bbob-biobj_f43_i10_d20 1",
  "bbob-biobj_f44_i10_d20 1",
  "bbob-biobj_f45_i10_d20 1",
  "bbob-biobj_f46_i10_d20 1",
  "bbob-biobj_f47_i10_d20 1",
  "bbob-biobj_f48_i10_d20 1",
  "bbob-biobj_f49_i10_d20 1",
  "bbob-biobj_f50_i10_d20 1",
  "bbob-biobj_f51_i10_d20 1",
  "bbob-biobj_f52_i10_d20 1",
  "bbob-biobj_f53_i10_d20 1",
  "bbob-biobj_f54_i10_d20 1",
  "bbob-biobj_f55_i10_d20 1",
  "bbob-biobj_f01_i06_d40 1",
  "bbob-biobj_f02_i06_d40 1",
  "bbob-biobj_f03_i06_d40 1",
  "bbob-biobj_f04_i06_d40 1",
  "bbob-biobj_f05_i06_d40 1",
  "bbob-biobj_f06_i06_d40 1",
  "bbob-biobj_f07_i06_d40 1",
  "bbob-biobj_f08_i06_d40 1",
  "bbob-biobj_f09_i06_d40 1",
  "bbob-biobj_f10_i06_d40 1",
  "bbob-biobj_f11_i06_d40 1",
  "bbob-biobj_f12_i06_d40 1",
  "bbob-biobj_f13_i06_d40 1",
  "bbob-biobj_f14_i06_d40 1",
  "bbob-biobj_f15_i06_d40 1",
  "bbob-biobj_f16_i06_d40 1",
  "bbob-biobj_f17_i06_d40 1",
  "bbob-biobj_f18_i06_d40 1",
  "bbob-biobj_f19_i06_d40 1",
  "bbob-biobj_f20_i06_d40 1",
  "bbob-biobj_f21_i06_d40 1",
  "bbob-biobj_f22_i06_d40 1",
  "bbob-biobj_f23_i06_d40 1",
  "bbob-biobj_f24_i06_d40 1",
  "bbob-biobj_f25_i06_d40 1",
  "bbob-biobj_f26_i06_d40 1",
  "bbob-biobj_f27_i06_d40 1",
  "bbob-biobj_f28_i06_d40 1",
  "bbob-biobj_f29_i06_d40 1",
  "bbob-biobj_f30_i06_d40 1",
  "bbob-biobj_f31_i06_d40 1",
  "bbob-biobj_f32_i06_d40 1",
  "bbob-biobj_f33_i06_d40 1",
  "bbob-biobj_f34_i06_d40 1",
  "bbob-biobj_f35_i06_d40 1",
  "bbob-biobj_f36_i06_d40 1",
  "bbob-biobj_f37_i06_d40 1",
  "bbob-biobj_f38_i06_d40 1",
  "bbob-biobj_f39_i06_d40 1",
  "bbob-biobj_f40_i06_d40 1",
  "bbob-biobj_f41_i06_d40 1",
  "bbob-biobj_f42_i06_d40 1",
  "bbob-biobj_f43_i06_d40 1",
  "bbob-biobj_f44_i06_d40 1",
  "bbob-biobj_f45_i06_d40 1",
  "bbob-biobj_f46_i06_d40 1",
  "bbob-biobj_f47_i06_d40 1",
  "bbob-biobj_f48_i06_d40 1",
  "bbob-biobj_f49_i06_d40 1",
  "bbob-biobj_f50_i06_d40 1",
  "bbob-biobj_f51_i06_d40 1",
  "bbob-biobj_f52_i06_d40 1",
  "bbob-biobj_f53_i06_d40 1",
  "bbob-biobj_f54_i06_d40 1",
  "bbob-biobj_f55_i06_d40 1",
  "bbob-biobj_f01_i07_d40 1",
  "bbob-biobj_f02_i07_d40 1",
  "bbob-biobj_f03_i07_d40 1",
  "bbob-biobj_f04_i07_d40 1",
  "bbob-biobj_f05_i07_d40 1",
  "bbob-biobj_f06_i07_d40 1",
  "bbob-biobj_f07_i07_d40 1",
  "bbob-biobj_f08_i07_d40 1",
  "bbob-biobj_f09_i07_d40 1",
  "bbob-biobj_f10_i07_d40 1",
  "bbob-biobj_f11_i07_d40 1",
  "bbob-biobj_f12_i07_d40 1",
  "bbob-biobj_f13_i07_d40 1",
  "bbob-biobj_f14_i07_d40 1",
  "bbob-biobj_f15_i07_d40 1",
  "bbob-biobj_f16_i07_d40 1",
  "bbob-biobj_f17_i07_d40 1",
  "bbob-biobj_f18_i07_d40 1",
  "bbob-biobj_f19_i07_d40 1",
  "bbob-biobj_f20_i07_d40 1",
  "bbob-biobj_f21_i07_d40 1",
  "bbob-biobj_f22_i07_d40 1",
  "bbob-biobj_f23_i07_d40 1",
  "bbob-biobj_f24_i07_d40 1",
  "bbob-biobj_f25_i07_d40 1",
  "bbob-biobj_f26_i07_d40 1",
  "bbob-biobj_f27_i07_d40 1",
  "bbob-biobj_f28_i07_d40 1",
  "bbob-biobj_f29_i07_d40 1",
  "bbob-biobj_f30_i07_d40 1",
  "bbob-biobj_f31_i07_d40 1",
  "bbob-biobj_f32_i07_d40 1",
  "bbob-biobj_f33_i07_d40 1",
  "bbob-biobj_f34_i07_d40 1",
  "bbob-biobj_f35_i07_d40 1",
  "bbob-biobj_f36_i07_d40 1",
  "bbob-biobj_f37_i07_d40 1",
  "bbob-biobj_f38_i07_d40 1",
  "bbob-biobj_f39_i07_d40 1",
  "bbob-biobj_f40_i07_d40 1",
  "bbob-biobj_f41_i07_d40 1",
  "bbob-biobj_f42_i07_d40 1",
  "bbob-biobj_f43_i07_d40 1",
  "bbob-biobj_f44_i07_d40 1",
  "bbob-biobj_f45_i07_d40 1",
  "bbob-biobj_f46_i07_d40 1",
  "bbob-biobj_f47_i07_d40 1",
  "bbob-biobj_f48_i07_d40 1",
  "bbob-biobj_f49_i07_d40 1",
  "bbob-biobj_f50_i07_d40 1",
  "bbob-biobj_f51_i07_d40 1",
  "bbob-biobj_f52_i07_d40 1",
  "bbob-biobj_f53_i07_d40 1",
  "bbob-biobj_f54_i07_d40 1",
  "bbob-biobj_f55_i07_d40 1",
  "bbob-biobj_f01_i08_d40 1",
  "bbob-biobj_f02_i08_d40 1",
  "bbob-biobj_f03_i08_d40 1",
  "bbob-biobj_f04_i08_d40 1",
  "bbob-biobj_f05_i08_d40 1",
  "bbob-biobj_f06_i08_d40 1",
  "bbob-biobj_f07_i08_d40 1",
  "bbob-biobj_f08_i08_d40 1",
  "bbob-biobj_f09_i08_d40 1",
  "bbob-biobj_f10_i08_d40 1",
  "bbob-biobj_f11_i08_d40 1",
  "bbob-biobj_f12_i08_d40 1",
  "bbob-biobj_f13_i08_d40 1",
  "bbob-biobj_f14_i08_d40 1",
  "bbob-biobj_f15_i08_d40 1",
  "bbob-biobj_f16_i08_d40 1",
  "bbob-biobj_f17_i08_d40 1",
  "bbob-biobj_f18_i08_d40 1",
  "bbob-biobj_f19_i08_d40 1",
  "bbob-biobj_f20_i08_d40 1",
  "bbob-biobj_f21_i08_d40 1",
  "bbob-biobj_f22_i08_d40 1",
  "bbob-biobj_f23_i08_d40 1",
  "bbob-biobj_f24_i08_d40 1",
  "bbob-biobj_f25_i08_d40 1",
  "bbob-biobj_f26_i08_d40 1",
  "bbob-biobj_f27_i08_d40 1",
  "bbob-biobj_f28_i08_d40 1",
  "bbob-biobj_f29_i08_d40 1",
  "bbob-biobj_f30_i08_d40 1",
  "bbob-biobj_f31_i08_d40 1",
  "bbob-biobj_f32_i08_d40 1",
  "bbob-biobj_f33_i08_d40 1",
  "bbob-biobj_f34_i08_d40 1",
  "bbob-biobj_f35_i08_d40 1",
  "bbob-biobj_f36_i08_d40 1",
  "bbob-biobj_f37_i08_d40 1",
  "bbob-biobj_f38_i08_d40 1",
  "bbob-biobj_f39_i08_d40 1",
  "bbob-biobj_f40_i08_d40 1",
  "bbob-biobj_f41_i08_d40 1",
  "bbob-biobj_f42_i08_d40 1",
  "bbob-biobj_f43_i08_d40 1",
  "bbob-biobj_f44_i08_d40 1",
  "bbob-biobj_f45_i08_d40 1",
  "bbob-biobj_f46_i08_d40 1",
  "bbob-biobj_f47_i08_d40 1",
  "bbob-biobj_f48_i08_d40 1",
  "bbob-biobj_f49_i08_d40 1",
  "bbob-biobj_f50_i08_d40 1",
  "bbob-biobj_f51_i08_d40 1",
  "bbob-biobj_f52_i08_d40 1",
  "bbob-biobj_f53_i08_d40 1",
  "bbob-biobj_f54_i08_d40 1",
  "bbob-biobj_f55_i08_d40 1",
  "bbob-biobj_f01_i09_d40 1",
  "bbob-biobj_f02_i09_d40 1",
  "bbob-biobj_f03_i09_d40 1",
  "bbob-biobj_f04_i09_d40 1",
  "bbob-biobj_f05_i09_d40 1",
  "bbob-biobj_f06_i09_d40 1",
  "bbob-biobj_f07_i09_d40 1",
  "bbob-biobj_f08_i09_d40 1",
  "bbob-biobj_f09_i09_d40 1",
  "bbob-biobj_f10_i09_d40 1",
  "bbob-biobj_f11_i09_d40 1",
  "bbob-biobj_f12_i09_d40 1",
  "bbob-biobj_f13_i09_d40 1",
  "bbob-biobj_f14_i09_d40 1",
  "bbob-biobj_f15_i09_d40 1",
  "bbob-biobj_f16_i09_d40 1",
  "bbob-biobj_f17_i09_d40 1",
  "bbob-biobj_f18_i09_d40 1",
  "bbob-biobj_f19_i09_d40 1",
  "bbob-biobj_f20_i09_d40 1",
  "bbob-biobj_f21_i09_d40 1",
  "bbob-biobj_f22_i09_d40 1",
  "bbob-biobj_f23_i09_d40 1",
  "bbob-biobj_f24_i09_d40 1",
  "bbob-biobj_f25_i09_d40 1",
  "bbob-biobj_f26_i09_d40 1",
  "bbob-biobj_f27_i09_d40 1",
  "bbob-biobj_f28_i09_d40 1",
  "bbob-biobj_f29_i09_d40 1",
  "bbob-biobj_f30_i09_d40 1",
  "bbob-biobj_f31_i09_d40 1",
  "bbob-biobj_f32_i09_d40 1",
  "bbob-biobj_f33_i09_d40 1",
  "bbob-biobj_f34_i09_d40 1",
  "bbob-biobj_f35_i09_d40 1",
  "bbob-biobj_f36_i09_d40 1",
  "bbob-biobj_f37_i09_d40 1",
  "bbob-biobj_f38_i09_d40 1",
  "bbob-biobj_f39_i09_d40 1",
  "bbob-biobj_f40_i09_d40 1",
  "bbob-biobj_f41_i09_d40 1",
  "bbob-biobj_f42_i09_d40 1",
  "bbob-biobj_f43_i09_d40 1",
  "bbob-biobj_f44_i09_d40 1",
  "bbob-biobj_f45_i09_d40 1",
  "bbob-biobj_f46_i09_d40 1",
  "bbob-biobj_f47_i09_d40 1",
  "bbob-biobj_f48_i09_d40 1",
  "bbob-biobj_f49_i09_d40 1",
  "bbob-biobj_f50_i09_d40 1",
  "bbob-biobj_f51_i09_d40 1",
  "bbob-biobj_f52_i09_d40 1",
  "bbob-biobj_f53_i09_d40 1",
  "bbob-biobj_f54_i09_d40 1",
  "bbob-biobj_f55_i09_d40 1",
  "bbob-biobj_f01_i10_d40 1",
  "bbob-biobj_f02_i10_d40 1",
  "bbob-biobj_f03_i10_d40 1",
  "bbob-biobj_f04_i10_d40 1",
  "bbob-biobj_f05_i10_d40 1",
  "bbob-biobj_f06_i10_d40 1",
  "bbob-biobj_f07_i10_d40 1",
  "bbob-biobj_f08_i10_d40 1",
  "bbob-biobj_f09_i10_d40 1",
  "bbob-biobj_f10_i10_d40 1",
  "bbob-biobj_f11_i10_d40 1",
  "bbob-biobj_f12_i10_d40 1",
  "bbob-biobj_f13_i10_d40 1",
  "bbob-biobj_f14_i10_d40 1",
  "bbob-biobj_f15_i10_d40 1",
  "bbob-biobj_f16_i10_d40 1",
  "bbob-biobj_f17_i10_d40 1",
  "bbob-biobj_f18_i10_d40 1",
  "bbob-biobj_f19_i10_d40 1",
  "bbob-biobj_f20_i10_d40 1",
  "bbob-biobj_f21_i10_d40 1",
  "bbob-biobj_f22_i10_d40 1",
  "bbob-biobj_f23_i10_d40 1",
  "bbob-biobj_f24_i10_d40 1",
  "bbob-biobj_f25_i10_d40 1",
  "bbob-biobj_f26_i10_d40 1",
  "bbob-biobj_f27_i10_d40 1",
  "bbob-biobj_f28_i10_d40 1",
  "bbob-biobj_f29_i10_d40 1",
  "bbob-biobj_f30_i10_d40 1",
  "bbob-biobj_f31_i10_d40 1",
  "bbob-biobj_f32_i10_d40 1",
  "bbob-biobj_f33_i10_d40 1",
  "bbob-biobj_f34_i10_d40 1",
  "bbob-biobj_f35_i10_d40 1",
  "bbob-biobj_f36_i10_d40 1",
  "bbob-biobj_f37_i10_d40 1",
  "bbob-biobj_f38_i10_d40 1",
  "bbob-biobj_f39_i10_d40 1",
  "bbob-biobj_f40_i10_d40 1",
  "bbob-biobj_f41_i10_d40 1",
  "bbob-biobj_f42_i10_d40 1",
  "bbob-biobj_f43_i10_d40 1",
  "bbob-biobj_f44_i10_d40 1",
  "bbob-biobj_f45_i10_d40 1",
  "bbob-biobj_f46_i10_d40 1",
  "bbob-biobj_f47_i10_d40 1",
  "bbob-biobj_f48_i10_d40 1",
  "bbob-biobj_f49_i10_d40 1",
  "bbob-biobj_f50_i10_d40 1",
  "bbob-biobj_f51_i10_d40 1",
  "bbob-biobj_f52_i10_d40 1",
  "bbob-biobj_f53_i10_d40 1",
  "bbob-biobj_f54_i10_d40 1",
  "bbob-biobj_f55_i10_d40 1"
};
#line 19 "code-experiments/src/suite_biobj.c"

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

  problem = coco_problem_stacked_allocate(problem1, problem2);

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
          problem = coco_problem_stacked_allocate(problem1, problem2);

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
      coco_problem_stacked_free(problem);
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
#line 20 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/suite_toy.c"
/**
 * @file suite_toy.c
 * @brief Implementation of a toy suite containing 6 noiseless "basic" single-objective functions in 5
 * dimensions.
 */

#line 8 "code-experiments/src/suite_toy.c"
#line 9 "code-experiments/src/suite_toy.c"
#line 10 "code-experiments/src/suite_toy.c"
#line 11 "code-experiments/src/suite_toy.c"
#line 12 "code-experiments/src/suite_toy.c"
#line 13 "code-experiments/src/suite_toy.c"
#line 14 "code-experiments/src/suite_toy.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the toy suite.
 */
static coco_suite_t *suite_toy_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20 };

  suite = coco_suite_allocate("toy", 6, 3, dimensions, "instances:1");

  return suite;
}

/**
 * @brief Returns the problem from the toy suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_toy_get_problem(coco_suite_t *suite,
                                             const size_t function_idx,
                                             const size_t dimension_idx,
                                             const size_t instance_idx) {


  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  if (function == 1) {
    problem = f_sphere_allocate(dimension);
  } else if (function == 2) {
    problem = f_ellipsoid_allocate(dimension);
  } else if (function == 3) {
    problem = f_rastrigin_allocate(dimension);
  } else if (function == 4) {
    problem = f_bueche_rastrigin_allocate(dimension);
  } else if (function == 5) {
    double xopt[40] = { 5.0 };
    problem = f_linear_slope_allocate(dimension, xopt);
  } else if (function == 6) {
    problem = f_rosenbrock_allocate(dimension);
  } else {
    coco_error("suite_toy_get_problem(): function %lu does not exist in this suite", function);
    return NULL; /* Never reached */
  }

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
#line 21 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/suite_largescale.c"
/**
 * @file suite_largescale.c
 * @brief Implementation of the bbob large-scale suite containing 1 function in 6 large dimensions.
 */

#line 7 "code-experiments/src/suite_largescale.c"

#line 9 "code-experiments/src/suite_largescale.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob large-scale suite.
 */
static coco_suite_t *suite_largescale_initialize(void) {
  
  coco_suite_t *suite;
  /*const size_t dimensions[] = { 8, 16, 32, 64, 128, 256,512,1024};*/
  const size_t dimensions[] = { 40, 80, 160, 320, 640, 1280};
  suite = coco_suite_allocate("bbob-largescale", 1, 6, dimensions, "instances:1-15");
  return suite;
}

/**
 * @brief Creates and returns a large-scale problem without needing the actual large-scale suite.
 */
static coco_problem_t *coco_get_largescale_problem(const size_t function,
                                                   const size_t dimension,
                                                   const size_t instance) {
  coco_problem_t *problem = NULL;

  const char *problem_id_template = "bbob_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "BBOB suite problem f%lu instance %lu in %luD";

  const long rseed = (long) (function + 10000 * instance);
  /*const long rseed_3 = (long) (3 + 10000 * instance);*/
  /*const long rseed_17 = (long) (17 + 10000 * instance);*/
  if (function == 1) {
    problem = f_ellipsoid_permblockdiag_bbob_problem_allocate(function, dimension, instance, rseed,
        problem_id_template, problem_name_template);
  } else {
    coco_error("coco_get_largescale_problem(): cannot retrieve problem f%lu instance %lu in %luD", function,
        instance, dimension);
    return NULL; /* Never reached */
  }

  return problem;
}

/**
 * @brief Returns the problem from the bbob large-scale suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_largescale_get_problem(coco_suite_t *suite,
                                                    const size_t function_idx,
                                                    const size_t dimension_idx,
                                                    const size_t instance_idx) {
  
  coco_problem_t *problem = NULL;
  
  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];
  
  problem = coco_get_largescale_problem(function, dimension, instance);
  
  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);
  
  return problem;
}
#line 22 "code-experiments/src/coco_suite.c"

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
    suite = suite_toy_initialize();
  } else if (strcmp(suite_name, "bbob") == 0) {
    suite = suite_bbob_initialize();
  } else if (strcmp(suite_name, "bbob-biobj") == 0) {
    suite = suite_biobj_initialize();
  } else if (strcmp(suite_name, "bbob-largescale") == 0) {
    suite = suite_largescale_initialize();
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
static const char *coco_suite_get_instances_by_year(const coco_suite_t *suite, const int year) {
  const char *year_string;

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
                                                           const size_t function_idx,
                                                           const size_t dimension_idx,
                                                           const size_t instance_idx) {

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
 * @brief Sets the suite instance to the given instance_numbers.
 */
static void coco_suite_set_instance(coco_suite_t *suite,
                                    const size_t *instance_numbers) {

  size_t i;

  if (!instance_numbers) {
    coco_error("coco_suite_set_instance(): no instance given");
    return;
  }

  suite->number_of_instances = coco_count_numbers(instance_numbers, COCO_MAX_INSTANCES, "suite instance numbers");
  suite->instances = coco_allocate_vector_size_t(suite->number_of_instances);
  for (i = 0; i < suite->number_of_instances; i++) {
    suite->instances[i] = instance_numbers[i];
  }

}

/**
 * @brief Filters the given items w.r.t. the given indices (starting from 1).
 *
 * Sets items[i - 1] to 0 for every i that cannot be found in indices (this function performs the conversion
 * from user-friendly indices starting from 1 to C-friendly indices starting from 0).
 */
static void coco_suite_filter_indices(size_t *items, const size_t number_of_items, const size_t *indices, const char *name) {

  size_t i, j;
  size_t count = coco_count_numbers(indices, COCO_MAX_INSTANCES, name);
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
  size_t count = coco_count_numbers(dimension_numbers, COCO_MAX_INSTANCES, "dimensions");
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
size_t coco_suite_get_function_from_function_index(const coco_suite_t *suite, const size_t function_idx) {

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
size_t coco_suite_get_dimension_from_dimension_index(const coco_suite_t *suite, const size_t dimension_idx) {

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
size_t coco_suite_get_instance_from_instance_index(const coco_suite_t *suite, const size_t instance_idx) {

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
coco_problem_t *coco_suite_get_problem(coco_suite_t *suite, const size_t problem_index) {

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
size_t coco_suite_get_number_of_problems(const coco_suite_t *suite) {
  return (suite->number_of_instances * suite->number_of_functions * suite->number_of_dimensions);
}


/**
 * @brief Returns the instances read from either a "year: YEAR" or "instances: NUMBERS" string.
 *
 * If both "year" and "instances" are given, the second is ignored (and a warning is raised). See the
 * coco_suite function for more information about the required format.
 */
static size_t *coco_suite_get_instance_indices(const coco_suite_t *suite, const char *suite_instance) {

  int year = -1;
  char *instances = NULL;
  const char *year_string = NULL;
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
      result = coco_string_parse_ranges(year_string, 1, 0, "instances", COCO_MAX_INSTANCES);
    } else {
      coco_warning("coco_suite_get_instance_indices(): problems parsing the 'year' suite_instance option, ignored");
    }
  }

  instances = coco_allocate_string(COCO_MAX_INSTANCES);
  if ((instances_found >= 0) && (parce_instances == 1)) {
    if (coco_options_read_values(suite_instance, "instances", instances) > 0) {
      result = coco_string_parse_ranges(instances, 1, 0, "instances", COCO_MAX_INSTANCES);
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
 * - "bbob-biobj" contains 55 <a href="http://numbbo.github.io/coco-doc/bbob-biobj/functions">bi-objective
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
 * - "dimension_indices: VALUES", where VALUES is a list or a range of dimension indices (starting from 1) to keep
 * in the suite, and
 * - "function_indices: VALUES", where VALUES is a list or a range of function indices (starting from 1) to keep
 * in the suite, and
 * - "instance_indices: VALUES", where VALUES is a list or a range of instance indices (starting from 1) to keep
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

  coco_option_keys_t *known_option_keys, *given_option_keys, *redundant_option_keys;

  /* Sets the valid keys for suite options and suite instance */
  const char *known_keys_o[] = { "dimensions", "dimension_indices", "function_indices", "instance_indices" };
  const char *known_keys_i[] = { "year", "instances" };

  /* Initialize the suite */
  suite = coco_suite_intialize(suite_name);

  /* Set the instance */
  if ((!suite_instance) || (strlen(suite_instance) == 0))
    instances = coco_suite_get_instance_indices(suite, suite->default_instances);
  else {
    instances = coco_suite_get_instance_indices(suite, suite_instance);

    if (!instances) {
      /* Something wrong in the suite_instance string, use default instead */
      instances = coco_suite_get_instance_indices(suite, suite->default_instances);
    }

    /* Check for redundant option keys for suite instance */
    known_option_keys = coco_option_keys_allocate(sizeof(known_keys_i) / sizeof(char *), known_keys_i);
    given_option_keys = coco_option_keys(suite_instance);

    if (given_option_keys) {
      redundant_option_keys = coco_option_keys_get_redundant(known_option_keys, given_option_keys);

      if ((redundant_option_keys != NULL) && (redundant_option_keys->count > 0)) {
        /* Warn the user that some of given options are being ignored and output the valid options */
        char *output_redundant = coco_option_keys_get_output_string(redundant_option_keys,
            "coco_suite(): Some keys in suite instance were ignored:\n");
        char *output_valid = coco_option_keys_get_output_string(known_option_keys,
            "Valid keys for suite instance are:\n");
        coco_warning("%s%s", output_redundant, output_valid);
        coco_free_memory(output_redundant);
        coco_free_memory(output_valid);
      }

      coco_option_keys_free(given_option_keys);
      coco_option_keys_free(redundant_option_keys);
    }
    coco_option_keys_free(known_option_keys);
  }
  coco_suite_set_instance(suite, instances);
  coco_free_memory(instances);

  /* Apply filter if any given by the suite_options */
  if ((suite_options) && (strlen(suite_options) > 0)) {
    option_string = coco_allocate_string(COCO_PATH_MAX);
    if (coco_options_read_values(suite_options, "function_indices", option_string) > 0) {
      indices = coco_string_parse_ranges(option_string, 1, suite->number_of_functions, "function_indices", COCO_MAX_INSTANCES);
      if (indices != NULL) {
        coco_suite_filter_indices(suite->functions, suite->number_of_functions, indices, "function_indices");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    option_string = coco_allocate_string(COCO_PATH_MAX);
    if (coco_options_read_values(suite_options, "instance_indices", option_string) > 0) {
      indices = coco_string_parse_ranges(option_string, 1, suite->number_of_instances, "instance_indices", COCO_MAX_INSTANCES);
      if (indices != NULL) {
        coco_suite_filter_indices(suite->instances, suite->number_of_instances, indices, "instance_indices");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    dim_found = coco_strfind(suite_options, "dimensions");
    dim_idx_found = coco_strfind(suite_options, "dimension_indices");

    if ((dim_found > 0) && (dim_idx_found > 0)) {
      if (dim_found < dim_idx_found) {
        parce_dim_idx = 0;
        coco_warning("coco_suite(): 'dimension_indices' suite option ignored because it follows 'dimensions'");
      }
      else {
        parce_dim = 0;
        coco_warning("coco_suite(): 'dimensions' suite option ignored because it follows 'dimension_indices'");
      }
    }

    option_string = coco_allocate_string(COCO_PATH_MAX);
    if ((dim_idx_found >= 0) && (parce_dim_idx == 1)
        && (coco_options_read_values(suite_options, "dimension_indices", option_string) > 0)) {
      indices = coco_string_parse_ranges(option_string, 1, suite->number_of_dimensions, "dimension_indices",
          COCO_MAX_INSTANCES);
      if (indices != NULL) {
        coco_suite_filter_indices(suite->dimensions, suite->number_of_dimensions, indices, "dimension_indices");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    option_string = coco_allocate_string(COCO_PATH_MAX);
    if ((dim_found >= 0) && (parce_dim == 1)
        && (coco_options_read_values(suite_options, "dimensions", option_string) > 0)) {
      ptr = option_string;
      /* Check for disallowed characters */
      while (*ptr != '\0') {
        if ((*ptr != ',') && !isdigit((unsigned char )*ptr)) {
          coco_warning("coco_suite(): 'dimensions' suite option ignored because of disallowed characters");
          return NULL;
        } else
          ptr++;
      }
      dimensions = coco_string_parse_ranges(option_string, suite->dimensions[0],
          suite->dimensions[suite->number_of_dimensions - 1], "dimensions", COCO_MAX_INSTANCES);
      if (dimensions != NULL) {
        coco_suite_filter_dimensions(suite, dimensions);
        coco_free_memory(dimensions);
      }
    }
    coco_free_memory(option_string);

    /* Check for redundant option keys for suite options */
    known_option_keys = coco_option_keys_allocate(sizeof(known_keys_o) / sizeof(char *), known_keys_o);
    given_option_keys = coco_option_keys(suite_options);

    if (given_option_keys) {
      redundant_option_keys = coco_option_keys_get_redundant(known_option_keys, given_option_keys);

      if ((redundant_option_keys != NULL) && (redundant_option_keys->count > 0)) {
        /* Warn the user that some of given options are being ignored and output the valid options */
        char *output_redundant = coco_option_keys_get_output_string(redundant_option_keys,
            "coco_suite(): Some keys in suite options were ignored:\n");
        char *output_valid = coco_option_keys_get_output_string(known_option_keys,
            "Valid keys for suite options are:\n");
        coco_warning("%s%s", output_redundant, output_valid);
        coco_free_memory(output_redundant);
        coco_free_memory(output_valid);
      }

      coco_option_keys_free(given_option_keys);
      coco_option_keys_free(redundant_option_keys);
    }
    coco_option_keys_free(known_option_keys);

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

  long previous_function_idx;
  long previous_dimension_idx;
  long previous_instance_idx;

  assert(suite != NULL);

  previous_function_idx = suite->current_function_idx;
  previous_dimension_idx = suite->current_dimension_idx;
  previous_instance_idx = suite->current_instance_idx;

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
      char *time_string = coco_current_time_get_string();
      if (dimension_idx > 0)
        coco_info_partial("done\n");
      else
        coco_info_partial("\n");
      coco_info_partial("COCO INFO: %s, d=%lu, running: f%02lu", time_string, suite->dimensions[dimension_idx], suite->functions[function_idx]);
      coco_free_memory(time_string);
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

/* See coco.h for more information on encoding and decoding problem index */

/**
 * @param suite The suite.
 * @param function_idx Index of the function (starting with 0).
 * @param dimension_idx Index of the dimension (starting with 0).
 * @param instance_idx Index of the insatnce (starting with 0).
 *
 * @return The problem index in the suite computed from function_idx, dimension_idx and instance_idx.
 */
size_t coco_suite_encode_problem_index(const coco_suite_t *suite,
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
void coco_suite_decode_problem_index(const coco_suite_t *suite,
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
#line 1 "code-experiments/src/coco_observer.c"
/**
 * @file coco_observer.c
 * @brief Definitions of functions regarding COCO observers.
 */

#line 7 "code-experiments/src/coco_observer.c"
#line 8 "code-experiments/src/coco_observer.c"
#include <limits.h>
#include <float.h>
#include <math.h>

/**
 * @brief The type for triggers based on target values.
 *
 * The target values that trigger logging are at every 10**(exponent/number_of_triggers) from positive
 * infinity down to precision and from -precision on with step -10**(exponent/number_of_triggers) until
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

static int coco_observer_targets_trigger(coco_observer_targets_t *targets, const double given_value) {

  int update_performed = 0;

  const double number_of_targets_double = (double) (long) targets->number_of_triggers;

  double verified_value = 0;
  int last_exponent = 0;
  int current_exponent = 0;
  int adjusted_exponent = 0;

  assert(targets != NULL);

  /* The given_value is positive */
  if (given_value > 0) {

    /* If close to zero, use precision instead of the given_value*/
    if (given_value < targets->precision) {
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
    adjusted_exponent = 2 * (int) (ceil(log10(targets->precision) * number_of_targets_double))
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

#line 1 "code-experiments/src/logger_bbob.c"
/**
 * @file logger_bbob.c
 * @brief Implementation of the bbob logger.
 *
 * Logs the performance of a single-objective optimizer on noisy or noiseless problems.
 * It produces four kinds of files:
 * - The "info" files ...
 * - The "dat" files ...
 * - The "tdat" files ...
 * - The "rdat" files ...
 */

/* TODO: Document this file in doxygen style! */

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <errno.h>

#line 23 "code-experiments/src/logger_bbob.c"

#line 25 "code-experiments/src/logger_bbob.c"
#line 26 "code-experiments/src/logger_bbob.c"
#line 27 "code-experiments/src/logger_bbob.c"
#line 1 "code-experiments/src/observer_bbob.c"
/**
 * @file observer_bbob.c
 * @brief Implementation of the bbob observer.
 */

#line 7 "code-experiments/src/observer_bbob.c"
#line 8 "code-experiments/src/observer_bbob.c"

static coco_problem_t *logger_bbob(coco_observer_t *observer, coco_problem_t *problem);
static void logger_bbob_free(void *logger);

/**
 * @brief The bbob observer data type.
 */
typedef struct {
  /* TODO: Can be used to store variables that need to be accessible during one run (i.e. for multiple
   * problems). For example, the following global variables from logger_bbob.c could be stored here: */
  size_t current_dim;
  size_t current_fun_id;
  /* ... and others */
} observer_bbob_data_t;

/**
 * @brief Initializes the bbob observer.
 */
static void observer_bbob(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer->logger_allocate_function = logger_bbob;
  observer->logger_free_function = logger_bbob_free;
  observer->data_free_function = NULL;
  observer->data = NULL;

  *option_keys = NULL;

  (void) options; /* To silence the compiler */
}
#line 28 "code-experiments/src/logger_bbob.c"

/*static const size_t bbob_nbpts_nbevals = 20; Wassim: tentative, are now observer options with these default values*/
/*static const size_t bbob_nbpts_fval = 5;*/
static size_t bbob_current_dim = 0;
static size_t bbob_current_funId = 0;
static size_t bbob_infoFile_firstInstance = 0;
char bbob_infoFile_firstInstance_char[3];
/* a possible solution: have a list of dims that are already in the file, if the ones we're about to log
 * is != bbob_current_dim and the funId is currend_funId, create a new .info file with as suffix the
 * number of the first instance */
static const int bbob_number_of_dimensions = 6;
static size_t bbob_dimensions_in_current_infoFile[6] = { 0, 0, 0, 0, 0, 0 }; /* TODO should use dimensions from the suite */

/* The current_... mechanism fails if several problems are open.
 * For the time being this should lead to an error.
 *
 * A possible solution: bbob_logger_is_open becomes a reference
 * counter and as long as another logger is open, always a new info
 * file is generated.
 * TODO: Shouldn't the new way of handling observers already fix this?
 */
static int bbob_logger_is_open = 0; /* this could become lock-list of .info files */

/* TODO: add possibility of adding a prefix to the index files (easy to do through observer options) */

/**
 * @brief The bbob logger data type.
 */
typedef struct {
  coco_observer_t *observer;
  int is_initialized;
  /*char *path;// relative path to the data folder. //Wassim: now fetched from the observer */
  /*const char *alg_name; the alg name, for now, temporarily the same as the path. Wassim: Now in the observer */
  FILE *index_file; /* index file */
  FILE *fdata_file; /* function value aligned data file */
  FILE *tdata_file; /* number of function evaluations aligned data file */
  FILE *rdata_file; /* restart info data file */
  size_t number_of_evaluations;
  double best_fvalue;
  double last_fvalue;
  short written_last_eval; /* allows writing the the data of the final fun eval in the .tdat file if not already written by the t_trigger*/
  double *best_solution;
  /* The following are to only pass data as a parameter in the free function. The
   * interface should probably be the same for all free functions so passing the
   * problem as a second parameter is not an option even though we need info
   * form it.*/
  size_t function_id; /*TODO: consider changing name*/
  size_t instance_id;
  size_t number_of_variables;
  double optimal_fvalue;

  coco_observer_targets_t *targets;          /**< @brief Triggers based on target values. */
  coco_observer_evaluations_t *evaluations;  /**< @brief Triggers based on the number of evaluations. */

} logger_bbob_data_t;

static const char *bbob_file_header_str = "%% function evaluation | "
    "noise-free fitness - Fopt (%13.12e) | "
    "best noise-free fitness - Fopt | "
    "measured fitness | "
    "best measured fitness | "
    "x1 | "
    "x2...\n";

/**
 * adds a formated line to a data file
 */
static void logger_bbob_write_data(FILE *target_file,
                                   size_t number_of_evaluations,
                                   double fvalue,
                                   double best_fvalue,
                                   double best_value,
                                   const double *x,
                                   size_t number_of_variables) {
  /* for some reason, it's %.0f in the old code instead of the 10.9e
   * in the documentation
   */
  fprintf(target_file, "%ld %+10.9e %+10.9e %+10.9e %+10.9e", number_of_evaluations, fvalue - best_value,
      best_fvalue - best_value, fvalue, best_fvalue);
  if (number_of_variables < 22) {
    size_t i;
    for (i = 0; i < number_of_variables; i++) {
      fprintf(target_file, " %+5.4e", x[i]);
    }
  }
  fprintf(target_file, "\n");
}

/**
 * Error when trying to create the file "path"
 */
static void logger_bbob_error_io(FILE *path, int errnum) {
  const char *error_format = "Error opening file: %s\n ";
  coco_error(error_format, strerror(errnum), path);
}

/**
 * Creates the data files or simply opens it
 */

/*
 calling sequence:
 logger_bbob_open_dataFile(&(logger->fdata_file), logger->observer->output_folder, dataFile_path,
 ".dat");
 */

static void logger_bbob_open_dataFile(FILE **target_file,
                                      const char *path,
                                      const char *dataFile_path,
                                      const char *file_extension) {
  char file_path[COCO_PATH_MAX] = { 0 };
  char relative_filePath[COCO_PATH_MAX] = { 0 };
  int errnum;
  strncpy(relative_filePath, dataFile_path,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  strncat(relative_filePath, file_extension,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
  if (*target_file == NULL) {
    *target_file = fopen(file_path, "a+");
    errnum = errno;
    if (*target_file == NULL) {
      logger_bbob_error_io(*target_file, errnum);
    }
  }
}

/*
static void logger_bbob_open_dataFile(FILE **target_file,
                                      const char *path,
                                      const char *dataFile_path,
                                      const char *file_extension) {
  char file_path[COCO_PATH_MAX] = { 0 };
  char relative_filePath[COCO_PATH_MAX] = { 0 };
  int errnum;
  strncpy(relative_filePath, dataFile_path,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  strncat(relative_filePath, file_extension,
  COCO_PATH_MAX - strlen(relative_filePath) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
  if (*target_file == NULL) {
    *target_file = fopen(file_path, "a+");
    errnum = errno;
    if (*target_file == NULL) {
      _bbob_logger_error_io(*target_file, errnum);
    }
  }
}*/

/**
 * Creates the index file fileName_prefix+problem_id+file_extension in
 * folde_path
 */
static void logger_bbob_openIndexFile(logger_bbob_data_t *logger,
                                      const char *folder_path,
                                      const char *indexFile_prefix,
                                      const char *function_id,
                                      const char *dataFile_path) {
  /* to add the instance number TODO: this should be done outside to avoid redoing this for the .*dat files */
  char used_dataFile_path[COCO_PATH_MAX] = { 0 };
  int errnum, newLine; /* newLine is at 1 if we need a new line in the info file */
  char function_id_char[3]; /* TODO: consider adding them to logger */
  char file_name[COCO_PATH_MAX] = { 0 };
  char file_path[COCO_PATH_MAX] = { 0 };
  FILE **target_file;
  FILE *tmp_file;
  strncpy(used_dataFile_path, dataFile_path, COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
  if (bbob_infoFile_firstInstance == 0) {
    bbob_infoFile_firstInstance = logger->instance_id;
  }
  sprintf(function_id_char, "%lu", logger->function_id);
  sprintf(bbob_infoFile_firstInstance_char, "%ld", bbob_infoFile_firstInstance);
  target_file = &(logger->index_file);
  tmp_file = NULL; /* to check whether the file already exists. Don't want to use target_file */
  strncpy(file_name, indexFile_prefix, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, "_f", COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, function_id_char, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, "_i", COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, bbob_infoFile_firstInstance_char, COCO_PATH_MAX - strlen(file_name) - 1);
  strncat(file_name, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
  coco_join_path(file_path, sizeof(file_path), folder_path, file_name, NULL);
  if (*target_file == NULL) {
    tmp_file = fopen(file_path, "r"); /* to check for existence */
    if ((tmp_file) && (bbob_current_dim == logger->number_of_variables)
        && (bbob_current_funId == logger->function_id)) {
        /* new instance of current funId and current dim */
      newLine = 0;
      *target_file = fopen(file_path, "a+");
      if (*target_file == NULL) {
        errnum = errno;
        logger_bbob_error_io(*target_file, errnum);
      }
      fclose(tmp_file);
    } else { /* either file doesn't exist (new funId) or new Dim */
      /* check that the dim was not already present earlier in the file, if so, create a new info file */
      if (bbob_current_dim != logger->number_of_variables) {
        int i, j;
        for (i = 0;
            i < bbob_number_of_dimensions && bbob_dimensions_in_current_infoFile[i] != 0
                && bbob_dimensions_in_current_infoFile[i] != logger->number_of_variables; i++) {
          ; /* checks whether dimension already present in the current infoFile */
        }
        if (i < bbob_number_of_dimensions && bbob_dimensions_in_current_infoFile[i] == 0) {
          /* new dimension seen for the first time */
          bbob_dimensions_in_current_infoFile[i] = logger->number_of_variables;
          newLine = 1;
        } else {
          if (i < bbob_number_of_dimensions) { /* dimension already present, need to create a new file */
            newLine = 0;
            file_path[strlen(file_path) - strlen(bbob_infoFile_firstInstance_char) - 7] = 0; /* truncate the instance part */
            bbob_infoFile_firstInstance = logger->instance_id;
            sprintf(bbob_infoFile_firstInstance_char, "%ld", bbob_infoFile_firstInstance);
            strncat(file_path, "_i", COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, bbob_infoFile_firstInstance_char, COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
          } else {/*we have all dimensions*/
            newLine = 1;
          }
          for (j = 0; j < bbob_number_of_dimensions; j++) { /* new info file, reinitialize list of dims */
            bbob_dimensions_in_current_infoFile[j] = 0;
          }
          bbob_dimensions_in_current_infoFile[i] = logger->number_of_variables;
        }
      } else {
        if ( bbob_current_funId != logger->function_id ) {
          /*new function in the same file */
          newLine = 1;
        }
      }
      *target_file = fopen(file_path, "a+"); /* in any case, we append */
      if (*target_file == NULL) {
        errnum = errno;
        logger_bbob_error_io(*target_file, errnum);
      }
      if (tmp_file) { /* File already exists, new dim so just a new line. Also, close the tmp_file */
        if (newLine) {
          fprintf(*target_file, "\n");
        }
        fclose(tmp_file);
      }

      fprintf(*target_file, "funcId = %d, DIM = %lu, Precision = %.3e, algId = '%s'\n",
          (int) strtol(function_id, NULL, 10), logger->number_of_variables, pow(10, -8),
          logger->observer->algorithm_name);
      fprintf(*target_file, "%%\n");
      strncat(used_dataFile_path, "_i", COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      strncat(used_dataFile_path, bbob_infoFile_firstInstance_char,
      COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      fprintf(*target_file, "%s.dat", used_dataFile_path); /* dataFile_path does not have the extension */
      bbob_current_dim = logger->number_of_variables;
      bbob_current_funId = logger->function_id;
    }
  }
}

/**
 * Generates the different files and folder needed by the logger to store the
 * data if these don't already exist
 */
static void logger_bbob_initialize(logger_bbob_data_t *logger, coco_problem_t *inner_problem) {
  /*
   Creates/opens the data and index files
   */
  char dataFile_path[COCO_PATH_MAX] = { 0 }; /* relative path to the .dat file from where the .info file is */
  char folder_path[COCO_PATH_MAX] = { 0 };
  char *tmpc_funId; /* serves to extract the function id as a char *. There should be a better way of doing this! */
  char *tmpc_dim; /* serves to extract the dimension as a char *. There should be a better way of doing this! */
  char indexFile_prefix[10] = "bbobexp"; /* TODO (minor): make the prefix bbobexp a parameter that the user can modify */
  size_t str_length_funId, str_length_dim;
  
  str_length_funId = coco_double_to_size_t(bbob2009_fmax(1, ceil(log10((double) coco_problem_get_suite_dep_function(inner_problem)))));
  str_length_dim = coco_double_to_size_t(bbob2009_fmax(1, ceil(log10((double) inner_problem->number_of_variables))));
  tmpc_funId = coco_allocate_string(str_length_funId);
  tmpc_dim = coco_allocate_string(str_length_dim);

  assert(logger != NULL);
  assert(inner_problem != NULL);
  assert(inner_problem->problem_id != NULL);

  sprintf(tmpc_funId, "%lu", coco_problem_get_suite_dep_function(inner_problem));
  sprintf(tmpc_dim, "%lu", (unsigned long) inner_problem->number_of_variables);

  /* prepare paths and names */
  strncpy(dataFile_path, "data_f", COCO_PATH_MAX);
  strncat(dataFile_path, tmpc_funId,
  COCO_PATH_MAX - strlen(dataFile_path) - 1);
  coco_join_path(folder_path, sizeof(folder_path), logger->observer->result_folder, dataFile_path,
  NULL);
  coco_create_directory(folder_path);
  strncat(dataFile_path, "/bbobexp_f",
  COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_funId,
  COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, "_DIM", COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_dim, COCO_PATH_MAX - strlen(dataFile_path) - 1);

  /* index/info file */
  logger_bbob_openIndexFile(logger, logger->observer->result_folder, indexFile_prefix, tmpc_funId,
      dataFile_path);
  fprintf(logger->index_file, ", %ld", coco_problem_get_suite_dep_instance(inner_problem));
  /* data files */
  /* TODO: definitely improvable but works for now */
  strncat(dataFile_path, "_i", COCO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, bbob_infoFile_firstInstance_char,
  COCO_PATH_MAX - strlen(dataFile_path) - 1);
  logger_bbob_open_dataFile(&(logger->fdata_file), logger->observer->result_folder, dataFile_path, ".dat");
  fprintf(logger->fdata_file, bbob_file_header_str, logger->optimal_fvalue);

  logger_bbob_open_dataFile(&(logger->tdata_file), logger->observer->result_folder, dataFile_path, ".tdat");
  fprintf(logger->tdata_file, bbob_file_header_str, logger->optimal_fvalue);

  logger_bbob_open_dataFile(&(logger->rdata_file), logger->observer->result_folder, dataFile_path, ".rdat");
  fprintf(logger->rdata_file, bbob_file_header_str, logger->optimal_fvalue);
  logger->is_initialized = 1;
  coco_free_memory(tmpc_dim);
  coco_free_memory(tmpc_funId);
}

/**
 * Layer added to the transformed-problem evaluate_function by the logger
 */
static void logger_bbob_evaluate(coco_problem_t *problem, const double *x, double *y) {
  logger_bbob_data_t *logger = (logger_bbob_data_t *) coco_problem_transformed_get_data(problem);
  coco_problem_t * inner_problem = coco_problem_transformed_get_inner_problem(problem);

  size_t i;

  if (!logger->is_initialized) {
    logger_bbob_initialize(logger, inner_problem);
  }
  if ((coco_log_level >= COCO_DEBUG) && logger->number_of_evaluations == 0) {
    coco_debug("%4ld: ", inner_problem->suite_dep_index);
    coco_debug("on problem %s ... ", coco_problem_get_id(inner_problem));
  }
  coco_evaluate_function(inner_problem, x, y);
  logger->last_fvalue = y[0];
  logger->written_last_eval = 0;
  if (logger->number_of_evaluations == 0 || y[0] < logger->best_fvalue) {
    logger->best_fvalue = y[0];
    for (i = 0; i < problem->number_of_variables; i++)
      logger->best_solution[i] = x[i];
  }
  logger->number_of_evaluations++;

  /* Add sanity check for optimal f value */
  assert(y[0] + 1e-13 >= logger->optimal_fvalue);

  /* Add a line in the .dat file for each logging target reached. */
    if (coco_observer_targets_trigger(logger->targets, y[0] - logger->optimal_fvalue)) {

    logger_bbob_write_data(logger->fdata_file, logger->number_of_evaluations, y[0], logger->best_fvalue,
        logger->optimal_fvalue, x, problem->number_of_variables);
  }

  /* Add a line in the .tdat file each time an fevals trigger is reached.*/
  if (coco_observer_evaluations_trigger(logger->evaluations, logger->number_of_evaluations)) {
    logger->written_last_eval = 1;
    logger_bbob_write_data(logger->tdata_file, logger->number_of_evaluations, y[0], logger->best_fvalue,
        logger->optimal_fvalue, x, problem->number_of_variables);
  }

  /* Flush output so that impatient users can see progress. */
  fflush(logger->fdata_file);
}

/**
 * Also serves as a finalize run method so. Must be called at the end
 * of Each run to correctly fill the index file
 *
 * TODO: make sure it is called at the end of each run or move the
 * writing into files to another function
 */
static void logger_bbob_free(void *stuff) {
  /* TODO: do all the "non simply freeing" stuff in another function
   * that can have problem as input
   */
  logger_bbob_data_t *logger = (logger_bbob_data_t *) stuff;

  if ((coco_log_level >= COCO_DEBUG) && logger && logger->number_of_evaluations > 0) {
    coco_debug("best f=%e after %ld fevals (done observing)\n", logger->best_fvalue,
        logger->number_of_evaluations);
  }
  if (logger->index_file != NULL) {
    fprintf(logger->index_file, ":%ld|%.1e", logger->number_of_evaluations,
        logger->best_fvalue - logger->optimal_fvalue);
    fclose(logger->index_file);
    logger->index_file = NULL;
  }
  if (logger->fdata_file != NULL) {
    fclose(logger->fdata_file);
    logger->fdata_file = NULL;
  }
  if (logger->tdata_file != NULL) {
    /* TODO: make sure it handles restarts well. i.e., it writes
     * at the end of a single run, not all the runs on a given
     * instance. Maybe start with forcing it to generate a new
     * "instance" of problem for each restart in the beginning
     */
    if (!logger->written_last_eval) {
      logger_bbob_write_data(logger->tdata_file, logger->number_of_evaluations, logger->last_fvalue,
          logger->best_fvalue, logger->optimal_fvalue, logger->best_solution, logger->number_of_variables);
    }
    fclose(logger->tdata_file);
    logger->tdata_file = NULL;
  }

  if (logger->rdata_file != NULL) {
    fclose(logger->rdata_file);
    logger->rdata_file = NULL;
  }

  if (logger->best_solution != NULL) {
    coco_free_memory(logger->best_solution);
    logger->best_solution = NULL;
  }

  if (logger->targets != NULL){
    coco_free_memory(logger->targets);
    logger->targets = NULL;
  }

  if (logger->evaluations != NULL){
    coco_observer_evaluations_free(logger->evaluations);
    logger->evaluations = NULL;
  }

  bbob_logger_is_open = 0;
}

static coco_problem_t *logger_bbob(coco_observer_t *observer, coco_problem_t *inner_problem) {
  logger_bbob_data_t *logger_bbob;
  coco_problem_t *problem;

  logger_bbob = (logger_bbob_data_t *) coco_allocate_memory(sizeof(*logger_bbob));
  logger_bbob->observer = observer;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_bbob(): The bbob logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }

  if (bbob_logger_is_open)
    coco_error("The current bbob_logger (observer) must be closed before a new one is opened");
  /* This is the name of the folder which happens to be the algName */
  /*logger->path = coco_strdup(observer->output_folder);*/
  logger_bbob->index_file = NULL;
  logger_bbob->fdata_file = NULL;
  logger_bbob->tdata_file = NULL;
  logger_bbob->rdata_file = NULL;
  logger_bbob->number_of_variables = inner_problem->number_of_variables;
  if (inner_problem->best_value == NULL) {
    /* coco_error("Optimal f value must be defined for each problem in order for the logger to work properly"); */
    /* Setting the value to 0 results in the assertion y>=optimal_fvalue being susceptible to failure */
    coco_warning("undefined optimal f value. Set to 0");
    logger_bbob->optimal_fvalue = 0;
  } else {
    logger_bbob->optimal_fvalue = *(inner_problem->best_value);
  }

  logger_bbob->number_of_evaluations = 0;
  logger_bbob->best_solution = coco_allocate_vector(inner_problem->number_of_variables);
  /* TODO: the following inits are just to be in the safe side and
   * should eventually be removed. Some fields of the bbob_logger struct
   * might be useless
   */
  logger_bbob->function_id = coco_problem_get_suite_dep_function(inner_problem);
  logger_bbob->instance_id = coco_problem_get_suite_dep_instance(inner_problem);
  logger_bbob->written_last_eval = 1;
  logger_bbob->last_fvalue = DBL_MAX;
  logger_bbob->is_initialized = 0;

  /* Initialize triggers based on target values and number of evaluations */
  logger_bbob->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  logger_bbob->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, logger_bbob, logger_bbob_free, observer->observer_name);

  problem->evaluate_function = logger_bbob_evaluate;
  bbob_logger_is_open = 1;
  return problem;
}

#line 361 "code-experiments/src/coco_observer.c"
#line 1 "code-experiments/src/logger_biobj.c"
/**
 * @file logger_biobj.c
 * @brief Implementation of the bbob-biobj logger.
 *
 * Logs the values of the implemented indicators and archives nondominated solutions.
 * Produces four kinds of files:
 * - The "info" files contain high-level information on the performed experiment. One .info file is created
 * for each problem group (and indicator type) and contains information on all the problems in that problem
 * group (and indicator type).
 * - The "dat" files contain function evaluations, indicator values and target hits for every target hit as
 * well as for the last evaluation. One .dat file is created for each problem function and dimension (and
 * indicator type) and contains information for all instances of that problem (and indicator type).
 * - The "tdat" files contain function evaluation and indicator values for every predefined evaluation
 * number as well as for the last evaluation. One .tdat file is created for each problem function and
 * dimension (and indicator type) and contains information for all instances of that problem (and indicator
 * type).
 * - The "adat" files are archive files that contain function evaluations, 2 objectives and dim variables
 * for every nondominated solution. Whether these files are created, at what point in time the logger writes
 * nondominated solutions to the archive and whether the decision variables are output or not depends on
 * the values of log_nondom_mode and log_nondom_mode. See the bi-objective observer constructor
 * observer_biobj() for more information. One .adat file is created for each problem function and dimension
 * and contains information for all instances of that problem.
 *
 * @note Whenever in this file a ROI is mentioned, it means the region of interest in the objective space.
 * The ROI is a rectangle with the ideal and nadir points as its two opposite vertices.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#line 33 "code-experiments/src/logger_biobj.c"
#line 34 "code-experiments/src/logger_biobj.c"

#line 36 "code-experiments/src/logger_biobj.c"
#line 37 "code-experiments/src/logger_biobj.c"
#line 38 "code-experiments/src/logger_biobj.c"
#line 1 "code-experiments/src/mo_avl_tree.c"
/*****************************************************************************

 avl.c - Source code for libavl

 Copyright (c) 1998  Michael H. Buselli <cosine@cosine.org>
 Copyright (c) 2000-2009  Wessel Dankers <wsl@fruit.je>

 This file is part of libavl.

 libavl is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as
 published by the Free Software Foundation, either version 3 of
 the License, or (at your option) any later version.

 libavl is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 You should have received a copy of the GNU General Public License
 and a copy of the GNU Lesser General Public License along with
 libavl.  If not, see <http://www.gnu.org/licenses/>.

 Augmented AVL-tree. Original by Michael H. Buselli <cosine@cosine.org>.

 Modified by Wessel Dankers <wsl@fruit.je> to add a bunch of bloat
 to the source code, change the interface and replace a few bugs.
 Mail him if you find any new bugs.

 Renamed and additionally modified by BOBBies to fit the COCO platform.

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/* In order to easily (un)comment unused functions */
#define AVL_TREE_COMMENT_UNUSED 1

/* We need either depths, counts or both (the latter being the default) */
#if !defined(AVL_DEPTH) && !defined(AVL_COUNT)
#define AVL_DEPTH
#define AVL_COUNT
#endif

/* User supplied function to compare two items like strcmp() does.
 * For example: compare(a,b) will return:
 *   -1  if a < b
 *    0  if a = b
 *    1  if a > b
 */
typedef int (*avl_compare_t)(const void *a, const void *b, void *userdata);

/* User supplied function to delete an item when a node is free()d.
 * If NULL, the item is not free()d.
 */
typedef void (*avl_free_t)(void *item, void *userdata);

#define AVL_CMP(a,b) ((a) < (b) ? -1 : (a) != (b))

#if defined(AVL_COUNT) && defined(AVL_DEPTH)
#define AVL_NODE_INITIALIZER(item) { 0, 0, 0, 0, 0, (item), 0, 0 }
#else
#define AVL_NODE_INITIALIZER(item) { 0, 0, 0, 0, 0, (item), 0 }
#endif

typedef struct avl_node {
  struct avl_node *prev;
  struct avl_node *next;
  struct avl_node *parent;
  struct avl_node *left;
  struct avl_node *right;
  void *item;
#ifdef AVL_COUNT
  unsigned long count;
#endif
#ifdef AVL_DEPTH
  unsigned char depth;
#endif
} avl_node_t;

#define AVL_TREE_INITIALIZER(cmp, free) { 0, 0, 0, (cmp), (free), {0}, 0, 0 }

typedef struct avl_tree {
  avl_node_t *top;
  avl_node_t *head;
  avl_node_t *tail;
  avl_compare_t cmpitem;
  avl_free_t freeitem;
  void *userdata;
  struct avl_allocator *allocator;
} avl_tree_t;

#define AVL_ALLOCATOR_INITIALIZER(alloc, dealloc) { (alloc), (dealloc) }

typedef avl_node_t *(*avl_allocate_t)(struct avl_allocator *);
typedef void (*avl_deallocate_t)(struct avl_allocator *, avl_node_t *);

typedef struct avl_allocator {
  avl_allocate_t allocate;
  avl_deallocate_t deallocate;
} avl_allocator_t;

static void avl_rebalance(avl_tree_t *, avl_node_t *);
static avl_node_t *avl_node_insert_after(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode);

#ifdef AVL_COUNT
#define AVL_NODE_COUNT(n)  ((n) ? (n)->count : 0)
#define AVL_L_COUNT(n)     (AVL_NODE_COUNT((n)->left))
#define AVL_R_COUNT(n)     (AVL_NODE_COUNT((n)->right))
#define AVL_CALC_COUNT(n)  (AVL_L_COUNT(n) + AVL_R_COUNT(n) + 1)
#endif

#ifdef AVL_DEPTH
#define AVL_NODE_DEPTH(n)  ((n) ? (n)->depth : 0)
#define AVL_L_DEPTH(n)     (AVL_NODE_DEPTH((n)->left))
#define AVL_R_DEPTH(n)     (AVL_NODE_DEPTH((n)->right))
#define AVL_CALC_DEPTH(n)  ((unsigned char)((AVL_L_DEPTH(n) > AVL_R_DEPTH(n) ? AVL_L_DEPTH(n) : AVL_R_DEPTH(n)) + 1))
#endif

const avl_node_t avl_node_0 = { 0, 0, 0, 0, 0, 0, 0, 0 };
const avl_tree_t avl_tree_0 = { 0, 0, 0, 0, 0, 0, 0 };
const avl_allocator_t avl_allocator_0 = { 0, 0 };

#define AVL_CONST_NODE(x) ((avl_node_t *)(x))
#define AVL_CONST_ITEM(x) ((void *)(x))

static int avl_check_balance(avl_node_t *avlnode) {
#ifdef AVL_DEPTH
  int d;
  d = AVL_R_DEPTH(avlnode) - AVL_L_DEPTH(avlnode);
  return d < -1 ? -1 : d > 1;
#else
  /*  int d;
   *  d = ffs(AVL_R_COUNT(avl_node)) - ffs(AVL_L_COUNT(avl_node));
   *  d = d < -1 ? -1 : d > 1;
   */
#ifdef AVL_COUNT
  int pl, r;

  pl = ffs(AVL_L_COUNT(avlnode));
  r = AVL_R_COUNT(avlnode);

  if (r >> pl + 1)
  return 1;
  if (pl < 2 || r >> pl - 2)
  return 0;
  return -1;
#else
#error No balancing possible.
#endif
#endif
}

#ifdef AVL_COUNT
static unsigned long avl_count(const avl_tree_t *avltree) {
  if (!avltree)
    return 0;
  return AVL_NODE_COUNT(avltree->top);
}

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_at(const avl_tree_t *avltree, unsigned long index) {
  avl_node_t *avlnode;
  unsigned long c;

  if (!avltree)
    return NULL;

  avlnode = avltree->top;

  while (avlnode) {
    c = AVL_L_COUNT(avlnode);

    if (index < c) {
      avlnode = avlnode->left;
    } else if (index > c) {
      avlnode = avlnode->right;
      index -= c + 1;
    } else {
      return avlnode;
    }
  }
  return NULL;
}
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static unsigned long avl_index(const avl_node_t *avlnode) {
  avl_node_t *next;
  unsigned long c;

  if (!avlnode)
    return 0;

  c = AVL_L_COUNT(avlnode);

  while ((next = avlnode->parent)) {
    if (avlnode == next->right)
      c += AVL_L_COUNT(next) + 1;
    avlnode = next;
  }

  return c;
}
#endif
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static const avl_node_t *avl_search_leftmost_equal(const avl_tree_t *tree, const avl_node_t *node,
    const void *item) {
  avl_compare_t cmp = tree->cmpitem;
  void *userdata = tree->userdata;
  const avl_node_t *r = node;

  for (;;) {
    for (;;) {
      node = node->left;
      if (!node)
        return r;
      if (cmp(item, node->item, userdata))
        break;
      r = node;
    }
    for (;;) {
      node = node->right;
      if (!node)
        return r;
      if (!cmp(item, node->item, userdata))
        break;
    }
    r = node;
  }

  return NULL; /* To silence the compiler */

}
#endif

static const avl_node_t *avl_search_rightmost_equal(const avl_tree_t *tree,
                                                    const avl_node_t *node,
                                                    const void *item) {
  avl_compare_t cmp = tree->cmpitem;
  void *userdata = tree->userdata;
  const avl_node_t *r = node;

  for (;;) {
    for (;;) {
      node = node->right;
      if (!node)
        return r;
      if (cmp(item, node->item, userdata))
        break;
      r = node;
    }
    for (;;) {
      node = node->left;
      if (!node)
        return r;
      if (!cmp(item, node->item, userdata))
        break;
    }
    r = node;
  }

  return NULL; /* To silence the compiler */
}

/* Searches for an item, returning either some exact
 * match, or (if no exact match could be found) the first (leftmost)
 * of the nodes that have an item larger than the search item.
 * If exact is not NULL, *exact will be set to:
 *    0  if the returned node is unequal or NULL
 *    1  if the returned node is equal
 * Returns NULL if no equal or larger element could be found.
 * O(lg n) */
#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_search_leftish(const avl_tree_t *tree, const void *item, int *exact) {
  avl_node_t *node;
  avl_compare_t cmp;
  void *userdata;
  int c;

  if (!exact)
    exact = &c;

  if (!tree)
    return *exact = 0, (avl_node_t *) NULL;

  node = tree->top;
  if (!node)
    return *exact = 0, (avl_node_t *) NULL;

  cmp = tree->cmpitem;
  userdata = tree->userdata;

  for (;;) {
    c = cmp(item, node->item, userdata);

    if (c < 0) {
      if (node->left)
        node = node->left;
      else
        return *exact = 0, node;
    } else if (c > 0) {
      if (node->right)
        node = node->right;
      else
        return *exact = 0, node->next;
    } else {
      return *exact = 1, node;
    }
  }

  return NULL; /* To silence the compiler */

}
#endif

/* Searches for an item, returning either some exact
 * match, or (if no exact match could be found) the last (rightmost)
 * of the nodes that have an item smaller than the search item.
 * If exact is not NULL, *exact will be set to:
 *    0  if the returned node is unequal or NULL
 *    1  if the returned node is equal
 * Returns NULL if no equal or smaller element could be found.
 * O(lg n) */
static avl_node_t *avl_search_rightish(const avl_tree_t *tree, const void *item, int *exact) {
  avl_node_t *node;
  avl_compare_t cmp;
  void *userdata;
  int c;

  if (!exact)
    exact = &c;

  if (!tree)
    return *exact = 0, (avl_node_t *) NULL;

  node = tree->top;
  if (!node)
    return *exact = 0, (avl_node_t *) NULL;

  cmp = tree->cmpitem;
  userdata = tree->userdata;

  for (;;) {
    c = cmp(item, node->item, userdata);

    if (c < 0) {
      if (node->left)
        node = node->left;
      else
        return *exact = 0, node->prev;
    } else if (c > 0) {
      if (node->right)
        node = node->right;
      else
        return *exact = 0, node;
    } else {
      return *exact = 1, node;
    }
  }

  return NULL; /* To silence the compiler */
}

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_item_search_left(const avl_tree_t *tree, const void *item, int *exact) {
  avl_node_t *node;
  int c;

  if (!exact)
    exact = &c;

  if (!tree)
    return *exact = 0, (avl_node_t *) NULL;

  node = avl_search_leftish(tree, item, exact);
  if (*exact)
    return AVL_CONST_NODE(avl_search_leftmost_equal(tree, node, item));

  return AVL_CONST_NODE(node);
}
#endif

/* Searches for an item, returning either the last (rightmost) exact
 * match, or (if no exact match could be found) the last (rightmost)
 * of the nodes that have an item smaller than the search item.
 * If exact is not NULL, *exact will be set to:
 *    0  if the returned node is inequal or NULL
 *    1  if the returned node is equal
 * Returns NULL if no equal or smaller element could be found.
 * O(lg n) */
static avl_node_t *avl_item_search_right(const avl_tree_t *tree, const void *item, int *exact) {
  const avl_node_t *node;
  int c;

  if (!exact)
    exact = &c;

  node = avl_search_rightish(tree, item, exact);
  if (*exact)
    return AVL_CONST_NODE(avl_search_rightmost_equal(tree, node, item));

  return AVL_CONST_NODE(node);
}

/* Searches for the item in the tree and returns a matching node if found
 * or NULL if not.
 * O(lg n) */
static avl_node_t *avl_item_search(const avl_tree_t *avltree, const void *item) {
  int c;
  avl_node_t *n;
  n = avl_search_rightish(avltree, item, &c);
  return c ? n : NULL;
}

/* Initializes a new tree for elements that will be ordered using
 * the supplied strcmp()-like function.
 * Returns the value of avltree (even if it's NULL).
 * O(1) */
static avl_tree_t *avl_tree_init(avl_tree_t *avltree, avl_compare_t cmp, avl_free_t free) {
  if (avltree) {
    avltree->head = NULL;
    avltree->tail = NULL;
    avltree->top = NULL;
    avltree->cmpitem = cmp;
    avltree->freeitem = free;
    avltree->userdata = NULL;
    avltree->allocator = NULL;
  }
  return avltree;
}

/* Allocates and initializes a new tree for elements that will be
 * ordered using the supplied strcmp()-like function.
 * Returns NULL if memory could not be allocated.
 * O(1) */
static avl_tree_t *avl_tree_construct(avl_compare_t cmp, avl_free_t free) {
  return avl_tree_init((avl_tree_t *) malloc(sizeof(avl_tree_t)), cmp, free);
}

/* Reinitializes the tree structure for reuse. Nothing is free()d.
 * Compare and free functions are left alone.
 * Returns the value of avltree (even if it's NULL).
 * O(1) */
static avl_tree_t *avl_tree_clear(avl_tree_t *avltree) {
  if (avltree)
    avltree->top = avltree->head = avltree->tail = NULL;
  return avltree;
}

static void avl_node_free(avl_tree_t *avltree, avl_node_t *node) {
  avl_allocator_t *allocator;
  avl_deallocate_t deallocate;

  allocator = avltree->allocator;
  if (allocator) {
    deallocate = allocator->deallocate;
    if (deallocate)
      deallocate(allocator, node);
  } else {
    free(node);
  }
}

/* Free()s all nodes in the tree but leaves the tree itself.
 * If the tree's free is not NULL it will be invoked on every item.
 * Returns the value of avltree (even if it's NULL).
 * O(n) */
static avl_tree_t *avl_tree_purge(avl_tree_t *avltree) {
  avl_node_t *node, *next;
  avl_free_t func;
  avl_allocator_t *allocator;
  avl_deallocate_t deallocate;
  void *userdata;

  if (!avltree)
    return NULL;

  userdata = avltree->userdata;

  func = avltree->freeitem;
  allocator = avltree->allocator;
  deallocate = allocator ? allocator->deallocate : (avl_deallocate_t) NULL;

  for (node = avltree->head; node; node = next) {
    next = node->next;
    if (func)
      func(node->item, userdata);
    if (allocator) {
      if (deallocate)
        deallocate(allocator, node);
    } else {
      free(node);
    }
  }

  return avl_tree_clear(avltree);
}

/* Frees the entire tree efficiently. Nodes will be free()d.
 * If the tree's free is not NULL it will be invoked on every item.
 * O(n) */
static void avl_tree_destruct(avl_tree_t *avltree) {
  if (!avltree)
    return;
  (void) avl_tree_purge(avltree);
  free(avltree);
}

static void avl_node_clear(avl_node_t *newnode) {
  newnode->left = newnode->right = NULL;
#   ifdef AVL_COUNT
  newnode->count = 1;
#   endif
#   ifdef AVL_DEPTH
  newnode->depth = 1;
#   endif
}

/* Initializes memory for use as a node.
 * Returns the value of avlnode (even if it's NULL).
 * O(1) */
static avl_node_t *avl_node_init(avl_node_t *newnode, const void *item) {
  if (newnode)
    newnode->item = AVL_CONST_ITEM(item);
  return newnode;
}

/* Allocates and initializes memory for use as a node.
 * Returns the value of avlnode (or NULL if the allocation failed).
 * O(1) */
static avl_node_t *avl_alloc(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;
  avl_allocator_t *allocator = avltree ? avltree->allocator : (avl_allocator_t *) NULL;
  avl_allocate_t allocate;
  if (allocator) {
    allocate = allocator->allocate;
    if (allocator) {
      newnode = allocate(allocator);
    } else {
      errno = ENOSYS;
      newnode = NULL;
    }
  } else {
    newnode = (avl_node_t *) malloc(sizeof *newnode);
  }
  return avl_node_init(newnode, item);
}

/* Insert a node in an empty tree. If avl_node is NULL, the tree will be
 * cleared and ready for re-use.
 * If the tree is not empty, the old nodes are left dangling.
 * O(1) */
static avl_node_t *avl_insert_top(avl_tree_t *avltree, avl_node_t *newnode) {
  avl_node_clear(newnode);
  newnode->prev = newnode->next = newnode->parent = NULL;
  avltree->head = avltree->tail = avltree->top = newnode;
  return newnode;
}

/* Insert a node before another node. Returns the new node.
 * If old is NULL, the item is appended to the tree.
 * O(lg n) */
static avl_node_t *avl_node_insert_before(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode) {
  if (!avltree || !newnode)
    return NULL;

  if (!node)
    return
        avltree->tail ?
            avl_node_insert_after(avltree, avltree->tail, newnode) : avl_insert_top(avltree, newnode);

  if (node->left)
    return avl_node_insert_after(avltree, node->prev, newnode);

  avl_node_clear(newnode);

  newnode->next = node;
  newnode->parent = node;

  newnode->prev = node->prev;
  if (node->prev)
    node->prev->next = newnode;
  else
    avltree->head = newnode;
  node->prev = newnode;

  node->left = newnode;
  avl_rebalance(avltree, node);
  return newnode;
}

/* Insert a node after another node. Returns the new node.
 * If old is NULL, the item is prepended to the tree.
 * O(lg n) */
static avl_node_t *avl_node_insert_after(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode) {
  if (!avltree || !newnode)
    return NULL;

  if (!node)
    return
        avltree->head ?
            avl_node_insert_before(avltree, avltree->head, newnode) : avl_insert_top(avltree, newnode);

  if (node->right)
    return avl_node_insert_before(avltree, node->next, newnode);

  avl_node_clear(newnode);

  newnode->prev = node;
  newnode->parent = node;

  newnode->next = node->next;
  if (node->next)
    node->next->prev = newnode;
  else
    avltree->tail = newnode;
  node->next = newnode;

  node->right = newnode;
  avl_rebalance(avltree, node);
  return newnode;
}

/* Insert a node into the tree and return it.
 * Returns NULL if an equal node is already in the tree.
 * O(lg n) */
static avl_node_t *avl_node_insert(avl_tree_t *avltree, avl_node_t *newnode) {
  avl_node_t *node;
  int c;

  node = avl_search_rightish(avltree, newnode->item, &c);
  return c ? NULL : avl_node_insert_after(avltree, node, newnode);
}


#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_node_insert_left(avl_tree_t *avltree, avl_node_t *newnode) {
  return avl_node_insert_before(avltree, avl_item_search_left(avltree, newnode->item, NULL), newnode);
}
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_node_insert_right(avl_tree_t *avltree, avl_node_t *newnode) {
  return avl_node_insert_after(avltree, avl_item_search_right(avltree, newnode->item, NULL), newnode);
}
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_node_insert_somewhere(avl_tree_t *avltree, avl_node_t *newnode) {
  return avl_node_insert_after(avltree, avl_search_rightish(avltree, newnode->item, NULL), newnode);
}
#endif

/* Insert an item into the tree and return the new node.
 * Returns NULL and sets errno if memory for the new node could not be
 * allocated or if the node is already in the tree (EEXIST).
 * O(lg n) */
static avl_node_t *avl_item_insert(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode) {
    if (avl_node_insert(avltree, newnode))
      return newnode;
    avl_node_free(avltree, newnode);
    errno = EEXIST;
  }
  return NULL;
}

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_item_insert_somewhere(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_somewhere(avltree, newnode);
  return NULL;
}
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_item_insert_before(avl_tree_t *avltree, avl_node_t *node, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_before(avltree, node, newnode);
  return NULL;
}
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_item_insert_after(avl_tree_t *avltree, avl_node_t *node, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_after(avltree, node, newnode);
  return NULL;
}
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_item_insert_left(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_left(avltree, newnode);
  return NULL;
}
#endif

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_item_insert_right(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_right(avltree, newnode);
  return NULL;
}
#endif

/* Deletes a node from the tree.
 * Returns the value of the node (even if it's NULL).
 * The item will NOT be free()d regardless of the tree's free handler.
 * This function comes in handy if you need to update the search key.
 * O(lg n) */
static avl_node_t *avl_node_unlink(avl_tree_t *avltree, avl_node_t *avlnode) {
  avl_node_t *parent;
  avl_node_t **superparent;
  avl_node_t *subst, *left, *right;
  avl_node_t *balnode;

  if (!avltree || !avlnode)
    return NULL;

  if (avlnode->prev)
    avlnode->prev->next = avlnode->next;
  else
    avltree->head = avlnode->next;

  if (avlnode->next)
    avlnode->next->prev = avlnode->prev;
  else
    avltree->tail = avlnode->prev;

  parent = avlnode->parent;

  superparent = parent ? avlnode == parent->left ? &parent->left : &parent->right : &avltree->top;

  left = avlnode->left;
  right = avlnode->right;
  if (!left) {
    *superparent = right;
    if (right)
      right->parent = parent;
    balnode = parent;
  } else if (!right) {
    *superparent = left;
    left->parent = parent;
    balnode = parent;
  } else {
    subst = avlnode->prev;
    if (subst == left) {
      balnode = subst;
    } else {
      balnode = subst->parent;
      balnode->right = subst->left;
      if (balnode->right)
        balnode->right->parent = balnode;
      subst->left = left;
      left->parent = subst;
    }
    subst->right = right;
    subst->parent = parent;
    right->parent = subst;
    *superparent = subst;
  }

  avl_rebalance(avltree, balnode);

  return avlnode;
}

/* Deletes a node from the tree. Returns immediately if the node is NULL.
 * If the tree's free is not NULL, it is invoked on the item.
 * If it is, returns the item. In all other cases returns NULL.
 * O(lg n) */
static void *avl_node_delete(avl_tree_t *avltree, avl_node_t *avlnode) {
  void *item = NULL;
  if (avlnode) {
    item = avlnode->item;
    (void) avl_node_unlink(avltree, avlnode);
    if (avltree->freeitem)
      avltree->freeitem(item, avltree->userdata);
    avl_node_free(avltree, avlnode);
  }
  return item;
}

/* Searches for an item in the tree and deletes it if found.
 * If the tree's free is not NULL, it is invoked on the item.
 * If it is, returns the item. In all other cases returns NULL.
 * O(lg n) */
static void *avl_item_delete(avl_tree_t *avltree, const void *item) {
  return avl_node_delete(avltree, avl_item_search(avltree, item));
}

#if (!AVL_TREE_COMMENT_UNUSED)
static avl_node_t *avl_node_fixup(avl_tree_t *avltree, avl_node_t *newnode) {
  avl_node_t *oldnode = NULL, *node;

  if (!avltree || !newnode)
    return NULL;

  node = newnode->prev;
  if (node) {
    oldnode = node->next;
    node->next = newnode;
  } else {
    avltree->head = newnode;
  }

  node = newnode->next;
  if (node) {
    oldnode = node->prev;
    node->prev = newnode;
  } else {
    avltree->tail = newnode;
  }

  node = newnode->parent;
  if (node) {
    if (node->left == oldnode)
      node->left = newnode;
    else
      node->right = newnode;
  } else {
    oldnode = avltree->top;
    avltree->top = newnode;
  }

  return oldnode;
}
#endif

/**
 * avl_rebalance:
 * Rebalances the tree if one side becomes too heavy.  This function
 * assumes that both subtrees are AVL-trees with consistent data.  The
 * function has the additional side effect of recalculating the count of
 * the tree at this node.  It should be noted that at the return of this
 * function, if a rebalance takes place, the top of this subtree is no
 * longer going to be the same node.
 */
static void avl_rebalance(avl_tree_t *avltree, avl_node_t *avlnode) {
  avl_node_t *child;
  avl_node_t *gchild;
  avl_node_t *parent;
  avl_node_t **superparent;

  parent = avlnode;

  while (avlnode) {
    parent = avlnode->parent;

    superparent = parent ? avlnode == parent->left ? &parent->left : &parent->right : &avltree->top;

    switch (avl_check_balance(avlnode)) {
    case -1:
      child = avlnode->left;
#           ifdef AVL_DEPTH
      if (AVL_L_DEPTH(child) >= AVL_R_DEPTH(child)) {
#           else
#           ifdef AVL_COUNT
        if (AVL_L_COUNT(child) >= AVL_R_COUNT(child)) {
#           else
#           error No balancing possible.
#           endif
#           endif
        avlnode->left = child->right;
        if (avlnode->left)
          avlnode->left->parent = avlnode;
        child->right = avlnode;
        avlnode->parent = child;
        *superparent = child;
        child->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = AVL_CALC_COUNT(avlnode);
        child->count = AVL_CALC_COUNT(child);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = AVL_CALC_DEPTH(avlnode);
        child->depth = AVL_CALC_DEPTH(child);
#               endif
      } else {
        gchild = child->right;
        avlnode->left = gchild->right;
        if (avlnode->left)
          avlnode->left->parent = avlnode;
        child->right = gchild->left;
        if (child->right)
          child->right->parent = child;
        gchild->right = avlnode;
        if (gchild->right)
          gchild->right->parent = gchild;
        gchild->left = child;
        if (gchild->left)
          gchild->left->parent = gchild;
        *superparent = gchild;
        gchild->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = AVL_CALC_COUNT(avlnode);
        child->count = AVL_CALC_COUNT(child);
        gchild->count = AVL_CALC_COUNT(gchild);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = AVL_CALC_DEPTH(avlnode);
        child->depth = AVL_CALC_DEPTH(child);
        gchild->depth = AVL_CALC_DEPTH(gchild);
#               endif
      }
      break;
    case 1:
      child = avlnode->right;
#           ifdef AVL_DEPTH
      if (AVL_R_DEPTH(child) >= AVL_L_DEPTH(child)) {
#           else
#           ifdef AVL_COUNT
        if (AVL_R_COUNT(child) >= AVL_L_COUNT(child)) {
#           else
#           error No balancing possible.
#           endif
#           endif
        avlnode->right = child->left;
        if (avlnode->right)
          avlnode->right->parent = avlnode;
        child->left = avlnode;
        avlnode->parent = child;
        *superparent = child;
        child->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = AVL_CALC_COUNT(avlnode);
        child->count = AVL_CALC_COUNT(child);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = AVL_CALC_DEPTH(avlnode);
        child->depth = AVL_CALC_DEPTH(child);
#               endif
      } else {
        gchild = child->left;
        avlnode->right = gchild->left;
        if (avlnode->right)
          avlnode->right->parent = avlnode;
        child->left = gchild->right;
        if (child->left)
          child->left->parent = child;
        gchild->left = avlnode;
        if (gchild->left)
          gchild->left->parent = gchild;
        gchild->right = child;
        if (gchild->right)
          gchild->right->parent = gchild;
        *superparent = gchild;
        gchild->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = AVL_CALC_COUNT(avlnode);
        child->count = AVL_CALC_COUNT(child);
        gchild->count = AVL_CALC_COUNT(gchild);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = AVL_CALC_DEPTH(avlnode);
        child->depth = AVL_CALC_DEPTH(child);
        gchild->depth = AVL_CALC_DEPTH(gchild);
#               endif
      }
      break;
    default:
#           ifdef AVL_COUNT
      avlnode->count = AVL_CALC_COUNT(avlnode);
#           endif
#           ifdef AVL_DEPTH
      avlnode->depth = AVL_CALC_DEPTH(avlnode);
#           endif
    }
    avlnode = parent;
  }
}
#line 39 "code-experiments/src/logger_biobj.c"
#line 1 "code-experiments/src/observer_biobj.c"
/**
 * @file observer_biobj.c
 * @brief Implementation of the bbob-biobj observer.
 */

#line 7 "code-experiments/src/observer_biobj.c"
#line 8 "code-experiments/src/observer_biobj.c"

#line 10 "code-experiments/src/observer_biobj.c"
#line 11 "code-experiments/src/observer_biobj.c"

/** @brief Enum for denoting the way in which the nondominated solutions are logged. */
typedef enum {
  LOG_NONDOM_NONE, LOG_NONDOM_FINAL, LOG_NONDOM_ALL
} observer_biobj_log_nondom_e;

/** @brief Enum for denoting the when the decision variables are logged. */
typedef enum {
  LOG_VARS_NEVER, LOG_VARS_LOW_DIM, LOG_VARS_ALWAYS
} observer_biobj_log_vars_e;

/**
 * @brief The bbob-biobj observer data type.
 */
typedef struct {
  observer_biobj_log_nondom_e log_nondom_mode; /**< @brief How the nondominated solutions are logged. */
  observer_biobj_log_vars_e log_vars_mode;     /**< @brief When the decision variables are logged. */

  int compute_indicators;                      /**< @brief Whether to compute indicators. */
  int produce_all_data;                        /**< @brief Whether to produce all data. */

  long previous_function;                      /**< @brief Information on the previous logged problem. */

} observer_biobj_data_t;

static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *problem);
static void logger_biobj_free(void *logger);

/**
 * @brief Initializes the bi-objective observer.
 *
 * Possible options:
 *
 * - "log_nondominated: STRING" determines which nondominated solutions to log. STRING can take on the
 * values "none" (don't log nondominated solutions), "final" (log only the final nondominated solutions) and
 * "all" (log every solution that is nondominated at creation time). The default value is "all".
 *
 * - "log_decision_variables: STRING" determines whether the decision variables are to be logged in addition
 * to the objective variables in the output of nondominated solutions. STRING can take on the values "none"
 * (don't output decision variables), "low_dim"(output decision variables only for dimensions lower or equal
 * to 5) and "all" (output all decision variables). The default value is "low_dim".
 *
 * - "compute_indicators: VALUE" determines whether to compute and output performance indicators (1) or not
 * (0). The default value is 1.
 *
 * - "produce_all_data: VALUE" determines whether to produce all data required for the workshop. If set to 1,
 * it overwrites some other options and is equivalent to setting "log_nondominated: all",
 * "log_decision_variables: low_dim" and "compute_indicators: 1". If set to 0, it does not change the values
 * of the other options. The default value is 0.
 */
static void observer_biobj(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_biobj_data_t *observer_biobj;
  char string_value[COCO_PATH_MAX];

  /* Sets the valid keys for bbob-biobj observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "log_nondominated", "log_decision_variables", "compute_indicators",
      "produce_all_data" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_biobj = (observer_biobj_data_t *) coco_allocate_memory(sizeof(*observer_biobj));

  observer_biobj->log_nondom_mode = LOG_NONDOM_ALL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_biobj->log_nondom_mode = LOG_NONDOM_NONE;
    else if (strcmp(string_value, "final") == 0)
      observer_biobj->log_nondom_mode = LOG_NONDOM_FINAL;
    else if (strcmp(string_value, "all") == 0)
      observer_biobj->log_nondom_mode = LOG_NONDOM_ALL;
  }

  observer_biobj->log_vars_mode = LOG_VARS_LOW_DIM;
  if (coco_options_read_string(options, "log_decision_variables", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_biobj->log_vars_mode = LOG_VARS_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_biobj->log_vars_mode = LOG_VARS_ALWAYS;
    else if (strcmp(string_value, "low_dim") == 0)
      observer_biobj->log_vars_mode = LOG_VARS_LOW_DIM;
  }

  if (coco_options_read_int(options, "compute_indicators", &(observer_biobj->compute_indicators)) == 0)
    observer_biobj->compute_indicators = 1;

  if (coco_options_read_int(options, "produce_all_data", &(observer_biobj->produce_all_data)) == 0)
    observer_biobj->produce_all_data = 0;

  if (observer_biobj->produce_all_data) {
    observer_biobj->compute_indicators = 1;
    observer_biobj->log_nondom_mode = LOG_NONDOM_ALL;
    observer_biobj->log_vars_mode = LOG_VARS_LOW_DIM;
  }

  if (observer_biobj->compute_indicators) {
    observer_biobj->previous_function = -1;
  }

  observer->logger_allocate_function = logger_biobj;
  observer->logger_free_function = logger_biobj_free;
  observer->data_free_function = NULL;
  observer->data = observer_biobj;

  if ((observer_biobj->log_nondom_mode == LOG_NONDOM_NONE) && (!observer_biobj->compute_indicators)) {
    /* No logging required */
    observer->is_active = 0;
  }
}
#line 40 "code-experiments/src/logger_biobj.c"

#line 42 "code-experiments/src/logger_biobj.c"

/** @brief Number of implemented indicators */
#define LOGGER_BIOBJ_NUMBER_OF_INDICATORS 1

/** @brief Names of implemented indicators
 *
 * "hyp" stands for the hypervolume indicator.
 * */
const char *logger_biobj_indicators[LOGGER_BIOBJ_NUMBER_OF_INDICATORS] = { "hyp" };

/**
 * @brief The indicator type.
 *
 * <B> The hypervolume indicator ("hyp") </B>
 *
 * The hypervolume indicator measures the volume of the portion of the ROI in the objective space that is
 * dominated by the current Pareto front approximation. Instead of logging the hypervolume indicator value,
 * this implementation logs the difference between the best know hypervolume indicator (a value stored in
 * best_value) and the hypervolume indicator of the current Pareto front approximation (current_value). The
 * current_value equals 0 if no solution is located in the ROI. In order to be able to register the
 * performance of an optimizer even before the ROI is reached, an additional value is computed when no
 * solutions are located inside the ROI. This value is stored in additional_penalty and equals the
 * normalized distance to the ROI of the solution closest to the ROI (additional_penalty is set to 0 as
 * soon as a solution reaches the ROI). The final value to be logged (overall_value) is therefore computed
 * in the following way:
 *
 * overall_value = best_value - current_value + additional_penalty
 *
 * @note Other indicators are yet to be implemented.
 */
typedef struct {

  char *name;                /**< @brief Name of the indicator used for identification and the output. */

  FILE *dat_file;            /**< @brief File for logging indicator values at predefined values. */
  FILE *tdat_file;           /**< @brief File for logging indicator values at predefined evaluations. */
  FILE *info_file;           /**< @brief File for logging summary information on algorithm performance. */

  int target_hit;            /**< @brief Whether the target was hit in the latest evaluation. */
  coco_observer_targets_t *targets;
                             /**< @brief Triggers based on target values. */
  int evaluation_logged;     /**< @brief Whether the whether the latest evaluation was logged. */
  coco_observer_evaluations_t *evaluations;
                             /**< @brief Triggers based on numbers of evaluations. */

  double best_value;         /**< @brief The best known indicator value for this problem. */
  double current_value;      /**< @brief The current indicator value. */
  double additional_penalty; /**< @brief Additional penalty for solutions outside the ROI. */
  double overall_value;      /**< @brief The overall value of the indicator tested for target hits. */

} logger_biobj_indicator_t;

/**
 * @brief The bi-objective logger data type.
 *
 * @note Some fields from the observers (coco_observer as well as observer_biobj) need to be copied here
 * because the observers can be deleted before the logger is finalized and we need these fields for
 * finalization.
 */
typedef struct {
  observer_biobj_log_nondom_e log_nondom_mode;
                                 /**< @brief Mode for archiving nondominated solutions. */
  FILE *adat_file;               /**< @brief File for archiving nondominated solutions (all or final). */

  int log_vars;                  /**< @brief Whether to log the decision values. */

  int precision_x;               /**< @brief Precision for outputting decision values. */
  int precision_f;               /**< @brief Precision for outputting objective values. */

  size_t number_of_evaluations;  /**< @brief The number of evaluations performed so far. */
  size_t number_of_variables;    /**< @brief Dimension of the problem. */
  size_t number_of_objectives;   /**< @brief Number of objectives (clearly equal to 2). */
  size_t suite_dep_instance;     /**< @brief Suite-dependent instance number of the observed problem. */

  avl_tree_t *archive_tree;      /**< @brief The tree keeping currently non-dominated solutions. */
  avl_tree_t *buffer_tree;       /**< @brief The tree with pointers to nondominated solutions that haven't
                                      been logged yet. */

  /* Indicators (TODO: Implement others!) */
  int compute_indicators;        /**< @brief Whether to compute the indicators. */
  logger_biobj_indicator_t *indicators[LOGGER_BIOBJ_NUMBER_OF_INDICATORS];
                                 /**< @brief The implemented indicators. */
} logger_biobj_data_t;

/**
 * @brief The type for the node's item in the AVL tree as used by the bi-objective logger.
 */
typedef struct {
  double *x;                 /**< @brief The decision values of this solution. */
  double *y;                 /**< @brief The values of objectives of this solution. */
  size_t evaluation_number;  /**< @brief The evaluation number of when the solution was created. */

  double indicator_contribution[LOGGER_BIOBJ_NUMBER_OF_INDICATORS];
                      /**< @brief The contribution of this solution to the overall indicator values. */
  int within_ROI;     /**< @brief Whether the solution is within the region of interest (ROI). */

} logger_biobj_avl_item_t;

/**
 * @brief Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static logger_biobj_avl_item_t* logger_biobj_node_create(const double *x,
                                                         const double *y,
                                                         const size_t evaluation_number,
                                                         const size_t dim,
                                                         const size_t num_obj) {

  size_t i;

  /* Allocate memory to hold the data structure logger_biobj_node_t */
  logger_biobj_avl_item_t *item = (logger_biobj_avl_item_t*) coco_allocate_memory(sizeof(*item));

  /* Allocate memory to store the (copied) data of the new node */
  item->x = coco_allocate_vector(dim);
  item->y = coco_allocate_vector(num_obj);

  /* Copy the data */
  for (i = 0; i < dim; i++)
    item->x[i] = x[i];
  for (i = 0; i < num_obj; i++)
    item->y[i] = y[i];
  item->evaluation_number = evaluation_number;
  for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++)
    item->indicator_contribution[i] = 0;
  item->within_ROI = 0;
  return item;
}

/**
 * @brief Frees the data of the given logger_biobj_avl_item_t.
 */
static void logger_biobj_node_free(logger_biobj_avl_item_t *item, void *userdata) {

  coco_free_memory(item->x);
  coco_free_memory(item->y);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Checks if the given node item is within the ROI and stores this information in its
 * within_ROI field.
 */
static void logger_biobj_check_within_ROI(const coco_problem_t *problem,
                                          logger_biobj_avl_item_t *node_item) {

  node_item->within_ROI = mo_solution_is_within_ROI(node_item->y, problem->best_value, problem->nadir_value,
      problem->number_of_objectives);
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the value of the last objective.
 *
 * @note This ordering is used by the archive_tree.
 */
static int avl_tree_compare_by_last_objective(const logger_biobj_avl_item_t *item1,
                                              const logger_biobj_avl_item_t *item2,
                                              void *userdata) {
  if (item1->y[1] < item2->y[1])
    return -1;
  else if (item1->y[1] > item2->y[1])
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the evaluation number (the time when the nodes were
 * created).
 *
 * @note This ordering is used by the buffer_tree.
 */
static int avl_tree_compare_by_eval_number(const logger_biobj_avl_item_t *item1,
                                           const logger_biobj_avl_item_t *item2,
                                           void *userdata) {
  if (item1->evaluation_number < item2->evaluation_number)
    return -1;
  else if (item1->evaluation_number > item2->evaluation_number)
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Outputs the AVL tree to the given file. Returns the number of nodes in the tree.
 */
static size_t logger_biobj_tree_output(FILE *file,
                                       const avl_tree_t *tree,
                                       const size_t dim,
                                       const size_t num_obj,
                                       const int log_vars,
                                       const int precision_x,
                                       const int precision_f) {

  avl_node_t *solution;
  size_t i;
  size_t j;
  size_t number_of_nodes = 0;

  if (tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = tree->head;
    while (solution != NULL) {
      fprintf(file, "%lu\t", ((logger_biobj_avl_item_t*) solution->item)->evaluation_number);
      for (j = 0; j < num_obj; j++)
        fprintf(file, "%.*e\t", precision_f, ((logger_biobj_avl_item_t*) solution->item)->y[j]);
      if (log_vars) {
        for (i = 0; i < dim; i++)
          fprintf(file, "%.*e\t", precision_x, ((logger_biobj_avl_item_t*) solution->item)->x[i]);
      }
      fprintf(file, "\n");
      solution = solution->next;
      number_of_nodes++;
    }
  }

  return number_of_nodes;
}

/**
 * @brief Updates the archive and buffer trees with the given node.
 *
 * Checks for domination and updates the archive tree and the values of the indicators if the given node is
 * not weakly dominated by existing nodes in the archive tree. This is where the main computation of
 * indicator values takes place.
 *
 * @return 1 if the update was performed and 0 otherwise.
 */
static int logger_biobj_tree_update(logger_biobj_data_t *logger,
                                    const coco_problem_t *problem,
                                    logger_biobj_avl_item_t *node_item) {

  avl_node_t *node, *next_node, *new_node;
  int trigger_update = 0;
  int dominance;
  size_t i;
  int previous_unavailable = 0;

  /* Find the first point that is not worse than the new point (NULL if such point does not exist) */
  node = avl_item_search_right(logger->archive_tree, node_item, NULL);

  if (node == NULL) {
    /* The new point is an extremal point */
    trigger_update = 1;
    next_node = logger->archive_tree->head;
  } else {
    dominance = mo_get_dominance(node_item->y, ((logger_biobj_avl_item_t*) node->item)->y,
        logger->number_of_objectives);
    if (dominance > -1) {
      trigger_update = 1;
      next_node = node->next;
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        if (logger->compute_indicators) {
          for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            logger->indicators[i]->current_value -= ((logger_biobj_avl_item_t*) node->item)->indicator_contribution[i];
          }
        }
        avl_item_delete(logger->buffer_tree, node->item);
        avl_node_delete(logger->archive_tree, node);
      }
    } else {
      /* The new point is dominated, nothing more to do */
      trigger_update = 0;
    }
  }

  if (!trigger_update) {
    logger_biobj_node_free(node_item, NULL);
  } else {
    /* Perform tree update */
    while (next_node != NULL) {
      /* Check the dominance relation between the new node and the next node. There are only two possibilities:
       * dominance = 0: the new node and the next node are nondominated
       * dominance = 1: the new node dominates the next node */
      node = next_node;
      dominance = mo_get_dominance(node_item->y, ((logger_biobj_avl_item_t*) node->item)->y,
          logger->number_of_objectives);
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        if (logger->compute_indicators) {
          for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            logger->indicators[i]->current_value -= ((logger_biobj_avl_item_t*) node->item)->indicator_contribution[i];
          }
        }
        next_node = node->next;
        avl_item_delete(logger->buffer_tree, node->item);
        avl_node_delete(logger->archive_tree, node);
      } else {
        break;
      }
    }

    new_node = avl_item_insert(logger->archive_tree, node_item);
    avl_item_insert(logger->buffer_tree, node_item);

    if (logger->compute_indicators) {
      logger_biobj_check_within_ROI(problem, node_item);
      if (node_item->within_ROI) {
        /* Compute indicator value for new node and update the indicator value of the affected nodes */
        logger_biobj_avl_item_t *next_item, *previous_item;

        if (new_node->next != NULL) {
          next_item = (logger_biobj_avl_item_t*) new_node->next->item;
          if (next_item->within_ROI) {
            for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              logger->indicators[i]->current_value -= next_item->indicator_contribution[i];
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                next_item->indicator_contribution[i] = (node_item->y[0] - next_item->y[0])
                    / (problem->nadir_value[0] - problem->best_value[0])
                    * (problem->nadir_value[1] - next_item->y[1])
                    / (problem->nadir_value[1] - problem->best_value[1]);
              } else {
                coco_error(
                    "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                    logger->indicators[i]->name);
              }
              logger->indicators[i]->current_value += next_item->indicator_contribution[i];
            }
          }
        }

        previous_unavailable = 0;
        if (new_node->prev != NULL) {
          previous_item = (logger_biobj_avl_item_t*) new_node->prev->item;
          if (previous_item->within_ROI) {
            for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                node_item->indicator_contribution[i] = (previous_item->y[0] - node_item->y[0])
                    / (problem->nadir_value[0] - problem->best_value[0])
                    * (problem->nadir_value[1] - node_item->y[1])
                    / (problem->nadir_value[1] - problem->best_value[1]);
              } else {
                coco_error(
                    "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                    logger->indicators[i]->name);
              }
            }
          } else {
            previous_unavailable = 1;
          }
        } else {
          previous_unavailable = 1;
        }

        if (previous_unavailable) {
          /* Previous item does not exist or is out of ROI, use reference point instead */
          for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
              node_item->indicator_contribution[i] = (problem->nadir_value[0] - node_item->y[0])
                  / (problem->nadir_value[0] - problem->best_value[0])
                  * (problem->nadir_value[1] - node_item->y[1])
                  / (problem->nadir_value[1] - problem->best_value[1]);
            } else {
              coco_error(
                  "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                  logger->indicators[i]->name);
            }
          }
        }

        for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
          logger->indicators[i]->current_value += node_item->indicator_contribution[i];
        }
      }
    }
  }

  return trigger_update;
}

/**
 * @brief Initializes the indicator with name indicator_name.
 *
 * Opens files for writing and resets counters.
 */
static logger_biobj_indicator_t *logger_biobj_indicator(const logger_biobj_data_t *logger,
                                                        const coco_observer_t *observer,
                                                        const coco_problem_t *problem,
                                                        const char *indicator_name) {

  observer_biobj_data_t *observer_biobj;
  logger_biobj_indicator_t *indicator;
  char *prefix, *file_name, *path_name;
  int info_file_exists = 0;

  indicator = (logger_biobj_indicator_t *) coco_allocate_memory(sizeof(*indicator));
  assert(observer);
  assert(observer->data);
  observer_biobj = (observer_biobj_data_t *) observer->data;

  indicator->name = coco_strdup(indicator_name);

  indicator->best_value = suite_biobj_get_best_value(indicator->name, problem->problem_id);
  indicator->target_hit = 0;
  indicator->evaluation_logged = 0;
  indicator->current_value = 0;
  indicator->additional_penalty = DBL_MAX;
  indicator->overall_value = 0;

  indicator->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  indicator->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, problem->number_of_variables);

  /* Prepare the info file */
  path_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_create_directory(path_name);
  file_name = coco_strdupf("%s_%s.info", problem->problem_type, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  info_file_exists = coco_file_exists(path_name);
  indicator->info_file = fopen(path_name, "a");
  if (indicator->info_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Prepare the tdat file */
  path_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_join_path(path_name, COCO_PATH_MAX, problem->problem_type, NULL);
  coco_create_directory(path_name);
  prefix = coco_remove_from_string(problem->problem_id, "_i", "_d");
  file_name = coco_strdupf("%s_%s.tdat", prefix, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  indicator->tdat_file = fopen(path_name, "a");
  if (indicator->tdat_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Prepare the dat file */
  path_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_join_path(path_name, COCO_PATH_MAX, problem->problem_type, NULL);
  coco_create_directory(path_name);
  file_name = coco_strdupf("%s_%s.dat", prefix, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  indicator->dat_file = fopen(path_name, "a");
  if (indicator->dat_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }

  /* Output header information to the info file */
  if (!info_file_exists) {
    /* Output algorithm name */
    fprintf(indicator->info_file, "algorithm = '%s', indicator = '%s', folder = '%s'\n%% %s", observer->algorithm_name,
        indicator_name, problem->problem_type, observer->algorithm_info);
  }
  if (observer_biobj->previous_function != problem->suite_dep_function) {
    fprintf(indicator->info_file, "\nfunction = %2lu, ", problem->suite_dep_function);
    fprintf(indicator->info_file, "dim = %2lu, ", problem->number_of_variables);
    fprintf(indicator->info_file, "%s", file_name);
  }

  coco_free_memory(prefix);
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Output header information to the dat file */
  fprintf(indicator->dat_file, "%%\n%% index = %ld, name = %s\n", problem->suite_dep_index, problem->problem_name);
  fprintf(indicator->dat_file, "%% instance = %ld, reference value = %.*e\n", problem->suite_dep_instance,
      logger->precision_f, indicator->best_value);
  fprintf(indicator->dat_file, "%% function evaluation | indicator value | target hit\n");

  /* Output header information to the tdat file */
  fprintf(indicator->tdat_file, "%%\n%% index = %ld, name = %s\n", problem->suite_dep_index, problem->problem_name);
  fprintf(indicator->tdat_file, "%% instance = %ld, reference value = %.*e\n", problem->suite_dep_instance,
      logger->precision_f, indicator->best_value);
  fprintf(indicator->tdat_file, "%% function eval_number | indicator value\n");

  return indicator;
}

/**
 * @brief Outputs the final information about this indicator.
 */
static void logger_biobj_indicator_finalize(logger_biobj_indicator_t *indicator, const logger_biobj_data_t *logger) {

  /* Log the last eval_number in the dat file if wasn't already logged */
  if (!indicator->target_hit) {
    fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
        indicator->overall_value, logger->precision_f, ((coco_observer_targets_t *) indicator->targets)->value);
  }

  /* Log the last eval_number in the tdat file if wasn't already logged */
  if (!indicator->evaluation_logged) {
    fprintf(indicator->tdat_file, "%lu\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
        indicator->overall_value);
  }

  /* Log the information in the info file */
  fprintf(indicator->info_file, ", %ld:%lu|%.1e", logger->suite_dep_instance, logger->number_of_evaluations,
      indicator->overall_value);
  fflush(indicator->info_file);
}

/**
 * @brief Frees the memory of the given indicator.
 */
static void logger_biobj_indicator_free(void *stuff) {

  logger_biobj_indicator_t *indicator;

  assert(stuff != NULL);
  indicator = (logger_biobj_indicator_t *) stuff;

  if (indicator->name != NULL) {
    coco_free_memory(indicator->name);
    indicator->name = NULL;
  }

  if (indicator->dat_file != NULL) {
    fclose(indicator->dat_file);
    indicator->dat_file = NULL;
  }

  if (indicator->tdat_file != NULL) {
    fclose(indicator->tdat_file);
    indicator->tdat_file = NULL;
  }

  if (indicator->info_file != NULL) {
    fclose(indicator->info_file);
    indicator->info_file = NULL;
  }

  if (indicator->targets != NULL){
    coco_free_memory(indicator->targets);
    indicator->targets = NULL;
  }

  if (indicator->evaluations != NULL){
    coco_observer_evaluations_free(indicator->evaluations);
    indicator->evaluations = NULL;
  }

  coco_free_memory(stuff);

}

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information based on the
 * observer options.
 */
static void logger_biobj_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_biobj_data_t *logger;

  logger_biobj_avl_item_t *node_item;
  logger_biobj_indicator_t *indicator;
  int update_performed;
  size_t i;

  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);

  /* Evaluate function */
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  logger->number_of_evaluations++;

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in
   * the archive */
  node_item = logger_biobj_node_create(x, y, logger->number_of_evaluations, logger->number_of_variables,
      logger->number_of_objectives);

  update_performed = logger_biobj_tree_update(logger, coco_problem_transformed_get_inner_problem(problem),
      node_item);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to
   * nondom_file */
  if (update_performed && (logger->log_nondom_mode == LOG_NONDOM_ALL)) {
    logger_biobj_tree_output(logger->adat_file, logger->buffer_tree, logger->number_of_variables,
        logger->number_of_objectives, logger->log_vars, logger->precision_x, logger->precision_f);
    avl_tree_purge(logger->buffer_tree);

    /* Flush output so that impatient users can see progress. */
    fflush(logger->adat_file);
  }

  /* Perform output to the:
   * - dat file, if the archive was updated and a new target was reached for an indicator;
   * - tdat file, if the number of evaluations matches one of the predefined numbers.
   *
   * Note that a target is reached when
   * best_value - current_value + additional_penalty <= relative_target_value
   *
   * The relative_target_value is a target for indicator difference, not the actual indicator value!
   */
  if (logger->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {

      indicator = logger->indicators[i];
      indicator->target_hit = 0;

      /* If the update was performed, update the overall indicator value */
      if (update_performed) {
        /* Compute the overall_value of an indicator */
        if (strcmp(indicator->name, "hyp") == 0) {
          if (indicator->current_value == 0) {
            /* Update the additional penalty for hypervolume (the minimal distance from the nondominated set
             * to the ROI) */
            double new_distance = mo_get_distance_to_ROI(node_item->y,
                problem->best_value, problem->nadir_value, problem->number_of_objectives);
            indicator->additional_penalty = coco_double_min(indicator->additional_penalty, new_distance);
            assert(indicator->additional_penalty >= 0);
          } else {
            indicator->additional_penalty = 0;
          }
          indicator->overall_value = indicator->best_value - indicator->current_value
              + indicator->additional_penalty;
        } else {
          coco_error("logger_biobj_evaluate(): Indicator computation not implemented yet for indicator %s",
              indicator->name);
        }

        /* Check whether a target was hit */
        indicator->target_hit = coco_observer_targets_trigger(indicator->targets, indicator->overall_value);
      }

      /* Log to the dat file if a target was hit */
      if (indicator->target_hit) {
        fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
            indicator->overall_value, logger->precision_f,
            ((coco_observer_targets_t *) indicator->targets)->value);
      }

      /* Log to the tdat file if the number of evaluations matches one of the predefined numbers */
      indicator->evaluation_logged = coco_observer_evaluations_trigger(indicator->evaluations,
          logger->number_of_evaluations);
      if (indicator->evaluation_logged) {
        fprintf(indicator->tdat_file, "%lu\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
            indicator->overall_value);
      }

    }
  }
}

/**
 * @brief Outputs the final nondominated solutions to the archive file.
 */
static void logger_biobj_finalize(logger_biobj_data_t *logger) {

  avl_tree_t *resorted_tree;
  avl_node_t *solution;

  /* Re-sort archive_tree according to time stamp and then output it */
  resorted_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_eval_number, NULL);

  if (logger->archive_tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = logger->archive_tree->head;
    while (solution != NULL) {
      avl_item_insert(resorted_tree, solution->item);
      solution = solution->next;
    }
  }

  logger_biobj_tree_output(logger->adat_file, resorted_tree, logger->number_of_variables,
      logger->number_of_objectives, logger->log_vars, logger->precision_x, logger->precision_f);

  avl_tree_destruct(resorted_tree);
}

/**
 * @brief Frees the memory of the given biobjective logger.
 */
static void logger_biobj_free(void *stuff) {

  logger_biobj_data_t *logger;
  size_t i;

  assert(stuff != NULL);
  logger = (logger_biobj_data_t *) stuff;

  if (logger->log_nondom_mode == LOG_NONDOM_FINAL) {
     logger_biobj_finalize(logger);
  }

  if (logger->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
      logger_biobj_indicator_finalize(logger->indicators[i], logger);
      logger_biobj_indicator_free(logger->indicators[i]);
    }
  }

  if ((logger->log_nondom_mode != LOG_NONDOM_NONE) && (logger->adat_file != NULL)) {
    fclose(logger->adat_file);
    logger->adat_file = NULL;
  }

  avl_tree_destruct(logger->archive_tree);
  avl_tree_destruct(logger->buffer_tree);

}

/**
 * @brief Initializes the biobjective logger.
 *
 * Copies all observer field values that are needed after initialization into logger field values for two
 * reasons:
 * - If the observer is deleted before the suite, the observer is not available anymore when the logger
 * is finalized.
 * - This reduces function calls.
 */
static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *inner_problem) {

  coco_problem_t *problem;
  logger_biobj_data_t *logger_biobj;
  observer_biobj_data_t *observer_biobj;
  const char nondom_folder_name[] = "archive";
  char *path_name, *file_name = NULL, *prefix;
  size_t i;

  if (inner_problem->number_of_objectives != 2) {
    coco_error("logger_biobj(): The bi-objective logger cannot log a problem with %d objective(s)",
        inner_problem->number_of_objectives);
    return NULL; /* Never reached. */
  }

  logger_biobj = (logger_biobj_data_t *) coco_allocate_memory(sizeof(*logger_biobj));

  logger_biobj->number_of_evaluations = 0;
  logger_biobj->number_of_variables = inner_problem->number_of_variables;
  logger_biobj->number_of_objectives = inner_problem->number_of_objectives;
  logger_biobj->suite_dep_instance = inner_problem->suite_dep_instance;

  observer_biobj = (observer_biobj_data_t *) observer->data;
  /* Copy values from the observes that you might need even if they do not exist any more */
  logger_biobj->log_nondom_mode = observer_biobj->log_nondom_mode;
  logger_biobj->compute_indicators = observer_biobj->compute_indicators;
  logger_biobj->precision_x = observer->precision_x;
  logger_biobj->precision_f = observer->precision_f;

  if (((observer_biobj->log_vars_mode == LOG_VARS_LOW_DIM) && (inner_problem->number_of_variables > 5))
      || (observer_biobj->log_vars_mode == LOG_VARS_NEVER))
    logger_biobj->log_vars = 0;
  else
    logger_biobj->log_vars = 1;

  /* Initialize logging of nondominated solutions into the archive file */
  if (logger_biobj->log_nondom_mode != LOG_NONDOM_NONE) {

    /* Create the path to the file */
    path_name = coco_allocate_string(COCO_PATH_MAX);
    memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
    coco_join_path(path_name, COCO_PATH_MAX, nondom_folder_name, NULL);
    coco_create_directory(path_name);

    /* Construct file name */
    prefix = coco_remove_from_string(inner_problem->problem_id, "_i", "_d");
    if (logger_biobj->log_nondom_mode == LOG_NONDOM_ALL)
      file_name = coco_strdupf("%s_nondom_all.adat", prefix);
    else if (logger_biobj->log_nondom_mode == LOG_NONDOM_FINAL)
      file_name = coco_strdupf("%s_nondom_final.adat", prefix);
    coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
    if (logger_biobj->log_nondom_mode != LOG_NONDOM_NONE)
      coco_free_memory(file_name);
    coco_free_memory(prefix);

    /* Open and initialize the archive file */
    logger_biobj->adat_file = fopen(path_name, "a");
    if (logger_biobj->adat_file == NULL) {
      coco_error("logger_biobj() failed to open file '%s'.", path_name);
      return NULL; /* Never reached */
    }
    coco_free_memory(path_name);

    /* Output header information */
    fprintf(logger_biobj->adat_file, "%% instance = %ld, name = %s\n", inner_problem->suite_dep_instance,
        inner_problem->problem_name);
    if (logger_biobj->log_vars) {
      fprintf(logger_biobj->adat_file, "%% function eval_number | %lu objectives | %lu variables\n",
          inner_problem->number_of_objectives, inner_problem->number_of_variables);
    } else {
      fprintf(logger_biobj->adat_file, "%% function eval_number | %lu objectives \n",
          inner_problem->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  logger_biobj->archive_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_last_objective,
      (avl_free_t) logger_biobj_node_free);
  logger_biobj->buffer_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_eval_number, NULL);

  /* Initialize the indicators */
  if (logger_biobj->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++)
      logger_biobj->indicators[i] = logger_biobj_indicator(logger_biobj, observer, inner_problem, logger_biobj_indicators[i]);

    observer_biobj->previous_function = (long) inner_problem->suite_dep_function;
  }

  problem = coco_problem_transformed_allocate(inner_problem, logger_biobj, logger_biobj_free, observer->observer_name);
  problem->evaluate_function = logger_biobj_evaluate;

  return problem;
}
#line 362 "code-experiments/src/coco_observer.c"
#line 1 "code-experiments/src/logger_toy.c"
/**
 * @file logger_toy.c
 * @brief Implementation of the toy logger.
 *
 * Logs the evaluation number, function value the target hit and all the variables each time a target has
 * been hit.
 */

#include <stdio.h>
#include <assert.h>

#line 13 "code-experiments/src/logger_toy.c"

#line 15 "code-experiments/src/logger_toy.c"
#line 16 "code-experiments/src/logger_toy.c"
#line 17 "code-experiments/src/logger_toy.c"
#line 1 "code-experiments/src/observer_toy.c"
/**
 * @file observer_toy.c
 * @brief Implementation of the toy observer.
 */

#line 7 "code-experiments/src/observer_toy.c"
#line 8 "code-experiments/src/observer_toy.c"

static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *problem);
static void logger_toy_free(void *logger);

/**
 * @brief The toy observer data type.
 */
typedef struct {
  FILE *log_file;            /**< @brief File used for logging. */
} observer_toy_data_t;

/**
 * @brief Frees memory of the toy observer data structure.
 */
static void observer_toy_free(void *stuff) {

  observer_toy_data_t *data;

  assert(stuff != NULL);
  data = (observer_toy_data_t *) stuff;

  if (data->log_file != NULL) {
    fclose(data->log_file);
    data->log_file = NULL;
  }

}

/**
 * @brief Initializes the toy observer.
 *
 * Possible options:
 * - file_name: string (name of the output file; default value is "first_hitting_times.dat")
 */
static void observer_toy(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_toy_data_t *observer_toy;
  char *string_value;
  char *file_name;

  /* Sets the valid keys for toy observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "file_name" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_toy = (observer_toy_data_t *) coco_allocate_memory(sizeof(*observer_toy));

  /* Read file_name and number_of_targets from the options and use them to initialize the observer */
  string_value = coco_allocate_string(COCO_PATH_MAX);
  if (coco_options_read_string(options, "file_name", string_value) == 0) {
    strcpy(string_value, "first_hitting_times.dat");
  }

  /* Open log_file */
  file_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(file_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_create_directory(file_name);
  coco_join_path(file_name, COCO_PATH_MAX, string_value, NULL);

  observer_toy->log_file = fopen(file_name, "a");
  if (observer_toy->log_file == NULL) {
    coco_error("observer_toy(): failed to open file %s.", file_name);
    return; /* Never reached */
  }

  coco_free_memory(string_value);
  coco_free_memory(file_name);

  observer->logger_allocate_function = logger_toy;
  observer->logger_free_function = logger_toy_free;
  observer->data_free_function = observer_toy_free;
  observer->data = observer_toy;
}
#line 18 "code-experiments/src/logger_toy.c"

/**
 * @brief The toy logger data type.
 */
typedef struct {
  FILE *log_file;                    /**< @brief Pointer to the file already prepared for logging. */
  coco_observer_targets_t *targets;  /**< @brief Triggers based on target values. */
  size_t number_of_evaluations;      /**< @brief The number of evaluations performed so far. */
  int precision_x;                   /**< @brief Precision for outputting decision values. */
  int precision_f;                   /**< @brief Precision for outputting objective values. */
} logger_toy_data_t;

/**
 * @brief Frees the memory of the given toy logger.
 */
static void logger_toy_free(void *stuff) {

  logger_toy_data_t *logger;

  assert(stuff != NULL);
  logger = (logger_toy_data_t *) stuff;

  if (logger->targets != NULL){
    coco_free_memory(logger->targets);
    logger->targets = NULL;
  }

}

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information based on the
 * targets that have been hit.
 */
static void logger_toy_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_toy_data_t *logger = (logger_toy_data_t *) coco_problem_transformed_get_data(problem);
  size_t i;

  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  logger->number_of_evaluations++;

  /* Output the solution when a new target that has been hit */
  if (coco_observer_targets_trigger(logger->targets, y[0])) {
    fprintf(logger->log_file, "%lu\t%.*e\t%.*e", logger->number_of_evaluations, logger->precision_f, y[0],
        logger->precision_f, logger->targets->value);
    for (i = 0; i < problem->number_of_variables; i++) {
      fprintf(logger->log_file, "\t%.*e", logger->precision_x, x[i]);
    }
    fprintf(logger->log_file, "\n");
  }

  /* Flush output so that impatient users can see the progress */
  fflush(logger->log_file);
}

/**
 * @brief Initializes the toy logger.
 */
static coco_problem_t *logger_toy(coco_observer_t *observer, coco_problem_t *inner_problem) {

  logger_toy_data_t *logger_toy;
  coco_problem_t *problem;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_toy(): The toy logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }

  /* Initialize the logger_toy_data_t object instance */
  logger_toy = (logger_toy_data_t *) coco_allocate_memory(sizeof(*logger_toy));
  logger_toy->number_of_evaluations = 0;
  logger_toy->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  logger_toy->log_file = ((observer_toy_data_t *) observer->data)->log_file;
  logger_toy->precision_x = observer->precision_x;
  logger_toy->precision_f = observer->precision_f;

  problem = coco_problem_transformed_allocate(inner_problem, logger_toy, logger_toy_free, observer->observer_name);
  problem->evaluate_function = logger_toy_evaluate;

  /* Output initial information */
  fprintf(logger_toy->log_file, "\n");
  fprintf(logger_toy->log_file, "%% problem_id = %s, problem_name = %s\n", coco_problem_get_id(inner_problem),
      coco_problem_get_name(inner_problem));
  fprintf(logger_toy->log_file, "%% evaluation number | function value | target hit | %lu variables \n",
      inner_problem->number_of_variables);

  return problem;
}
#line 363 "code-experiments/src/coco_observer.c"

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
  } else if (0 == strcmp(observer_name, "bbob")) {
    observer_bbob(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "bbob-biobj")) {
    observer_biobj(observer, observer_options, &additional_option_keys);
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
#line 1 "code-experiments/src/coco_archive.c"
/**
 * @file coco_acrhive.c
 * @brief Definitions of functions regarding COCO archives.
 *
 * COCO archives are used to do some pre-processing on the bi-objective archive files. Namely, through a
 * wrapper written in Python, these functions are used to merge archives and compute their hypervolumes.
 */

#line 10 "code-experiments/src/coco_archive.c"
#line 11 "code-experiments/src/coco_archive.c"
#line 12 "code-experiments/src/coco_archive.c"
#line 13 "code-experiments/src/coco_archive.c"

/**
 * @brief The COCO archive structure.
 *
 * The archive structure is used for pre-processing archives of non-dominated solutions.
 */
struct coco_archive_s {

  avl_tree_t *tree;              /**< @brief The AVL tree with non-dominated solutions. */
  double *ideal;                 /**< @brief The ideal point. */
  double *nadir;                 /**< @brief The nadir point. */

  size_t number_of_objectives;   /**< @brief Number of objectives (clearly equal to 2). */

  int is_up_to_date;             /**< @brief Whether archive fields have been updated since last addition. */
  size_t number_of_solutions;    /**< @brief Number of solutions in the archive. */
  double hypervolume;            /**< @brief Hypervolume of the solutions in the archive. */

  avl_node_t *current_solution;  /**< @brief Current solution (to return). */
  avl_node_t *extreme1;          /**< @brief Pointer to the first extreme solution. */
  avl_node_t *extreme2;          /**< @brief Pointer to the second extreme solution. */
  int extremes_already_returned; /**< @brief Whether the extreme solutions have already been returned. */
};

/**
 * @brief The type for the node's item in the AVL tree used by the archive.
 */
typedef struct {

  double f1;           /**< @brief The value of the first objective of this solution. */
  double f2;           /**< @brief The value of the second objective of this solution. */
  char *text;          /**< @brief The text describing the solution (the whole line of the archive). */

} coco_archive_avl_item_t;

/**
 * @brief Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static coco_archive_avl_item_t* coco_archive_node_item_create(const double f1,
                                                              const double f2,
                                                              const char *text) {

  /* Allocate memory to hold the data structure mo_preprocessing_avl_item_t */
  coco_archive_avl_item_t *item = (coco_archive_avl_item_t*) coco_allocate_memory(sizeof(*item));

  item->f1 = f1;
  item->f2 = f2;
  item->text = coco_strdup(text);

  return item;
}

/**
 * @brief Returns f1 and f2 in a vector of two doubles. Memory needs to be freed by the caller.
 */
static double *coco_archive_node_item_get_vector(coco_archive_avl_item_t *item) {
  double *result = coco_allocate_vector(2);
  result[0] = item->f1;
  result[1] = item->f2;
  return result;
}

/**
 * @brief Frees the data of the given coco_archive_avl_item_t.
 */
static void coco_archive_node_item_free(coco_archive_avl_item_t *item, void *userdata) {
  coco_free_memory(item->text);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the value of the last objective.
 */
static int coco_archive_compare_by_last_objective(const coco_archive_avl_item_t *item1,
                                                  const coco_archive_avl_item_t *item2,
                                                  void *userdata) {
  if (item1->f2 < item2->f2)
    return -1;
  else if (item1->f2 > item2->f2)
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Allocates memory for the archive and initializes its fields.
 */
static coco_archive_t *coco_archive_allocate(void) {

  /* Allocate memory to hold the data structure coco_archive_t */
  coco_archive_t *archive = (coco_archive_t*) coco_allocate_memory(sizeof(*archive));

  /* Initialize the AVL tree */
  archive->tree = avl_tree_construct((avl_compare_t) coco_archive_compare_by_last_objective,
      (avl_free_t) coco_archive_node_item_free);

  archive->ideal = NULL;                /* To be allocated in coco_archive() */
  archive->nadir = NULL;                /* To be allocated in coco_archive() */
  archive->number_of_objectives = 2;
  archive->is_up_to_date = 0;
  archive->number_of_solutions = 0;
  archive->hypervolume = 0.0;

  archive->current_solution = NULL;
  archive->extreme1 = NULL;             /* To be set in coco_archive() */
  archive->extreme2 = NULL;             /* To be set in coco_archive() */
  archive->extremes_already_returned = 0;

  return archive;
}

/**
 * The archive always contains the two extreme solutions
 */
coco_archive_t *coco_archive(const char *suite_name,
                             const size_t function,
                             const size_t dimension,
                             const size_t instance) {

  coco_archive_t *archive = coco_archive_allocate();
  int output_precision = 15;
  coco_suite_t *suite;
  char *suite_instance = coco_strdupf("instances: %lu", instance);
  char *suite_options = coco_strdupf("dimensions: %lu function_indices: %lu", dimension, function);
  coco_problem_t *problem;
  char *text;
  int update;

  suite = coco_suite(suite_name, suite_instance, suite_options);
  if (suite == NULL) {
    coco_error("coco_archive(): cannot create suite '%s'", suite_name);
    return NULL; /* Never reached */
  }
  problem = coco_suite_get_next_problem(suite, NULL);
  if (problem == NULL) {
    coco_error("coco_archive(): cannot create problem f%02lu_i%02lu_d%02lu in suite '%s'", function, instance,
        dimension, suite_name);
    return NULL; /* Never reached */
  }

  /* Store the ideal and nadir points */
  archive->ideal = coco_duplicate_vector(problem->best_value, 2);
  archive->nadir = coco_duplicate_vector(problem->nadir_value, 2);

  /* Add the extreme points to the archive */
  text = coco_strdupf("0\t%.*e\t%.*e\n", output_precision, archive->nadir[0], output_precision, archive->ideal[1]);
  update = coco_archive_add_solution(archive, archive->nadir[0], archive->ideal[1], text);
  coco_free_memory(text);
  assert(update == 1);

  text = coco_strdupf("0\t%.*e\t%.*e\n", output_precision, archive->ideal[0], output_precision, archive->nadir[1]);
  update = coco_archive_add_solution(archive, archive->ideal[0], archive->nadir[1], text);
  coco_free_memory(text);
  assert(update == 1);

  archive->extreme1 = archive->tree->head;
  archive->extreme2 = archive->tree->tail;
  assert(archive->extreme1 != archive->extreme2);

  coco_free_memory(suite_instance);
  coco_free_memory(suite_options);

  coco_suite_free(suite);

  return archive;
}

int coco_archive_add_solution(coco_archive_t *archive, const double f1, const double f2, const char *text) {

  coco_archive_avl_item_t* insert_item = coco_archive_node_item_create(f1, f2, text);
  double *insert_objectives, *node_objectives;
  avl_node_t *node, *next_node;
  int update = 0;
  int dominance;

  insert_objectives = coco_archive_node_item_get_vector(insert_item);

  /* Find the first point that is not worse than the new point (NULL if such point does not exist) */
  node = avl_item_search_right(archive->tree, insert_item, NULL);

  if (node == NULL) {
    /* The new point is an extremal point */
    update = 1;
    next_node = archive->tree->head;
  } else {
    node_objectives = coco_archive_node_item_get_vector((coco_archive_avl_item_t*) node->item);
    dominance = mo_get_dominance(insert_objectives, node_objectives, archive->number_of_objectives);
    coco_free_memory(node_objectives);
    if (dominance > -1) {
      update = 1;
      next_node = node->next;
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        avl_node_delete(archive->tree, node);
      }
    } else {
      /* The new point is dominated, nothing more to do */
      update = 0;
    }
  }

  if (!update) {
    coco_archive_node_item_free(insert_item, NULL);
  } else {
    /* Perform tree update */
    while (next_node != NULL) {
      /* Check the dominance relation between the new node and the next node. There are only two possibilities:
       * dominance = 0: the new node and the next node are nondominated
       * dominance = 1: the new node dominates the next node */
      node = next_node;
      node_objectives = coco_archive_node_item_get_vector((coco_archive_avl_item_t*) node->item);
      dominance = mo_get_dominance(insert_objectives, node_objectives, archive->number_of_objectives);
      coco_free_memory(node_objectives);
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        next_node = node->next;
        avl_node_delete(archive->tree, node);
      } else {
        break;
      }
    }

    avl_item_insert(archive->tree, insert_item);
    archive->is_up_to_date = 0;
  }

  coco_free_memory(insert_objectives);
  return update;
}

/**
 * @brief Updates the archive fields returned by the getters.
 */
static void coco_archive_update(coco_archive_t *archive) {

  if (!archive->is_up_to_date) {

    avl_node_t *node, *left_node;
    coco_archive_avl_item_t *node_item, *left_node_item;
    double *node_objectives, *left_node_objectives;

    /* Updates number_of_solutions */

    archive->number_of_solutions = avl_count(archive->tree);

    /* Updates hypervolume */

    node = archive->tree->head;
    archive->hypervolume = 0; /* Hypervolume of the extreme point equals 0 */
    while (node->next) {
      /* Add hypervolume contributions of the other points that are within ROI */
      left_node = node->next;
      node_item = (coco_archive_avl_item_t *) node->item;
      left_node_item = (coco_archive_avl_item_t *) left_node->item;
      node_objectives = coco_archive_node_item_get_vector(node_item);
      left_node_objectives = coco_archive_node_item_get_vector(left_node_item);
      if (mo_solution_is_within_ROI(left_node_objectives, archive->ideal, archive->nadir, archive->number_of_objectives)) {
        if (mo_solution_is_within_ROI(node_objectives, archive->ideal, archive->nadir, archive->number_of_objectives))
          archive->hypervolume += (node_item->f1 - left_node_item->f1) * (archive->nadir[1] - left_node_item->f2);
        else
          archive->hypervolume += (archive->nadir[0] - left_node_item->f1) * (archive->nadir[1] - left_node_item->f2);
      }
      coco_free_memory(node_objectives);
      coco_free_memory(left_node_objectives);
      node = left_node;
    }
    /* Performs normalization */
    archive->hypervolume /= ((archive->nadir[0] - archive->ideal[0]) * (archive->nadir[1] - archive->ideal[1]));

    archive->is_up_to_date = 1;
    archive->current_solution = NULL;
    archive->extremes_already_returned = 0;
  }

}

const char *coco_archive_get_next_solution_text(coco_archive_t *archive) {

  char *text;

  coco_archive_update(archive);

  if (!archive->extremes_already_returned) {

    if (archive->current_solution == NULL) {
      /* Return the first extreme */
      text = ((coco_archive_avl_item_t *) archive->extreme1->item)->text;
      archive->current_solution = archive->extreme2;
      return text;
    }

    if (archive->current_solution == archive->extreme2) {
      /* Return the second extreme */
      text = ((coco_archive_avl_item_t *) archive->extreme2->item)->text;
      archive->extremes_already_returned = 1;
      archive->current_solution = archive->tree->head;
      return text;
    }

  } else {

    if (archive->current_solution == NULL)
      return "";

    if ((archive->current_solution == archive->extreme1) || (archive->current_solution == archive->extreme2)) {
      /* Skip this one */
      archive->current_solution = archive->current_solution->next;
      return coco_archive_get_next_solution_text(archive);
    }

    /* Return the current solution and move to the next */
    text = ((coco_archive_avl_item_t *) archive->current_solution->item)->text;
    archive->current_solution = archive->current_solution->next;
    return text;
  }

  return NULL; /* This point should never be reached. */
}

size_t coco_archive_get_number_of_solutions(coco_archive_t *archive) {
  coco_archive_update(archive);
  return archive->number_of_solutions;
}

double coco_archive_get_hypervolume(coco_archive_t *archive) {
  coco_archive_update(archive);
  return archive->hypervolume;
}

void coco_archive_free(coco_archive_t *archive) {

  assert(archive != NULL);

  avl_tree_destruct(archive->tree);
  coco_free_memory(archive->ideal);
  coco_free_memory(archive->nadir);
  coco_free_memory(archive);

}
#line 1 "code-experiments/src/coco_runtime_c.c"
/**
 * @file coco_runtime_c.c
 * @brief Generic COCO runtime implementation for the C language.
 *
 * Other language interfaces might want to replace this so that memory allocation and error handling go
 * through the respective language runtime.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#line 14 "code-experiments/src/coco_runtime_c.c"
#line 15 "code-experiments/src/coco_runtime_c.c"

void coco_error(const char *message, ...) {
  va_list args;

  fprintf(stderr, "COCO FATAL ERROR: ");
  va_start(args, message);
  vfprintf(stderr, message, args);
  va_end(args);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

void coco_warning(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_WARNING) {
    fprintf(stderr, "COCO WARNING: ");
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, "\n");
  }
}

void coco_info(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_INFO) {
    fprintf(stdout, "COCO INFO: ");
    va_start(args, message);
    vfprintf(stdout, message, args);
    va_end(args);
    fprintf(stdout, "\n");
    fflush(stdout);
  }
}

/**
 * A function similar to coco_info that prints only the given message without any prefix and without
 * adding a new line.
 */
void coco_info_partial(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_INFO) {
    va_start(args, message);
    vfprintf(stdout, message, args);
    va_end(args);
    fflush(stdout);
  }
}

void coco_debug(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_DEBUG) {
    fprintf(stdout, "COCO DEBUG: ");
    va_start(args, message);
    vfprintf(stdout, message, args);
    va_end(args);
    fprintf(stdout, "\n");
    fflush(stdout);
  }
}

void *coco_allocate_memory(const size_t size) {
  void *data;
  if (size == 0) {
    coco_error("coco_allocate_memory() called with 0 size.");
    return NULL; /* never reached */
  }
  data = malloc(size);
  if (data == NULL)
    coco_error("coco_allocate_memory() failed.");
  return data;
}

void coco_free_memory(void *data) {
  free(data);
}
