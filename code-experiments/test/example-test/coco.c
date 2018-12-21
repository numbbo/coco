
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
#include <float.h>
#ifndef NAN
/** @brief Definition of NAN to be used only if undefined by the included headers */
#define NAN 8.8888e88
#endif
#ifndef isnan
/** @brief Definition of isnan to be used only if undefined by the included headers */
#define isnan(x) (0)
#endif
#ifndef INFINITY
/** @brief Definition of INFINITY to be used only if undefined by the included headers */
#define INFINITY 1e22
/* why not using 1e99? */
#endif
#ifndef isinf
/** @brief Definition of isinf to be used only if undefined by the included headers */
#define isinf(x) (0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief COCO's version.
 *
 * Automatically updated by do.py.
 */
/**@{*/
static const char coco_version[32] = "2.2.1.565";
/**@}*/

/***********************************************************************************************************/
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
 * @brief Returns the first problem of the suite defined by function, dimension and instance numbers.
 */
coco_problem_t *coco_suite_get_problem_by_function_dimension_instance(coco_suite_t *suite,
                                                                      const size_t function,
                                                                      const size_t dimension,
                                                                      const size_t instance);

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
                                     size_t *dimension_idx,
                                     size_t *instance_idx);
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

/**
 * @brief Returns result folder name, where logger output is written. 
 */
const char *coco_observer_get_result_folder(const coco_observer_t *observer);

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
 * @brief Recommends a solution as the current best guesses to the problem. Not implemented yet.
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
 * @brief Returns the type of the problem.
 */
const char *coco_problem_get_type(const coco_problem_t *problem);

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
 * @brief Returns the number of objective function evaluations done on the problem.
 */
size_t coco_problem_get_evaluations(const coco_problem_t *problem);

/**
 * @brief Returns the number of constraint function evaluations done on the problem.
 */
size_t coco_problem_get_evaluations_constraints(const coco_problem_t *problem);

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
 * @brief Returns the number of integer variables. If > 0, all integer variables come before any
 * continuous ones.
 */
size_t coco_problem_get_number_of_integer_variables(const coco_problem_t *problem);

/**
 * @brief For multi-objective problems, returns a vector of largest values of interest in each objective.
 * Currently, this equals the nadir point. For single-objective problems it raises an error.
 */
const double *coco_problem_get_largest_fvalues_of_interest(const coco_problem_t *problem);

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
 * @name Methods regarding COCO archives and log files (used when pre-processing MO data)
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
 * @brief Adds a solution with objectives (y1, y2) to the archive if none of the existing solutions in the
 * archive dominates it. In this case, returns 1, otherwise the archive is not updated and the method
 * returns 0.
 */
int coco_archive_add_solution(coco_archive_t *archive, const double y1, const double y2, const char *text);

/**
 * @brief Returns the number of (non-dominated) solutions in the archive (computed first, if needed).
 */
size_t coco_archive_get_number_of_solutions(coco_archive_t *archive);

/**
 * @brief Returns the hypervolume of the archive (computed first, if needed).
 */
double coco_archive_get_hypervolume(coco_archive_t *archive);

/**
 * @brief Returns the text of the next (non-dominated) solution in the archive and "" when there are no
 * solutions left. The first two solutions are always the extreme ones.
 */
const char *coco_archive_get_next_solution_text(coco_archive_t *archive);

/**
 * @brief Frees the archive.
 */
void coco_archive_free(coco_archive_t *archive);

/**
 * @brief Feeds the solution to the bi-objective logger for logger output reconstruction purposes.
 */
int coco_logger_biobj_feed_solution(coco_problem_t *problem, const size_t evaluation, const double *y);
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
#include <stdio.h>

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
  /* printf("coco_random_new(): %u\n", seed); */
  /* Expand seed to fill initial state array. */
  for (i = 0; i < COCO_LONG_LAG; ++i) {
    /* Uses uint64_t to silence the compiler ("shift count negative or too big, undefined behavior" warning) */
    state->x[i] = ((double) seed) / (double) (((uint64_t) 1UL << 32) - 1);
    /* Advance seed based on simple RNG from TAOCP */
    seed = (uint32_t) 1812433253UL * (seed ^ (seed >> 30)) + ((uint32_t) i + 1);
  }
  state->index = 12;
  /* coco_random_generate(state); */
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
 * @brief The get problem function type.
 *
 * This is a template for functions that return a problem based on function, dimension and instance.
 */
typedef coco_problem_t *(*coco_get_problem_function_t)(const size_t function,
                                                       const size_t dimension,
                                                       const size_t instance);


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

  coco_evaluate_function_t evaluate_function;         /**< @brief  The function for evaluating the problem. */
  coco_evaluate_function_t evaluate_constraint;       /**< @brief  The function for evaluating the constraints. */
  coco_evaluate_function_t evaluate_gradient;         /**< @brief  The function for evaluating the constraints. */
  coco_recommend_function_t recommend_solution;       /**< @brief  The function for recommending a solution. */
  coco_problem_free_function_t problem_free_function; /**< @brief  The function for freeing this problem. */

  size_t number_of_variables;          /**< @brief Number of variables expected by the function, i.e.
                                       problem dimension */
  size_t number_of_objectives;         /**< @brief Number of objectives. */
  size_t number_of_constraints;        /**< @brief Number of constraints. */

  double *smallest_values_of_interest; /**< @brief The lower bounds of the ROI in the decision space. */
  double *largest_values_of_interest;  /**< @brief The upper bounds of the ROI in the decision space. */
  size_t number_of_integer_variables;  /**< @brief Number of integer variables (if > 0, all integer variables come
                                       before any continuous ones). */

  double *initial_solution;            /**< @brief Initial feasible solution. */
  double *best_value;                  /**< @brief Optimal (smallest) function value */
  double *nadir_value;                 /**< @brief The nadir point (defined when number_of_objectives > 1) */
  double *best_parameter;              /**< @brief Optimal decision vector (defined only when unique) */

  char *problem_name;                  /**< @brief Problem name. */
  char *problem_id;                    /**< @brief Problem ID (unique in the containing suite) */
  char *problem_type;                  /**< @brief Problem type */

  size_t evaluations;                  /**< @brief Number of objective function evaluations performed on the problem. */
  size_t evaluations_constraints;      /**< @brief Number of constraint function evaluations performed on the problem. */

  /* Convenience fields for output generation */
  /* If at some point in time these arrays are changed to pointers, checks need to be added in the code to make sure
   * they are not NULL.*/

  double final_target_delta[1];        /**< @brief Final target delta. */
  double best_observed_fvalue[1];      /**< @brief The best observed value so far. */
  size_t best_observed_evaluation[1];  /**< @brief The evaluation when the best value so far was achieved. */

  /* Fields depending on the containing benchmark suite */

  coco_suite_t *suite;                 /**< @brief Pointer to the containing suite (NULL if not given) */
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
  int precision_g;           /**< @brief Output precision for constraint values. */
  int log_discrete_as_int;   /**< @brief Whether to output discrete variables in int or double format. */
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

static void bbob_evaluate_gradient(coco_problem_t *problem, const double *x, double *y);

void bbob_problem_best_parameter_print(const coco_problem_t *problem);

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
#define USES_CREATEPROCESS
#elif defined(__gnu_linux__)
#include <sys/stat.h>
#include <sys/types.h>
#include <linux/limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#include <unistd.h>
#include <sys/wait.h>
#define USES_EXECVP
#elif defined(__APPLE__)
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syslimits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#include <unistd.h>
#include <sys/wait.h>
#define USES_EXECVP
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
#include <errno.h>
#include <limits.h>

#line 18 "code-experiments/src/coco_utilities.c"
#line 19 "code-experiments/src/coco_utilities.c"
#line 1 "code-experiments/src/coco_string.c"
/**
 * @file coco_string.c
 * @brief Definitions of functions that manipulate strings.
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>

#line 12 "code-experiments/src/coco_string.c"

static size_t *coco_allocate_vector_size_t(const size_t number_of_elements);
static char *coco_allocate_string(const size_t number_of_elements);

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
          coco_error("coco_strfind(): strange values observed i=%lu, j=%lu, strlen(base)=%lu",
          		(unsigned long) i, (unsigned long) j, (unsigned long) strlen(base));
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
 * The caller is responsible for freeing the allocated memory using:
 *
 *  for (i = 0; *(result + i); i++)
 *    coco_free_memory(*(result + i));
 *  coco_free_memory(*(result + i));    <- This is needed!
 *  coco_free_memory(result);
 *
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
    coco_free_memory(str);
    return NULL;
  }

  ptr = str;
  /* Check for disallowed characters */
  while (*ptr != '\0') {
    if ((*ptr != '-') && (*ptr != ',') && !isdigit((unsigned char )*ptr)) {
      coco_warning("coco_string_parse_ranges(): problem parsing '%s' - cannot parse ranges with '%c'", str,
          *ptr);
      coco_free_memory(str);
      return NULL;
    } else
      ptr++;
  }

  /* Check for incorrect boundaries */
  if ((max > 0) && (min > max)) {
    coco_warning("coco_string_parse_ranges(): incorrect boundaries");
    coco_free_memory(str);
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
        coco_warning("coco_string_parse_ranges(): '%s' ranges adjusted to be >= %lu", name,
        		(unsigned long) min);
      }
      if ((max > 0) && (num[1] > max)) {
        num[1] = max;
        coco_warning("coco_string_parse_ranges(): '%s' ranges adjusted to be <= %lu", name,
        		(unsigned long) max);
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

/**
 * @brief Trims the given string (removes any leading and trailing spaces).
 *
 * If the string contains any leading spaces, the contents are shifted so that if it was dynamically
 * allocated, it can be still freed on the returned pointer.
 */
static char *coco_string_trim(char *string) {
	size_t i, len = 0;
	int all_whitespaces = 1;
	char *frontp = string;
	char *endp = NULL;

	if (string == NULL) {
		return NULL;
	}
	if (string[0] == '\0') {
		return string;
	}

	len = strlen(string);
	endp = string + len;

	for (i = 0; ((i < len) && all_whitespaces); i++)
		all_whitespaces = all_whitespaces && isspace(string[i]);
	if (all_whitespaces) {
	  string[0] = '\0';
		return string;
	}

	/* Move the front and back pointers to address the first non-whitespace characters from each end. */
	while (isspace((unsigned char) *frontp)) {
		++frontp;
	}
	if (endp != frontp) {
		while (isspace((unsigned char) *(--endp)) && endp != frontp) {
		}
	}

	if (string + len - 1 != endp)
		*(endp + 1) = '\0';
	else if (frontp != string && endp == frontp)
		*string = '\0';

	/* Shift the string. Note the reuse of endp to mean the front of the string buffer now. */
	endp = string;
	if (frontp != string) {
		while (*frontp) {
			*endp++ = *frontp++;
		}
		*endp = '\0';
	}

	return string;
}

#line 20 "code-experiments/src/coco_utilities.c"


/***********************************************************************************************************/

/**
 * @brief Sets the constant chosen_precision to 1e-9.
 */
static const double chosen_precision = 1e-9;

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
  DWORD dwAttrib = GetFileAttributesA(path);
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
  DWORD dwAttrib = GetFileAttributesA(path);
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
 * @brief Calls the right mkdir() method (depending on the platform) with full privileges for the user. 
 * If the created directory has not existed before, returns 0, otherwise returns 1. If the directory has 
 * not been created, a coco_error is raised. 
 *
 * @param path The directory path.
 *
 * @return 0 if the created directory has not existed before and 1 otherwise.
 */
static int coco_mkdir(const char *path) {
  int result = 0;

#if _MSC_VER
  result = _mkdir(path);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  result = mkdir(path);
#else
  result = mkdir(path, S_IRWXU);
#endif

  if (result == 0)
    return 0;
  else if (errno == EEXIST)
    return 1;
  else 
    coco_error("coco_mkdir(): unable to create %s, mkdir error: %s", path, strerror(errno));
    return 1; /* Never reached */
}

/**
 * @brief Creates a directory (possibly having to create nested directories). If the last created directory 
 * has not existed before, returns 0, otherwise returns 1.
 *
 * @param path The directory path.
 *
 * @return 0 if the created directory has not existed before and 1 otherwise.
 */
static int coco_create_directory(const char *path) {
  char *path_copy = NULL;
  char *tmp, *p;
  char path_sep = coco_path_separator[0];
  size_t len = strlen(path);

  int result = 0;

  path_copy = coco_strdup(path);
  tmp = path_copy;

  /* Remove possible leading and trailing (back)slash */
  if (tmp[len - 1] == path_sep)
    tmp[len - 1] = 0;
  if (tmp[0] == path_sep)
    tmp++;

  /* Iterate through nested directories (does nothing if directories are not nested) */
  for (p = tmp; *p; p++) {
    if (*p == path_sep) {
      *p = 0;
      coco_mkdir(tmp);
      *p = path_sep;
    }
  }
  
  /* Create the last nested or only directory */
  result = coco_mkdir(tmp);
  coco_free_memory(path_copy);
  return result;
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
 * @brief Creates a directory that has not existed before.
 *
 * If the given path does not yet exit, it is left as is, otherwise it is changed(!) by appending a number
 * to it. If path already exists, path-001 will be tried. If this one exists as well, path-002 will be tried,
 * and so on. If path-999 exists as well, an error is raised.
 */
static void coco_create_unique_directory(char **path) {

  int counter = 1;
  char *new_path;

  if (coco_create_directory(*path) == 0) {
	/* Directory created */
    return;
  }

  while (counter < 999) {

    new_path = coco_strdupf("%s-%03d", *path, counter);

    if (coco_create_directory(new_path) == 0) {
      /* Directory created */
      coco_free_memory(*path);
      *path = new_path;
      return;
    } else {
      counter++;
      coco_free_memory(new_path);
    }

  }

  coco_error("coco_create_unique_directory(): unable to create unique directory %s", *path);
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



/**
 * The method should work across different platforms/compilers.
 *
 * @file_name The path to the file
 *
 * @return 0 on successful completion, and -1 on error.
 */
int coco_remove_file(const char *file_name) {
#if _MSC_VER
  int r = -1;
  /* Try to delete the file */
  /* Careful, DeleteFile returns 0 if it fails and nonzero otherwise! */
  r = -(DeleteFile(file_name) == 0);
  return r;
#else
  int r = -1;
  /* Try to delete the file */
  r = unlink(file_name);
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
 * @brief Allocates memory for a vector and sets all its elements to value.
 */
static double *coco_allocate_vector_with_value(const size_t number_of_elements, double value) {
  const size_t block_size = number_of_elements * sizeof(double);
  double *vector = (double *) coco_allocate_memory(block_size);
  size_t i;

  for (i = 0; i < number_of_elements; i++)
  	vector[i] = value;

  return vector;
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

  size_t i;
  char **keys;
  coco_option_keys_t *option_keys = NULL;
  char *string_to_parse, *key;

  /* Check for empty string */
  if ((option_string == NULL) || (strlen(option_string) == 0)) {
	    return NULL;
  }

  /* Split the options w.r.t ':' */
  keys = coco_string_split(option_string, ':');

  if (keys) {
    /* Keys now contain something like this: "values_of_previous_key this_key" except for the first, which
     * contains only the key and the last, which contains only the previous values */
    for (i = 0; *(keys + i); i++) {
      string_to_parse = coco_strdup(*(keys + i));

      /* Remove any leading and trailing spaces */
      string_to_parse = coco_string_trim(string_to_parse);

      /* Stop if this is the last substring (contains a value and no key) */
      if ((i > 0) && (*(keys + i + 1) == NULL)) {
        coco_free_memory(string_to_parse);
        break;
      }

      /* Disregard everything before the last space */
      key = strrchr(string_to_parse, ' ');
      if ((key == NULL) || (i == 0)) {
        /* No spaces left (or this is the first key), everything is the key */
        key = string_to_parse;
      } else {
        /* Move to the start of the key (one char after the space) */
        key++;
      }

      /* Put the key in keys */
      coco_free_memory(*(keys + i));
      *(keys + i) = coco_strdup(key);
      coco_free_memory(string_to_parse);
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
  char *string = NULL, *new_string;

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
 * i.e. reads all characters from the corresponding name up to the next alphabetic character or end of string,
 * ignoring white-space characters.
 *
 * Formatting requirements:
 * - names have to start with alphabetic characters
 * - values cannot include alphabetic characters
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

  if (i2 <= i1) {
    return 0;
  }

  i = 0;
  while (!isalpha((unsigned char) options[i2 + i]) && (options[i2 + i] != '\0')) {
    if(isspace((unsigned char) options[i2 + i])) {
        i2++;
    } else {
        pointer[i] = options[i2 + i];
        i++;
    }
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
 *
 * TODO: This method could (should?) check for overflow when casting (similarly as is done in
 * coco_double_to_int()).
 */
static size_t coco_double_to_size_t(const double number) {
  return (size_t) coco_double_round(number);
}

/**
 * @brief Rounds the given double to the nearest integer (returns the number in int type)
 */
static int coco_double_to_int(const double number) {
  if (number > (double)INT_MAX) {
    coco_error("coco_double_to_int(): Cannot cast %f to the nearest integer, max %d allowed",
        number, INT_MAX);
    return -1; /* Never reached */
  }
  else if (number < (double)INT_MIN) {
    coco_error("coco_double_to_int(): Cannot cast %f to the nearest integer, min %d allowed",
        number, INT_MIN);
    return -1; /* Never reached */
  }
  else {
    return (int)(number + 0.5);
  }
}

/**
 * @brief  Returns 1 if |a - b| < precision and 0 otherwise.
 */
static int coco_double_almost_equal(const double a, const double b, const double precision) {
  return (fabs(a - b) < precision);
}

/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods handling NAN and INFINITY
 */
/**@{*/

/**
 * @brief Returns 1 if x is NAN and 0 otherwise.
 */
static int coco_is_nan(const double x) {
  return (isnan(x) || (x != x) || !(x == x) || ((x >= NAN / (1 + chosen_precision)) && (x <= NAN * (1 + chosen_precision))));
}

/**
 * @brief Returns 1 if the input vector of dimension dim contains any NAN values and 0 otherwise.
 */
static int coco_vector_contains_nan(const double *x, const size_t dim) {
	size_t i;
	for (i = 0; i < dim; i++) {
		if (coco_is_nan(x[i]))
		  return 1;
	}
	return 0;
}

/**
 * @brief Sets all dim values of y to NAN.
 */
static void coco_vector_set_to_nan(double *y, const size_t dim) {
	size_t i;
	for (i = 0; i < dim; i++) {
		y[i] = NAN;
	}
}

/**
 * @brief Returns 1 if x is INFINITY and 0 otherwise.
 */
static int coco_is_inf(const double x) {
	if (coco_is_nan(x))
		return 0;
	return (isinf(x) || (x <= -INFINITY) || (x >= INFINITY));
}

/**
 * @brief Returns 1 if the input vector of dimension dim contains no NaN of inf values, and 0 otherwise.
 */
static int coco_vector_isfinite(const double *x, const size_t dim) {
	size_t i;
	for (i = 0; i < dim; i++) {
		if (coco_is_nan(x[i]) || coco_is_inf(x[i]))
		  return 0;
	}
	return 1;
}

/**
 * @brief Returns 1 if the point x is feasible, and 0 otherwise.
 *
 * Allows constraint_values == NULL, otherwise constraint_values
 * must be a valid double* pointer and contains the g-values of x
 * on "return".
 * 
 * Any point x containing NaN or inf values is considered infeasible.
 *
 * This function is (and should be) used internally only, and does not
 * increase the counter of constraint function evaluations.
 *
 * @param problem The given COCO problem.
 * @param x Decision vector.
 * @param constraint_values Vector of contraints values resulting from evaluation.
 */
static int coco_is_feasible(coco_problem_t *problem,
                     const double *x,
                     double *constraint_values) {

  size_t i;
  double *cons_values = constraint_values;
  int ret_val = 1;

  /* Return 0 if the decision vector contains any INFINITY or NaN values */
  if (!coco_vector_isfinite(x, coco_problem_get_dimension(problem)))
    return 0;

  if (coco_problem_get_number_of_constraints(problem) <= 0)
    return 1;

  assert(problem != NULL);
  assert(problem->evaluate_constraint != NULL);
  
  if (constraint_values == NULL)
     cons_values = coco_allocate_vector(problem->number_of_constraints);

  problem->evaluate_constraint(problem, x, cons_values);
  /* coco_evaluate_constraint(problem, x, cons_values) increments problem->evaluations_constraints counter */

  for(i = 0; i < coco_problem_get_number_of_constraints(problem); ++i) {
    if (cons_values[i] > 0.0) {
      ret_val = 0;
      break;
    }
  }

  if (constraint_values == NULL)
    coco_free_memory(cons_values);
  return ret_val;
}

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
    coco_error("coco_count_numbers(): over %lu numbers in %s", (unsigned long) max_count, name);
    return 0; /* Never reached*/
  }

  return count;
}

/**
 * @brief multiply each componenent by nom/denom or by nom if denom == 0.
 *
 * return used scaling factor, usually nom/denom.
 *
 * Example: coco_vector_scale(x, dimension, 1, coco_vector_norm(x, dimension));
 */
static double coco_vector_scale(double *x, size_t dimension, double nom, double denom) {

  size_t i;

  assert(x);

  if (denom != 0)
    nom /= denom;

  for (i = 0; i < dimension; ++i)
      x[i] *= nom;
  return nom;
}

/**
 * @brief return norm of vector x.
 *
 */
static double coco_vector_norm(const double *x, size_t dimension) {

  size_t i;
  double ssum = 0.0;

  assert(x);

  for (i = 0; i < dimension; ++i)
    ssum += x[i] * x[i];

  return sqrt(ssum);
}

/**
 * @brief Checks if a given matrix M is orthogonal by (partially) computing M * M^T.
 * If M is a square matrix and M * M^T is close enough to the identity matrix
 * (up to a chosen precision), the function returns 1. Otherwise, it returns 0.
 * The matrix M must be represented as an array of doubles.
 */
static int coco_is_orthogonal(const double *M, const size_t nb_rows, const size_t nb_columns) {

  size_t i, j, z;
  double sum;

  if (nb_rows != nb_columns)
    return 0;

  for (i = 0; i < nb_rows; ++i) {
    for (j = 0; j < nb_rows; ++j) {
        /* Compute the dot product of the ith row of M
         * and the jth column of M^T (i.e. jth row of M)
         */
        sum = 0.0;
        for (z = 0; z < nb_rows; ++z) {
            sum += M[i * nb_rows + z] * M[j * nb_rows + z];
        }

        /* Check if the dot product is 1 (resp. 0) when the row and the column
         * indices are the same (resp. different)
         */
        if (((i == j) && !coco_double_almost_equal(sum, 1, chosen_precision)) ||
            ((i != j) && !coco_double_almost_equal(sum, 0, chosen_precision)))
                return 0;

    }
  }
  return 1;
}

/**
 * @brief Returns 1 if the input vector x is (close to) zero and 0 otherwise.
 */
static int coco_vector_is_zero(const double *x, const size_t dim) {
  size_t i = 0;
  int is_zero = 1;

  if (coco_vector_contains_nan(x, dim))
    return 0;

  while (i < dim && is_zero) {
    is_zero = coco_double_almost_equal(x[i], 0, chosen_precision);
    i++;
  }

  return is_zero;
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
  size_t i, j;
  int is_feasible;
  double *z;
  
  assert(problem != NULL);
  assert(problem->evaluate_function != NULL);
  
  /* Set objective vector to INFINITY if the decision vector contains any INFINITY values */
  for (i = 0; i < coco_problem_get_dimension(problem); i++) {
    if (coco_is_inf(x[i])) {
      for (j = 0; j < coco_problem_get_number_of_objectives(problem); j++) {
        y[j] = fabs(x[i]);
      }
      return;
    }
  }
  
  /* Set objective vector to NAN if the decision vector contains any NAN values */
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  problem->evaluate_function(problem, x, y);
  problem->evaluations++; /* each derived class has its own counter, only the most outer will be visible */

  /* A little bit of bookkeeping */
  if (y[0] < problem->best_observed_fvalue[0]) {
    is_feasible = 1;
    if (coco_problem_get_number_of_constraints(problem) > 0) {
      z = coco_allocate_vector(coco_problem_get_number_of_constraints(problem));
      is_feasible = coco_is_feasible(problem, x, z);
      coco_free_memory(z);
    }
    if (is_feasible) {
      problem->best_observed_fvalue[0] = y[0];
      problem->best_observed_evaluation[0] = problem->evaluations;
    }
  }
}

/**
 * Evaluates the problem constraint.
 * 
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The vector of constraints that is the result of the evaluation.
 */
void coco_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of problem->evaluate(problem, x, y) */
  size_t i, j;
  assert(problem != NULL);
  if (problem->evaluate_constraint == NULL) {
    coco_error("coco_evaluate_constraint(): No constraint function implemented for problem %s",
        problem->problem_id);
  }
  
  /* Set constraints vector to INFINITY if the decision vector contains any INFINITY values */
  for (i = 0; i < coco_problem_get_dimension(problem); i++) {
    if (coco_is_inf(x[i])) {
      for (j = 0; j < coco_problem_get_number_of_constraints(problem); j++) {
        y[j] = fabs(x[i]);
      }
      return;
    }
  }
  
  /* Set constraints vector to NAN if the decision vector contains any NAN values */
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
    return;
  }
  
  problem->evaluate_constraint(problem, x, y);
  problem->evaluations_constraints++;
}

/**
 * @note Both x and y must point to correctly sized allocated memory regions.
 *
 * @param problem The given COCO problem.
 * @param x The decision vector.
 * @param y The gradient of the function evaluated at the point x.
 */
static void bbob_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  /* implements a safer version of problem->evaluate_gradient(problem, x, y) */
  assert(problem != NULL);
  if (problem->evaluate_gradient == NULL) {
    coco_error("bbob_evaluate_gradient(): No gradient function implemented for problem %s",
        problem->problem_id);
  }
  problem->evaluate_gradient(problem, x, y);
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
  problem->evaluate_gradient = NULL;
  problem->recommend_solution = NULL;
  problem->problem_free_function = NULL;
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = number_of_objectives;
  problem->number_of_constraints = number_of_constraints;
  problem->smallest_values_of_interest = coco_allocate_vector(number_of_variables);
  problem->largest_values_of_interest = coco_allocate_vector(number_of_variables);
  problem->number_of_integer_variables = 0; /* No integer variables by default */

  if (number_of_objectives > 1) {
    problem->best_parameter = NULL;
    problem->best_value = coco_allocate_vector(number_of_objectives);
    problem->nadir_value = coco_allocate_vector(number_of_objectives);
  }
  else {
    problem->best_parameter = coco_allocate_vector(number_of_variables);
    problem->best_value = coco_allocate_vector(1);
    problem->nadir_value = NULL;
  }
  problem->problem_name = NULL;
  problem->problem_id = NULL;
  problem->problem_type = NULL;
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
  problem->final_target_delta[0] = 1e-8; /* in case to be modified by the benchmark */
  problem->best_observed_fvalue[0] = DBL_MAX;
  problem->best_observed_evaluation[0] = 0;
  problem->suite = NULL; /* To be initialized in the coco_suite_get_problem_from_indices() function */
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
  problem->number_of_integer_variables = other->number_of_integer_variables;

  if (other->initial_solution)
    problem->initial_solution = coco_duplicate_vector(other->initial_solution, other->number_of_variables);

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
  problem->evaluations_constraints = other->evaluations_constraints;
  problem->final_target_delta[0] = other->final_target_delta[0];
  problem->best_observed_fvalue[0] = other->best_observed_fvalue[0];
  problem->best_observed_evaluation[0] = other->best_observed_evaluation[0];

  problem->suite = other->suite;
  problem->suite_dep_index = other->suite_dep_index;
  problem->suite_dep_function = other->suite_dep_function;
  problem->suite_dep_instance = other->suite_dep_instance;

  problem->data = NULL;

  return problem;
}

/**
 * @brief Allocates a problem using scalar values for smallest_value_of_interest, largest_value_of_interest
 * and best_parameter. Assumes all variables are continuous.
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
  problem->number_of_integer_variables = 0;
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
    if (problem->initial_solution != NULL)
      coco_free_memory(problem->initial_solution);
    problem->smallest_values_of_interest = NULL;
    problem->largest_values_of_interest = NULL;
    problem->best_parameter = NULL;
    problem->best_value = NULL;
    problem->nadir_value = NULL;
    problem->suite = NULL;
    problem->data = NULL;
    problem->initial_solution = NULL;
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

size_t coco_problem_get_evaluations_constraints(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->evaluations_constraints;
}

/**
 * @brief Returns 1 if the best parameter is not (close to) zero and 0 otherwise.
 */
static int coco_problem_best_parameter_not_zero(const coco_problem_t *problem) {
  size_t i = 0;
  int best_is_zero = 1;

  if (coco_vector_contains_nan(problem->best_parameter, problem->number_of_variables))
    return 1;

  while (i < problem->number_of_variables && best_is_zero) {
    best_is_zero = coco_double_almost_equal(problem->best_parameter[i], 0, 1e-9);
    i++;
  }

  return !best_is_zero;
}

/**
 * @note Can be used to prevent unnecessary burning of CPU time.
 */
int coco_problem_final_target_hit(const coco_problem_t *problem) {
  assert(problem != NULL);
  if (coco_problem_get_number_of_objectives(problem) != 1 ||
      coco_problem_get_evaluations(problem) < 1) 
    return 0;
  if (problem->best_value == NULL)
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
 * @brief Returns the optimal function value of the problem
 */
double coco_problem_get_best_value(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->best_value != NULL);
  return problem->best_value[0];
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

size_t coco_problem_get_number_of_integer_variables(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->number_of_integer_variables;
}

const double *coco_problem_get_largest_fvalues_of_interest(const coco_problem_t *problem) {
  assert(problem != NULL);
  if (problem->number_of_objectives == 1)
    coco_error("coco_problem_get_largest_fvalues_of_interest(): f-values of interest undefined for single-objective problems");
  if (problem->nadir_value == NULL)
    coco_error("coco_problem_get_largest_fvalues_of_interest(): f-values of interest undefined");
  return problem->nadir_value;
}

/**
 * Copies problem->initial_solution into initial_solution if not null, 
 * otherwise the center of the problem's region of interest is the 
 * initial solution. Takes care of rounding the solution in case of integer variables.
 * 
 * @param problem The given COCO problem.
 * @param initial_solution The pointer to the initial solution being set by this method.
 */
void coco_problem_get_initial_solution(const coco_problem_t *problem, double *initial_solution) {
  
  size_t i; 
   
  assert(problem != NULL);
  if (problem->initial_solution != NULL) {
    for (i = 0; i < problem->number_of_variables; ++i)
      initial_solution[i] = problem->initial_solution[i];
  } else {
    assert(problem->smallest_values_of_interest != NULL);
    assert(problem->largest_values_of_interest != NULL);
    for (i = 0; i < problem->number_of_variables; ++i)
      initial_solution[i] = problem->smallest_values_of_interest[i] + 0.5
          * (problem->largest_values_of_interest[i] - problem->smallest_values_of_interest[i]);
    if (problem->number_of_integer_variables > 0) {
      for (i = 0; i < problem->number_of_integer_variables; ++i) {
        initial_solution[i] = coco_double_round(initial_solution[i]);
      }
    }
  }
}

static coco_suite_t *coco_problem_get_suite(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->suite;
}

static void coco_problem_set_suite(coco_problem_t *problem, coco_suite_t *suite) {
  assert(problem != NULL);
  problem->suite = suite;
}

size_t coco_problem_get_suite_dep_index(const coco_problem_t *problem) {
  assert(problem != NULL);
  return problem->suite_dep_index;
}

static size_t coco_problem_get_suite_dep_function(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_function > 0);
  return problem->suite_dep_function;
}

static size_t coco_problem_get_suite_dep_instance(const coco_problem_t *problem) {
  assert(problem != NULL);
  assert(problem->suite_dep_instance > 0);
  return problem->suite_dep_instance;
}
/**@}*/

void bbob_problem_best_parameter_print(const coco_problem_t *problem) {
  size_t i;
  FILE *file;
  assert(problem != NULL);
  assert(problem->best_parameter != NULL);
  file = fopen("._bbob_problem_best_parameter.txt", "w");
  if (file != NULL) {
    for (i = 0; i < problem->number_of_variables; ++i)
      fprintf(file, " %.16f ", problem->best_parameter[i]);
    fclose(file);
  }
}

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

static void bbob_problem_transformed_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_transformed_data_t *data;
  assert(problem != NULL);
  assert(problem->data != NULL);
  data = (coco_problem_transformed_data_t *) problem->data;
  assert(data->inner_problem != NULL);

  bbob_evaluate_gradient(data->inner_problem, x, y);
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
  inner_copy->evaluate_gradient = bbob_problem_transformed_evaluate_gradient;
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

  const size_t number_of_objectives_problem1 = coco_problem_get_number_of_objectives(data->problem1);
  const size_t number_of_objectives_problem2 = coco_problem_get_number_of_objectives(data->problem2);
  double *cons_values = NULL;
  int is_feasible;
    
  assert(coco_problem_get_number_of_objectives(problem)
      == number_of_objectives_problem1 + number_of_objectives_problem2);
  
  if (number_of_objectives_problem1 > 0)
     coco_evaluate_function(data->problem1, x, &y[0]);
  if (number_of_objectives_problem2 > 0)
     coco_evaluate_function(data->problem2, x, &y[number_of_objectives_problem1]);

  /* Make sure that no feasible point has a function value lower
   * than the minimum's.
   */
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);   
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
}

/**
 * @brief Calls the coco_evaluate_constraint function on the underlying problems.
 */
static void coco_problem_stacked_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  coco_problem_stacked_data_t* data = (coco_problem_stacked_data_t*) problem->data;

  const size_t number_of_constraints_problem1 = coco_problem_get_number_of_constraints(data->problem1);
  const size_t number_of_constraints_problem2 = coco_problem_get_number_of_constraints(data->problem2);
  assert(coco_problem_get_number_of_constraints(problem)
      == number_of_constraints_problem1 + number_of_constraints_problem2);

  if (number_of_constraints_problem1 > 0)
    coco_evaluate_constraint(data->problem1, x, y);
  if (number_of_constraints_problem2 > 0)
    coco_evaluate_constraint(data->problem2, x, &y[number_of_constraints_problem1]);
  
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
static coco_problem_t *coco_problem_stacked_allocate(coco_problem_t *problem1, 
                                                     coco_problem_t *problem2,
                                                     const double *smallest_values_of_interest,
                                                     const double *largest_values_of_interest) {

  size_t number_of_variables, number_of_objectives, number_of_constraints;
  size_t i;
  char *s;
  coco_problem_stacked_data_t *data;
  coco_problem_t *problem; /* the new coco problem */

  assert(problem1);
  assert(problem2);
  assert(coco_problem_get_dimension(problem1) == coco_problem_get_dimension(problem2));

  number_of_variables = coco_problem_get_dimension(problem1);
  number_of_objectives = coco_problem_get_number_of_objectives(problem1)
      + coco_problem_get_number_of_objectives(problem2);
  number_of_constraints = coco_problem_get_number_of_constraints(problem1)
      + coco_problem_get_number_of_constraints(problem2);

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

  assert(smallest_values_of_interest);
  assert(largest_values_of_interest);
  
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] = largest_values_of_interest[i];
  }
  assert(problem1->number_of_integer_variables == problem2->number_of_integer_variables);
  problem->number_of_integer_variables = problem1->number_of_integer_variables;

  assert(problem->best_value);
    
  if (number_of_constraints > 0) {
     
    /* The best_value must be set up afterwards in suite_cons_bbob_problems.c */
    problem->best_value[0] = -FLT_MAX;
    
    /* Define problem->initial_solution as problem2->initial_solution */
    if (coco_problem_get_number_of_constraints(problem2) > 0 && problem2->initial_solution)
      problem->initial_solution = coco_duplicate_vector(problem2->initial_solution, number_of_variables);
      
  }
  else {
     
    /* Compute the ideal and nadir values */
    assert(problem->nadir_value);
    
    problem->best_value[0] = problem1->best_value[0];
    problem->best_value[1] = problem2->best_value[0];
    coco_evaluate_function(problem1, problem2->best_parameter, &problem->nadir_value[0]);
    coco_evaluate_function(problem2, problem1->best_parameter, &problem->nadir_value[1]);
    
  }

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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }
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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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
 * @brief Evaluates the transformed function.
 */
static void transform_obj_shift_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_shift_data_t *data;
  double *cons_values;
  int is_feasible;
  size_t i;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }
  
  data = (transform_obj_shift_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  
  for (i = 0; i < problem->number_of_objectives; i++)
    y[i] += data->offset;
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the gradient of the transformed function at x
 */
static void transform_obj_shift_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }
  
  bbob_evaluate_gradient(coco_problem_transformed_get_inner_problem(problem), x, y);
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

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    NULL, "transform_obj_shift");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_obj_shift_evaluate_function;
    
  problem->evaluate_gradient = transform_obj_shift_evaluate_gradient;  /* TODO (NH): why do we need a new function pointer here? */
  
  for (i = 0; i < problem->number_of_objectives; i++)
    problem->best_value[i] += offset;
    
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
 *
 * Currently, the best parameter is transformed correctly only in the simple 
 * cases where M is orthogonal which is always the case for the `bbob`
 * functions. How to code this for general transformations of the above form,
 * see https://github.com/numbbo/coco/issues/814#issuecomment-303724400
 */

#include <assert.h>

#line 17 "code-experiments/src/transform_vars_affine.c"
#line 18 "code-experiments/src/transform_vars_affine.c"

/**
 * @brief Data type for transform_vars_affine.
 */
typedef struct {
  double *M, *b, *x;
} transform_vars_affine_data_t;

/**
 * @brief Evaluates the transformed objective function.
 */
static void transform_vars_affine_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j;
  double *cons_values;
  int is_feasible;
  transform_vars_affine_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed constraint.
 */
static void transform_vars_affine_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j;  
  transform_vars_affine_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

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
  coco_evaluate_constraint(inner_problem, data->x, y);
}

/**
 * @brief Evaluates the gradient of the transformed function.
 */
static void transform_vars_affine_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i, j;
  transform_vars_affine_data_t *data;
  coco_problem_t *inner_problem;
  double *current_row;
  double *gradient;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }
  
  data = (transform_vars_affine_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  gradient = coco_allocate_vector(inner_problem->number_of_variables);
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
    gradient[i] = 0.0;

  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    /* data->M has problem->number_of_variables columns and inner_problem->number_of_variables rows. */
    current_row = data->M + i * problem->number_of_variables;
    data->x[i] = data->b[i];
    for (j = 0; j < problem->number_of_variables; ++j) {
      data->x[i] += x[j] * current_row[j];
    }
  }
  
  bbob_evaluate_gradient(inner_problem, data->x, y);
  
  /* grad_(f o g )(x), where g(x) = M * x + b, equals to
   * M^T * grad_f(M *x + b) 
   */
  for (j = 0; j < inner_problem->number_of_variables; ++j) {
    for (i = 0; i < inner_problem->number_of_variables; ++i) {
       current_row = data->M + i * problem->number_of_variables;
       gradient[j] += y[i] * current_row[j];
    }
  }
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
     y[i] = gradient[i];
  
  current_row = NULL;
  coco_free_memory(gradient);
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

  size_t i, j;
  coco_problem_t *problem;
  transform_vars_affine_data_t *data;
  size_t entries_in_M;

  entries_in_M = inner_problem->number_of_variables * number_of_variables;
  data = (transform_vars_affine_data_t *) coco_allocate_memory(sizeof(*data));
  data->M = coco_duplicate_vector(M, entries_in_M);
  data->b = coco_duplicate_vector(b, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_affine_free, "transform_vars_affine");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_affine_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_vars_affine_evaluate_constraint;
    
  problem->evaluate_gradient = transform_vars_affine_evaluate_gradient;

  /* Update the best parameter by computing
     problem->best_parameter = M^T * (inner_problem->best_parameter - b).

     The update takes place only if the best parameter or b are different than zero
     and the transformation matrix M is orthogonal.
  */
  if (coco_problem_best_parameter_not_zero(inner_problem) || !coco_vector_is_zero(data->b, inner_problem->number_of_variables)) {
    if (!coco_is_orthogonal(data->M, problem->number_of_variables, inner_problem->number_of_variables))
        coco_warning("transform_vars_affine(): rotation matrix is not orthogonal. Best parameter not updated");
    else {
        for (i = 0; i < inner_problem->number_of_variables; ++i) {
            data->x[i] = inner_problem->best_parameter[i] - data->b[i];
        }
        for (i = 0; i < problem->number_of_variables; ++i) {
            problem->best_parameter[i] = 0;
            for (j = 0; j < inner_problem->number_of_variables; ++j) {
                problem->best_parameter[i] += data->M[j * problem->number_of_variables + i] * data->x[j];
            }
        }
    }
  }

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
 * @brief Evaluates the transformed objective function.
 */
static void transform_vars_shift_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double *cons_values;
  int is_feasible;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_shift_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  
  coco_evaluate_function(inner_problem, data->shifted_x, y);
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed constraint function.
 */
static void transform_vars_shift_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_shift_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  coco_evaluate_constraint(inner_problem, data->shifted_x, y);
}

/**
 * @brief Evaluates the gradient of the transformed function at x
 */
static void transform_vars_shift_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_shift_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_shift_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
		  
  for (i = 0; i < problem->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  bbob_evaluate_gradient(inner_problem, data->shifted_x, y);

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

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_shift_free, "transform_vars_shift");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_shift_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0)
    problem->evaluate_constraint = transform_vars_shift_evaluate_constraint;
    
  problem->evaluate_gradient = transform_vars_shift_evaluate_gradient;
  
  /* Update the best parameter */
  for (i = 0; i < problem->number_of_variables; i++)
    problem->best_parameter[i] += data->offset[i];
    
  /* Update the initial solution if any */
  if (problem->initial_solution)
    for (i = 0; i < problem->number_of_variables; i++)
      problem->initial_solution[i] += data->offset[i];
      
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
 * @brief Evaluates the transformed function.
 */
static void transform_vars_asymmetric_evaluate_function(coco_problem_t *problem, 
                                                        const double *x, 
                                                        double *y) {
  size_t i;
  double exponent, *cons_values;
  int is_feasible;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent = 1.0
          + ((data->beta * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0)) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  
  coco_evaluate_function(inner_problem, data->x, y);
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed constraint.
 */
static void transform_vars_asymmetric_evaluate_constraint(coco_problem_t *problem, 
                                                          const double *x, 
                                                          double *y) {
  size_t i;
  double exponent;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

  data = (transform_vars_asymmetric_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  for (i = 0; i < problem->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent = 1.0
          + ((data->beta * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0)) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  coco_evaluate_constraint(inner_problem, data->x, y);
}

static void transform_vars_asymmetric_free(void *thing) {
  transform_vars_asymmetric_data_t *data = (transform_vars_asymmetric_data_t *) thing;
  coco_free_memory(data->x);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_asymmetric(coco_problem_t *inner_problem, const double beta) {
  
  size_t i;
  int is_feasible;
  double alpha, *cons_values;
  transform_vars_asymmetric_data_t *data;
  coco_problem_t *problem;
  
  data = (transform_vars_asymmetric_data_t *) coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->beta = beta;
  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_asymmetric_free, "transform_vars_asymmetric");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_asymmetric_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0) {
	  
    problem->evaluate_constraint = transform_vars_asymmetric_evaluate_constraint;
    
    /* Check if the initial solution remains feasible after
     * the transformation. If not, do a backtracking
     * towards the origin until it becomes feasible.
     */
    if (inner_problem->initial_solution) {
      cons_values = coco_allocate_vector(problem->number_of_constraints);
      is_feasible = coco_is_feasible(problem, inner_problem->initial_solution, cons_values);
      alpha = 0.9;
      i = 0;
      while (!is_feasible) {
        problem->initial_solution[i] *= alpha;
        is_feasible = coco_is_feasible(problem, problem->initial_solution, cons_values);
        i = (i + 1) % inner_problem->number_of_variables;
      }
      coco_free_memory(cons_values);
    }
  }
  
  if (inner_problem->number_of_objectives > 0 && coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_warning("transform_vars_asymmetric(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }
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
  
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

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
 * @brief Evaluates the gradient of the bent cigar function.
 */
static void f_bent_cigar_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  static const double condition = 1.0e6;
  size_t i;

  y[0] = 2.0 * x[0];
  for (i = 1; i < problem->number_of_variables; ++i)
    y[i] = 2.0 * condition * x[i];

}

/**
 * @brief Allocates the basic bent cigar problem.
 */
static coco_problem_t *f_bent_cigar_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("bent cigar function",
      f_bent_cigar_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_bent_cigar_evaluate_gradient;
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

/**
 * @brief Creates the bent cigar problem for the constrained BBOB suite.
 */
static coco_problem_t *f_bent_cigar_cons_bbob_problem_allocate(const size_t function,
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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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

  if (coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_warning("transform_vars_brs(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }
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
 * @brief Evaluates the transformed objective functions.
 */
static void transform_vars_oscillate_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  static const double alpha = 0.1;
  double tmp, base, *oscillated_x, *cons_values;
  int is_feasible;
  size_t i;
  transform_vars_oscillate_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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
  
  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);    
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the transformed constraints.
 */
static void transform_vars_oscillate_evaluate_constraint(coco_problem_t *problem, const double *x, double *y) {
  static const double alpha = 0.1;
  double tmp, base, *oscillated_x;
  size_t i;
  transform_vars_oscillate_data_t *data;
  coco_problem_t *inner_problem;
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_constraints(problem));
  	return;
  }

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
  coco_evaluate_constraint(inner_problem, oscillated_x, y);
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
	
  size_t i;
  int is_feasible;
  double alpha, *cons_values;
  transform_vars_oscillate_data_t *data;
  coco_problem_t *problem;
  data = (transform_vars_oscillate_data_t *) coco_allocate_memory(sizeof(*data));
  data->oscillated_x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, 
    transform_vars_oscillate_free, "transform_vars_oscillate");
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_oscillate_evaluate_function;
    
  if (inner_problem->number_of_constraints > 0) {
    problem->evaluate_constraint = transform_vars_oscillate_evaluate_constraint;
    
    /* Check if the initial solution remains feasible after
     * the transformation. If not, do a backtracking
     * towards the origin until it becomes feasible.
     */
    if (inner_problem->initial_solution) {
      cons_values = coco_allocate_vector(problem->number_of_constraints);
      is_feasible = coco_is_feasible(problem, inner_problem->initial_solution, cons_values);
      alpha = 0.9;
      i = 0;
      while (!is_feasible) {
        problem->initial_solution[i] *= alpha;
        is_feasible = coco_is_feasible(problem, problem->initial_solution, cons_values);
        i = (i + 1) % inner_problem->number_of_variables;
      }
      coco_free_memory(cons_values);
    }
  }
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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
  
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;
    
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
 * @brief Implements the sign function.
 */
double sign(double x) {
  
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

/**
 * @brief Evaluates the gradient of the function "different powers".
 */
static void f_different_powers_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  size_t i;
  double sum = 0.0;
  double aux;

  for (i = 0; i < problem->number_of_variables; ++i) {
    aux = 2.0 + (4.0 * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0);
    sum += pow(fabs(x[i]), aux);
  }
  
  for (i = 0; i < problem->number_of_variables; ++i) {
    aux = 2.0 + (4.0 * (double) (long) i) / ((double) (long) problem->number_of_variables - 1.0);
	 y[i] = 0.5 * (aux)/(sum);
    aux -= 1.0;
    y[i] *= pow(fabs(x[i]), aux) * sign(x[i]);
  }
  
}

/**
 * @brief Allocates the basic different powers problem.
 */
static coco_problem_t *f_different_powers_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("different powers function",
      f_different_powers_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_different_powers_evaluate_gradient;
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

/**
 * @brief Creates the bbob-constrained different powers problem.
 */
static coco_problem_t *f_different_powers_bbob_constrained_problem_allocate(const size_t function,
                                                                const size_t dimension,
                                                                const size_t instance,
                                                                const long rseed,
                                                                const char *problem_id_template,
                                                                const char *problem_name_template) {
  /* Different powers function used in bbob-constrained test suite.
   * In this version, the (unconstrained) optimum, xopt, is set to
   * a distance of 1e-2 to the origin. By doing so, the optimum of
   * the constrained problem is at a "reasonable" distance from
   * the unconstrained one and, hence, the constrained problem is not too easy.
   */

  size_t i;
  double *xopt, fopt, result;
  coco_problem_t *problem = NULL;

  double *M = coco_allocate_vector(dimension * dimension);
  double *b = coco_allocate_vector(dimension);
  double **rot1;

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);

  /* Compute Euclidean norm of xopt */
  /* CAVEAT: this implementation is not ideal for large dimensions */
  result = 0.0;
  for (i = 0; i < dimension; ++i) {
    result += xopt[i] * xopt[i];
  }
  result = sqrt(result);

  /* Scale xopt such that the distance to the origin is 1e-2 */
  for (i = 0; i < dimension; ++i) {
    xopt[i] *= 1e-2 / result;
  }

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
  
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;
    
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
 * @brief Evaluates the gradient of the discus function.
 */
static void f_discus_evaluate_gradient(coco_problem_t *problem, 
                                       const double *x, 
                                       double *y) {

  static const double condition = 1.0e6;
  size_t i;

  y[0] = condition * 2.0 * x[0];
  for (i = 1; i < problem->number_of_variables; ++i)
    y[i] = 2.0 * x[i];

}

/**
 * @brief Allocates the basic discus problem.
 */
static coco_problem_t *f_discus_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("discus function",
      f_discus_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_discus_evaluate_gradient;
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

/**
 * @brief Creates the discus problem for the constrained BBOB suite.
 */
static coco_problem_t *f_discus_cons_bbob_problem_allocate(const size_t function,
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
  unsigned long i;
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
  unsigned long i, idx_swap;
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
  size_t i;
  
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
  
  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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
    
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

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
 * @brief Evaluates the gradient of the ellipsoid function.
 */
static void f_ellipsoid_evaluate_gradient(coco_problem_t *problem, 
                                          const double *x, 
                                          double *y) {

  static const double condition = 1.0e6;
  double exponent;
  size_t i = 0;
  
  for (i = 0; i < problem->number_of_variables; ++i) {
    exponent = 1.0 * (double) (long) i / ((double) (long) problem->number_of_variables - 1.0);
    y[i] = 2.0*pow(condition, exponent) * x[i];
  }
 
}

/**
 * @brief Allocates the basic ellipsoid problem.
 */
static coco_problem_t *f_ellipsoid_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("ellipsoid function",
      f_ellipsoid_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_ellipsoid_evaluate_gradient;
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

/**
 * @brief Creates the ellipsoid problem for the constrained BBOB suite
 */
static coco_problem_t *f_ellipsoid_cons_bbob_problem_allocate(const size_t function,
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
  /* TODO (NH): fopt -= problem->evaluate(all_zeros(dimension)) */
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

  coco_free_memory(xopt);
  return problem;
}

/**
 * @brief Creates the rotated ellipsoid problem for the constrained
 *        BBOB suite
 */
static coco_problem_t *f_ellipsoid_rotated_cons_bbob_problem_allocate(const size_t function,
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
    		(unsigned long) number_of_peaks);
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
      f_katsuura_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
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
  
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;
    
  for (i = 0; i < number_of_variables; ++i) {
    double base, exponent, si;

    base = sqrt(alpha);
    exponent = (double) (long) i / ((double) (long) number_of_variables - 1);
    if (best_parameter[i] > 0.0) {
      si = pow(base, exponent);
    } else {
      si = -pow(base, exponent);
    }
    /* boundary handling */
    if (x[i] * best_parameter[i] < 25.0) {
      result += 5.0 * fabs(si) - si * x[i];
    } else {
      result += 5.0 * fabs(si) - si * best_parameter[i];
    }
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
 * @brief Evaluates the gradient of the linear slope function.
 */
static void f_linear_slope_evaluate_gradient(coco_problem_t *problem, 
                                             const double *x, 
                                             double *y) {

  static const double alpha = 100.0;
  double base, exponent, si;
  size_t i;

  (void)x; /* silence (C89) compiliers */
  for (i = 0; i < problem->number_of_variables; ++i) {
    base = sqrt(alpha);
    exponent = (double) (long) i / ((double) (long) problem->number_of_variables - 1);
    if (problem->best_parameter[i] > 0.0) {
      si = pow(base, exponent);
    } else {
      si = -pow(base, exponent);
    }
    y[i] = -si;
  }
}

/**
 * @brief Allocates the basic linear slope problem.
 */
static coco_problem_t *f_linear_slope_allocate(const size_t number_of_variables, const double *best_parameter) {

  size_t i;
  /* best_parameter will be overwritten below */
  coco_problem_t *problem = coco_problem_allocate_from_scalars("linear slope function",
      f_linear_slope_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_linear_slope_evaluate_gradient;
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
#line 14 "code-experiments/src/f_rastrigin.c"
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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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

/**
 * @brief Evaluates the gradient of the transformed function.
 */
static void transform_vars_conditioning_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_conditioning_data_t *data;
  coco_problem_t *inner_problem;
  double *gradient;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

  data = (transform_vars_conditioning_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  gradient = coco_allocate_vector(inner_problem->number_of_variables);
  
  for (i = 0; i < problem->number_of_variables; ++i) {
    gradient[i] = pow(data->alpha, 0.5 * (double) (long) i / ((double) (long) problem->number_of_variables - 1.0));
    data->x[i] = gradient[i] * x[i];
  }
  bbob_evaluate_gradient(inner_problem, data->x, y);
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
    gradient[i] *= y[i];
  
  for (i = 0; i < inner_problem->number_of_variables; ++i)
    y[i] = gradient[i];
    
  coco_free_memory(gradient);
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
  problem->evaluate_gradient = transform_vars_conditioning_evaluate_gradient;

  if (coco_problem_best_parameter_not_zero(inner_problem)) {
    coco_warning("transform_vars_conditioning(): 'best_parameter' not updated, set to NAN");
    coco_vector_set_to_nan(inner_problem->best_parameter, inner_problem->number_of_variables);
  }  return problem;
}
#line 15 "code-experiments/src/f_rastrigin.c"
#line 16 "code-experiments/src/f_rastrigin.c"
#line 17 "code-experiments/src/f_rastrigin.c"
#line 18 "code-experiments/src/f_rastrigin.c"
#line 19 "code-experiments/src/f_rastrigin.c"
#line 20 "code-experiments/src/f_rastrigin.c"

/**
 * @brief Implements the Rastrigin function without connections to any COCO structures.
 */
static double f_rastrigin_raw(const double *x, const size_t number_of_variables) {

  size_t i = 0;
  double result;
  double sum1 = 0.0, sum2 = 0.0;
    
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

  for (i = 0; i < number_of_variables; ++i) {
    sum1 += cos(coco_two_pi * x[i]);
    sum2 += x[i] * x[i];
  }
  if (coco_is_inf(sum2)) /* cos(inf) -> nan */
    return sum2;
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
 * @brief Evaluates the gradient of the raw Rastrigin function.
 */
static void f_rastrigin_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  size_t i;

  for (i = 0; i < problem->number_of_variables; ++i) {
    y[i] = 2.0 * (10. * coco_pi * sin(coco_two_pi * x[i]) + x[i]);
  }
}

/**
 * @brief Allocates the basic Rastrigin problem.
 */
static coco_problem_t *f_rastrigin_allocate(const size_t number_of_variables) {

  coco_problem_t *problem = coco_problem_allocate_from_scalars("Rastrigin function",
      f_rastrigin_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  /* TODO: make sure the gradient is computed correctly for the rotated Rastrigin */
  problem->evaluate_gradient = f_rastrigin_evaluate_gradient;
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

/**
 * @brief Computes xopt for constrained Rastrigin (alternative to bbob2009_compute_xopt())
 * xopt is a vector of dim uniform random integers
 */
static void f_rastrigin_cons_compute_xopt(double *xopt, const long rseed, const size_t dim) {

  size_t i;

  bbob2009_unif(xopt, dim, rseed);

  for (i = 0; i < dim; ++i) {
    xopt[i] = 10 * xopt[i] - 5;
    xopt[i] = (int) xopt[i];
  }

  /* In case (0, ..., 0) is sampled, set xopt to a different value */
  if (coco_vector_is_zero(xopt, dim))
    for (i = 0; i < dim; ++i) {
        xopt[i] = (int) (i % 9) - 4;
    }
}

/**
 * @brief Creates the Rastrigin problem for the constrained BBOB suite.
 */
static coco_problem_t *f_rastrigin_cons_bbob_problem_allocate(const size_t function,
                                                         const size_t dimension,
                                                         const size_t instance,
                                                         const long rseed,
                                                         const char *problem_id_template,
                                                         const char *problem_name_template) {

  double *xopt, fopt;
  coco_problem_t *problem = NULL;

  xopt = coco_allocate_vector(dimension);
  f_rastrigin_cons_compute_xopt(xopt, rseed, dimension);
  fopt = bbob2009_compute_fopt(function, instance);

  problem = f_rastrigin_allocate(dimension);
  problem = transform_vars_shift(problem, xopt, 0);
  problem = transform_obj_shift(problem, fopt);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "1-separable");

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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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
  size_t i;
  data = (transform_vars_scale_data_t *) coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_scale_free, "transform_vars_scale");
  problem->evaluate_function = transform_vars_scale_evaluate;
  /* Compute best parameter */
  if (data->factor != 0.) {
      for (i = 0; i < problem->number_of_variables; i++) {
          problem->best_parameter[i] /= data->factor;
      }
  } /* else error? */
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
      current_row[column] = factor * rot1[row][column];
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  /* Computation core */
  result = 0.0;
  for (i = 0; i < number_of_variables - 1; ++i) {
    const double tmp = x[i] * x[i] + x[i + 1] * x[i + 1];
    if (coco_is_inf(tmp) && coco_is_nan(sin(50.0 * pow(tmp, 0.1))))  /* sin(inf) -> nan */
      /* the second condition is necessary to pass the integration tests under Windows and Linux */
      return tmp;
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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

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
  /* TODO: implement best_parameter transformation if needed in the case of not zero:
     see also issue #814.
  The correct update of best_parameter seems not too difficult and should not anymore
  break the current implementation of the Schwefel function. 
   coco_warning("transform_vars_z_hat(): 'best_parameter' not updated"); 
 
  This:
  
  size_t i;
  if (problem->best_parameter != NULL)
	for (i = 1; i < problem->number_of_variables; ++i)
	  problem->best_parameter[i] -= 0.25 * (problem->best_parameter[i - 1] - 2.0 * fabs(data->xopt[i - 1]));

  should do, but gives
  COCO INFO: ..., d=2, running: f18.Assertion failed: (about_equal_value(hypervolume, 8.1699208579037619e-05)), function test_coco_archive_extreme_solutions, file ./test_coco_archive.c, line 123.

  */
  if (strstr(coco_problem_get_id(inner_problem), "schwefel") == NULL) {
    coco_warning("transform_vars_z_hat(): 'best_parameter' not updated, set to NAN.");
    coco_vector_set_to_nan(problem->best_parameter, problem->number_of_variables);
  }

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

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
  	coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
  	return;
  }

 data = (transform_vars_x_hat_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  do {
    bbob2009_unif(data->x, problem->number_of_variables, data->seed);

    for (i = 0; i < problem->number_of_variables; ++i) {
      if (data->x[i] < 0.5) {
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
  if (coco_problem_best_parameter_not_zero(problem)) {
    bbob2009_unif(data->x, problem->number_of_variables, data->seed);
	for (i = 0; i < problem->number_of_variables; ++i)
	  if (data->x[i] < 0.5)  /* with probability 1/2 */
		problem->best_parameter[i] *= -1;
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
  size_t i;

  const double condition = 10.;

  double *tmp1 = coco_allocate_vector(dimension);
  double *tmp2 = coco_allocate_vector(dimension);

  xopt = coco_allocate_vector(dimension);
  fopt = bbob2009_compute_fopt(function, instance);
  bbob2009_unif(tmp1, dimension, rseed);
  for (i = 0; i < dimension; ++i) {
    xopt[i] = (tmp1[i] < 0.5 ? -1 : 1) * 0.5 * 4.2096874637;
  }

  for (i = 0; i < dimension; ++i) {
    tmp1[i] = -2 * fabs(xopt[i]);
    tmp2[i] = 2 * fabs(xopt[i]);
  }

  problem = f_schwefel_allocate(dimension);
  problem = transform_obj_shift(problem, fopt);
  problem = transform_vars_scale(problem, 100);
  problem = transform_vars_shift(problem, tmp1, 0);
  /* problem = transform_vars_affine(problem, M, b, dimension); */
  problem = transform_vars_conditioning(problem, condition);
  problem = transform_vars_shift(problem, tmp2, 0);
  problem = transform_vars_z_hat(problem, xopt); /* only for the correct xopt the best_parameter is not changed */
  problem = transform_vars_scale(problem, 2);
  problem = transform_vars_x_hat(problem, rseed);

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "5-weakly-structured");

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
  const double vars_40 = 1; /* generalized: number_of_variables <= 40 ? 1 : number_of_variables / 40.0; */
  size_t i = 0;
  double result;

  assert(number_of_variables > 1);

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

  result = 0.0;
  for (i = coco_double_to_size_t(ceil(vars_40)); i < number_of_variables; ++i) {
    result += x[i] * x[i];
  }
  result = alpha * sqrt(result / vars_40);
  for (i = 0; i < ceil(vars_40); ++i)
    result += x[i] * x[i] / vars_40;

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
    
  if (coco_vector_contains_nan(x, number_of_variables))
    return NAN;

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
 * @brief Evaluates the gradient of the sphere function.
 */
static void f_sphere_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {

  size_t i;

  for (i = 0; i < problem->number_of_variables; ++i) {
    y[i] = 2.0 * x[i];
  }
}

/**
 * @brief Allocates the basic sphere problem.
 */
static coco_problem_t *f_sphere_allocate(const size_t number_of_variables) {
	
  coco_problem_t *problem = coco_problem_allocate_from_scalars("sphere function",
     f_sphere_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.0);
  problem->evaluate_gradient = f_sphere_evaluate_gradient;
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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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

  if (coco_vector_contains_nan(x, number_of_variables))
  	return NAN;

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
      f_weierstrass_evaluate, NULL, number_of_variables, -5.0, 5.0, 0.);
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
  suite = coco_suite_allocate("bbob", 24, 6, dimensions, "year: 2019");

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
  else if ((year == 2016) || (year == 0000)) { /* test case */
    return "1-5,51-60";
  }
  else if (year == 2017) {
    return "1-5,61-70";
  }
  else if (year == 2018) {
    return "1-5,71-80";
  }
  else if (year == 2019) {
    return "1-5,81-90";
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
    coco_error("coco_get_bbob_problem(): cannot retrieve problem f%lu instance %lu in %luD",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension);
    return NULL; /* Never reached */
  }

  return problem;
}

/**
 * TODO: A mock of the function to call large-scale BBOB functions. To be replaced by the right one
 * when it is made available.
 */
static coco_problem_t *mock_coco_get_largescale_problem(const size_t function,
                                                        const size_t dimension,
                                                        const size_t instance) {
  return coco_get_bbob_problem(function, dimension, instance);
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
#line 1 "code-experiments/src/suite_bbob_mixint.c"
/**
 * @file suite_bbob_mixint.c
 * @brief Implementation of a suite with mixed-integer bbob problems (based on the 24 bbob functions,
 * but using their large-scale implementations instead of the original ones).
 */

#line 8 "code-experiments/src/suite_bbob_mixint.c"
#line 9 "code-experiments/src/suite_bbob_mixint.c"
#line 1 "code-experiments/src/transform_vars_discretize.c"
/**
 * @file transform_vars_discretize.c
 *
 * @brief Implementation of transforming a continuous problem to a mixed-integer problem by making some
 * of its variables discrete. The integer variables are considered as bounded (any variable outside the
 * decision space is mapped to the closest boundary point), while the continuous ones are treated as
 * unbounded.
 *
 * @note The first problem->number_of_integer_variables are integer, while the rest are continuous.
 *
 * The discretization works as follows. Consider the case where the interval [l, u] of the inner problem
 * needs to be discretized to n integer values of the outer problem. First, [l, u] is discretized to n
 * integers by placing the integers so that there is a (u-l)/(n+1) distance between them (and the border
 * points). Then, the transformation is shifted so that the optimum aligns with the closest integer. In
 * this way, we make sure that the all the shifted points are still within [l, u].
 *
 * When evaluating such a problem, the x values of the integer variables are first discretized. Any value
 * x < 0 is mapped to 0 and any value x > (n-1) is mapped to (n-1).
 */

#include <assert.h>

#line 24 "code-experiments/src/transform_vars_discretize.c"
#line 25 "code-experiments/src/transform_vars_discretize.c"

/**
 * @brief Data type for transform_vars_discretize.
 */
typedef struct {
  double *offset;
} transform_vars_discretize_data_t;

/**
 * @brief Evaluates the transformed objective function.
 */
static void transform_vars_discretize_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_vars_discretize_data_t *data;
  coco_problem_t *inner_problem;
  double *discretized_x;
  double l, u, inner_l, inner_u, outer_l, outer_u;
  int n;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  data = (transform_vars_discretize_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* Transform x to fit in the discretized space */
  discretized_x = coco_duplicate_vector(x, problem->number_of_variables);
  for (i = 0; i < problem->number_of_integer_variables; ++i) {
    outer_l = problem->smallest_values_of_interest[i];
    outer_u = problem->largest_values_of_interest[i];
    l = inner_problem->smallest_values_of_interest[i];
    u = inner_problem->largest_values_of_interest[i];
    n = coco_double_to_int(outer_u) - coco_double_to_int(outer_l) + 1; /* number of integer values in this coordinate */
    assert(n > 1);
    inner_l = l + (u - l) / (n + 1);
    inner_u = u - (u - l) / (n + 1);
    /* Make sure you the bounds are respected */
    discretized_x[i] = coco_double_round(x[i]);
    if (discretized_x[i] < outer_l)
      discretized_x[i] = outer_l;
    if (discretized_x[i] > outer_u)
      discretized_x[i] = outer_u;
    discretized_x[i] = inner_l + (inner_u - inner_l) * (discretized_x[i] - outer_l) / (outer_u - outer_l) - data->offset[i];
  }

  coco_evaluate_function(inner_problem, discretized_x, y);
  coco_free_memory(discretized_x);
}

/**
 * @brief Frees the data object.
 */
static void transform_vars_discretize_free(void *thing) {
  transform_vars_discretize_data_t *data = (transform_vars_discretize_data_t *) thing;
  coco_free_memory(data->offset);
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_vars_discretize(coco_problem_t *inner_problem,
                                                 const double *smallest_values_of_interest,
                                                 const double *largest_values_of_interest,
                                                 const size_t number_of_integer_variables) {
  transform_vars_discretize_data_t *data;
  coco_problem_t *problem = NULL;
  double l, u, inner_l, inner_u, outer_l, outer_u;
  double outer_xopt, inner_xopt, inner_approx_xopt;
  int n;
  size_t i;

  data = (transform_vars_discretize_data_t *) coco_allocate_memory(sizeof(*data));
  data->offset = coco_allocate_vector(inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, data, transform_vars_discretize_free, "transform_vars_discretize");
  assert(number_of_integer_variables > 0);
  problem->number_of_integer_variables = number_of_integer_variables;

  for (i = 0; i < problem->number_of_variables; i++) {
    assert(smallest_values_of_interest[i] < largest_values_of_interest[i]);
    problem->smallest_values_of_interest[i] = smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] = largest_values_of_interest[i];
    data->offset[i] = 0;
    if (i < number_of_integer_variables) {
      /* Compute the offset for integer variables */
      outer_l = problem->smallest_values_of_interest[i];
      outer_u = problem->largest_values_of_interest[i];
      l = inner_problem->smallest_values_of_interest[i];
      u = inner_problem->largest_values_of_interest[i];
      n = coco_double_to_int(outer_u) - coco_double_to_int(outer_l) + 1; /* number of integer values */
      assert(n > 1);
      inner_l = l + (u - l) / (n + 1);
      inner_u = u - (u - l) / (n + 1);
      /* Find the location of the optimum in the coordinates of the outer problem */
      inner_xopt = inner_problem->best_parameter[i];
      outer_xopt = outer_l + (outer_u - outer_l) * (inner_xopt - inner_l) / (inner_u - inner_l);
      outer_xopt = coco_double_round(outer_xopt);
      /* Make sure you the bounds are respected */
      if (outer_xopt < outer_l)
        outer_xopt = outer_l;
      if (outer_xopt > outer_u)
        outer_xopt = outer_u;
      problem->best_parameter[i] = outer_xopt;
      /* Find the value corresponding to outer_xopt in the coordinates of the inner problem */
      inner_approx_xopt = inner_l + (inner_u - inner_l) * (outer_xopt - outer_l) / (outer_u - outer_l);
      /* Compute the difference between the inner_approx_xopt and inner_xopt */
      data->offset[i] = inner_approx_xopt - inner_xopt;
    }
  }
    
  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_vars_discretize_evaluate_function;

  if (problem->number_of_constraints > 0)
    coco_error("transform_vars_discretize(): Constraints not supported yet.");

  problem->evaluate_constraint = NULL; /* TODO? */
  problem->evaluate_gradient = NULL;   /* TODO? */
      
  return problem;
}
#line 10 "code-experiments/src/suite_bbob_mixint.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob-mixint suite.
 */
static coco_suite_t *suite_bbob_mixint_initialize(const char *suite_name) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 5, 10, 20, 40, 80, 160 };
  /* TODO: Use also dimensions 80 and 160 (change the 4 below into a 6) */
  suite = coco_suite_allocate(suite_name, 24, 4, dimensions, "year: 2019");

  return suite;
}

/**
 * @brief Creates and returns a mixed-integer bbob problem without needing the actual bbob-mixint
 * suite.
 *
 * @param function Function
 * @param dimension Dimension
 * @param instance Instance
 * @param coco_get_problem_function The function that is used to access the continuous problem.
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *coco_get_bbob_mixint_problem(const size_t function,
                                                    const size_t dimension,
                                                    const size_t instance,
                                                    const coco_get_problem_function_t coco_get_problem_function,
                                                    const char *suite_name) {
  coco_problem_t *problem = NULL;

  /* The cardinality of variables (0 = continuous variables should always come last) */
  /* TODO: Use just one (and delete the suite_name parameter) */
  const size_t variable_cardinality_1[] = { 2, 4, 8, 16, 0 };
  const size_t variable_cardinality_2[] = { 2, 6, 18, 0, 0 };

  double *smallest_values_of_interest = coco_allocate_vector(dimension);
  double *largest_values_of_interest = coco_allocate_vector(dimension);
  char *inner_problem_id;

  size_t i, j;
  size_t cardinality = 0;
  size_t num_integer = dimension;
  if (dimension % 5 != 0)
    coco_error("coco_get_bbob_mixint_problem(): dimension %lu not supported for suite_bbob_mixint", dimension);

  /* Sets the ROI according to the given cardinality of variables */
  for (i = 0; i < dimension; i++) {
    j = i / (dimension / 5);
    if (strcmp(suite_name, "bbob-mixint-1") == 0)
      cardinality = variable_cardinality_1[j];
    else
      cardinality = variable_cardinality_2[j];
    if (cardinality == 0) {
      smallest_values_of_interest[i] = -5;
      largest_values_of_interest[i] = 5;
      if (num_integer == dimension)
        num_integer = i;
    }
    else {
      smallest_values_of_interest[i] = 0;
      largest_values_of_interest[i] = (double)cardinality - 1;
    }
  }

  problem = coco_get_problem_function(function, dimension, instance);

  assert(problem != NULL);
  inner_problem_id = problem->problem_id;

  problem = transform_vars_discretize(problem, smallest_values_of_interest,
      largest_values_of_interest, num_integer);

  coco_problem_set_id(problem, "bbob-mixint_f%03lu_i%02lu_d%02lu", function, instance, dimension);
  coco_problem_set_name(problem, "mixint(%s)", inner_problem_id);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}

/**
 * @brief Returns the problem from the bbob-mixint suite that corresponds to the given parameters.
 *
 * Uses large-scale bbob functions if dimension is equal or larger than the hard-coded dim_large_scale
 * value (50).
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_bbob_mixint_get_problem(coco_suite_t *suite,
                                                     const size_t function_idx,
                                                     const size_t dimension_idx,
                                                     const size_t instance_idx) {

  coco_problem_t *problem = NULL;
  const size_t dim_large_scale = 50; /* Switch to large-scale functions for dimensions over 50 */

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  if (dimension < dim_large_scale)
    problem = coco_get_bbob_mixint_problem(function, dimension, instance, coco_get_bbob_problem,
        suite->suite_name);
  else
    problem = coco_get_bbob_mixint_problem(function, dimension, instance, mock_coco_get_largescale_problem,
        suite->suite_name);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
#line 20 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/suite_biobj.c"
/**
 * @file suite_biobj.c
 * @brief Implementation of two bi-objective suites created by combining two single-objective problems
 * from the bbob suite:
 * - bbob-biobj contains 55 functions and 6 dimensions
 * - bbob-biobj-ext contains 55 + 37 functions and 6 dimensions
 *
 * The 55 functions of the bbob-biobj suite are created by combining any two single-objective bbob functions
 * i,j (where i<j) from a subset of 10 functions.
 *
 * The first 55 functions of the bbob-biobj-ext suite are the same as in the original bbob-biobj test suite
 * to which 37 functions are added. Those additional functions are constructed by combining all not yet
 * contained in-group combinations (i,j) of single-objective bbob functions i and j such that i<j (i.e. in
 * particular not all combinations (i,i) are included in this bbob-biobj-ext suite), with the exception of
 * the Weierstrass function (f16) for which the optimum is not unique and thus a nadir point is difficult
 * to compute, see http://numbbo.github.io/coco-doc/bbob-biobj/functions/ for details.
 *
 * @note See file suite_biobj_utilities.c for the implementation of the bi-objective problems and the handling
 * of new instances.
 */

#line 23 "code-experiments/src/suite_biobj.c"
#line 1 "code-experiments/src/mo_utilities.c"
/**
 * @file mo_utilities.c
 * @brief Definitions of miscellaneous functions used for multi-objective problems.
 */

#include <stdlib.h>
#include <stdio.h>
#line 9 "code-experiments/src/mo_utilities.c"

/**
 * @brief Precision used when comparing multi-objective solutions.
 *
 * Two solutions are considered equal in objective space when their normalized difference is smaller than
 * mo_precision.
 *
 * @note mo_precision needs to be smaller than mo_discretization
 */
static const double mo_precision = 1e-13;

/**
 * @brief Discretization interval used for rounding normalized multi-objective solutions.
 *
 * @note mo_discretization needs to be larger than mo_precision
 */
static const double mo_discretization = 5 * 1e-13;

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
 * @brief Creates a rounded normalized version of the given solution w.r.t. the given ROI.
 *
 * If the solution seems to be better than the extremes it is corrected (2 objectives are assumed).
 * The caller is responsible for freeing the allocated memory using coco_free_memory().
 */
static double *mo_normalize(const double *y, const double *ideal, const double *nadir, const size_t num_obj) {

  size_t i;
  double *normalized_y = coco_allocate_vector(num_obj);

  for (i = 0; i < num_obj; i++) {
    assert((nadir[i] - ideal[i]) > mo_discretization);
    normalized_y[i] = (y[i] - ideal[i]) / (nadir[i] - ideal[i]);
    normalized_y[i] = coco_double_round(normalized_y[i] / mo_discretization) * mo_discretization;
    if (normalized_y[i] < 0) {
      coco_debug("Adjusting %.15e to %.15e", y[i], ideal[i]);
      normalized_y[i] = 0;
    }
  }

  for (i = 0; i < num_obj; i++) {
    assert(num_obj == 2);
    if (coco_double_almost_equal(normalized_y[i], 0, mo_precision) && (normalized_y[1-i] < 1)) {
      coco_debug("Adjusting %.15e to %.15e", y[1-i], nadir[1-i]);
      normalized_y[1-i] = 1;
    }
  }

  return normalized_y;
}

/**
 * @brief Checks the dominance relation in the unconstrained minimization case between two normalized
 * solutions in the objective space.
 *
 * If two values are closer together than mo_precision, they are treated as equal.
 *
 * @return
 *  1 if normalized_y1 dominates normalized_y2 <br>
 *  0 if normalized_y1 and normalized_y2 are non-dominated <br>
 * -1 if normalized_y2 dominates normalized_y1 <br>
 * -2 if normalized_y1 is identical to normalized_y2
 */
static int mo_get_dominance(const double *normalized_y1, const double *normalized_y2, const size_t num_obj) {

  size_t i;
  int flag1 = 0;
  int flag2 = 0;

  for (i = 0; i < num_obj; i++) {
    if (coco_double_almost_equal(normalized_y1[i], normalized_y2[i], mo_precision)) {
      continue;
    } else if (normalized_y1[i] < normalized_y2[i]) {
      flag1 = 1;
    } else if (normalized_y1[i] > normalized_y2[i]) {
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
 * @brief Checks whether the normalized solution is within [0, 1]^num_obj.
 */
static int mo_is_within_ROI(const double *normalized_y, const size_t num_obj) {

  size_t i;
  int within = 1;

  for (i = 0; i < num_obj; i++) {
    if (coco_double_almost_equal(normalized_y[i], 0, mo_precision) ||
        coco_double_almost_equal(normalized_y[i], 1, mo_precision) ||
        (normalized_y[i] > 0 && normalized_y[i] < 1))
      continue;
    else
      within = 0;
  }
  return within;
}

/**
 * @brief Computes and returns the minimal normalized distance of the point normalized_y from the ROI
 * (equals 0 if within the ROI).
 *
 *  @note Assumes num_obj = 2 and normalized_y >= 0
 */
static double mo_get_distance_to_ROI(const double *normalized_y, const size_t num_obj) {

  double diff_0, diff_1;

  if (mo_is_within_ROI(normalized_y, num_obj))
    return 0;

  assert(num_obj == 2);
  assert(normalized_y[0] >= 0);
  assert(normalized_y[1] >= 0);

  diff_0 = normalized_y[0] - 1;
  diff_1 = normalized_y[1] - 1;
  if ((diff_0 > 0) && (diff_1 > 0)) {
    return sqrt(pow(diff_0, 2) + pow(diff_1, 2));
  }
  else if (diff_0 > 0)
    return diff_0;
  else
    return diff_1;
}
#line 24 "code-experiments/src/suite_biobj.c"
#line 1 "code-experiments/src/suite_biobj_utilities.c"
/**
 * @file suite_biobj_utilities.c
 * @brief Implementation of some functions (mostly handling instances) used by the bi-objective suites.
 *
 * @note Because some bi-objective problems constructed from two single-objective ones have a single optimal
 * value, some care must be taken when selecting the instances. The already verified instances are stored in
 * suite_biobj_instances. If a new instance of the problem is called, a check ensures that the two underlying
 * single-objective instances create a true bi-objective problem. However, these new instances need to be
 * manually added to suite_biobj_instances, otherwise they will be computed each time the suite constructor
 * is invoked with these instances.
 */

#line 14 "code-experiments/src/suite_biobj_utilities.c"
#line 1 "code-experiments/src/suite_biobj_best_values_hyp.c"
/**
 * @file suite_biobj_best_values_hyp.c
 * @brief Contains the best known hypervolume values for the bob-biobj and bbob-biobj-ext
 * suite's problems.
 * @note For now, the hypervolume reference values for the problems not in the bbob-biobj
 * suite are 1.0 as well as for the new instances larger than 10 (as of 2017/01/20).
 */

/**
 * @brief The best known hypervolume values for the bbob-biobj and bbob-biobj-ext suite problems.
 *
 * @note Because this file is used for automatically retrieving the existing best hypervolume values for
 * pre-processing purposes, its formatting should not be altered. This means that there must be exactly one
 * string per line, the first string appearing on the next line after "static const char..." (no comments 
 * allowed in between). Nothing should be placed on the last line (line with };).
 */
static const char *suite_biobj_best_values_hyp[] = { /* Best values on 29.01.2017 16:30:00, copied from: best values current data, 10.07.2016 */
  "bbob-biobj_f01_i01_d02 0.833332923849452",
  "bbob-biobj_f01_i01_d03 0.833332590193468",
  "bbob-biobj_f01_i01_d05 0.833332871970114",
  "bbob-biobj_f01_i01_d10 0.833332983507963",
  "bbob-biobj_f01_i01_d20 0.833332656879612",
  "bbob-biobj_f01_i01_d40 0.833314650345829",
  "bbob-biobj_f01_i02_d02 0.833332931307885",
  "bbob-biobj_f01_i02_d03 0.833332733855334",
  "bbob-biobj_f01_i02_d05 0.833332480608307",
  "bbob-biobj_f01_i02_d10 0.833332975206005",
  "bbob-biobj_f01_i02_d20 0.833333117544167",
  "bbob-biobj_f01_i02_d40 0.833314493712438",
  "bbob-biobj_f01_i03_d02 0.833332922354197",
  "bbob-biobj_f01_i03_d03 0.833332806949774",
  "bbob-biobj_f01_i03_d05 0.833332842547238",
  "bbob-biobj_f01_i03_d10 0.833332977573570",
  "bbob-biobj_f01_i03_d20 0.833333116146967",
  "bbob-biobj_f01_i03_d40 0.833314058645234",
  "bbob-biobj_f01_i04_d02 0.833332867309178",
  "bbob-biobj_f01_i04_d03 0.833332594337647",
  "bbob-biobj_f01_i04_d05 0.833332839988488",
  "bbob-biobj_f01_i04_d10 0.833332988612530",
  "bbob-biobj_f01_i04_d20 0.833333104916722",
  "bbob-biobj_f01_i04_d40 0.833314378132203",
  "bbob-biobj_f01_i05_d02 0.833332945296347",
  "bbob-biobj_f01_i05_d03 0.833332652221029",
  "bbob-biobj_f01_i05_d05 0.833332837022086",
  "bbob-biobj_f01_i05_d10 0.833332990781764",
  "bbob-biobj_f01_i05_d20 0.833333113205696",
  "bbob-biobj_f01_i05_d40 0.833314458103617",
  "bbob-biobj_f01_i06_d02 0.833332799189226",
  "bbob-biobj_f01_i06_d03 0.833332521560045",
  "bbob-biobj_f01_i06_d05 0.833332854255201",
  "bbob-biobj_f01_i06_d10 0.833332431471356",
  "bbob-biobj_f01_i06_d20 0.833333105010663",
  "bbob-biobj_f01_i06_d40 0.833140980202184",
  "bbob-biobj_f01_i07_d02 0.833332799479431",
  "bbob-biobj_f01_i07_d03 0.833332509886665",
  "bbob-biobj_f01_i07_d05 0.833332870930573",
  "bbob-biobj_f01_i07_d10 0.833332985899831",
  "bbob-biobj_f01_i07_d20 0.833332882157948",
  "bbob-biobj_f01_i07_d40 0.833025632391428",
  "bbob-biobj_f01_i08_d02 0.833332644870580",
  "bbob-biobj_f01_i08_d03 0.833332802814295",
  "bbob-biobj_f01_i08_d05 0.833332837685789",
  "bbob-biobj_f01_i08_d10 0.833332980221363",
  "bbob-biobj_f01_i08_d20 0.833333103071380",
  "bbob-biobj_f01_i08_d40 0.833065287510734",
  "bbob-biobj_f01_i09_d02 0.833332608159240",
  "bbob-biobj_f01_i09_d03 0.833332813941859",
  "bbob-biobj_f01_i09_d05 0.833332858149369",
  "bbob-biobj_f01_i09_d10 0.833332976673314",
  "bbob-biobj_f01_i09_d20 0.833333112988244",
  "bbob-biobj_f01_i09_d40 0.833017475872989",
  "bbob-biobj_f01_i10_d02 0.833332793837110",
  "bbob-biobj_f01_i10_d03 0.833332797396180",
  "bbob-biobj_f01_i10_d05 0.833332479626792",
  "bbob-biobj_f01_i10_d10 0.833332975447230",
  "bbob-biobj_f01_i10_d20 0.833332903259388",
  "bbob-biobj_f01_i10_d40 0.833049367302132",
  "bbob-biobj_f01_i11_d02 1.0",
  "bbob-biobj_f01_i11_d03 1.0",
  "bbob-biobj_f01_i11_d05 1.0",
  "bbob-biobj_f01_i11_d10 1.0",
  "bbob-biobj_f01_i11_d20 1.0",
  "bbob-biobj_f01_i11_d40 1.0",
  "bbob-biobj_f01_i12_d02 1.0",
  "bbob-biobj_f01_i12_d03 1.0",
  "bbob-biobj_f01_i12_d05 1.0",
  "bbob-biobj_f01_i12_d10 1.0",
  "bbob-biobj_f01_i12_d20 1.0",
  "bbob-biobj_f01_i12_d40 1.0",
  "bbob-biobj_f01_i13_d02 1.0",
  "bbob-biobj_f01_i13_d03 1.0",
  "bbob-biobj_f01_i13_d05 1.0",
  "bbob-biobj_f01_i13_d10 1.0",
  "bbob-biobj_f01_i13_d20 1.0",
  "bbob-biobj_f01_i13_d40 1.0",
  "bbob-biobj_f01_i14_d02 1.0",
  "bbob-biobj_f01_i14_d03 1.0",
  "bbob-biobj_f01_i14_d05 1.0",
  "bbob-biobj_f01_i14_d10 1.0",
  "bbob-biobj_f01_i14_d20 1.0",
  "bbob-biobj_f01_i14_d40 1.0",
  "bbob-biobj_f01_i15_d02 1.0",
  "bbob-biobj_f01_i15_d03 1.0",
  "bbob-biobj_f01_i15_d05 1.0",
  "bbob-biobj_f01_i15_d10 1.0",
  "bbob-biobj_f01_i15_d20 1.0",
  "bbob-biobj_f01_i15_d40 1.0",
  "bbob-biobj_f02_i01_d02 0.995822561023240",
  "bbob-biobj_f02_i01_d03 0.879310431864156",
  "bbob-biobj_f02_i01_d05 0.953384084311242",
  "bbob-biobj_f02_i01_d10 0.978189643915096",
  "bbob-biobj_f02_i01_d20 0.951903142351403",
  "bbob-biobj_f02_i01_d40 0.949824650187359",
  "bbob-biobj_f02_i02_d02 0.917892814584968",
  "bbob-biobj_f02_i02_d03 0.981135909601684",
  "bbob-biobj_f02_i02_d05 0.966473231322075",
  "bbob-biobj_f02_i02_d10 0.954022219761768",
  "bbob-biobj_f02_i02_d20 0.980915316952865",
  "bbob-biobj_f02_i02_d40 0.967820288072087",
  "bbob-biobj_f02_i03_d02 0.990979165827488",
  "bbob-biobj_f02_i03_d03 0.952601084184924",
  "bbob-biobj_f02_i03_d05 0.950364402842151",
  "bbob-biobj_f02_i03_d10 0.890666848167831",
  "bbob-biobj_f02_i03_d20 0.971903877089018",
  "bbob-biobj_f02_i03_d40 0.962240289831067",
  "bbob-biobj_f02_i04_d02 0.956280253169872",
  "bbob-biobj_f02_i04_d03 0.889457145687917",
  "bbob-biobj_f02_i04_d05 0.881363552894039",
  "bbob-biobj_f02_i04_d10 0.972589938217805",
  "bbob-biobj_f02_i04_d20 0.977355424486821",
  "bbob-biobj_f02_i04_d40 0.976521983119385",
  "bbob-biobj_f02_i05_d02 0.960749326556107",
  "bbob-biobj_f02_i05_d03 0.924022607296279",
  "bbob-biobj_f02_i05_d05 0.861593353038038",
  "bbob-biobj_f02_i05_d10 0.933915127767411",
  "bbob-biobj_f02_i05_d20 0.962923115039428",
  "bbob-biobj_f02_i05_d40 0.960311958395483",
  "bbob-biobj_f02_i06_d02 0.875386399710729",
  "bbob-biobj_f02_i06_d03 0.948381052595089",
  "bbob-biobj_f02_i06_d05 0.910439280190353",
  "bbob-biobj_f02_i06_d10 0.977165791278565",
  "bbob-biobj_f02_i06_d20 0.971870013922785",
  "bbob-biobj_f02_i06_d40 0.972319995645285",
  "bbob-biobj_f02_i07_d02 0.832403909990292",
  "bbob-biobj_f02_i07_d03 0.875220188815227",
  "bbob-biobj_f02_i07_d05 0.912587284654367",
  "bbob-biobj_f02_i07_d10 0.958804666468614",
  "bbob-biobj_f02_i07_d20 0.977380321260569",
  "bbob-biobj_f02_i07_d40 0.968045916312069",
  "bbob-biobj_f02_i08_d02 0.829465051906365",
  "bbob-biobj_f02_i08_d03 0.974523475625668",
  "bbob-biobj_f02_i08_d05 0.936067436058678",
  "bbob-biobj_f02_i08_d10 0.922560458802353",
  "bbob-biobj_f02_i08_d20 0.967069143117920",
  "bbob-biobj_f02_i08_d40 0.963122260805039",
  "bbob-biobj_f02_i09_d02 0.991879825579866",
  "bbob-biobj_f02_i09_d03 0.978472046555062",
  "bbob-biobj_f02_i09_d05 0.942549982385051",
  "bbob-biobj_f02_i09_d10 0.953997088214724",
  "bbob-biobj_f02_i09_d20 0.961031595278923",
  "bbob-biobj_f02_i09_d40 0.973807415276184",
  "bbob-biobj_f02_i10_d02 0.947432430910857",
  "bbob-biobj_f02_i10_d03 0.991603200976025",
  "bbob-biobj_f02_i10_d05 0.948442017309507",
  "bbob-biobj_f02_i10_d10 0.965514596960421",
  "bbob-biobj_f02_i10_d20 0.970944129483383",
  "bbob-biobj_f02_i10_d40 0.963693463826435",
  "bbob-biobj_f02_i11_d02 1.0",
  "bbob-biobj_f02_i11_d03 1.0",
  "bbob-biobj_f02_i11_d05 1.0",
  "bbob-biobj_f02_i11_d10 1.0",
  "bbob-biobj_f02_i11_d20 1.0",
  "bbob-biobj_f02_i11_d40 1.0",
  "bbob-biobj_f02_i12_d02 1.0",
  "bbob-biobj_f02_i12_d03 1.0",
  "bbob-biobj_f02_i12_d05 1.0",
  "bbob-biobj_f02_i12_d10 1.0",
  "bbob-biobj_f02_i12_d20 1.0",
  "bbob-biobj_f02_i12_d40 1.0",
  "bbob-biobj_f02_i13_d02 1.0",
  "bbob-biobj_f02_i13_d03 1.0",
  "bbob-biobj_f02_i13_d05 1.0",
  "bbob-biobj_f02_i13_d10 1.0",
  "bbob-biobj_f02_i13_d20 1.0",
  "bbob-biobj_f02_i13_d40 1.0",
  "bbob-biobj_f02_i14_d02 1.0",
  "bbob-biobj_f02_i14_d03 1.0",
  "bbob-biobj_f02_i14_d05 1.0",
  "bbob-biobj_f02_i14_d10 1.0",
  "bbob-biobj_f02_i14_d20 1.0",
  "bbob-biobj_f02_i14_d40 1.0",
  "bbob-biobj_f02_i15_d02 1.0",
  "bbob-biobj_f02_i15_d03 1.0",
  "bbob-biobj_f02_i15_d05 1.0",
  "bbob-biobj_f02_i15_d10 1.0",
  "bbob-biobj_f02_i15_d20 1.0",
  "bbob-biobj_f02_i15_d40 1.0",
  "bbob-biobj_f03_i01_d02 0.811764227919237",
  "bbob-biobj_f03_i01_d03 0.974987686406398",
  "bbob-biobj_f03_i01_d05 0.846755821475001",
  "bbob-biobj_f03_i01_d10 0.916897326744567",
  "bbob-biobj_f03_i01_d20 0.887093378512445",
  "bbob-biobj_f03_i01_d40 0.877301209968849",
  "bbob-biobj_f03_i02_d02 0.870719320227517",
  "bbob-biobj_f03_i02_d03 0.845126230251145",
  "bbob-biobj_f03_i02_d05 0.961675547477218",
  "bbob-biobj_f03_i02_d10 0.980164167551993",
  "bbob-biobj_f03_i02_d20 0.950054357943145",
  "bbob-biobj_f03_i02_d40 0.941149877177212",
  "bbob-biobj_f03_i03_d02 0.843027854657315",
  "bbob-biobj_f03_i03_d03 0.860130827007275",
  "bbob-biobj_f03_i03_d05 0.836876534262302",
  "bbob-biobj_f03_i03_d10 0.985500464678158",
  "bbob-biobj_f03_i03_d20 0.867661654006730",
  "bbob-biobj_f03_i03_d40 0.885731348325649",
  "bbob-biobj_f03_i04_d02 0.816337758672414",
  "bbob-biobj_f03_i04_d03 0.965017447879510",
  "bbob-biobj_f03_i04_d05 0.832979226365889",
  "bbob-biobj_f03_i04_d10 0.908658717348390",
  "bbob-biobj_f03_i04_d20 0.932782033778145",
  "bbob-biobj_f03_i04_d40 0.911122582165054",
  "bbob-biobj_f03_i05_d02 0.854019720086261",
  "bbob-biobj_f03_i05_d03 0.879236357635921",
  "bbob-biobj_f03_i05_d05 0.959277121197834",
  "bbob-biobj_f03_i05_d10 0.881588676730308",
  "bbob-biobj_f03_i05_d20 0.875617568043751",
  "bbob-biobj_f03_i05_d40 0.897748784259607",
  "bbob-biobj_f03_i06_d02 0.830343055064749",
  "bbob-biobj_f03_i06_d03 0.944959812797630",
  "bbob-biobj_f03_i06_d05 0.970527337180708",
  "bbob-biobj_f03_i06_d10 0.844333191961208",
  "bbob-biobj_f03_i06_d20 0.910362127777293",
  "bbob-biobj_f03_i06_d40 0.913082625433224",
  "bbob-biobj_f03_i07_d02 0.868151338090149",
  "bbob-biobj_f03_i07_d03 0.869382169348494",
  "bbob-biobj_f03_i07_d05 0.903912429165581",
  "bbob-biobj_f03_i07_d10 0.845699441056724",
  "bbob-biobj_f03_i07_d20 0.923299654935261",
  "bbob-biobj_f03_i07_d40 0.907453227472945",
  "bbob-biobj_f03_i08_d02 0.990911524871468",
  "bbob-biobj_f03_i08_d03 0.835187869372628",
  "bbob-biobj_f03_i08_d05 0.922645354726611",
  "bbob-biobj_f03_i08_d10 0.886866796169689",
  "bbob-biobj_f03_i08_d20 0.924597832045464",
  "bbob-biobj_f03_i08_d40 0.901676028146734",
  "bbob-biobj_f03_i09_d02 0.813085530790926",
  "bbob-biobj_f03_i09_d03 0.929449915590067",
  "bbob-biobj_f03_i09_d05 0.852770622828177",
  "bbob-biobj_f03_i09_d10 0.988149709245385",
  "bbob-biobj_f03_i09_d20 0.891263868986904",
  "bbob-biobj_f03_i09_d40 0.960313249884054",
  "bbob-biobj_f03_i10_d02 0.806842793172941",
  "bbob-biobj_f03_i10_d03 0.889858712567055",
  "bbob-biobj_f03_i10_d05 0.872339918273572",
  "bbob-biobj_f03_i10_d10 0.838785952959289",
  "bbob-biobj_f03_i10_d20 0.932837871417158",
  "bbob-biobj_f03_i10_d40 0.919291853546905",
  "bbob-biobj_f03_i11_d02 1.0",
  "bbob-biobj_f03_i11_d03 1.0",
  "bbob-biobj_f03_i11_d05 1.0",
  "bbob-biobj_f03_i11_d10 1.0",
  "bbob-biobj_f03_i11_d20 1.0",
  "bbob-biobj_f03_i11_d40 1.0",
  "bbob-biobj_f03_i12_d02 1.0",
  "bbob-biobj_f03_i12_d03 1.0",
  "bbob-biobj_f03_i12_d05 1.0",
  "bbob-biobj_f03_i12_d10 1.0",
  "bbob-biobj_f03_i12_d20 1.0",
  "bbob-biobj_f03_i12_d40 1.0",
  "bbob-biobj_f03_i13_d02 1.0",
  "bbob-biobj_f03_i13_d03 1.0",
  "bbob-biobj_f03_i13_d05 1.0",
  "bbob-biobj_f03_i13_d10 1.0",
  "bbob-biobj_f03_i13_d20 1.0",
  "bbob-biobj_f03_i13_d40 1.0",
  "bbob-biobj_f03_i14_d02 1.0",
  "bbob-biobj_f03_i14_d03 1.0",
  "bbob-biobj_f03_i14_d05 1.0",
  "bbob-biobj_f03_i14_d10 1.0",
  "bbob-biobj_f03_i14_d20 1.0",
  "bbob-biobj_f03_i14_d40 1.0",
  "bbob-biobj_f03_i15_d02 1.0",
  "bbob-biobj_f03_i15_d03 1.0",
  "bbob-biobj_f03_i15_d05 1.0",
  "bbob-biobj_f03_i15_d10 1.0",
  "bbob-biobj_f03_i15_d20 1.0",
  "bbob-biobj_f03_i15_d40 1.0",
  "bbob-biobj_f04_i01_d02 0.965338726794382",
  "bbob-biobj_f04_i01_d03 0.968687951515035",
  "bbob-biobj_f04_i01_d05 0.943989863409768",
  "bbob-biobj_f04_i01_d10 0.944838255687357",
  "bbob-biobj_f04_i01_d20 0.935875503168791",
  "bbob-biobj_f04_i01_d40 0.938346564293152",
  "bbob-biobj_f04_i02_d02 0.970390735576566",
  "bbob-biobj_f04_i02_d03 0.955014817645219",
  "bbob-biobj_f04_i02_d05 0.963230490151800",
  "bbob-biobj_f04_i02_d10 0.954675551328523",
  "bbob-biobj_f04_i02_d20 0.941764787741878",
  "bbob-biobj_f04_i02_d40 0.942548357947242",
  "bbob-biobj_f04_i03_d02 0.971131699210568",
  "bbob-biobj_f04_i03_d03 0.910481413771928",
  "bbob-biobj_f04_i03_d05 0.937880321716215",
  "bbob-biobj_f04_i03_d10 0.951455403991113",
  "bbob-biobj_f04_i03_d20 0.931490116714832",
  "bbob-biobj_f04_i03_d40 0.935912840885351",
  "bbob-biobj_f04_i04_d02 0.977012543483185",
  "bbob-biobj_f04_i04_d03 0.994699731867687",
  "bbob-biobj_f04_i04_d05 0.944471116749840",
  "bbob-biobj_f04_i04_d10 0.936538418626468",
  "bbob-biobj_f04_i04_d20 0.942534311803126",
  "bbob-biobj_f04_i04_d40 0.936539435934015",
  "bbob-biobj_f04_i05_d02 0.924874420000501",
  "bbob-biobj_f04_i05_d03 0.923161919375749",
  "bbob-biobj_f04_i05_d05 0.942091615902321",
  "bbob-biobj_f04_i05_d10 0.941633909892918",
  "bbob-biobj_f04_i05_d20 0.948833719064138",
  "bbob-biobj_f04_i05_d40 0.941340383563524",
  "bbob-biobj_f04_i06_d02 0.972712023086024",
  "bbob-biobj_f04_i06_d03 0.943259154470174",
  "bbob-biobj_f04_i06_d05 0.950605294323082",
  "bbob-biobj_f04_i06_d10 0.952990287497578",
  "bbob-biobj_f04_i06_d20 0.950812808533448",
  "bbob-biobj_f04_i06_d40 0.941972851884953",
  "bbob-biobj_f04_i07_d02 0.955670149154298",
  "bbob-biobj_f04_i07_d03 0.967708425195390",
  "bbob-biobj_f04_i07_d05 0.954782909290990",
  "bbob-biobj_f04_i07_d10 0.963225833055543",
  "bbob-biobj_f04_i07_d20 0.945577888150042",
  "bbob-biobj_f04_i07_d40 0.946077143098894",
  "bbob-biobj_f04_i08_d02 0.907747209528065",
  "bbob-biobj_f04_i08_d03 0.921849147005426",
  "bbob-biobj_f04_i08_d05 0.959565746288096",
  "bbob-biobj_f04_i08_d10 0.948097764116263",
  "bbob-biobj_f04_i08_d20 0.941570417466490",
  "bbob-biobj_f04_i08_d40 0.934063924497575",
  "bbob-biobj_f04_i09_d02 0.810228484370594",
  "bbob-biobj_f04_i09_d03 0.940463955038713",
  "bbob-biobj_f04_i09_d05 0.933584265077007",
  "bbob-biobj_f04_i09_d10 0.942896781839956",
  "bbob-biobj_f04_i09_d20 0.933010807630098",
  "bbob-biobj_f04_i09_d40 0.935798291522789",
  "bbob-biobj_f04_i10_d02 0.954309122443056",
  "bbob-biobj_f04_i10_d03 0.918874808389024",
  "bbob-biobj_f04_i10_d05 0.935604060678033",
  "bbob-biobj_f04_i10_d10 0.926101932209592",
  "bbob-biobj_f04_i10_d20 0.933741140736711",
  "bbob-biobj_f04_i10_d40 0.935392338253246",
  "bbob-biobj_f04_i11_d02 1.0",
  "bbob-biobj_f04_i11_d03 1.0",
  "bbob-biobj_f04_i11_d05 1.0",
  "bbob-biobj_f04_i11_d10 1.0",
  "bbob-biobj_f04_i11_d20 1.0",
  "bbob-biobj_f04_i11_d40 1.0",
  "bbob-biobj_f04_i12_d02 1.0",
  "bbob-biobj_f04_i12_d03 1.0",
  "bbob-biobj_f04_i12_d05 1.0",
  "bbob-biobj_f04_i12_d10 1.0",
  "bbob-biobj_f04_i12_d20 1.0",
  "bbob-biobj_f04_i12_d40 1.0",
  "bbob-biobj_f04_i13_d02 1.0",
  "bbob-biobj_f04_i13_d03 1.0",
  "bbob-biobj_f04_i13_d05 1.0",
  "bbob-biobj_f04_i13_d10 1.0",
  "bbob-biobj_f04_i13_d20 1.0",
  "bbob-biobj_f04_i13_d40 1.0",
  "bbob-biobj_f04_i14_d02 1.0",
  "bbob-biobj_f04_i14_d03 1.0",
  "bbob-biobj_f04_i14_d05 1.0",
  "bbob-biobj_f04_i14_d10 1.0",
  "bbob-biobj_f04_i14_d20 1.0",
  "bbob-biobj_f04_i14_d40 1.0",
  "bbob-biobj_f04_i15_d02 1.0",
  "bbob-biobj_f04_i15_d03 1.0",
  "bbob-biobj_f04_i15_d05 1.0",
  "bbob-biobj_f04_i15_d10 1.0",
  "bbob-biobj_f04_i15_d20 1.0",
  "bbob-biobj_f04_i15_d40 1.0",
  "bbob-biobj_f05_i01_d02 0.754128843554063",
  "bbob-biobj_f05_i01_d03 0.728475348474197",
  "bbob-biobj_f05_i01_d05 0.732294673842556",
  "bbob-biobj_f05_i01_d10 0.714081601978928",
  "bbob-biobj_f05_i01_d20 0.694491269656484",
  "bbob-biobj_f05_i01_d40 0.709897599941799",
  "bbob-biobj_f05_i02_d02 0.954990573126856",
  "bbob-biobj_f05_i02_d03 0.688043584247301",
  "bbob-biobj_f05_i02_d05 0.714739513483451",
  "bbob-biobj_f05_i02_d10 0.730746486457995",
  "bbob-biobj_f05_i02_d20 0.689211189126236",
  "bbob-biobj_f05_i02_d40 0.698143247686728",
  "bbob-biobj_f05_i03_d02 0.684482936855002",
  "bbob-biobj_f05_i03_d03 0.802889224282279",
  "bbob-biobj_f05_i03_d05 0.699898464202400",
  "bbob-biobj_f05_i03_d10 0.683945382318722",
  "bbob-biobj_f05_i03_d20 0.697145640312030",
  "bbob-biobj_f05_i03_d40 0.694224764943005",
  "bbob-biobj_f05_i04_d02 0.878631790550396",
  "bbob-biobj_f05_i04_d03 0.744997244910120",
  "bbob-biobj_f05_i04_d05 0.776093236181555",
  "bbob-biobj_f05_i04_d10 0.716317814175257",
  "bbob-biobj_f05_i04_d20 0.705280280057881",
  "bbob-biobj_f05_i04_d40 0.699990681454391",
  "bbob-biobj_f05_i05_d02 0.926275561653235",
  "bbob-biobj_f05_i05_d03 0.701518937377523",
  "bbob-biobj_f05_i05_d05 0.737170729705834",
  "bbob-biobj_f05_i05_d10 0.749504681924043",
  "bbob-biobj_f05_i05_d20 0.695500720128268",
  "bbob-biobj_f05_i05_d40 0.698208388110590",
  "bbob-biobj_f05_i06_d02 0.885993423655518",
  "bbob-biobj_f05_i06_d03 0.753054842415797",
  "bbob-biobj_f05_i06_d05 0.777207864089930",
  "bbob-biobj_f05_i06_d10 0.760062484063981",
  "bbob-biobj_f05_i06_d20 0.701429647524717",
  "bbob-biobj_f05_i06_d40 0.703399229288760",
  "bbob-biobj_f05_i07_d02 0.733322624337346",
  "bbob-biobj_f05_i07_d03 0.837353794951257",
  "bbob-biobj_f05_i07_d05 0.732469624019470",
  "bbob-biobj_f05_i07_d10 0.704094291056772",
  "bbob-biobj_f05_i07_d20 0.714293887126808",
  "bbob-biobj_f05_i07_d40 0.702091851169133",
  "bbob-biobj_f05_i08_d02 0.720852977164468",
  "bbob-biobj_f05_i08_d03 0.718993847264250",
  "bbob-biobj_f05_i08_d05 0.719205565720365",
  "bbob-biobj_f05_i08_d10 0.695251774184546",
  "bbob-biobj_f05_i08_d20 0.721402091348061",
  "bbob-biobj_f05_i08_d40 0.698735224096643",
  "bbob-biobj_f05_i09_d02 0.843425481280734",
  "bbob-biobj_f05_i09_d03 0.678680980278178",
  "bbob-biobj_f05_i09_d05 0.774526799831117",
  "bbob-biobj_f05_i09_d10 0.702694287379912",
  "bbob-biobj_f05_i09_d20 0.700114467616452",
  "bbob-biobj_f05_i09_d40 0.695637998755007",
  "bbob-biobj_f05_i10_d02 0.781317669360153",
  "bbob-biobj_f05_i10_d03 0.935431588217515",
  "bbob-biobj_f05_i10_d05 0.765296061050919",
  "bbob-biobj_f05_i10_d10 0.703665546454890",
  "bbob-biobj_f05_i10_d20 0.695348707691516",
  "bbob-biobj_f05_i10_d40 0.699213843205399",
  "bbob-biobj_f05_i11_d02 1.0",
  "bbob-biobj_f05_i11_d03 1.0",
  "bbob-biobj_f05_i11_d05 1.0",
  "bbob-biobj_f05_i11_d10 1.0",
  "bbob-biobj_f05_i11_d20 1.0",
  "bbob-biobj_f05_i11_d40 1.0",
  "bbob-biobj_f05_i12_d02 1.0",
  "bbob-biobj_f05_i12_d03 1.0",
  "bbob-biobj_f05_i12_d05 1.0",
  "bbob-biobj_f05_i12_d10 1.0",
  "bbob-biobj_f05_i12_d20 1.0",
  "bbob-biobj_f05_i12_d40 1.0",
  "bbob-biobj_f05_i13_d02 1.0",
  "bbob-biobj_f05_i13_d03 1.0",
  "bbob-biobj_f05_i13_d05 1.0",
  "bbob-biobj_f05_i13_d10 1.0",
  "bbob-biobj_f05_i13_d20 1.0",
  "bbob-biobj_f05_i13_d40 1.0",
  "bbob-biobj_f05_i14_d02 1.0",
  "bbob-biobj_f05_i14_d03 1.0",
  "bbob-biobj_f05_i14_d05 1.0",
  "bbob-biobj_f05_i14_d10 1.0",
  "bbob-biobj_f05_i14_d20 1.0",
  "bbob-biobj_f05_i14_d40 1.0",
  "bbob-biobj_f05_i15_d02 1.0",
  "bbob-biobj_f05_i15_d03 1.0",
  "bbob-biobj_f05_i15_d05 1.0",
  "bbob-biobj_f05_i15_d10 1.0",
  "bbob-biobj_f05_i15_d20 1.0",
  "bbob-biobj_f05_i15_d40 1.0",
  "bbob-biobj_f06_i01_d02 0.667254444373576",
  "bbob-biobj_f06_i01_d03 0.954292615388024",
  "bbob-biobj_f06_i01_d05 0.846018108006070",
  "bbob-biobj_f06_i01_d10 0.937011171661415",
  "bbob-biobj_f06_i01_d20 0.931062491929114",
  "bbob-biobj_f06_i01_d40 0.923107376979132",
  "bbob-biobj_f06_i02_d02 0.901470541064455",
  "bbob-biobj_f06_i02_d03 0.863894446994245",
  "bbob-biobj_f06_i02_d05 0.867726820357307",
  "bbob-biobj_f06_i02_d10 0.875243417373076",
  "bbob-biobj_f06_i02_d20 0.910534831484962",
  "bbob-biobj_f06_i02_d40 0.943043761978475",
  "bbob-biobj_f06_i03_d02 0.884300125495847",
  "bbob-biobj_f06_i03_d03 0.833639337871505",
  "bbob-biobj_f06_i03_d05 0.848595402210255",
  "bbob-biobj_f06_i03_d10 0.895943187528683",
  "bbob-biobj_f06_i03_d20 0.932804873601530",
  "bbob-biobj_f06_i03_d40 0.902613635336386",
  "bbob-biobj_f06_i04_d02 0.945804377878614",
  "bbob-biobj_f06_i04_d03 0.921371854303564",
  "bbob-biobj_f06_i04_d05 0.935945307405495",
  "bbob-biobj_f06_i04_d10 0.930064024521303",
  "bbob-biobj_f06_i04_d20 0.901083959683698",
  "bbob-biobj_f06_i04_d40 0.876264281037954",
  "bbob-biobj_f06_i05_d02 0.942603371250243",
  "bbob-biobj_f06_i05_d03 0.899137550818265",
  "bbob-biobj_f06_i05_d05 0.930569530909069",
  "bbob-biobj_f06_i05_d10 0.743374275978944",
  "bbob-biobj_f06_i05_d20 0.918107792651112",
  "bbob-biobj_f06_i05_d40 0.858546452709913",
  "bbob-biobj_f06_i06_d02 0.899088058493637",
  "bbob-biobj_f06_i06_d03 0.836273396344679",
  "bbob-biobj_f06_i06_d05 0.811666818853098",
  "bbob-biobj_f06_i06_d10 0.928896921971501",
  "bbob-biobj_f06_i06_d20 0.868880077191445",
  "bbob-biobj_f06_i06_d40 0.914009310714674",
  "bbob-biobj_f06_i07_d02 0.813378211274640",
  "bbob-biobj_f06_i07_d03 0.899764157891963",
  "bbob-biobj_f06_i07_d05 0.877044131516540",
  "bbob-biobj_f06_i07_d10 0.815103763392029",
  "bbob-biobj_f06_i07_d20 0.935546623083071",
  "bbob-biobj_f06_i07_d40 0.918203928752294",
  "bbob-biobj_f06_i08_d02 0.910565241257933",
  "bbob-biobj_f06_i08_d03 0.667404103609139",
  "bbob-biobj_f06_i08_d05 0.937319850801338",
  "bbob-biobj_f06_i08_d10 0.930156422143394",
  "bbob-biobj_f06_i08_d20 0.910955371810017",
  "bbob-biobj_f06_i08_d40 0.925981149845227",
  "bbob-biobj_f06_i09_d02 0.675947010648316",
  "bbob-biobj_f06_i09_d03 0.867855866365995",
  "bbob-biobj_f06_i09_d05 0.897324697379564",
  "bbob-biobj_f06_i09_d10 0.845127370421703",
  "bbob-biobj_f06_i09_d20 0.949720407857437",
  "bbob-biobj_f06_i09_d40 0.945411493883968",
  "bbob-biobj_f06_i10_d02 0.882457686653892",
  "bbob-biobj_f06_i10_d03 0.907422193376687",
  "bbob-biobj_f06_i10_d05 0.905219354104915",
  "bbob-biobj_f06_i10_d10 0.906496647250774",
  "bbob-biobj_f06_i10_d20 0.901309728619030",
  "bbob-biobj_f06_i10_d40 0.921989810690995",
  "bbob-biobj_f06_i11_d02 1.0",
  "bbob-biobj_f06_i11_d03 1.0",
  "bbob-biobj_f06_i11_d05 1.0",
  "bbob-biobj_f06_i11_d10 1.0",
  "bbob-biobj_f06_i11_d20 1.0",
  "bbob-biobj_f06_i11_d40 1.0",
  "bbob-biobj_f06_i12_d02 1.0",
  "bbob-biobj_f06_i12_d03 1.0",
  "bbob-biobj_f06_i12_d05 1.0",
  "bbob-biobj_f06_i12_d10 1.0",
  "bbob-biobj_f06_i12_d20 1.0",
  "bbob-biobj_f06_i12_d40 1.0",
  "bbob-biobj_f06_i13_d02 1.0",
  "bbob-biobj_f06_i13_d03 1.0",
  "bbob-biobj_f06_i13_d05 1.0",
  "bbob-biobj_f06_i13_d10 1.0",
  "bbob-biobj_f06_i13_d20 1.0",
  "bbob-biobj_f06_i13_d40 1.0",
  "bbob-biobj_f06_i14_d02 1.0",
  "bbob-biobj_f06_i14_d03 1.0",
  "bbob-biobj_f06_i14_d05 1.0",
  "bbob-biobj_f06_i14_d10 1.0",
  "bbob-biobj_f06_i14_d20 1.0",
  "bbob-biobj_f06_i14_d40 1.0",
  "bbob-biobj_f06_i15_d02 1.0",
  "bbob-biobj_f06_i15_d03 1.0",
  "bbob-biobj_f06_i15_d05 1.0",
  "bbob-biobj_f06_i15_d10 1.0",
  "bbob-biobj_f06_i15_d20 1.0",
  "bbob-biobj_f06_i15_d40 1.0",
  "bbob-biobj_f07_i01_d02 0.936972575085523",
  "bbob-biobj_f07_i01_d03 0.937571783523299",
  "bbob-biobj_f07_i01_d05 0.860222475841532",
  "bbob-biobj_f07_i01_d10 0.897609185174781",
  "bbob-biobj_f07_i01_d20 0.942671659576489",
  "bbob-biobj_f07_i01_d40 0.910485687990751",
  "bbob-biobj_f07_i02_d02 0.906340900885370",
  "bbob-biobj_f07_i02_d03 0.923761423740148",
  "bbob-biobj_f07_i02_d05 0.893388606223265",
  "bbob-biobj_f07_i02_d10 0.896254312553792",
  "bbob-biobj_f07_i02_d20 0.900483922216280",
  "bbob-biobj_f07_i02_d40 0.865795482651835",
  "bbob-biobj_f07_i03_d02 0.886134331413053",
  "bbob-biobj_f07_i03_d03 0.921398210499996",
  "bbob-biobj_f07_i03_d05 0.868192249749793",
  "bbob-biobj_f07_i03_d10 0.894160344923831",
  "bbob-biobj_f07_i03_d20 0.898091141981344",
  "bbob-biobj_f07_i03_d40 0.889534578187546",
  "bbob-biobj_f07_i04_d02 0.870759950441604",
  "bbob-biobj_f07_i04_d03 0.933982487161439",
  "bbob-biobj_f07_i04_d05 0.870207948851616",
  "bbob-biobj_f07_i04_d10 0.884508750253254",
  "bbob-biobj_f07_i04_d20 0.894023396251866",
  "bbob-biobj_f07_i04_d40 0.905164921817996",
  "bbob-biobj_f07_i05_d02 0.911523577984129",
  "bbob-biobj_f07_i05_d03 0.887628704517619",
  "bbob-biobj_f07_i05_d05 0.911689477358187",
  "bbob-biobj_f07_i05_d10 0.868303705443327",
  "bbob-biobj_f07_i05_d20 0.888205577785785",
  "bbob-biobj_f07_i05_d40 0.912756246768348",
  "bbob-biobj_f07_i06_d02 0.937861505018401",
  "bbob-biobj_f07_i06_d03 0.945942900173665",
  "bbob-biobj_f07_i06_d05 0.915705285244223",
  "bbob-biobj_f07_i06_d10 0.891206796700426",
  "bbob-biobj_f07_i06_d20 0.884437403163543",
  "bbob-biobj_f07_i06_d40 0.883827465078616",
  "bbob-biobj_f07_i07_d02 0.871209654272761",
  "bbob-biobj_f07_i07_d03 0.911170901726427",
  "bbob-biobj_f07_i07_d05 0.885356348557851",
  "bbob-biobj_f07_i07_d10 0.895861833448791",
  "bbob-biobj_f07_i07_d20 0.894942450950192",
  "bbob-biobj_f07_i07_d40 0.854575042999799",
  "bbob-biobj_f07_i08_d02 0.849406116302883",
  "bbob-biobj_f07_i08_d03 0.909560273134834",
  "bbob-biobj_f07_i08_d05 0.846611979908353",
  "bbob-biobj_f07_i08_d10 0.916959924844533",
  "bbob-biobj_f07_i08_d20 0.891124123593114",
  "bbob-biobj_f07_i08_d40 0.905849069194381",
  "bbob-biobj_f07_i09_d02 0.877457921891718",
  "bbob-biobj_f07_i09_d03 0.928694938524451",
  "bbob-biobj_f07_i09_d05 0.890504447918247",
  "bbob-biobj_f07_i09_d10 0.911171424258652",
  "bbob-biobj_f07_i09_d20 0.898982426181766",
  "bbob-biobj_f07_i09_d40 0.867135361861662",
  "bbob-biobj_f07_i10_d02 0.907426576520353",
  "bbob-biobj_f07_i10_d03 0.920344140918204",
  "bbob-biobj_f07_i10_d05 0.897609858978946",
  "bbob-biobj_f07_i10_d10 0.902681008338079",
  "bbob-biobj_f07_i10_d20 0.933248995366967",
  "bbob-biobj_f07_i10_d40 0.893566872525990",
  "bbob-biobj_f07_i11_d02 1.0",
  "bbob-biobj_f07_i11_d03 1.0",
  "bbob-biobj_f07_i11_d05 1.0",
  "bbob-biobj_f07_i11_d10 1.0",
  "bbob-biobj_f07_i11_d20 1.0",
  "bbob-biobj_f07_i11_d40 1.0",
  "bbob-biobj_f07_i12_d02 1.0",
  "bbob-biobj_f07_i12_d03 1.0",
  "bbob-biobj_f07_i12_d05 1.0",
  "bbob-biobj_f07_i12_d10 1.0",
  "bbob-biobj_f07_i12_d20 1.0",
  "bbob-biobj_f07_i12_d40 1.0",
  "bbob-biobj_f07_i13_d02 1.0",
  "bbob-biobj_f07_i13_d03 1.0",
  "bbob-biobj_f07_i13_d05 1.0",
  "bbob-biobj_f07_i13_d10 1.0",
  "bbob-biobj_f07_i13_d20 1.0",
  "bbob-biobj_f07_i13_d40 1.0",
  "bbob-biobj_f07_i14_d02 1.0",
  "bbob-biobj_f07_i14_d03 1.0",
  "bbob-biobj_f07_i14_d05 1.0",
  "bbob-biobj_f07_i14_d10 1.0",
  "bbob-biobj_f07_i14_d20 1.0",
  "bbob-biobj_f07_i14_d40 1.0",
  "bbob-biobj_f07_i15_d02 1.0",
  "bbob-biobj_f07_i15_d03 1.0",
  "bbob-biobj_f07_i15_d05 1.0",
  "bbob-biobj_f07_i15_d10 1.0",
  "bbob-biobj_f07_i15_d20 1.0",
  "bbob-biobj_f07_i15_d40 1.0",
  "bbob-biobj_f08_i01_d02 0.903849381293032",
  "bbob-biobj_f08_i01_d03 0.911799446711896",
  "bbob-biobj_f08_i01_d05 0.942810865795991",
  "bbob-biobj_f08_i01_d10 0.982310271948219",
  "bbob-biobj_f08_i01_d20 0.969081195591175",
  "bbob-biobj_f08_i01_d40 0.941602861002529",
  "bbob-biobj_f08_i02_d02 0.784765299142009",
  "bbob-biobj_f08_i02_d03 0.882040732286310",
  "bbob-biobj_f08_i02_d05 0.909602249858625",
  "bbob-biobj_f08_i02_d10 0.916464603099051",
  "bbob-biobj_f08_i02_d20 0.905238001542862",
  "bbob-biobj_f08_i02_d40 0.942706996827259",
  "bbob-biobj_f08_i03_d02 0.748604947410058",
  "bbob-biobj_f08_i03_d03 0.850955572510764",
  "bbob-biobj_f08_i03_d05 0.805508853750290",
  "bbob-biobj_f08_i03_d10 0.931243739617921",
  "bbob-biobj_f08_i03_d20 0.950573205092055",
  "bbob-biobj_f08_i03_d40 0.882551515361396",
  "bbob-biobj_f08_i04_d02 0.743267364302322",
  "bbob-biobj_f08_i04_d03 0.667007078367313",
  "bbob-biobj_f08_i04_d05 0.927892758832404",
  "bbob-biobj_f08_i04_d10 0.951140771552574",
  "bbob-biobj_f08_i04_d20 0.956543860986805",
  "bbob-biobj_f08_i04_d40 0.924431301334235",
  "bbob-biobj_f08_i05_d02 0.865136714427976",
  "bbob-biobj_f08_i05_d03 0.893998699750253",
  "bbob-biobj_f08_i05_d05 0.917998781925269",
  "bbob-biobj_f08_i05_d10 0.930536026397557",
  "bbob-biobj_f08_i05_d20 0.910195152512765",
  "bbob-biobj_f08_i05_d40 0.944277101801299",
  "bbob-biobj_f08_i06_d02 0.829476441539014",
  "bbob-biobj_f08_i06_d03 0.895408933722067",
  "bbob-biobj_f08_i06_d05 0.889128202192725",
  "bbob-biobj_f08_i06_d10 0.912648355358455",
  "bbob-biobj_f08_i06_d20 0.929751402682770",
  "bbob-biobj_f08_i06_d40 0.912322766676420",
  "bbob-biobj_f08_i07_d02 0.933714327146467",
  "bbob-biobj_f08_i07_d03 0.911121051771586",
  "bbob-biobj_f08_i07_d05 0.901891452886488",
  "bbob-biobj_f08_i07_d10 0.876772709102926",
  "bbob-biobj_f08_i07_d20 0.948895341508442",
  "bbob-biobj_f08_i07_d40 0.871778232072206",
  "bbob-biobj_f08_i08_d02 0.901615894941159",
  "bbob-biobj_f08_i08_d03 0.935771577639889",
  "bbob-biobj_f08_i08_d05 0.913263769388604",
  "bbob-biobj_f08_i08_d10 0.924413116450644",
  "bbob-biobj_f08_i08_d20 0.942886981112154",
  "bbob-biobj_f08_i08_d40 0.832711391708555",
  "bbob-biobj_f08_i09_d02 0.904705113882673",
  "bbob-biobj_f08_i09_d03 0.941784988928111",
  "bbob-biobj_f08_i09_d05 0.833932222374868",
  "bbob-biobj_f08_i09_d10 0.920117214003409",
  "bbob-biobj_f08_i09_d20 0.913696814139621",
  "bbob-biobj_f08_i09_d40 0.935225792568577",
  "bbob-biobj_f08_i10_d02 0.904870851433164",
  "bbob-biobj_f08_i10_d03 0.934442126180304",
  "bbob-biobj_f08_i10_d05 0.952539863227733",
  "bbob-biobj_f08_i10_d10 0.924171076528587",
  "bbob-biobj_f08_i10_d20 0.918327947939103",
  "bbob-biobj_f08_i10_d40 0.962723183344110",
  "bbob-biobj_f08_i11_d02 1.0",
  "bbob-biobj_f08_i11_d03 1.0",
  "bbob-biobj_f08_i11_d05 1.0",
  "bbob-biobj_f08_i11_d10 1.0",
  "bbob-biobj_f08_i11_d20 1.0",
  "bbob-biobj_f08_i11_d40 1.0",
  "bbob-biobj_f08_i12_d02 1.0",
  "bbob-biobj_f08_i12_d03 1.0",
  "bbob-biobj_f08_i12_d05 1.0",
  "bbob-biobj_f08_i12_d10 1.0",
  "bbob-biobj_f08_i12_d20 1.0",
  "bbob-biobj_f08_i12_d40 1.0",
  "bbob-biobj_f08_i13_d02 1.0",
  "bbob-biobj_f08_i13_d03 1.0",
  "bbob-biobj_f08_i13_d05 1.0",
  "bbob-biobj_f08_i13_d10 1.0",
  "bbob-biobj_f08_i13_d20 1.0",
  "bbob-biobj_f08_i13_d40 1.0",
  "bbob-biobj_f08_i14_d02 1.0",
  "bbob-biobj_f08_i14_d03 1.0",
  "bbob-biobj_f08_i14_d05 1.0",
  "bbob-biobj_f08_i14_d10 1.0",
  "bbob-biobj_f08_i14_d20 1.0",
  "bbob-biobj_f08_i14_d40 1.0",
  "bbob-biobj_f08_i15_d02 1.0",
  "bbob-biobj_f08_i15_d03 1.0",
  "bbob-biobj_f08_i15_d05 1.0",
  "bbob-biobj_f08_i15_d10 1.0",
  "bbob-biobj_f08_i15_d20 1.0",
  "bbob-biobj_f08_i15_d40 1.0",
  "bbob-biobj_f09_i01_d02 0.925657814170223",
  "bbob-biobj_f09_i01_d03 0.904197423117471",
  "bbob-biobj_f09_i01_d05 0.932181427920137",
  "bbob-biobj_f09_i01_d10 0.940801691617986",
  "bbob-biobj_f09_i01_d20 0.960316065946310",
  "bbob-biobj_f09_i01_d40 0.966662588954428",
  "bbob-biobj_f09_i02_d02 0.977793751262295",
  "bbob-biobj_f09_i02_d03 0.992207394369088",
  "bbob-biobj_f09_i02_d05 0.961854320874622",
  "bbob-biobj_f09_i02_d10 0.975991497292509",
  "bbob-biobj_f09_i02_d20 0.962608726627843",
  "bbob-biobj_f09_i02_d40 0.963762685971772",
  "bbob-biobj_f09_i03_d02 0.968705844447316",
  "bbob-biobj_f09_i03_d03 0.986085524471979",
  "bbob-biobj_f09_i03_d05 0.930451696324880",
  "bbob-biobj_f09_i03_d10 0.955044825826449",
  "bbob-biobj_f09_i03_d20 0.970332351664872",
  "bbob-biobj_f09_i03_d40 0.969394339815251",
  "bbob-biobj_f09_i04_d02 0.948342668463238",
  "bbob-biobj_f09_i04_d03 0.940844214602507",
  "bbob-biobj_f09_i04_d05 0.950696199447794",
  "bbob-biobj_f09_i04_d10 0.944322425851199",
  "bbob-biobj_f09_i04_d20 0.961728387760400",
  "bbob-biobj_f09_i04_d40 0.963103853915457",
  "bbob-biobj_f09_i05_d02 0.860780709822583",
  "bbob-biobj_f09_i05_d03 0.939789647695892",
  "bbob-biobj_f09_i05_d05 0.968561825403842",
  "bbob-biobj_f09_i05_d10 0.940734136756500",
  "bbob-biobj_f09_i05_d20 0.963736063540811",
  "bbob-biobj_f09_i05_d40 0.967143960180269",
  "bbob-biobj_f09_i06_d02 0.957398241336098",
  "bbob-biobj_f09_i06_d03 0.987802627245661",
  "bbob-biobj_f09_i06_d05 0.910234055809238",
  "bbob-biobj_f09_i06_d10 0.958616029953390",
  "bbob-biobj_f09_i06_d20 0.969292910703259",
  "bbob-biobj_f09_i06_d40 0.960785018243888",
  "bbob-biobj_f09_i07_d02 0.990548357014945",
  "bbob-biobj_f09_i07_d03 0.962705179569111",
  "bbob-biobj_f09_i07_d05 0.974605803220911",
  "bbob-biobj_f09_i07_d10 0.974607189329728",
  "bbob-biobj_f09_i07_d20 0.975338881485032",
  "bbob-biobj_f09_i07_d40 0.972104542550424",
  "bbob-biobj_f09_i08_d02 0.889502838059927",
  "bbob-biobj_f09_i08_d03 0.964124022283451",
  "bbob-biobj_f09_i08_d05 0.959709821835848",
  "bbob-biobj_f09_i08_d10 0.954117820334898",
  "bbob-biobj_f09_i08_d20 0.950915150582074",
  "bbob-biobj_f09_i08_d40 0.959892153059548",
  "bbob-biobj_f09_i09_d02 0.950460190109210",
  "bbob-biobj_f09_i09_d03 0.983671960174457",
  "bbob-biobj_f09_i09_d05 0.964992545695910",
  "bbob-biobj_f09_i09_d10 0.979833476209204",
  "bbob-biobj_f09_i09_d20 0.978347861561944",
  "bbob-biobj_f09_i09_d40 0.967598075906950",
  "bbob-biobj_f09_i10_d02 0.869881009407849",
  "bbob-biobj_f09_i10_d03 0.943447288797987",
  "bbob-biobj_f09_i10_d05 0.920154801176530",
  "bbob-biobj_f09_i10_d10 0.944146078493198",
  "bbob-biobj_f09_i10_d20 0.957505476179459",
  "bbob-biobj_f09_i10_d40 0.964026681541125",
  "bbob-biobj_f09_i11_d02 1.0",
  "bbob-biobj_f09_i11_d03 1.0",
  "bbob-biobj_f09_i11_d05 1.0",
  "bbob-biobj_f09_i11_d10 1.0",
  "bbob-biobj_f09_i11_d20 1.0",
  "bbob-biobj_f09_i11_d40 1.0",
  "bbob-biobj_f09_i12_d02 1.0",
  "bbob-biobj_f09_i12_d03 1.0",
  "bbob-biobj_f09_i12_d05 1.0",
  "bbob-biobj_f09_i12_d10 1.0",
  "bbob-biobj_f09_i12_d20 1.0",
  "bbob-biobj_f09_i12_d40 1.0",
  "bbob-biobj_f09_i13_d02 1.0",
  "bbob-biobj_f09_i13_d03 1.0",
  "bbob-biobj_f09_i13_d05 1.0",
  "bbob-biobj_f09_i13_d10 1.0",
  "bbob-biobj_f09_i13_d20 1.0",
  "bbob-biobj_f09_i13_d40 1.0",
  "bbob-biobj_f09_i14_d02 1.0",
  "bbob-biobj_f09_i14_d03 1.0",
  "bbob-biobj_f09_i14_d05 1.0",
  "bbob-biobj_f09_i14_d10 1.0",
  "bbob-biobj_f09_i14_d20 1.0",
  "bbob-biobj_f09_i14_d40 1.0",
  "bbob-biobj_f09_i15_d02 1.0",
  "bbob-biobj_f09_i15_d03 1.0",
  "bbob-biobj_f09_i15_d05 1.0",
  "bbob-biobj_f09_i15_d10 1.0",
  "bbob-biobj_f09_i15_d20 1.0",
  "bbob-biobj_f09_i15_d40 1.0",
  "bbob-biobj_f10_i01_d02 0.922987888165046",
  "bbob-biobj_f10_i01_d03 0.927568092282234",
  "bbob-biobj_f10_i01_d05 0.867913922452004",
  "bbob-biobj_f10_i01_d10 0.879667617760632",
  "bbob-biobj_f10_i01_d20 0.840992193517756",
  "bbob-biobj_f10_i01_d40 0.730862112175264",
  "bbob-biobj_f10_i02_d02 0.889244712003482",
  "bbob-biobj_f10_i02_d03 0.883788429112751",
  "bbob-biobj_f10_i02_d05 0.898119402291299",
  "bbob-biobj_f10_i02_d10 0.798834395729540",
  "bbob-biobj_f10_i02_d20 0.812130174416182",
  "bbob-biobj_f10_i02_d40 0.663856072359917",
  "bbob-biobj_f10_i03_d02 0.921418301089463",
  "bbob-biobj_f10_i03_d03 0.901135310123523",
  "bbob-biobj_f10_i03_d05 0.890601509335293",
  "bbob-biobj_f10_i03_d10 0.710848076679431",
  "bbob-biobj_f10_i03_d20 0.798315835129164",
  "bbob-biobj_f10_i03_d40 0.747904986400848",
  "bbob-biobj_f10_i04_d02 0.942089494535008",
  "bbob-biobj_f10_i04_d03 0.932702272655739",
  "bbob-biobj_f10_i04_d05 0.946972264766413",
  "bbob-biobj_f10_i04_d10 0.906395334139304",
  "bbob-biobj_f10_i04_d20 0.800335891125016",
  "bbob-biobj_f10_i04_d40 0.638845426804670",
  "bbob-biobj_f10_i05_d02 0.940003750569188",
  "bbob-biobj_f10_i05_d03 0.934364067558401",
  "bbob-biobj_f10_i05_d05 0.934949844294875",
  "bbob-biobj_f10_i05_d10 0.842838836991676",
  "bbob-biobj_f10_i05_d20 0.778548919281115",
  "bbob-biobj_f10_i05_d40 0.640157170132133",
  "bbob-biobj_f10_i06_d02 0.884512896964566",
  "bbob-biobj_f10_i06_d03 0.929345926122951",
  "bbob-biobj_f10_i06_d05 0.937753565304302",
  "bbob-biobj_f10_i06_d10 0.916302417802770",
  "bbob-biobj_f10_i06_d20 0.804216801427507",
  "bbob-biobj_f10_i06_d40 0.690918114180113",
  "bbob-biobj_f10_i07_d02 0.972916090945292",
  "bbob-biobj_f10_i07_d03 0.971450633117566",
  "bbob-biobj_f10_i07_d05 0.941727082076548",
  "bbob-biobj_f10_i07_d10 0.817927199823897",
  "bbob-biobj_f10_i07_d20 0.763258585381718",
  "bbob-biobj_f10_i07_d40 0.739991394439648",
  "bbob-biobj_f10_i08_d02 0.926524440495943",
  "bbob-biobj_f10_i08_d03 0.950976461393930",
  "bbob-biobj_f10_i08_d05 0.975346535085332",
  "bbob-biobj_f10_i08_d10 0.916401249392711",
  "bbob-biobj_f10_i08_d20 0.857675602487971",
  "bbob-biobj_f10_i08_d40 0.775583712514456",
  "bbob-biobj_f10_i09_d02 0.663726926800968",
  "bbob-biobj_f10_i09_d03 0.879016548436657",
  "bbob-biobj_f10_i09_d05 0.942218870289438",
  "bbob-biobj_f10_i09_d10 0.904838192250101",
  "bbob-biobj_f10_i09_d20 0.792568176822578",
  "bbob-biobj_f10_i09_d40 0.722036310804938",
  "bbob-biobj_f10_i10_d02 0.909615354906003",
  "bbob-biobj_f10_i10_d03 0.950985467384464",
  "bbob-biobj_f10_i10_d05 0.937047585618269",
  "bbob-biobj_f10_i10_d10 0.894128956510904",
  "bbob-biobj_f10_i10_d20 0.748700702453785",
  "bbob-biobj_f10_i10_d40 0.532852626529976",
  "bbob-biobj_f10_i11_d02 1.0",
  "bbob-biobj_f10_i11_d03 1.0",
  "bbob-biobj_f10_i11_d05 1.0",
  "bbob-biobj_f10_i11_d10 1.0",
  "bbob-biobj_f10_i11_d20 1.0",
  "bbob-biobj_f10_i11_d40 1.0",
  "bbob-biobj_f10_i12_d02 1.0",
  "bbob-biobj_f10_i12_d03 1.0",
  "bbob-biobj_f10_i12_d05 1.0",
  "bbob-biobj_f10_i12_d10 1.0",
  "bbob-biobj_f10_i12_d20 1.0",
  "bbob-biobj_f10_i12_d40 1.0",
  "bbob-biobj_f10_i13_d02 1.0",
  "bbob-biobj_f10_i13_d03 1.0",
  "bbob-biobj_f10_i13_d05 1.0",
  "bbob-biobj_f10_i13_d10 1.0",
  "bbob-biobj_f10_i13_d20 1.0",
  "bbob-biobj_f10_i13_d40 1.0",
  "bbob-biobj_f10_i14_d02 1.0",
  "bbob-biobj_f10_i14_d03 1.0",
  "bbob-biobj_f10_i14_d05 1.0",
  "bbob-biobj_f10_i14_d10 1.0",
  "bbob-biobj_f10_i14_d20 1.0",
  "bbob-biobj_f10_i14_d40 1.0",
  "bbob-biobj_f10_i15_d02 1.0",
  "bbob-biobj_f10_i15_d03 1.0",
  "bbob-biobj_f10_i15_d05 1.0",
  "bbob-biobj_f10_i15_d10 1.0",
  "bbob-biobj_f10_i15_d20 1.0",
  "bbob-biobj_f10_i15_d40 1.0",
  "bbob-biobj_f11_i01_d02 0.823972812562388",
  "bbob-biobj_f11_i01_d03 0.878621203194964",
  "bbob-biobj_f11_i01_d05 0.812586648665059",
  "bbob-biobj_f11_i01_d10 0.836592516333072",
  "bbob-biobj_f11_i01_d20 0.836424820585753",
  "bbob-biobj_f11_i01_d40 0.826157745787494",
  "bbob-biobj_f11_i02_d02 0.834474640589775",
  "bbob-biobj_f11_i02_d03 0.833334234596160",
  "bbob-biobj_f11_i02_d05 0.813664713043024",
  "bbob-biobj_f11_i02_d10 0.829501419795512",
  "bbob-biobj_f11_i02_d20 0.835498021013514",
  "bbob-biobj_f11_i02_d40 0.836861505054914",
  "bbob-biobj_f11_i03_d02 0.817436951100827",
  "bbob-biobj_f11_i03_d03 0.827370208306986",
  "bbob-biobj_f11_i03_d05 0.841073303364040",
  "bbob-biobj_f11_i03_d10 0.821941232889153",
  "bbob-biobj_f11_i03_d20 0.835250205841700",
  "bbob-biobj_f11_i03_d40 0.842912564518251",
  "bbob-biobj_f11_i04_d02 0.883087616206635",
  "bbob-biobj_f11_i04_d03 0.841524279183524",
  "bbob-biobj_f11_i04_d05 0.886492170474548",
  "bbob-biobj_f11_i04_d10 0.834701874121709",
  "bbob-biobj_f11_i04_d20 0.838340761867027",
  "bbob-biobj_f11_i04_d40 0.836846910424672",
  "bbob-biobj_f11_i05_d02 0.849344297999467",
  "bbob-biobj_f11_i05_d03 0.775581024465445",
  "bbob-biobj_f11_i05_d05 0.834556160894165",
  "bbob-biobj_f11_i05_d10 0.840911958504972",
  "bbob-biobj_f11_i05_d20 0.841666619239648",
  "bbob-biobj_f11_i05_d40 0.833867420433639",
  "bbob-biobj_f11_i06_d02 0.826928090908244",
  "bbob-biobj_f11_i06_d03 0.829311542360003",
  "bbob-biobj_f11_i06_d05 0.835849754824295",
  "bbob-biobj_f11_i06_d10 0.834281888590491",
  "bbob-biobj_f11_i06_d20 0.835890818567694",
  "bbob-biobj_f11_i06_d40 0.833909224886551",
  "bbob-biobj_f11_i07_d02 0.827124524692875",
  "bbob-biobj_f11_i07_d03 0.834101026397339",
  "bbob-biobj_f11_i07_d05 0.822553859716167",
  "bbob-biobj_f11_i07_d10 0.828565235659249",
  "bbob-biobj_f11_i07_d20 0.847995795586526",
  "bbob-biobj_f11_i07_d40 0.839429981523613",
  "bbob-biobj_f11_i08_d02 0.816785490280523",
  "bbob-biobj_f11_i08_d03 0.828801608657967",
  "bbob-biobj_f11_i08_d05 0.821726221706212",
  "bbob-biobj_f11_i08_d10 0.837631789427020",
  "bbob-biobj_f11_i08_d20 0.842921662278686",
  "bbob-biobj_f11_i08_d40 0.840655475524800",
  "bbob-biobj_f11_i09_d02 0.832761545730916",
  "bbob-biobj_f11_i09_d03 0.824043534074746",
  "bbob-biobj_f11_i09_d05 0.792984808020167",
  "bbob-biobj_f11_i09_d10 0.824441637294302",
  "bbob-biobj_f11_i09_d20 0.829359805622782",
  "bbob-biobj_f11_i09_d40 0.826982642746262",
  "bbob-biobj_f11_i10_d02 0.826872158038815",
  "bbob-biobj_f11_i10_d03 0.815632322967258",
  "bbob-biobj_f11_i10_d05 0.813174586531673",
  "bbob-biobj_f11_i10_d10 0.845302354695332",
  "bbob-biobj_f11_i10_d20 0.827523794821044",
  "bbob-biobj_f11_i10_d40 0.834513234276812",
  "bbob-biobj_f11_i11_d02 1.0",
  "bbob-biobj_f11_i11_d03 1.0",
  "bbob-biobj_f11_i11_d05 1.0",
  "bbob-biobj_f11_i11_d10 1.0",
  "bbob-biobj_f11_i11_d20 1.0",
  "bbob-biobj_f11_i11_d40 1.0",
  "bbob-biobj_f11_i12_d02 1.0",
  "bbob-biobj_f11_i12_d03 1.0",
  "bbob-biobj_f11_i12_d05 1.0",
  "bbob-biobj_f11_i12_d10 1.0",
  "bbob-biobj_f11_i12_d20 1.0",
  "bbob-biobj_f11_i12_d40 1.0",
  "bbob-biobj_f11_i13_d02 1.0",
  "bbob-biobj_f11_i13_d03 1.0",
  "bbob-biobj_f11_i13_d05 1.0",
  "bbob-biobj_f11_i13_d10 1.0",
  "bbob-biobj_f11_i13_d20 1.0",
  "bbob-biobj_f11_i13_d40 1.0",
  "bbob-biobj_f11_i14_d02 1.0",
  "bbob-biobj_f11_i14_d03 1.0",
  "bbob-biobj_f11_i14_d05 1.0",
  "bbob-biobj_f11_i14_d10 1.0",
  "bbob-biobj_f11_i14_d20 1.0",
  "bbob-biobj_f11_i14_d40 1.0",
  "bbob-biobj_f11_i15_d02 1.0",
  "bbob-biobj_f11_i15_d03 1.0",
  "bbob-biobj_f11_i15_d05 1.0",
  "bbob-biobj_f11_i15_d10 1.0",
  "bbob-biobj_f11_i15_d20 1.0",
  "bbob-biobj_f11_i15_d40 1.0",
  "bbob-biobj_f12_i01_d02 0.981019021891034",
  "bbob-biobj_f12_i01_d03 0.997359054292220",
  "bbob-biobj_f12_i01_d05 0.999992980228967",
  "bbob-biobj_f12_i01_d10 0.997430774247223",
  "bbob-biobj_f12_i01_d20 0.999555509973294",
  "bbob-biobj_f12_i01_d40 0.999389239789896",
  "bbob-biobj_f12_i02_d02 0.999911394690186",
  "bbob-biobj_f12_i02_d03 0.999935184954457",
  "bbob-biobj_f12_i02_d05 0.999910225023882",
  "bbob-biobj_f12_i02_d10 0.999751444384452",
  "bbob-biobj_f12_i02_d20 0.999573550016330",
  "bbob-biobj_f12_i02_d40 0.999835569504230",
  "bbob-biobj_f12_i03_d02 0.999915740303757",
  "bbob-biobj_f12_i03_d03 0.999886885606420",
  "bbob-biobj_f12_i03_d05 0.947369433553171",
  "bbob-biobj_f12_i03_d10 0.999905015565585",
  "bbob-biobj_f12_i03_d20 0.999773283216776",
  "bbob-biobj_f12_i03_d40 0.999807886346771",
  "bbob-biobj_f12_i04_d02 0.972347947492247",
  "bbob-biobj_f12_i04_d03 0.934336745551181",
  "bbob-biobj_f12_i04_d05 0.973688344818477",
  "bbob-biobj_f12_i04_d10 0.999926003977452",
  "bbob-biobj_f12_i04_d20 0.999900680810457",
  "bbob-biobj_f12_i04_d40 0.999665196282814",
  "bbob-biobj_f12_i05_d02 0.908760849106752",
  "bbob-biobj_f12_i05_d03 0.934983963630235",
  "bbob-biobj_f12_i05_d05 0.999480774733750",
  "bbob-biobj_f12_i05_d10 0.999713953626639",
  "bbob-biobj_f12_i05_d20 0.999230870967443",
  "bbob-biobj_f12_i05_d40 0.995378202120478",
  "bbob-biobj_f12_i06_d02 0.951004550116135",
  "bbob-biobj_f12_i06_d03 0.994674719769856",
  "bbob-biobj_f12_i06_d05 0.999941998845874",
  "bbob-biobj_f12_i06_d10 0.998789935943491",
  "bbob-biobj_f12_i06_d20 0.999917048335107",
  "bbob-biobj_f12_i06_d40 0.998885474175385",
  "bbob-biobj_f12_i07_d02 0.999977426010364",
  "bbob-biobj_f12_i07_d03 0.999953601829309",
  "bbob-biobj_f12_i07_d05 0.969731923038145",
  "bbob-biobj_f12_i07_d10 0.999958334143000",
  "bbob-biobj_f12_i07_d20 0.999849852199318",
  "bbob-biobj_f12_i07_d40 0.999855587362279",
  "bbob-biobj_f12_i08_d02 0.995578051763076",
  "bbob-biobj_f12_i08_d03 0.848219286038717",
  "bbob-biobj_f12_i08_d05 0.999934633214172",
  "bbob-biobj_f12_i08_d10 0.999745727463896",
  "bbob-biobj_f12_i08_d20 0.999509720168614",
  "bbob-biobj_f12_i08_d40 0.994977499522366",
  "bbob-biobj_f12_i09_d02 0.927643402901094",
  "bbob-biobj_f12_i09_d03 0.999275263954279",
  "bbob-biobj_f12_i09_d05 0.999973997783679",
  "bbob-biobj_f12_i09_d10 0.999920783244525",
  "bbob-biobj_f12_i09_d20 0.999724134576184",
  "bbob-biobj_f12_i09_d40 0.998416448289325",
  "bbob-biobj_f12_i10_d02 0.865823542233124",
  "bbob-biobj_f12_i10_d03 0.999979025107695",
  "bbob-biobj_f12_i10_d05 0.999720439186375",
  "bbob-biobj_f12_i10_d10 0.998457916849131",
  "bbob-biobj_f12_i10_d20 0.998159410929698",
  "bbob-biobj_f12_i10_d40 0.999763956776766",
  "bbob-biobj_f12_i11_d02 1.0",
  "bbob-biobj_f12_i11_d03 1.0",
  "bbob-biobj_f12_i11_d05 1.0",
  "bbob-biobj_f12_i11_d10 1.0",
  "bbob-biobj_f12_i11_d20 1.0",
  "bbob-biobj_f12_i11_d40 1.0",
  "bbob-biobj_f12_i12_d02 1.0",
  "bbob-biobj_f12_i12_d03 1.0",
  "bbob-biobj_f12_i12_d05 1.0",
  "bbob-biobj_f12_i12_d10 1.0",
  "bbob-biobj_f12_i12_d20 1.0",
  "bbob-biobj_f12_i12_d40 1.0",
  "bbob-biobj_f12_i13_d02 1.0",
  "bbob-biobj_f12_i13_d03 1.0",
  "bbob-biobj_f12_i13_d05 1.0",
  "bbob-biobj_f12_i13_d10 1.0",
  "bbob-biobj_f12_i13_d20 1.0",
  "bbob-biobj_f12_i13_d40 1.0",
  "bbob-biobj_f12_i14_d02 1.0",
  "bbob-biobj_f12_i14_d03 1.0",
  "bbob-biobj_f12_i14_d05 1.0",
  "bbob-biobj_f12_i14_d10 1.0",
  "bbob-biobj_f12_i14_d20 1.0",
  "bbob-biobj_f12_i14_d40 1.0",
  "bbob-biobj_f12_i15_d02 1.0",
  "bbob-biobj_f12_i15_d03 1.0",
  "bbob-biobj_f12_i15_d05 1.0",
  "bbob-biobj_f12_i15_d10 1.0",
  "bbob-biobj_f12_i15_d20 1.0",
  "bbob-biobj_f12_i15_d40 1.0",
  "bbob-biobj_f13_i01_d02 0.944888481495913",
  "bbob-biobj_f13_i01_d03 0.999495787566523",
  "bbob-biobj_f13_i01_d05 0.999803029691490",
  "bbob-biobj_f13_i01_d10 0.999291777318943",
  "bbob-biobj_f13_i01_d20 0.950020375722517",
  "bbob-biobj_f13_i01_d40 0.988891641941712",
  "bbob-biobj_f13_i02_d02 0.999759637668710",
  "bbob-biobj_f13_i02_d03 0.998700086450386",
  "bbob-biobj_f13_i02_d05 0.998548467880571",
  "bbob-biobj_f13_i02_d10 0.999205394321604",
  "bbob-biobj_f13_i02_d20 0.970036413257736",
  "bbob-biobj_f13_i02_d40 0.990113397448570",
  "bbob-biobj_f13_i03_d02 0.999951136530969",
  "bbob-biobj_f13_i03_d03 0.962146148205073",
  "bbob-biobj_f13_i03_d05 0.999819062980181",
  "bbob-biobj_f13_i03_d10 0.999219213475386",
  "bbob-biobj_f13_i03_d20 0.945700443762299",
  "bbob-biobj_f13_i03_d40 0.991648066114144",
  "bbob-biobj_f13_i04_d02 0.999963191976562",
  "bbob-biobj_f13_i04_d03 0.999978301764668",
  "bbob-biobj_f13_i04_d05 0.999810945856148",
  "bbob-biobj_f13_i04_d10 0.991558939642803",
  "bbob-biobj_f13_i04_d20 0.997758867066474",
  "bbob-biobj_f13_i04_d40 0.986681645879782",
  "bbob-biobj_f13_i05_d02 0.999818304544863",
  "bbob-biobj_f13_i05_d03 0.999920718261273",
  "bbob-biobj_f13_i05_d05 0.999370197155656",
  "bbob-biobj_f13_i05_d10 0.994191582583059",
  "bbob-biobj_f13_i05_d20 0.998411773863768",
  "bbob-biobj_f13_i05_d40 0.986683196829484",
  "bbob-biobj_f13_i06_d02 0.999786254118461",
  "bbob-biobj_f13_i06_d03 0.999444869224228",
  "bbob-biobj_f13_i06_d05 0.996137458986228",
  "bbob-biobj_f13_i06_d10 0.999373545789059",
  "bbob-biobj_f13_i06_d20 0.971364486469867",
  "bbob-biobj_f13_i06_d40 0.976307325115025",
  "bbob-biobj_f13_i07_d02 0.998281928552673",
  "bbob-biobj_f13_i07_d03 0.999947731590753",
  "bbob-biobj_f13_i07_d05 0.957830089646704",
  "bbob-biobj_f13_i07_d10 0.997028847143220",
  "bbob-biobj_f13_i07_d20 0.986328888917346",
  "bbob-biobj_f13_i07_d40 0.993096262654648",
  "bbob-biobj_f13_i08_d02 0.986263129922459",
  "bbob-biobj_f13_i08_d03 0.987267926473285",
  "bbob-biobj_f13_i08_d05 0.998553113394840",
  "bbob-biobj_f13_i08_d10 0.997550762872402",
  "bbob-biobj_f13_i08_d20 0.992896154059295",
  "bbob-biobj_f13_i08_d40 0.993156766863208",
  "bbob-biobj_f13_i09_d02 0.999837952938782",
  "bbob-biobj_f13_i09_d03 0.999565585857190",
  "bbob-biobj_f13_i09_d05 0.993083126723309",
  "bbob-biobj_f13_i09_d10 0.996849023496994",
  "bbob-biobj_f13_i09_d20 0.998252533454138",
  "bbob-biobj_f13_i09_d40 0.985232252457824",
  "bbob-biobj_f13_i10_d02 0.999981682852072",
  "bbob-biobj_f13_i10_d03 0.998118852374587",
  "bbob-biobj_f13_i10_d05 0.993827383263134",
  "bbob-biobj_f13_i10_d10 0.999188102145963",
  "bbob-biobj_f13_i10_d20 0.999301545811323",
  "bbob-biobj_f13_i10_d40 0.988746713126202",
  "bbob-biobj_f13_i11_d02 1.0",
  "bbob-biobj_f13_i11_d03 1.0",
  "bbob-biobj_f13_i11_d05 1.0",
  "bbob-biobj_f13_i11_d10 1.0",
  "bbob-biobj_f13_i11_d20 1.0",
  "bbob-biobj_f13_i11_d40 1.0",
  "bbob-biobj_f13_i12_d02 1.0",
  "bbob-biobj_f13_i12_d03 1.0",
  "bbob-biobj_f13_i12_d05 1.0",
  "bbob-biobj_f13_i12_d10 1.0",
  "bbob-biobj_f13_i12_d20 1.0",
  "bbob-biobj_f13_i12_d40 1.0",
  "bbob-biobj_f13_i13_d02 1.0",
  "bbob-biobj_f13_i13_d03 1.0",
  "bbob-biobj_f13_i13_d05 1.0",
  "bbob-biobj_f13_i13_d10 1.0",
  "bbob-biobj_f13_i13_d20 1.0",
  "bbob-biobj_f13_i13_d40 1.0",
  "bbob-biobj_f13_i14_d02 1.0",
  "bbob-biobj_f13_i14_d03 1.0",
  "bbob-biobj_f13_i14_d05 1.0",
  "bbob-biobj_f13_i14_d10 1.0",
  "bbob-biobj_f13_i14_d20 1.0",
  "bbob-biobj_f13_i14_d40 1.0",
  "bbob-biobj_f13_i15_d02 1.0",
  "bbob-biobj_f13_i15_d03 1.0",
  "bbob-biobj_f13_i15_d05 1.0",
  "bbob-biobj_f13_i15_d10 1.0",
  "bbob-biobj_f13_i15_d20 1.0",
  "bbob-biobj_f13_i15_d40 1.0",
  "bbob-biobj_f14_i01_d02 0.912474502758044",
  "bbob-biobj_f14_i01_d03 0.832552981303613",
  "bbob-biobj_f14_i01_d05 0.935814561082043",
  "bbob-biobj_f14_i01_d10 0.856132946104561",
  "bbob-biobj_f14_i01_d20 0.838350394077427",
  "bbob-biobj_f14_i01_d40 0.858947036385989",
  "bbob-biobj_f14_i02_d02 0.997754601564265",
  "bbob-biobj_f14_i02_d03 0.996014288588860",
  "bbob-biobj_f14_i02_d05 0.842217695273373",
  "bbob-biobj_f14_i02_d10 0.880401726227052",
  "bbob-biobj_f14_i02_d20 0.909312507872731",
  "bbob-biobj_f14_i02_d40 0.872697649399011",
  "bbob-biobj_f14_i03_d02 0.966433483456690",
  "bbob-biobj_f14_i03_d03 0.983421791352917",
  "bbob-biobj_f14_i03_d05 0.792220674897986",
  "bbob-biobj_f14_i03_d10 0.931296898791649",
  "bbob-biobj_f14_i03_d20 0.918558442682632",
  "bbob-biobj_f14_i03_d40 0.866794368082230",
  "bbob-biobj_f14_i04_d02 0.982900075173571",
  "bbob-biobj_f14_i04_d03 0.999128374327137",
  "bbob-biobj_f14_i04_d05 0.854119062144995",
  "bbob-biobj_f14_i04_d10 0.871684724976058",
  "bbob-biobj_f14_i04_d20 0.902486214917498",
  "bbob-biobj_f14_i04_d40 0.905428785220358",
  "bbob-biobj_f14_i05_d02 0.987028385153950",
  "bbob-biobj_f14_i05_d03 0.986703979079455",
  "bbob-biobj_f14_i05_d05 0.966176301186096",
  "bbob-biobj_f14_i05_d10 0.940533252230839",
  "bbob-biobj_f14_i05_d20 0.885673184125202",
  "bbob-biobj_f14_i05_d40 0.880595320603629",
  "bbob-biobj_f14_i06_d02 0.989626579264181",
  "bbob-biobj_f14_i06_d03 0.967890488352312",
  "bbob-biobj_f14_i06_d05 0.951167651281388",
  "bbob-biobj_f14_i06_d10 0.969713179805528",
  "bbob-biobj_f14_i06_d20 0.877598942889784",
  "bbob-biobj_f14_i06_d40 0.884063329838707",
  "bbob-biobj_f14_i07_d02 0.888026642624433",
  "bbob-biobj_f14_i07_d03 0.994848097170878",
  "bbob-biobj_f14_i07_d05 0.993474418345262",
  "bbob-biobj_f14_i07_d10 0.918352873191297",
  "bbob-biobj_f14_i07_d20 0.846020394087209",
  "bbob-biobj_f14_i07_d40 0.886227765574340",
  "bbob-biobj_f14_i08_d02 0.785490555803731",
  "bbob-biobj_f14_i08_d03 0.812199702852755",
  "bbob-biobj_f14_i08_d05 0.963234245077621",
  "bbob-biobj_f14_i08_d10 0.837741778995617",
  "bbob-biobj_f14_i08_d20 0.864659883079803",
  "bbob-biobj_f14_i08_d40 0.888056073328803",
  "bbob-biobj_f14_i09_d02 0.999595374320139",
  "bbob-biobj_f14_i09_d03 0.836230878758369",
  "bbob-biobj_f14_i09_d05 0.828145060746739",
  "bbob-biobj_f14_i09_d10 0.903161047421889",
  "bbob-biobj_f14_i09_d20 0.900002248950141",
  "bbob-biobj_f14_i09_d40 0.862656054750155",
  "bbob-biobj_f14_i10_d02 0.980393179492728",
  "bbob-biobj_f14_i10_d03 0.871033820732873",
  "bbob-biobj_f14_i10_d05 0.987722998698588",
  "bbob-biobj_f14_i10_d10 0.804358216871116",
  "bbob-biobj_f14_i10_d20 0.948265232823403",
  "bbob-biobj_f14_i10_d40 0.879075253366757",
  "bbob-biobj_f14_i11_d02 1.0",
  "bbob-biobj_f14_i11_d03 1.0",
  "bbob-biobj_f14_i11_d05 1.0",
  "bbob-biobj_f14_i11_d10 1.0",
  "bbob-biobj_f14_i11_d20 1.0",
  "bbob-biobj_f14_i11_d40 1.0",
  "bbob-biobj_f14_i12_d02 1.0",
  "bbob-biobj_f14_i12_d03 1.0",
  "bbob-biobj_f14_i12_d05 1.0",
  "bbob-biobj_f14_i12_d10 1.0",
  "bbob-biobj_f14_i12_d20 1.0",
  "bbob-biobj_f14_i12_d40 1.0",
  "bbob-biobj_f14_i13_d02 1.0",
  "bbob-biobj_f14_i13_d03 1.0",
  "bbob-biobj_f14_i13_d05 1.0",
  "bbob-biobj_f14_i13_d10 1.0",
  "bbob-biobj_f14_i13_d20 1.0",
  "bbob-biobj_f14_i13_d40 1.0",
  "bbob-biobj_f14_i14_d02 1.0",
  "bbob-biobj_f14_i14_d03 1.0",
  "bbob-biobj_f14_i14_d05 1.0",
  "bbob-biobj_f14_i14_d10 1.0",
  "bbob-biobj_f14_i14_d20 1.0",
  "bbob-biobj_f14_i14_d40 1.0",
  "bbob-biobj_f14_i15_d02 1.0",
  "bbob-biobj_f14_i15_d03 1.0",
  "bbob-biobj_f14_i15_d05 1.0",
  "bbob-biobj_f14_i15_d10 1.0",
  "bbob-biobj_f14_i15_d20 1.0",
  "bbob-biobj_f14_i15_d40 1.0",
  "bbob-biobj_f15_i01_d02 0.978329958464595",
  "bbob-biobj_f15_i01_d03 0.928763486090370",
  "bbob-biobj_f15_i01_d05 0.948217515777184",
  "bbob-biobj_f15_i01_d10 0.965014252434939",
  "bbob-biobj_f15_i01_d20 0.988172010920960",
  "bbob-biobj_f15_i01_d40 0.979685330107244",
  "bbob-biobj_f15_i02_d02 0.941436984787556",
  "bbob-biobj_f15_i02_d03 0.954422322416083",
  "bbob-biobj_f15_i02_d05 0.979714682662946",
  "bbob-biobj_f15_i02_d10 0.992135269377142",
  "bbob-biobj_f15_i02_d20 0.957738383640000",
  "bbob-biobj_f15_i02_d40 0.979390495405073",
  "bbob-biobj_f15_i03_d02 0.998672057695252",
  "bbob-biobj_f15_i03_d03 0.994036428905887",
  "bbob-biobj_f15_i03_d05 0.980685118388359",
  "bbob-biobj_f15_i03_d10 0.991363095374915",
  "bbob-biobj_f15_i03_d20 0.982822742987328",
  "bbob-biobj_f15_i03_d40 0.980720242095878",
  "bbob-biobj_f15_i04_d02 0.825079630387599",
  "bbob-biobj_f15_i04_d03 0.975014914860254",
  "bbob-biobj_f15_i04_d05 0.944764896851763",
  "bbob-biobj_f15_i04_d10 0.994605465945425",
  "bbob-biobj_f15_i04_d20 0.973604110003641",
  "bbob-biobj_f15_i04_d40 0.993852848505890",
  "bbob-biobj_f15_i05_d02 0.988337565201807",
  "bbob-biobj_f15_i05_d03 0.932553406858140",
  "bbob-biobj_f15_i05_d05 0.867422911031452",
  "bbob-biobj_f15_i05_d10 0.985327846602946",
  "bbob-biobj_f15_i05_d20 0.988082796248327",
  "bbob-biobj_f15_i05_d40 0.992448395031419",
  "bbob-biobj_f15_i06_d02 0.969099569124418",
  "bbob-biobj_f15_i06_d03 0.955415507003273",
  "bbob-biobj_f15_i06_d05 0.975696462708374",
  "bbob-biobj_f15_i06_d10 0.985311334071489",
  "bbob-biobj_f15_i06_d20 0.985018339453307",
  "bbob-biobj_f15_i06_d40 0.977396751957974",
  "bbob-biobj_f15_i07_d02 0.982639481792181",
  "bbob-biobj_f15_i07_d03 0.904575831150473",
  "bbob-biobj_f15_i07_d05 0.993550969605167",
  "bbob-biobj_f15_i07_d10 0.956852344383798",
  "bbob-biobj_f15_i07_d20 0.977291264392811",
  "bbob-biobj_f15_i07_d40 0.982446927782356",
  "bbob-biobj_f15_i08_d02 0.925538670864120",
  "bbob-biobj_f15_i08_d03 0.886713197832339",
  "bbob-biobj_f15_i08_d05 0.977590338126398",
  "bbob-biobj_f15_i08_d10 0.987530968987965",
  "bbob-biobj_f15_i08_d20 0.972980219387174",
  "bbob-biobj_f15_i08_d40 0.987718098552715",
  "bbob-biobj_f15_i09_d02 0.985677211419677",
  "bbob-biobj_f15_i09_d03 0.988758882716527",
  "bbob-biobj_f15_i09_d05 0.932574336998797",
  "bbob-biobj_f15_i09_d10 0.974249913721194",
  "bbob-biobj_f15_i09_d20 0.951580788703233",
  "bbob-biobj_f15_i09_d40 0.981177296001791",
  "bbob-biobj_f15_i10_d02 0.932483359039571",
  "bbob-biobj_f15_i10_d03 0.760369148662153",
  "bbob-biobj_f15_i10_d05 0.962419327722367",
  "bbob-biobj_f15_i10_d10 0.968517586729314",
  "bbob-biobj_f15_i10_d20 0.983982954449852",
  "bbob-biobj_f15_i10_d40 0.989503840372814",
  "bbob-biobj_f15_i11_d02 1.0",
  "bbob-biobj_f15_i11_d03 1.0",
  "bbob-biobj_f15_i11_d05 1.0",
  "bbob-biobj_f15_i11_d10 1.0",
  "bbob-biobj_f15_i11_d20 1.0",
  "bbob-biobj_f15_i11_d40 1.0",
  "bbob-biobj_f15_i12_d02 1.0",
  "bbob-biobj_f15_i12_d03 1.0",
  "bbob-biobj_f15_i12_d05 1.0",
  "bbob-biobj_f15_i12_d10 1.0",
  "bbob-biobj_f15_i12_d20 1.0",
  "bbob-biobj_f15_i12_d40 1.0",
  "bbob-biobj_f15_i13_d02 1.0",
  "bbob-biobj_f15_i13_d03 1.0",
  "bbob-biobj_f15_i13_d05 1.0",
  "bbob-biobj_f15_i13_d10 1.0",
  "bbob-biobj_f15_i13_d20 1.0",
  "bbob-biobj_f15_i13_d40 1.0",
  "bbob-biobj_f15_i14_d02 1.0",
  "bbob-biobj_f15_i14_d03 1.0",
  "bbob-biobj_f15_i14_d05 1.0",
  "bbob-biobj_f15_i14_d10 1.0",
  "bbob-biobj_f15_i14_d20 1.0",
  "bbob-biobj_f15_i14_d40 1.0",
  "bbob-biobj_f15_i15_d02 1.0",
  "bbob-biobj_f15_i15_d03 1.0",
  "bbob-biobj_f15_i15_d05 1.0",
  "bbob-biobj_f15_i15_d10 1.0",
  "bbob-biobj_f15_i15_d20 1.0",
  "bbob-biobj_f15_i15_d40 1.0",
  "bbob-biobj_f16_i01_d02 0.981136805943563",
  "bbob-biobj_f16_i01_d03 0.999116528856403",
  "bbob-biobj_f16_i01_d05 0.991930799152053",
  "bbob-biobj_f16_i01_d10 0.985572038817464",
  "bbob-biobj_f16_i01_d20 0.975714609419363",
  "bbob-biobj_f16_i01_d40 0.956990166830896",
  "bbob-biobj_f16_i02_d02 0.965606293624667",
  "bbob-biobj_f16_i02_d03 0.934172295812466",
  "bbob-biobj_f16_i02_d05 0.954688360663971",
  "bbob-biobj_f16_i02_d10 0.962101153358912",
  "bbob-biobj_f16_i02_d20 0.979237967235992",
  "bbob-biobj_f16_i02_d40 0.958279379670396",
  "bbob-biobj_f16_i03_d02 0.959401186688791",
  "bbob-biobj_f16_i03_d03 0.943745810462135",
  "bbob-biobj_f16_i03_d05 0.990038348199901",
  "bbob-biobj_f16_i03_d10 0.968591811327548",
  "bbob-biobj_f16_i03_d20 0.989124335582215",
  "bbob-biobj_f16_i03_d40 0.970723944014379",
  "bbob-biobj_f16_i04_d02 0.995345570126699",
  "bbob-biobj_f16_i04_d03 0.997825564404976",
  "bbob-biobj_f16_i04_d05 0.984839714537455",
  "bbob-biobj_f16_i04_d10 0.985360159508032",
  "bbob-biobj_f16_i04_d20 0.980330681207284",
  "bbob-biobj_f16_i04_d40 0.966902504082311",
  "bbob-biobj_f16_i05_d02 0.999043230290520",
  "bbob-biobj_f16_i05_d03 0.953343574406745",
  "bbob-biobj_f16_i05_d05 0.995257304189170",
  "bbob-biobj_f16_i05_d10 0.974600005218726",
  "bbob-biobj_f16_i05_d20 0.978627892686058",
  "bbob-biobj_f16_i05_d40 0.981349905490900",
  "bbob-biobj_f16_i06_d02 0.139769187387063",
  "bbob-biobj_f16_i06_d03 0.932548729932145",
  "bbob-biobj_f16_i06_d05 0.966456880272056",
  "bbob-biobj_f16_i06_d10 0.978734131874224",
  "bbob-biobj_f16_i06_d20 0.984064160757473",
  "bbob-biobj_f16_i06_d40 0.936638336407728",
  "bbob-biobj_f16_i07_d02 0.997713548522536",
  "bbob-biobj_f16_i07_d03 0.955359679744685",
  "bbob-biobj_f16_i07_d05 0.994067502225735",
  "bbob-biobj_f16_i07_d10 0.954136637460491",
  "bbob-biobj_f16_i07_d20 0.972292039636678",
  "bbob-biobj_f16_i07_d40 0.946972200685131",
  "bbob-biobj_f16_i08_d02 0.968081586517578",
  "bbob-biobj_f16_i08_d03 0.985370098786813",
  "bbob-biobj_f16_i08_d05 0.982997262020142",
  "bbob-biobj_f16_i08_d10 0.983981918026254",
  "bbob-biobj_f16_i08_d20 0.992281167041330",
  "bbob-biobj_f16_i08_d40 0.906801714715668",
  "bbob-biobj_f16_i09_d02 0.971216964728232",
  "bbob-biobj_f16_i09_d03 0.973148118267147",
  "bbob-biobj_f16_i09_d05 0.975600577622356",
  "bbob-biobj_f16_i09_d10 0.993920099434157",
  "bbob-biobj_f16_i09_d20 0.980655093439768",
  "bbob-biobj_f16_i09_d40 0.964135748091886",
  "bbob-biobj_f16_i10_d02 0.928290354838167",
  "bbob-biobj_f16_i10_d03 0.992105778427930",
  "bbob-biobj_f16_i10_d05 0.955140928713979",
  "bbob-biobj_f16_i10_d10 0.993839635641078",
  "bbob-biobj_f16_i10_d20 0.985730027740502",
  "bbob-biobj_f16_i10_d40 0.949573101002550",
  "bbob-biobj_f16_i11_d02 1.0",
  "bbob-biobj_f16_i11_d03 1.0",
  "bbob-biobj_f16_i11_d05 1.0",
  "bbob-biobj_f16_i11_d10 1.0",
  "bbob-biobj_f16_i11_d20 1.0",
  "bbob-biobj_f16_i11_d40 1.0",
  "bbob-biobj_f16_i12_d02 1.0",
  "bbob-biobj_f16_i12_d03 1.0",
  "bbob-biobj_f16_i12_d05 1.0",
  "bbob-biobj_f16_i12_d10 1.0",
  "bbob-biobj_f16_i12_d20 1.0",
  "bbob-biobj_f16_i12_d40 1.0",
  "bbob-biobj_f16_i13_d02 1.0",
  "bbob-biobj_f16_i13_d03 1.0",
  "bbob-biobj_f16_i13_d05 1.0",
  "bbob-biobj_f16_i13_d10 1.0",
  "bbob-biobj_f16_i13_d20 1.0",
  "bbob-biobj_f16_i13_d40 1.0",
  "bbob-biobj_f16_i14_d02 1.0",
  "bbob-biobj_f16_i14_d03 1.0",
  "bbob-biobj_f16_i14_d05 1.0",
  "bbob-biobj_f16_i14_d10 1.0",
  "bbob-biobj_f16_i14_d20 1.0",
  "bbob-biobj_f16_i14_d40 1.0",
  "bbob-biobj_f16_i15_d02 1.0",
  "bbob-biobj_f16_i15_d03 1.0",
  "bbob-biobj_f16_i15_d05 1.0",
  "bbob-biobj_f16_i15_d10 1.0",
  "bbob-biobj_f16_i15_d20 1.0",
  "bbob-biobj_f16_i15_d40 1.0",
  "bbob-biobj_f17_i01_d02 0.979958736285246",
  "bbob-biobj_f17_i01_d03 0.899963827746614",
  "bbob-biobj_f17_i01_d05 0.980383162441116",
  "bbob-biobj_f17_i01_d10 0.991667904464048",
  "bbob-biobj_f17_i01_d20 0.976040274918623",
  "bbob-biobj_f17_i01_d40 0.977329603108451",
  "bbob-biobj_f17_i02_d02 0.942990267586704",
  "bbob-biobj_f17_i02_d03 0.931457132513428",
  "bbob-biobj_f17_i02_d05 0.971736847883754",
  "bbob-biobj_f17_i02_d10 0.982592256494084",
  "bbob-biobj_f17_i02_d20 0.977069573194543",
  "bbob-biobj_f17_i02_d40 0.977917762018740",
  "bbob-biobj_f17_i03_d02 0.941726980158077",
  "bbob-biobj_f17_i03_d03 0.956417557572814",
  "bbob-biobj_f17_i03_d05 0.967908237120145",
  "bbob-biobj_f17_i03_d10 0.990870743467526",
  "bbob-biobj_f17_i03_d20 0.988448519986969",
  "bbob-biobj_f17_i03_d40 0.956758020414636",
  "bbob-biobj_f17_i04_d02 0.766767768354889",
  "bbob-biobj_f17_i04_d03 0.861040742950137",
  "bbob-biobj_f17_i04_d05 0.995694983895964",
  "bbob-biobj_f17_i04_d10 0.993525252835279",
  "bbob-biobj_f17_i04_d20 0.994425581013768",
  "bbob-biobj_f17_i04_d40 0.987502438468208",
  "bbob-biobj_f17_i05_d02 0.940686879206378",
  "bbob-biobj_f17_i05_d03 0.855795658694083",
  "bbob-biobj_f17_i05_d05 0.989699773758120",
  "bbob-biobj_f17_i05_d10 0.988141216910086",
  "bbob-biobj_f17_i05_d20 0.998383309638694",
  "bbob-biobj_f17_i05_d40 0.970878934399012",
  "bbob-biobj_f17_i06_d02 0.987721346183989",
  "bbob-biobj_f17_i06_d03 0.933842682645463",
  "bbob-biobj_f17_i06_d05 0.995114738748214",
  "bbob-biobj_f17_i06_d10 0.988337346516209",
  "bbob-biobj_f17_i06_d20 0.986706062517203",
  "bbob-biobj_f17_i06_d40 0.929788434606291",
  "bbob-biobj_f17_i07_d02 0.996488999370768",
  "bbob-biobj_f17_i07_d03 0.955505256269616",
  "bbob-biobj_f17_i07_d05 0.997336955837073",
  "bbob-biobj_f17_i07_d10 0.994446054246338",
  "bbob-biobj_f17_i07_d20 0.998927058918791",
  "bbob-biobj_f17_i07_d40 0.960619879218413",
  "bbob-biobj_f17_i08_d02 0.889890847545346",
  "bbob-biobj_f17_i08_d03 0.847970499485007",
  "bbob-biobj_f17_i08_d05 0.935311368226691",
  "bbob-biobj_f17_i08_d10 0.992365339808906",
  "bbob-biobj_f17_i08_d20 0.982187603171574",
  "bbob-biobj_f17_i08_d40 0.946639360992838",
  "bbob-biobj_f17_i09_d02 0.939038516134587",
  "bbob-biobj_f17_i09_d03 0.986156728736634",
  "bbob-biobj_f17_i09_d05 0.953446290144602",
  "bbob-biobj_f17_i09_d10 0.991419467988624",
  "bbob-biobj_f17_i09_d20 0.994472378706110",
  "bbob-biobj_f17_i09_d40 0.919518078858342",
  "bbob-biobj_f17_i10_d02 0.998456596594862",
  "bbob-biobj_f17_i10_d03 0.954549550539068",
  "bbob-biobj_f17_i10_d05 0.982859786689298",
  "bbob-biobj_f17_i10_d10 0.994326972525940",
  "bbob-biobj_f17_i10_d20 0.989077431151756",
  "bbob-biobj_f17_i10_d40 0.946069625906736",
  "bbob-biobj_f17_i11_d02 1.0",
  "bbob-biobj_f17_i11_d03 1.0",
  "bbob-biobj_f17_i11_d05 1.0",
  "bbob-biobj_f17_i11_d10 1.0",
  "bbob-biobj_f17_i11_d20 1.0",
  "bbob-biobj_f17_i11_d40 1.0",
  "bbob-biobj_f17_i12_d02 1.0",
  "bbob-biobj_f17_i12_d03 1.0",
  "bbob-biobj_f17_i12_d05 1.0",
  "bbob-biobj_f17_i12_d10 1.0",
  "bbob-biobj_f17_i12_d20 1.0",
  "bbob-biobj_f17_i12_d40 1.0",
  "bbob-biobj_f17_i13_d02 1.0",
  "bbob-biobj_f17_i13_d03 1.0",
  "bbob-biobj_f17_i13_d05 1.0",
  "bbob-biobj_f17_i13_d10 1.0",
  "bbob-biobj_f17_i13_d20 1.0",
  "bbob-biobj_f17_i13_d40 1.0",
  "bbob-biobj_f17_i14_d02 1.0",
  "bbob-biobj_f17_i14_d03 1.0",
  "bbob-biobj_f17_i14_d05 1.0",
  "bbob-biobj_f17_i14_d10 1.0",
  "bbob-biobj_f17_i14_d20 1.0",
  "bbob-biobj_f17_i14_d40 1.0",
  "bbob-biobj_f17_i15_d02 1.0",
  "bbob-biobj_f17_i15_d03 1.0",
  "bbob-biobj_f17_i15_d05 1.0",
  "bbob-biobj_f17_i15_d10 1.0",
  "bbob-biobj_f17_i15_d20 1.0",
  "bbob-biobj_f17_i15_d40 1.0",
  "bbob-biobj_f18_i01_d02 0.969204955463581",
  "bbob-biobj_f18_i01_d03 0.998688677345057",
  "bbob-biobj_f18_i01_d05 0.998492990918753",
  "bbob-biobj_f18_i01_d10 0.992631755064189",
  "bbob-biobj_f18_i01_d20 0.947928180148035",
  "bbob-biobj_f18_i01_d40 0.946549388502231",
  "bbob-biobj_f18_i02_d02 0.953409672783533",
  "bbob-biobj_f18_i02_d03 0.993777613644610",
  "bbob-biobj_f18_i02_d05 0.991743086947726",
  "bbob-biobj_f18_i02_d10 0.962770435464250",
  "bbob-biobj_f18_i02_d20 0.950150835780217",
  "bbob-biobj_f18_i02_d40 0.985237661051139",
  "bbob-biobj_f18_i03_d02 0.999403474100123",
  "bbob-biobj_f18_i03_d03 0.999304357758777",
  "bbob-biobj_f18_i03_d05 0.946583934581978",
  "bbob-biobj_f18_i03_d10 0.992475805892152",
  "bbob-biobj_f18_i03_d20 0.960674247077403",
  "bbob-biobj_f18_i03_d40 0.977803190233087",
  "bbob-biobj_f18_i04_d02 0.990015009660456",
  "bbob-biobj_f18_i04_d03 0.998331133141258",
  "bbob-biobj_f18_i04_d05 0.973769141045947",
  "bbob-biobj_f18_i04_d10 0.950033214296972",
  "bbob-biobj_f18_i04_d20 0.933611850388737",
  "bbob-biobj_f18_i04_d40 0.983210373781807",
  "bbob-biobj_f18_i05_d02 0.999935348038347",
  "bbob-biobj_f18_i05_d03 0.975931995071483",
  "bbob-biobj_f18_i05_d05 0.941223676186121",
  "bbob-biobj_f18_i05_d10 0.954466530642470",
  "bbob-biobj_f18_i05_d20 0.975454397797652",
  "bbob-biobj_f18_i05_d40 0.972762716762444",
  "bbob-biobj_f18_i06_d02 0.999890664800389",
  "bbob-biobj_f18_i06_d03 0.951984261631391",
  "bbob-biobj_f18_i06_d05 0.972237437217906",
  "bbob-biobj_f18_i06_d10 0.983832662814702",
  "bbob-biobj_f18_i06_d20 0.988509799002198",
  "bbob-biobj_f18_i06_d40 0.943847382150639",
  "bbob-biobj_f18_i07_d02 0.999722916185884",
  "bbob-biobj_f18_i07_d03 0.939354361285139",
  "bbob-biobj_f18_i07_d05 0.973505022362569",
  "bbob-biobj_f18_i07_d10 0.956653936959765",
  "bbob-biobj_f18_i07_d20 0.975130531493727",
  "bbob-biobj_f18_i07_d40 0.963260507409640",
  "bbob-biobj_f18_i08_d02 0.979522633177982",
  "bbob-biobj_f18_i08_d03 0.983582859809175",
  "bbob-biobj_f18_i08_d05 0.972341214517837",
  "bbob-biobj_f18_i08_d10 0.943622394038077",
  "bbob-biobj_f18_i08_d20 0.985012761018232",
  "bbob-biobj_f18_i08_d40 0.967820102584116",
  "bbob-biobj_f18_i09_d02 0.826540269777394",
  "bbob-biobj_f18_i09_d03 0.955266931664022",
  "bbob-biobj_f18_i09_d05 0.990400188081170",
  "bbob-biobj_f18_i09_d10 0.973719812711369",
  "bbob-biobj_f18_i09_d20 0.927184296777493",
  "bbob-biobj_f18_i09_d40 0.940826217021873",
  "bbob-biobj_f18_i10_d02 0.981170857987276",
  "bbob-biobj_f18_i10_d03 0.977487404321554",
  "bbob-biobj_f18_i10_d05 0.973161994860676",
  "bbob-biobj_f18_i10_d10 0.930485909829110",
  "bbob-biobj_f18_i10_d20 0.958214791812810",
  "bbob-biobj_f18_i10_d40 0.936687847583049",
  "bbob-biobj_f18_i11_d02 1.0",
  "bbob-biobj_f18_i11_d03 1.0",
  "bbob-biobj_f18_i11_d05 1.0",
  "bbob-biobj_f18_i11_d10 1.0",
  "bbob-biobj_f18_i11_d20 1.0",
  "bbob-biobj_f18_i11_d40 1.0",
  "bbob-biobj_f18_i12_d02 1.0",
  "bbob-biobj_f18_i12_d03 1.0",
  "bbob-biobj_f18_i12_d05 1.0",
  "bbob-biobj_f18_i12_d10 1.0",
  "bbob-biobj_f18_i12_d20 1.0",
  "bbob-biobj_f18_i12_d40 1.0",
  "bbob-biobj_f18_i13_d02 1.0",
  "bbob-biobj_f18_i13_d03 1.0",
  "bbob-biobj_f18_i13_d05 1.0",
  "bbob-biobj_f18_i13_d10 1.0",
  "bbob-biobj_f18_i13_d20 1.0",
  "bbob-biobj_f18_i13_d40 1.0",
  "bbob-biobj_f18_i14_d02 1.0",
  "bbob-biobj_f18_i14_d03 1.0",
  "bbob-biobj_f18_i14_d05 1.0",
  "bbob-biobj_f18_i14_d10 1.0",
  "bbob-biobj_f18_i14_d20 1.0",
  "bbob-biobj_f18_i14_d40 1.0",
  "bbob-biobj_f18_i15_d02 1.0",
  "bbob-biobj_f18_i15_d03 1.0",
  "bbob-biobj_f18_i15_d05 1.0",
  "bbob-biobj_f18_i15_d10 1.0",
  "bbob-biobj_f18_i15_d20 1.0",
  "bbob-biobj_f18_i15_d40 1.0",
  "bbob-biobj_f19_i01_d02 0.865811770731921",
  "bbob-biobj_f19_i01_d03 0.973017247057512",
  "bbob-biobj_f19_i01_d05 0.992521544728759",
  "bbob-biobj_f19_i01_d10 0.992015676750431",
  "bbob-biobj_f19_i01_d20 0.985727207095561",
  "bbob-biobj_f19_i01_d40 0.968708111444335",
  "bbob-biobj_f19_i02_d02 0.920899455182180",
  "bbob-biobj_f19_i02_d03 0.986110943385011",
  "bbob-biobj_f19_i02_d05 0.989654348796015",
  "bbob-biobj_f19_i02_d10 0.998069566869596",
  "bbob-biobj_f19_i02_d20 0.971946796974695",
  "bbob-biobj_f19_i02_d40 0.950410504770783",
  "bbob-biobj_f19_i03_d02 0.904545299274402",
  "bbob-biobj_f19_i03_d03 0.957587365401640",
  "bbob-biobj_f19_i03_d05 0.982449904527118",
  "bbob-biobj_f19_i03_d10 0.991760867868237",
  "bbob-biobj_f19_i03_d20 0.993345216975816",
  "bbob-biobj_f19_i03_d40 0.963213924146087",
  "bbob-biobj_f19_i04_d02 0.999884821364810",
  "bbob-biobj_f19_i04_d03 0.996381011503881",
  "bbob-biobj_f19_i04_d05 0.992031985964079",
  "bbob-biobj_f19_i04_d10 0.992986860540315",
  "bbob-biobj_f19_i04_d20 0.976801738275965",
  "bbob-biobj_f19_i04_d40 0.972957229459967",
  "bbob-biobj_f19_i05_d02 0.997258709261627",
  "bbob-biobj_f19_i05_d03 0.959361859257914",
  "bbob-biobj_f19_i05_d05 0.993758691718374",
  "bbob-biobj_f19_i05_d10 0.992128970471116",
  "bbob-biobj_f19_i05_d20 0.984470434825430",
  "bbob-biobj_f19_i05_d40 0.939395739172570",
  "bbob-biobj_f19_i06_d02 0.955211504788045",
  "bbob-biobj_f19_i06_d03 0.991795027816023",
  "bbob-biobj_f19_i06_d05 0.995294664426083",
  "bbob-biobj_f19_i06_d10 0.993453063414076",
  "bbob-biobj_f19_i06_d20 0.988881226912081",
  "bbob-biobj_f19_i06_d40 0.930889367257548",
  "bbob-biobj_f19_i07_d02 0.916748358468225",
  "bbob-biobj_f19_i07_d03 0.976697911415346",
  "bbob-biobj_f19_i07_d05 0.998897439560711",
  "bbob-biobj_f19_i07_d10 0.985337158797574",
  "bbob-biobj_f19_i07_d20 0.984294297663750",
  "bbob-biobj_f19_i07_d40 0.977668769343021",
  "bbob-biobj_f19_i08_d02 0.954028862879188",
  "bbob-biobj_f19_i08_d03 0.958851958123935",
  "bbob-biobj_f19_i08_d05 0.981617688258936",
  "bbob-biobj_f19_i08_d10 0.991957920230594",
  "bbob-biobj_f19_i08_d20 0.993384581430229",
  "bbob-biobj_f19_i08_d40 0.931400004959781",
  "bbob-biobj_f19_i09_d02 0.949454854397529",
  "bbob-biobj_f19_i09_d03 0.968348328188755",
  "bbob-biobj_f19_i09_d05 0.957151920037808",
  "bbob-biobj_f19_i09_d10 0.990791529652153",
  "bbob-biobj_f19_i09_d20 0.981446422252370",
  "bbob-biobj_f19_i09_d40 0.961629256198723",
  "bbob-biobj_f19_i10_d02 0.946066683377767",
  "bbob-biobj_f19_i10_d03 0.980578550259608",
  "bbob-biobj_f19_i10_d05 0.999356136024621",
  "bbob-biobj_f19_i10_d10 0.988518494846568",
  "bbob-biobj_f19_i10_d20 0.988884026870925",
  "bbob-biobj_f19_i10_d40 0.896565501410778",
  "bbob-biobj_f19_i11_d02 1.0",
  "bbob-biobj_f19_i11_d03 1.0",
  "bbob-biobj_f19_i11_d05 1.0",
  "bbob-biobj_f19_i11_d10 1.0",
  "bbob-biobj_f19_i11_d20 1.0",
  "bbob-biobj_f19_i11_d40 1.0",
  "bbob-biobj_f19_i12_d02 1.0",
  "bbob-biobj_f19_i12_d03 1.0",
  "bbob-biobj_f19_i12_d05 1.0",
  "bbob-biobj_f19_i12_d10 1.0",
  "bbob-biobj_f19_i12_d20 1.0",
  "bbob-biobj_f19_i12_d40 1.0",
  "bbob-biobj_f19_i13_d02 1.0",
  "bbob-biobj_f19_i13_d03 1.0",
  "bbob-biobj_f19_i13_d05 1.0",
  "bbob-biobj_f19_i13_d10 1.0",
  "bbob-biobj_f19_i13_d20 1.0",
  "bbob-biobj_f19_i13_d40 1.0",
  "bbob-biobj_f19_i14_d02 1.0",
  "bbob-biobj_f19_i14_d03 1.0",
  "bbob-biobj_f19_i14_d05 1.0",
  "bbob-biobj_f19_i14_d10 1.0",
  "bbob-biobj_f19_i14_d20 1.0",
  "bbob-biobj_f19_i14_d40 1.0",
  "bbob-biobj_f19_i15_d02 1.0",
  "bbob-biobj_f19_i15_d03 1.0",
  "bbob-biobj_f19_i15_d05 1.0",
  "bbob-biobj_f19_i15_d10 1.0",
  "bbob-biobj_f19_i15_d20 1.0",
  "bbob-biobj_f19_i15_d40 1.0",
  "bbob-biobj_f20_i01_d02 0.995718026456638",
  "bbob-biobj_f20_i01_d03 0.813230762836442",
  "bbob-biobj_f20_i01_d05 0.967871002432375",
  "bbob-biobj_f20_i01_d10 0.999902739804120",
  "bbob-biobj_f20_i01_d20 0.999775497498613",
  "bbob-biobj_f20_i01_d40 0.999664116664435",
  "bbob-biobj_f20_i02_d02 0.910960748030164",
  "bbob-biobj_f20_i02_d03 0.974017477797567",
  "bbob-biobj_f20_i02_d05 0.814003461767120",
  "bbob-biobj_f20_i02_d10 0.999677569102844",
  "bbob-biobj_f20_i02_d20 0.999654004306429",
  "bbob-biobj_f20_i02_d40 0.999790560996347",
  "bbob-biobj_f20_i03_d02 0.985777930049723",
  "bbob-biobj_f20_i03_d03 0.999442800070996",
  "bbob-biobj_f20_i03_d05 0.892413297548470",
  "bbob-biobj_f20_i03_d10 0.972685717809694",
  "bbob-biobj_f20_i03_d20 0.999810203444123",
  "bbob-biobj_f20_i03_d40 0.997306868880074",
  "bbob-biobj_f20_i04_d02 0.890180220604319",
  "bbob-biobj_f20_i04_d03 0.925392845818037",
  "bbob-biobj_f20_i04_d05 0.999111564516406",
  "bbob-biobj_f20_i04_d10 0.999930995483195",
  "bbob-biobj_f20_i04_d20 0.999631609770450",
  "bbob-biobj_f20_i04_d40 0.999889174125538",
  "bbob-biobj_f20_i05_d02 0.980549386581144",
  "bbob-biobj_f20_i05_d03 0.963675021169463",
  "bbob-biobj_f20_i05_d05 0.946995300479762",
  "bbob-biobj_f20_i05_d10 0.999846864012334",
  "bbob-biobj_f20_i05_d20 0.958608264570341",
  "bbob-biobj_f20_i05_d40 0.999836960921451",
  "bbob-biobj_f20_i06_d02 0.869043334279351",
  "bbob-biobj_f20_i06_d03 0.999660960854221",
  "bbob-biobj_f20_i06_d05 0.888668538214566",
  "bbob-biobj_f20_i06_d10 0.921065924714680",
  "bbob-biobj_f20_i06_d20 0.999867187743562",
  "bbob-biobj_f20_i06_d40 0.999694714553354",
  "bbob-biobj_f20_i07_d02 0.951115687828409",
  "bbob-biobj_f20_i07_d03 0.864351550293221",
  "bbob-biobj_f20_i07_d05 0.999632572524201",
  "bbob-biobj_f20_i07_d10 0.972934973625756",
  "bbob-biobj_f20_i07_d20 0.982969587225086",
  "bbob-biobj_f20_i07_d40 0.999588862627488",
  "bbob-biobj_f20_i08_d02 0.840742965786629",
  "bbob-biobj_f20_i08_d03 0.849646747099705",
  "bbob-biobj_f20_i08_d05 0.994543846130131",
  "bbob-biobj_f20_i08_d10 0.999724942401950",
  "bbob-biobj_f20_i08_d20 0.999806100292651",
  "bbob-biobj_f20_i08_d40 0.999263843535335",
  "bbob-biobj_f20_i09_d02 0.998339920553951",
  "bbob-biobj_f20_i09_d03 0.999067290809170",
  "bbob-biobj_f20_i09_d05 0.963580072588558",
  "bbob-biobj_f20_i09_d10 0.999433695629111",
  "bbob-biobj_f20_i09_d20 0.999767702792318",
  "bbob-biobj_f20_i09_d40 0.999829335473340",
  "bbob-biobj_f20_i10_d02 0.994832672081393",
  "bbob-biobj_f20_i10_d03 0.945569229589977",
  "bbob-biobj_f20_i10_d05 0.859870669342559",
  "bbob-biobj_f20_i10_d10 0.999803407505220",
  "bbob-biobj_f20_i10_d20 0.999323836369105",
  "bbob-biobj_f20_i10_d40 0.998019183408066",
  "bbob-biobj_f20_i11_d02 1.0",
  "bbob-biobj_f20_i11_d03 1.0",
  "bbob-biobj_f20_i11_d05 1.0",
  "bbob-biobj_f20_i11_d10 1.0",
  "bbob-biobj_f20_i11_d20 1.0",
  "bbob-biobj_f20_i11_d40 1.0",
  "bbob-biobj_f20_i12_d02 1.0",
  "bbob-biobj_f20_i12_d03 1.0",
  "bbob-biobj_f20_i12_d05 1.0",
  "bbob-biobj_f20_i12_d10 1.0",
  "bbob-biobj_f20_i12_d20 1.0",
  "bbob-biobj_f20_i12_d40 1.0",
  "bbob-biobj_f20_i13_d02 1.0",
  "bbob-biobj_f20_i13_d03 1.0",
  "bbob-biobj_f20_i13_d05 1.0",
  "bbob-biobj_f20_i13_d10 1.0",
  "bbob-biobj_f20_i13_d20 1.0",
  "bbob-biobj_f20_i13_d40 1.0",
  "bbob-biobj_f20_i14_d02 1.0",
  "bbob-biobj_f20_i14_d03 1.0",
  "bbob-biobj_f20_i14_d05 1.0",
  "bbob-biobj_f20_i14_d10 1.0",
  "bbob-biobj_f20_i14_d20 1.0",
  "bbob-biobj_f20_i14_d40 1.0",
  "bbob-biobj_f20_i15_d02 1.0",
  "bbob-biobj_f20_i15_d03 1.0",
  "bbob-biobj_f20_i15_d05 1.0",
  "bbob-biobj_f20_i15_d10 1.0",
  "bbob-biobj_f20_i15_d20 1.0",
  "bbob-biobj_f20_i15_d40 1.0",
  "bbob-biobj_f21_i01_d02 0.999736618189728",
  "bbob-biobj_f21_i01_d03 0.911543728863680",
  "bbob-biobj_f21_i01_d05 0.912545592283940",
  "bbob-biobj_f21_i01_d10 0.993980399482700",
  "bbob-biobj_f21_i01_d20 0.981413338740226",
  "bbob-biobj_f21_i01_d40 0.983728574746616",
  "bbob-biobj_f21_i02_d02 0.985471502176322",
  "bbob-biobj_f21_i02_d03 0.980784705794501",
  "bbob-biobj_f21_i02_d05 0.998160647068493",
  "bbob-biobj_f21_i02_d10 0.997301495041459",
  "bbob-biobj_f21_i02_d20 0.993496201263574",
  "bbob-biobj_f21_i02_d40 0.996307110176717",
  "bbob-biobj_f21_i03_d02 0.973889905170571",
  "bbob-biobj_f21_i03_d03 0.968954320323192",
  "bbob-biobj_f21_i03_d05 0.929908020902274",
  "bbob-biobj_f21_i03_d10 0.998453798823771",
  "bbob-biobj_f21_i03_d20 0.993066368397647",
  "bbob-biobj_f21_i03_d40 0.995963598347175",
  "bbob-biobj_f21_i04_d02 0.999788754145244",
  "bbob-biobj_f21_i04_d03 0.954074968597077",
  "bbob-biobj_f21_i04_d05 0.928306129173680",
  "bbob-biobj_f21_i04_d10 0.904334442460224",
  "bbob-biobj_f21_i04_d20 0.996041198064085",
  "bbob-biobj_f21_i04_d40 0.968559650834793",
  "bbob-biobj_f21_i05_d02 0.890725681076297",
  "bbob-biobj_f21_i05_d03 0.999573377249035",
  "bbob-biobj_f21_i05_d05 0.997872881727144",
  "bbob-biobj_f21_i05_d10 0.958085510372521",
  "bbob-biobj_f21_i05_d20 0.982091917659635",
  "bbob-biobj_f21_i05_d40 0.985788879474571",
  "bbob-biobj_f21_i06_d02 0.998906461755895",
  "bbob-biobj_f21_i06_d03 0.898698179313327",
  "bbob-biobj_f21_i06_d05 0.999922910143998",
  "bbob-biobj_f21_i06_d10 0.993121495443903",
  "bbob-biobj_f21_i06_d20 0.995678603147139",
  "bbob-biobj_f21_i06_d40 0.990854319668004",
  "bbob-biobj_f21_i07_d02 0.862088180810070",
  "bbob-biobj_f21_i07_d03 0.994766867282409",
  "bbob-biobj_f21_i07_d05 0.999835182167843",
  "bbob-biobj_f21_i07_d10 0.997671966993714",
  "bbob-biobj_f21_i07_d20 0.953953966750356",
  "bbob-biobj_f21_i07_d40 0.978273920139613",
  "bbob-biobj_f21_i08_d02 0.986058931076662",
  "bbob-biobj_f21_i08_d03 0.999631463214444",
  "bbob-biobj_f21_i08_d05 0.942384927863932",
  "bbob-biobj_f21_i08_d10 0.996932728163747",
  "bbob-biobj_f21_i08_d20 0.988146384617908",
  "bbob-biobj_f21_i08_d40 0.998385840941191",
  "bbob-biobj_f21_i09_d02 0.972494611963315",
  "bbob-biobj_f21_i09_d03 0.998500373745383",
  "bbob-biobj_f21_i09_d05 0.998994382749117",
  "bbob-biobj_f21_i09_d10 0.981106741445144",
  "bbob-biobj_f21_i09_d20 0.981978280713147",
  "bbob-biobj_f21_i09_d40 0.989400835138566",
  "bbob-biobj_f21_i10_d02 0.940893377121817",
  "bbob-biobj_f21_i10_d03 0.949363879049512",
  "bbob-biobj_f21_i10_d05 0.979604346471867",
  "bbob-biobj_f21_i10_d10 0.977275303898697",
  "bbob-biobj_f21_i10_d20 0.972982656541522",
  "bbob-biobj_f21_i10_d40 0.997081448699098",  
  "bbob-biobj_f21_i11_d02 1.0",
  "bbob-biobj_f21_i11_d03 1.0",
  "bbob-biobj_f21_i11_d05 1.0",
  "bbob-biobj_f21_i11_d10 1.0",
  "bbob-biobj_f21_i11_d20 1.0",
  "bbob-biobj_f21_i11_d40 1.0",
  "bbob-biobj_f21_i12_d02 1.0",
  "bbob-biobj_f21_i12_d03 1.0",
  "bbob-biobj_f21_i12_d05 1.0",
  "bbob-biobj_f21_i12_d10 1.0",
  "bbob-biobj_f21_i12_d20 1.0",
  "bbob-biobj_f21_i12_d40 1.0",
  "bbob-biobj_f21_i13_d02 1.0",
  "bbob-biobj_f21_i13_d03 1.0",
  "bbob-biobj_f21_i13_d05 1.0",
  "bbob-biobj_f21_i13_d10 1.0",
  "bbob-biobj_f21_i13_d20 1.0",
  "bbob-biobj_f21_i13_d40 1.0",
  "bbob-biobj_f21_i14_d02 1.0",
  "bbob-biobj_f21_i14_d03 1.0",
  "bbob-biobj_f21_i14_d05 1.0",
  "bbob-biobj_f21_i14_d10 1.0",
  "bbob-biobj_f21_i14_d20 1.0",
  "bbob-biobj_f21_i14_d40 1.0",
  "bbob-biobj_f21_i15_d02 1.0",
  "bbob-biobj_f21_i15_d03 1.0",
  "bbob-biobj_f21_i15_d05 1.0",
  "bbob-biobj_f21_i15_d10 1.0",
  "bbob-biobj_f21_i15_d20 1.0",
  "bbob-biobj_f21_i15_d40 1.0",
  "bbob-biobj_f22_i01_d02 0.700916438458949",
  "bbob-biobj_f22_i01_d03 0.694484080130242",
  "bbob-biobj_f22_i01_d05 0.986978393520645",
  "bbob-biobj_f22_i01_d10 0.837909044739840",
  "bbob-biobj_f22_i01_d20 0.771234269693943",
  "bbob-biobj_f22_i01_d40 0.795805716891725",
  "bbob-biobj_f22_i02_d02 0.999055968888817",
  "bbob-biobj_f22_i02_d03 0.742603415635723",
  "bbob-biobj_f22_i02_d05 0.764278616146223",
  "bbob-biobj_f22_i02_d10 0.728760863755530",
  "bbob-biobj_f22_i02_d20 0.756692138841068",
  "bbob-biobj_f22_i02_d40 0.854998423517715",
  "bbob-biobj_f22_i03_d02 0.678523358470699",
  "bbob-biobj_f22_i03_d03 0.951238744037570",
  "bbob-biobj_f22_i03_d05 0.735426416509445",
  "bbob-biobj_f22_i03_d10 0.862594276552713",
  "bbob-biobj_f22_i03_d20 0.863135740546530",
  "bbob-biobj_f22_i03_d40 0.775341469507760",
  "bbob-biobj_f22_i04_d02 0.846367984738874",
  "bbob-biobj_f22_i04_d03 0.803806833320501",
  "bbob-biobj_f22_i04_d05 0.834646162414875",
  "bbob-biobj_f22_i04_d10 0.842317802659711",
  "bbob-biobj_f22_i04_d20 0.915887707245297",
  "bbob-biobj_f22_i04_d40 0.805194046957110",
  "bbob-biobj_f22_i05_d02 0.856039950461191",
  "bbob-biobj_f22_i05_d03 0.929859265278102",
  "bbob-biobj_f22_i05_d05 0.892887709868479",
  "bbob-biobj_f22_i05_d10 0.819642006415066",
  "bbob-biobj_f22_i05_d20 0.789452742291418",
  "bbob-biobj_f22_i05_d40 0.764365637153030",
  "bbob-biobj_f22_i06_d02 0.977995690715274",
  "bbob-biobj_f22_i06_d03 0.724548962936434",
  "bbob-biobj_f22_i06_d05 0.812081047098926",
  "bbob-biobj_f22_i06_d10 0.861080402018885",
  "bbob-biobj_f22_i06_d20 0.761773450745187",
  "bbob-biobj_f22_i06_d40 0.890819565561763",
  "bbob-biobj_f22_i07_d02 0.910044699639924",
  "bbob-biobj_f22_i07_d03 0.691497046586329",
  "bbob-biobj_f22_i07_d05 0.722689208326945",
  "bbob-biobj_f22_i07_d10 0.786295128946995",
  "bbob-biobj_f22_i07_d20 0.763717376257625",
  "bbob-biobj_f22_i07_d40 0.771665444522524",
  "bbob-biobj_f22_i08_d02 0.906996575764602",
  "bbob-biobj_f22_i08_d03 0.835949129023862",
  "bbob-biobj_f22_i08_d05 0.772262918155847",
  "bbob-biobj_f22_i08_d10 0.940087182487452",
  "bbob-biobj_f22_i08_d20 0.769339400709221",
  "bbob-biobj_f22_i08_d40 0.784553862839583",
  "bbob-biobj_f22_i09_d02 0.968992795251485",
  "bbob-biobj_f22_i09_d03 0.916981949396889",
  "bbob-biobj_f22_i09_d05 0.950507879748341",
  "bbob-biobj_f22_i09_d10 0.899124450109738",
  "bbob-biobj_f22_i09_d20 0.812683075569951",
  "bbob-biobj_f22_i09_d40 0.825849882521296",
  "bbob-biobj_f22_i10_d02 0.928391619677447",
  "bbob-biobj_f22_i10_d03 0.652494766185522",
  "bbob-biobj_f22_i10_d05 0.761558277329775",
  "bbob-biobj_f22_i10_d10 0.743060120366039",
  "bbob-biobj_f22_i10_d20 0.700275027790970",
  "bbob-biobj_f22_i10_d40 0.822934283135951",
  "bbob-biobj_f22_i11_d02 1.0",
  "bbob-biobj_f22_i11_d03 1.0",
  "bbob-biobj_f22_i11_d05 1.0",
  "bbob-biobj_f22_i11_d10 1.0",
  "bbob-biobj_f22_i11_d20 1.0",
  "bbob-biobj_f22_i11_d40 1.0",
  "bbob-biobj_f22_i12_d02 1.0",
  "bbob-biobj_f22_i12_d03 1.0",
  "bbob-biobj_f22_i12_d05 1.0",
  "bbob-biobj_f22_i12_d10 1.0",
  "bbob-biobj_f22_i12_d20 1.0",
  "bbob-biobj_f22_i12_d40 1.0",
  "bbob-biobj_f22_i13_d02 1.0",
  "bbob-biobj_f22_i13_d03 1.0",
  "bbob-biobj_f22_i13_d05 1.0",
  "bbob-biobj_f22_i13_d10 1.0",
  "bbob-biobj_f22_i13_d20 1.0",
  "bbob-biobj_f22_i13_d40 1.0",
  "bbob-biobj_f22_i14_d02 1.0",
  "bbob-biobj_f22_i14_d03 1.0",
  "bbob-biobj_f22_i14_d05 1.0",
  "bbob-biobj_f22_i14_d10 1.0",
  "bbob-biobj_f22_i14_d20 1.0",
  "bbob-biobj_f22_i14_d40 1.0",
  "bbob-biobj_f22_i15_d02 1.0",
  "bbob-biobj_f22_i15_d03 1.0",
  "bbob-biobj_f22_i15_d05 1.0",
  "bbob-biobj_f22_i15_d10 1.0",
  "bbob-biobj_f22_i15_d20 1.0",
  "bbob-biobj_f22_i15_d40 1.0",
  "bbob-biobj_f23_i01_d02 0.992534054268812",
  "bbob-biobj_f23_i01_d03 0.872237030138405",
  "bbob-biobj_f23_i01_d05 0.980182962061889",
  "bbob-biobj_f23_i01_d10 0.991308488908461",
  "bbob-biobj_f23_i01_d20 0.941127578039942",
  "bbob-biobj_f23_i01_d40 0.969136640388886",
  "bbob-biobj_f23_i02_d02 0.996445292624505",
  "bbob-biobj_f23_i02_d03 0.862095706452471",
  "bbob-biobj_f23_i02_d05 0.919722216097581",
  "bbob-biobj_f23_i02_d10 0.957106291743395",
  "bbob-biobj_f23_i02_d20 0.916213075021448",
  "bbob-biobj_f23_i02_d40 0.976239423933783",
  "bbob-biobj_f23_i03_d02 0.928112249455155",
  "bbob-biobj_f23_i03_d03 0.880344320238549",
  "bbob-biobj_f23_i03_d05 0.894403398636602",
  "bbob-biobj_f23_i03_d10 0.985390261740675",
  "bbob-biobj_f23_i03_d20 0.969070196929476",
  "bbob-biobj_f23_i03_d40 0.978589794415086",
  "bbob-biobj_f23_i04_d02 0.965419264936455",
  "bbob-biobj_f23_i04_d03 0.925757047133570",
  "bbob-biobj_f23_i04_d05 0.887977988719971",
  "bbob-biobj_f23_i04_d10 0.980016070326204",
  "bbob-biobj_f23_i04_d20 0.983100676088016",
  "bbob-biobj_f23_i04_d40 0.970816589257367",
  "bbob-biobj_f23_i05_d02 0.738909732737773",
  "bbob-biobj_f23_i05_d03 0.840966518574642",
  "bbob-biobj_f23_i05_d05 0.891343027607885",
  "bbob-biobj_f23_i05_d10 0.954458858293685",
  "bbob-biobj_f23_i05_d20 0.958522644454065",
  "bbob-biobj_f23_i05_d40 0.952036620183425",
  "bbob-biobj_f23_i06_d02 0.897625000364078",
  "bbob-biobj_f23_i06_d03 0.850455709554025",
  "bbob-biobj_f23_i06_d05 0.902953778767981",
  "bbob-biobj_f23_i06_d10 0.953871508298359",
  "bbob-biobj_f23_i06_d20 0.961806271708952",
  "bbob-biobj_f23_i06_d40 0.973529155239116",
  "bbob-biobj_f23_i07_d02 0.911732420138085",
  "bbob-biobj_f23_i07_d03 0.945122055110574",
  "bbob-biobj_f23_i07_d05 0.984532827314458",
  "bbob-biobj_f23_i07_d10 0.892809064107139",
  "bbob-biobj_f23_i07_d20 0.885739321899932",
  "bbob-biobj_f23_i07_d40 0.957880351545702",
  "bbob-biobj_f23_i08_d02 0.980413763708752",
  "bbob-biobj_f23_i08_d03 0.940189636105499",
  "bbob-biobj_f23_i08_d05 0.912738863710758",
  "bbob-biobj_f23_i08_d10 0.957188314008682",
  "bbob-biobj_f23_i08_d20 0.968134176347444",
  "bbob-biobj_f23_i08_d40 0.974280055391648",
  "bbob-biobj_f23_i09_d02 0.965467065790784",
  "bbob-biobj_f23_i09_d03 0.949242147095511",
  "bbob-biobj_f23_i09_d05 0.975255259508653",
  "bbob-biobj_f23_i09_d10 0.957152835866277",
  "bbob-biobj_f23_i09_d20 0.930236450073095",
  "bbob-biobj_f23_i09_d40 0.972137109059265",
  "bbob-biobj_f23_i10_d02 0.965992184639191",
  "bbob-biobj_f23_i10_d03 0.935204676725171",
  "bbob-biobj_f23_i10_d05 0.842935224036951",
  "bbob-biobj_f23_i10_d10 0.926191647516182",
  "bbob-biobj_f23_i10_d20 0.896701726989031",
  "bbob-biobj_f23_i10_d40 0.969163906891611",
  "bbob-biobj_f23_i11_d02 1.0",
  "bbob-biobj_f23_i11_d03 1.0",
  "bbob-biobj_f23_i11_d05 1.0",
  "bbob-biobj_f23_i11_d10 1.0",
  "bbob-biobj_f23_i11_d20 1.0",
  "bbob-biobj_f23_i11_d40 1.0",
  "bbob-biobj_f23_i12_d02 1.0",
  "bbob-biobj_f23_i12_d03 1.0",
  "bbob-biobj_f23_i12_d05 1.0",
  "bbob-biobj_f23_i12_d10 1.0",
  "bbob-biobj_f23_i12_d20 1.0",
  "bbob-biobj_f23_i12_d40 1.0",
  "bbob-biobj_f23_i13_d02 1.0",
  "bbob-biobj_f23_i13_d03 1.0",
  "bbob-biobj_f23_i13_d05 1.0",
  "bbob-biobj_f23_i13_d10 1.0",
  "bbob-biobj_f23_i13_d20 1.0",
  "bbob-biobj_f23_i13_d40 1.0",
  "bbob-biobj_f23_i14_d02 1.0",
  "bbob-biobj_f23_i14_d03 1.0",
  "bbob-biobj_f23_i14_d05 1.0",
  "bbob-biobj_f23_i14_d10 1.0",
  "bbob-biobj_f23_i14_d20 1.0",
  "bbob-biobj_f23_i14_d40 1.0",
  "bbob-biobj_f23_i15_d02 1.0",
  "bbob-biobj_f23_i15_d03 1.0",
  "bbob-biobj_f23_i15_d05 1.0",
  "bbob-biobj_f23_i15_d10 1.0",
  "bbob-biobj_f23_i15_d20 1.0",
  "bbob-biobj_f23_i15_d40 1.0",
  "bbob-biobj_f24_i01_d02 0.886263607678189",
  "bbob-biobj_f24_i01_d03 0.987114738235866",
  "bbob-biobj_f24_i01_d05 0.869900434826347",
  "bbob-biobj_f24_i01_d10 0.954785088308314",
  "bbob-biobj_f24_i01_d20 0.943800367180536",
  "bbob-biobj_f24_i01_d40 0.949986153251480",
  "bbob-biobj_f24_i02_d02 0.988139648340057",
  "bbob-biobj_f24_i02_d03 0.906308249603081",
  "bbob-biobj_f24_i02_d05 0.948010371784443",
  "bbob-biobj_f24_i02_d10 0.897865958220916",
  "bbob-biobj_f24_i02_d20 0.987059232687201",
  "bbob-biobj_f24_i02_d40 0.968007269361948",
  "bbob-biobj_f24_i03_d02 0.952811649213530",
  "bbob-biobj_f24_i03_d03 0.855068095684849",
  "bbob-biobj_f24_i03_d05 0.935534198962083",
  "bbob-biobj_f24_i03_d10 0.911447630079168",
  "bbob-biobj_f24_i03_d20 0.982501571019476",
  "bbob-biobj_f24_i03_d40 0.932892937400739",
  "bbob-biobj_f24_i04_d02 0.953732580833603",
  "bbob-biobj_f24_i04_d03 0.931783237315413",
  "bbob-biobj_f24_i04_d05 0.881151127927997",
  "bbob-biobj_f24_i04_d10 0.968856602598105",
  "bbob-biobj_f24_i04_d20 0.950459841911081",
  "bbob-biobj_f24_i04_d40 0.944023946290275",
  "bbob-biobj_f24_i05_d02 0.823877137496916",
  "bbob-biobj_f24_i05_d03 0.892824055307102",
  "bbob-biobj_f24_i05_d05 0.958081666884363",
  "bbob-biobj_f24_i05_d10 0.939871414605463",
  "bbob-biobj_f24_i05_d20 0.927432042163679",
  "bbob-biobj_f24_i05_d40 0.949016725327886",
  "bbob-biobj_f24_i06_d02 0.961551593219125",
  "bbob-biobj_f24_i06_d03 0.993394862477095",
  "bbob-biobj_f24_i06_d05 0.874370681463446",
  "bbob-biobj_f24_i06_d10 0.955515530362279",
  "bbob-biobj_f24_i06_d20 0.969428910758238",
  "bbob-biobj_f24_i06_d40 0.966430959788954",
  "bbob-biobj_f24_i07_d02 0.881087719966057",
  "bbob-biobj_f24_i07_d03 0.927025246802620",
  "bbob-biobj_f24_i07_d05 0.953592183500972",
  "bbob-biobj_f24_i07_d10 0.930995458287158",
  "bbob-biobj_f24_i07_d20 0.937478402473053",
  "bbob-biobj_f24_i07_d40 0.925844990010482",
  "bbob-biobj_f24_i08_d02 0.746749823443233",
  "bbob-biobj_f24_i08_d03 0.975106631219541",
  "bbob-biobj_f24_i08_d05 0.910590009382779",
  "bbob-biobj_f24_i08_d10 0.976367018095116",
  "bbob-biobj_f24_i08_d20 0.910905968470937",
  "bbob-biobj_f24_i08_d40 0.903930930125308",
  "bbob-biobj_f24_i09_d02 0.941309282349682",
  "bbob-biobj_f24_i09_d03 0.997025018274453",
  "bbob-biobj_f24_i09_d05 0.972035209899614",
  "bbob-biobj_f24_i09_d10 0.910417850467829",
  "bbob-biobj_f24_i09_d20 0.942610674256181",
  "bbob-biobj_f24_i09_d40 0.900887613079224",
  "bbob-biobj_f24_i10_d02 0.929447945801003",
  "bbob-biobj_f24_i10_d03 0.995092437929069",
  "bbob-biobj_f24_i10_d05 0.877392585145231",
  "bbob-biobj_f24_i10_d10 0.948433552062243",
  "bbob-biobj_f24_i10_d20 0.958192162368286",
  "bbob-biobj_f24_i10_d40 0.929477278156345",
  "bbob-biobj_f24_i11_d02 1.0",
  "bbob-biobj_f24_i11_d03 1.0",
  "bbob-biobj_f24_i11_d05 1.0",
  "bbob-biobj_f24_i11_d10 1.0",
  "bbob-biobj_f24_i11_d20 1.0",
  "bbob-biobj_f24_i11_d40 1.0",
  "bbob-biobj_f24_i12_d02 1.0",
  "bbob-biobj_f24_i12_d03 1.0",
  "bbob-biobj_f24_i12_d05 1.0",
  "bbob-biobj_f24_i12_d10 1.0",
  "bbob-biobj_f24_i12_d20 1.0",
  "bbob-biobj_f24_i12_d40 1.0",
  "bbob-biobj_f24_i13_d02 1.0",
  "bbob-biobj_f24_i13_d03 1.0",
  "bbob-biobj_f24_i13_d05 1.0",
  "bbob-biobj_f24_i13_d10 1.0",
  "bbob-biobj_f24_i13_d20 1.0",
  "bbob-biobj_f24_i13_d40 1.0",
  "bbob-biobj_f24_i14_d02 1.0",
  "bbob-biobj_f24_i14_d03 1.0",
  "bbob-biobj_f24_i14_d05 1.0",
  "bbob-biobj_f24_i14_d10 1.0",
  "bbob-biobj_f24_i14_d20 1.0",
  "bbob-biobj_f24_i14_d40 1.0",
  "bbob-biobj_f24_i15_d02 1.0",
  "bbob-biobj_f24_i15_d03 1.0",
  "bbob-biobj_f24_i15_d05 1.0",
  "bbob-biobj_f24_i15_d10 1.0",
  "bbob-biobj_f24_i15_d20 1.0",
  "bbob-biobj_f24_i15_d40 1.0",
  "bbob-biobj_f25_i01_d02 0.890543038798975",
  "bbob-biobj_f25_i01_d03 0.996743415999830",
  "bbob-biobj_f25_i01_d05 0.993207593187196",
  "bbob-biobj_f25_i01_d10 0.986814215720081",
  "bbob-biobj_f25_i01_d20 0.980467784715618",
  "bbob-biobj_f25_i01_d40 0.965373602633378",
  "bbob-biobj_f25_i02_d02 0.944370927074528",
  "bbob-biobj_f25_i02_d03 0.961712928294577",
  "bbob-biobj_f25_i02_d05 0.983975393346627",
  "bbob-biobj_f25_i02_d10 0.972675484610256",
  "bbob-biobj_f25_i02_d20 0.937188891611961",
  "bbob-biobj_f25_i02_d40 0.980258323163874",
  "bbob-biobj_f25_i03_d02 0.978560043300318",
  "bbob-biobj_f25_i03_d03 0.957789042519712",
  "bbob-biobj_f25_i03_d05 0.970084792278332",
  "bbob-biobj_f25_i03_d10 0.988227800329668",
  "bbob-biobj_f25_i03_d20 0.986540713045901",
  "bbob-biobj_f25_i03_d40 0.966498227782967",
  "bbob-biobj_f25_i04_d02 0.827380671367650",
  "bbob-biobj_f25_i04_d03 0.985284603938040",
  "bbob-biobj_f25_i04_d05 0.992678058338896",
  "bbob-biobj_f25_i04_d10 0.972909138535233",
  "bbob-biobj_f25_i04_d20 0.992295634647113",
  "bbob-biobj_f25_i04_d40 0.955202373496471",
  "bbob-biobj_f25_i05_d02 0.903625214843076",
  "bbob-biobj_f25_i05_d03 0.987244839347358",
  "bbob-biobj_f25_i05_d05 0.938620430685910",
  "bbob-biobj_f25_i05_d10 0.957995840802201",
  "bbob-biobj_f25_i05_d20 0.961012519647219",
  "bbob-biobj_f25_i05_d40 0.936720256813014",
  "bbob-biobj_f25_i06_d02 0.953597929830999",
  "bbob-biobj_f25_i06_d03 0.973854013837476",
  "bbob-biobj_f25_i06_d05 0.795906462798231",
  "bbob-biobj_f25_i06_d10 0.962497101024527",
  "bbob-biobj_f25_i06_d20 0.996988957184134",
  "bbob-biobj_f25_i06_d40 0.889472473043087",
  "bbob-biobj_f25_i07_d02 0.723624129010671",
  "bbob-biobj_f25_i07_d03 0.964012619854987",
  "bbob-biobj_f25_i07_d05 0.909500148998780",
  "bbob-biobj_f25_i07_d10 0.993126174158272",
  "bbob-biobj_f25_i07_d20 0.996350558340747",
  "bbob-biobj_f25_i07_d40 0.958555839241515",
  "bbob-biobj_f25_i08_d02 0.877837312082041",
  "bbob-biobj_f25_i08_d03 0.873300181393398",
  "bbob-biobj_f25_i08_d05 0.991399141885842",
  "bbob-biobj_f25_i08_d10 0.965310117151957",
  "bbob-biobj_f25_i08_d20 0.996417564685983",
  "bbob-biobj_f25_i08_d40 0.954034666543565",
  "bbob-biobj_f25_i09_d02 0.865140328123062",
  "bbob-biobj_f25_i09_d03 0.995947820347842",
  "bbob-biobj_f25_i09_d05 0.964898722604236",
  "bbob-biobj_f25_i09_d10 0.972570074149609",
  "bbob-biobj_f25_i09_d20 0.946782160081355",
  "bbob-biobj_f25_i09_d40 0.917252796575144",
  "bbob-biobj_f25_i10_d02 0.885990743877604",
  "bbob-biobj_f25_i10_d03 0.983929386161724",
  "bbob-biobj_f25_i10_d05 0.942512487443841",
  "bbob-biobj_f25_i10_d10 0.998291464917026",
  "bbob-biobj_f25_i10_d20 0.971696871369013",
  "bbob-biobj_f25_i10_d40 0.951976806873060",
  "bbob-biobj_f25_i11_d02 1.0",
  "bbob-biobj_f25_i11_d03 1.0",
  "bbob-biobj_f25_i11_d05 1.0",
  "bbob-biobj_f25_i11_d10 1.0",
  "bbob-biobj_f25_i11_d20 1.0",
  "bbob-biobj_f25_i11_d40 1.0",
  "bbob-biobj_f25_i12_d02 1.0",
  "bbob-biobj_f25_i12_d03 1.0",
  "bbob-biobj_f25_i12_d05 1.0",
  "bbob-biobj_f25_i12_d10 1.0",
  "bbob-biobj_f25_i12_d20 1.0",
  "bbob-biobj_f25_i12_d40 1.0",
  "bbob-biobj_f25_i13_d02 1.0",
  "bbob-biobj_f25_i13_d03 1.0",
  "bbob-biobj_f25_i13_d05 1.0",
  "bbob-biobj_f25_i13_d10 1.0",
  "bbob-biobj_f25_i13_d20 1.0",
  "bbob-biobj_f25_i13_d40 1.0",
  "bbob-biobj_f25_i14_d02 1.0",
  "bbob-biobj_f25_i14_d03 1.0",
  "bbob-biobj_f25_i14_d05 1.0",
  "bbob-biobj_f25_i14_d10 1.0",
  "bbob-biobj_f25_i14_d20 1.0",
  "bbob-biobj_f25_i14_d40 1.0",
  "bbob-biobj_f25_i15_d02 1.0",
  "bbob-biobj_f25_i15_d03 1.0",
  "bbob-biobj_f25_i15_d05 1.0",
  "bbob-biobj_f25_i15_d10 1.0",
  "bbob-biobj_f25_i15_d20 1.0",
  "bbob-biobj_f25_i15_d40 1.0",
  "bbob-biobj_f26_i01_d02 0.978921982681011",
  "bbob-biobj_f26_i01_d03 0.999860007752057",
  "bbob-biobj_f26_i01_d05 0.949035194489816",
  "bbob-biobj_f26_i01_d10 0.999623861039334",
  "bbob-biobj_f26_i01_d20 0.999902274088330",
  "bbob-biobj_f26_i01_d40 0.999593261593257",
  "bbob-biobj_f26_i02_d02 0.994494713596938",
  "bbob-biobj_f26_i02_d03 0.988829353431115",
  "bbob-biobj_f26_i02_d05 0.979222106784557",
  "bbob-biobj_f26_i02_d10 0.999635351294537",
  "bbob-biobj_f26_i02_d20 0.996348467425628",
  "bbob-biobj_f26_i02_d40 0.997476701245914",
  "bbob-biobj_f26_i03_d02 0.999888275992974",
  "bbob-biobj_f26_i03_d03 0.996486879482429",
  "bbob-biobj_f26_i03_d05 0.984010781647230",
  "bbob-biobj_f26_i03_d10 0.997283808894049",
  "bbob-biobj_f26_i03_d20 0.999942747534485",
  "bbob-biobj_f26_i03_d40 0.999675454511395",
  "bbob-biobj_f26_i04_d02 0.929919263339422",
  "bbob-biobj_f26_i04_d03 0.996886251406220",
  "bbob-biobj_f26_i04_d05 0.965393641764490",
  "bbob-biobj_f26_i04_d10 0.999902015865428",
  "bbob-biobj_f26_i04_d20 0.999910685806054",
  "bbob-biobj_f26_i04_d40 0.992570055689473",
  "bbob-biobj_f26_i05_d02 0.732359031442187",
  "bbob-biobj_f26_i05_d03 0.919257197619109",
  "bbob-biobj_f26_i05_d05 0.999565024330862",
  "bbob-biobj_f26_i05_d10 0.998146858129032",
  "bbob-biobj_f26_i05_d20 0.994309396955916",
  "bbob-biobj_f26_i05_d40 0.995553407285418",
  "bbob-biobj_f26_i06_d02 0.998259269935287",
  "bbob-biobj_f26_i06_d03 0.999730545891778",
  "bbob-biobj_f26_i06_d05 0.999950884709179",
  "bbob-biobj_f26_i06_d10 0.999961381926083",
  "bbob-biobj_f26_i06_d20 0.996029693542029",
  "bbob-biobj_f26_i06_d40 0.999915832962006",
  "bbob-biobj_f26_i07_d02 0.958138168639729",
  "bbob-biobj_f26_i07_d03 0.982256290624622",
  "bbob-biobj_f26_i07_d05 0.999648489126283",
  "bbob-biobj_f26_i07_d10 0.997384707121421",
  "bbob-biobj_f26_i07_d20 0.970139253020355",
  "bbob-biobj_f26_i07_d40 0.988318689116821",
  "bbob-biobj_f26_i08_d02 0.893980110702659",
  "bbob-biobj_f26_i08_d03 0.904151590886388",
  "bbob-biobj_f26_i08_d05 0.993795024317420",
  "bbob-biobj_f26_i08_d10 0.995847245270051",
  "bbob-biobj_f26_i08_d20 0.999614815433601",
  "bbob-biobj_f26_i08_d40 0.999580757097517",
  "bbob-biobj_f26_i09_d02 0.928186501872019",
  "bbob-biobj_f26_i09_d03 0.948229931197733",
  "bbob-biobj_f26_i09_d05 0.999708624389292",
  "bbob-biobj_f26_i09_d10 0.976182017061119",
  "bbob-biobj_f26_i09_d20 0.991850094961123",
  "bbob-biobj_f26_i09_d40 0.999529662180170",
  "bbob-biobj_f26_i10_d02 0.975780733014457",
  "bbob-biobj_f26_i10_d03 0.792423375244880",
  "bbob-biobj_f26_i10_d05 0.994495887173813",
  "bbob-biobj_f26_i10_d10 0.999418944318252",
  "bbob-biobj_f26_i10_d20 0.986079733726734",
  "bbob-biobj_f26_i10_d40 0.999900773346504",
  "bbob-biobj_f26_i11_d02 1.0",
  "bbob-biobj_f26_i11_d03 1.0",
  "bbob-biobj_f26_i11_d05 1.0",
  "bbob-biobj_f26_i11_d10 1.0",
  "bbob-biobj_f26_i11_d20 1.0",
  "bbob-biobj_f26_i11_d40 1.0",
  "bbob-biobj_f26_i12_d02 1.0",
  "bbob-biobj_f26_i12_d03 1.0",
  "bbob-biobj_f26_i12_d05 1.0",
  "bbob-biobj_f26_i12_d10 1.0",
  "bbob-biobj_f26_i12_d20 1.0",
  "bbob-biobj_f26_i12_d40 1.0",
  "bbob-biobj_f26_i13_d02 1.0",
  "bbob-biobj_f26_i13_d03 1.0",
  "bbob-biobj_f26_i13_d05 1.0",
  "bbob-biobj_f26_i13_d10 1.0",
  "bbob-biobj_f26_i13_d20 1.0",
  "bbob-biobj_f26_i13_d40 1.0",
  "bbob-biobj_f26_i14_d02 1.0",
  "bbob-biobj_f26_i14_d03 1.0",
  "bbob-biobj_f26_i14_d05 1.0",
  "bbob-biobj_f26_i14_d10 1.0",
  "bbob-biobj_f26_i14_d20 1.0",
  "bbob-biobj_f26_i14_d40 1.0",
  "bbob-biobj_f26_i15_d02 1.0",
  "bbob-biobj_f26_i15_d03 1.0",
  "bbob-biobj_f26_i15_d05 1.0",
  "bbob-biobj_f26_i15_d10 1.0",
  "bbob-biobj_f26_i15_d20 1.0",
  "bbob-biobj_f26_i15_d40 1.0",
  "bbob-biobj_f27_i01_d02 0.903502360859576",
  "bbob-biobj_f27_i01_d03 0.987126659857804",
  "bbob-biobj_f27_i01_d05 0.992204039648331",
  "bbob-biobj_f27_i01_d10 0.981633765867045",
  "bbob-biobj_f27_i01_d20 0.977456099993406",
  "bbob-biobj_f27_i01_d40 0.933361039015257",
  "bbob-biobj_f27_i02_d02 0.951959020655848",
  "bbob-biobj_f27_i02_d03 0.953826363356710",
  "bbob-biobj_f27_i02_d05 0.932663039854700",
  "bbob-biobj_f27_i02_d10 0.903607612085008",
  "bbob-biobj_f27_i02_d20 0.966846358922970",
  "bbob-biobj_f27_i02_d40 0.895128052744702",
  "bbob-biobj_f27_i03_d02 0.957759418302702",
  "bbob-biobj_f27_i03_d03 0.964059704651274",
  "bbob-biobj_f27_i03_d05 0.978491185290137",
  "bbob-biobj_f27_i03_d10 0.991362976843701",
  "bbob-biobj_f27_i03_d20 0.966360698252087",
  "bbob-biobj_f27_i03_d40 0.922382792737987",
  "bbob-biobj_f27_i04_d02 0.960857862516853",
  "bbob-biobj_f27_i04_d03 0.952867716096942",
  "bbob-biobj_f27_i04_d05 0.981450553799186",
  "bbob-biobj_f27_i04_d10 0.987221945469656",
  "bbob-biobj_f27_i04_d20 0.978986516494064",
  "bbob-biobj_f27_i04_d40 0.877932504329183",
  "bbob-biobj_f27_i05_d02 0.930274907344689",
  "bbob-biobj_f27_i05_d03 0.960952865988819",
  "bbob-biobj_f27_i05_d05 0.975426290474826",
  "bbob-biobj_f27_i05_d10 0.981754257654295",
  "bbob-biobj_f27_i05_d20 0.926159038181840",
  "bbob-biobj_f27_i05_d40 0.842053796842232",
  "bbob-biobj_f27_i06_d02 0.962264097364663",
  "bbob-biobj_f27_i06_d03 0.993501202677761",
  "bbob-biobj_f27_i06_d05 0.964516742175064",
  "bbob-biobj_f27_i06_d10 0.952513248144868",
  "bbob-biobj_f27_i06_d20 0.951520512743833",
  "bbob-biobj_f27_i06_d40 0.885894360808665",
  "bbob-biobj_f27_i07_d02 0.950865324455741",
  "bbob-biobj_f27_i07_d03 0.946807607449187",
  "bbob-biobj_f27_i07_d05 0.997102178147074",
  "bbob-biobj_f27_i07_d10 0.966057902059931",
  "bbob-biobj_f27_i07_d20 0.933505493507962",
  "bbob-biobj_f27_i07_d40 0.811723833872017",
  "bbob-biobj_f27_i08_d02 0.989196266545765",
  "bbob-biobj_f27_i08_d03 0.944526899527912",
  "bbob-biobj_f27_i08_d05 0.982111603130709",
  "bbob-biobj_f27_i08_d10 0.995676001826639",
  "bbob-biobj_f27_i08_d20 0.940450662237770",
  "bbob-biobj_f27_i08_d40 0.760563946752818",
  "bbob-biobj_f27_i09_d02 0.931460125810749",
  "bbob-biobj_f27_i09_d03 0.986123409450877",
  "bbob-biobj_f27_i09_d05 0.999683816880195",
  "bbob-biobj_f27_i09_d10 0.925587074782249",
  "bbob-biobj_f27_i09_d20 0.983192589804257",
  "bbob-biobj_f27_i09_d40 0.854360098666703",
  "bbob-biobj_f27_i10_d02 0.976309654875098",
  "bbob-biobj_f27_i10_d03 0.995372072303659",
  "bbob-biobj_f27_i10_d05 0.937317068589800",
  "bbob-biobj_f27_i10_d10 0.959888207198668",
  "bbob-biobj_f27_i10_d20 0.980521651346730",
  "bbob-biobj_f27_i10_d40 0.787348659613970",
  "bbob-biobj_f27_i11_d02 1.0",
  "bbob-biobj_f27_i11_d03 1.0",
  "bbob-biobj_f27_i11_d05 1.0",
  "bbob-biobj_f27_i11_d10 1.0",
  "bbob-biobj_f27_i11_d20 1.0",
  "bbob-biobj_f27_i11_d40 1.0",
  "bbob-biobj_f27_i12_d02 1.0",
  "bbob-biobj_f27_i12_d03 1.0",
  "bbob-biobj_f27_i12_d05 1.0",
  "bbob-biobj_f27_i12_d10 1.0",
  "bbob-biobj_f27_i12_d20 1.0",
  "bbob-biobj_f27_i12_d40 1.0",
  "bbob-biobj_f27_i13_d02 1.0",
  "bbob-biobj_f27_i13_d03 1.0",
  "bbob-biobj_f27_i13_d05 1.0",
  "bbob-biobj_f27_i13_d10 1.0",
  "bbob-biobj_f27_i13_d20 1.0",
  "bbob-biobj_f27_i13_d40 1.0",
  "bbob-biobj_f27_i14_d02 1.0",
  "bbob-biobj_f27_i14_d03 1.0",
  "bbob-biobj_f27_i14_d05 1.0",
  "bbob-biobj_f27_i14_d10 1.0",
  "bbob-biobj_f27_i14_d20 1.0",
  "bbob-biobj_f27_i14_d40 1.0",
  "bbob-biobj_f27_i15_d02 1.0",
  "bbob-biobj_f27_i15_d03 1.0",
  "bbob-biobj_f27_i15_d05 1.0",
  "bbob-biobj_f27_i15_d10 1.0",
  "bbob-biobj_f27_i15_d20 1.0",
  "bbob-biobj_f27_i15_d40 1.0",
  "bbob-biobj_f28_i01_d02 0.977401999066625",
  "bbob-biobj_f28_i01_d03 0.998639738664743",
  "bbob-biobj_f28_i01_d05 0.995557746846828",
  "bbob-biobj_f28_i01_d10 0.994072439832833",
  "bbob-biobj_f28_i01_d20 0.992463218537431",
  "bbob-biobj_f28_i01_d40 0.992414213096006",
  "bbob-biobj_f28_i02_d02 0.998930207760655",
  "bbob-biobj_f28_i02_d03 0.993519710437286",
  "bbob-biobj_f28_i02_d05 0.991806547665404",
  "bbob-biobj_f28_i02_d10 0.992230831814733",
  "bbob-biobj_f28_i02_d20 0.990009499042260",
  "bbob-biobj_f28_i02_d40 0.990106398998538",
  "bbob-biobj_f28_i03_d02 0.999666285076891",
  "bbob-biobj_f28_i03_d03 0.977207039636466",
  "bbob-biobj_f28_i03_d05 0.993682678054200",
  "bbob-biobj_f28_i03_d10 0.994082775916324",
  "bbob-biobj_f28_i03_d20 0.993735723579743",
  "bbob-biobj_f28_i03_d40 0.993556096096843",
  "bbob-biobj_f28_i04_d02 0.984721520157451",
  "bbob-biobj_f28_i04_d03 0.992074459843054",
  "bbob-biobj_f28_i04_d05 0.997393288512045",
  "bbob-biobj_f28_i04_d10 0.992803283337928",
  "bbob-biobj_f28_i04_d20 0.994521653588152",
  "bbob-biobj_f28_i04_d40 0.992190072632095",
  "bbob-biobj_f28_i05_d02 0.999901258252883",
  "bbob-biobj_f28_i05_d03 0.990127198930246",
  "bbob-biobj_f28_i05_d05 0.987358603286700",
  "bbob-biobj_f28_i05_d10 0.993739971345883",
  "bbob-biobj_f28_i05_d20 0.995300696754144",
  "bbob-biobj_f28_i05_d40 0.992485959020777",
  "bbob-biobj_f28_i06_d02 0.999626447634610",
  "bbob-biobj_f28_i06_d03 0.958212010427058",
  "bbob-biobj_f28_i06_d05 0.997899397583847",
  "bbob-biobj_f28_i06_d10 0.993885165299602",
  "bbob-biobj_f28_i06_d20 0.992419988958514",
  "bbob-biobj_f28_i06_d40 0.991015746943900",
  "bbob-biobj_f28_i07_d02 0.997315263406170",
  "bbob-biobj_f28_i07_d03 0.991252216118587",
  "bbob-biobj_f28_i07_d05 0.991916981494586",
  "bbob-biobj_f28_i07_d10 0.995119902702797",
  "bbob-biobj_f28_i07_d20 0.991092804449868",
  "bbob-biobj_f28_i07_d40 0.991380726848034",
  "bbob-biobj_f28_i08_d02 0.985044198727011",
  "bbob-biobj_f28_i08_d03 0.970025788092634",
  "bbob-biobj_f28_i08_d05 0.991797875341854",
  "bbob-biobj_f28_i08_d10 0.994125424477083",
  "bbob-biobj_f28_i08_d20 0.990425211605881",
  "bbob-biobj_f28_i08_d40 0.991812429475186",
  "bbob-biobj_f28_i09_d02 0.992233419283827",
  "bbob-biobj_f28_i09_d03 0.978709253453134",
  "bbob-biobj_f28_i09_d05 0.995106604135360",
  "bbob-biobj_f28_i09_d10 0.995406461110909",
  "bbob-biobj_f28_i09_d20 0.992532195946441",
  "bbob-biobj_f28_i09_d40 0.992693013689914",
  "bbob-biobj_f28_i10_d02 0.999840936604087",
  "bbob-biobj_f28_i10_d03 0.994850417699833",
  "bbob-biobj_f28_i10_d05 0.989213621719788",
  "bbob-biobj_f28_i10_d10 0.996382743554660",
  "bbob-biobj_f28_i10_d20 0.993636432001135",
  "bbob-biobj_f28_i10_d40 0.989841821448926",
  "bbob-biobj_f28_i11_d02 1.0",
  "bbob-biobj_f28_i11_d03 1.0",
  "bbob-biobj_f28_i11_d05 1.0",
  "bbob-biobj_f28_i11_d10 1.0",
  "bbob-biobj_f28_i11_d20 1.0",
  "bbob-biobj_f28_i11_d40 1.0",
  "bbob-biobj_f28_i12_d02 1.0",
  "bbob-biobj_f28_i12_d03 1.0",
  "bbob-biobj_f28_i12_d05 1.0",
  "bbob-biobj_f28_i12_d10 1.0",
  "bbob-biobj_f28_i12_d20 1.0",
  "bbob-biobj_f28_i12_d40 1.0",
  "bbob-biobj_f28_i13_d02 1.0",
  "bbob-biobj_f28_i13_d03 1.0",
  "bbob-biobj_f28_i13_d05 1.0",
  "bbob-biobj_f28_i13_d10 1.0",
  "bbob-biobj_f28_i13_d20 1.0",
  "bbob-biobj_f28_i13_d40 1.0",
  "bbob-biobj_f28_i14_d02 1.0",
  "bbob-biobj_f28_i14_d03 1.0",
  "bbob-biobj_f28_i14_d05 1.0",
  "bbob-biobj_f28_i14_d10 1.0",
  "bbob-biobj_f28_i14_d20 1.0",
  "bbob-biobj_f28_i14_d40 1.0",
  "bbob-biobj_f28_i15_d02 1.0",
  "bbob-biobj_f28_i15_d03 1.0",
  "bbob-biobj_f28_i15_d05 1.0",
  "bbob-biobj_f28_i15_d10 1.0",
  "bbob-biobj_f28_i15_d20 1.0",
  "bbob-biobj_f28_i15_d40 1.0",
  "bbob-biobj_f29_i01_d02 0.972457455413490",
  "bbob-biobj_f29_i01_d03 0.866370346657890",
  "bbob-biobj_f29_i01_d05 0.870445739074006",
  "bbob-biobj_f29_i01_d10 0.894388363117107",
  "bbob-biobj_f29_i01_d20 0.808762122504279",
  "bbob-biobj_f29_i01_d40 0.840453851920017",
  "bbob-biobj_f29_i02_d02 0.999165568899664",
  "bbob-biobj_f29_i02_d03 0.939195283003279",
  "bbob-biobj_f29_i02_d05 0.882828302937317",
  "bbob-biobj_f29_i02_d10 0.804966339230319",
  "bbob-biobj_f29_i02_d20 0.859893815195278",
  "bbob-biobj_f29_i02_d40 0.835724630030829",
  "bbob-biobj_f29_i03_d02 0.993280459075654",
  "bbob-biobj_f29_i03_d03 0.980543061350425",
  "bbob-biobj_f29_i03_d05 0.830303971380969",
  "bbob-biobj_f29_i03_d10 0.851093726256976",
  "bbob-biobj_f29_i03_d20 0.830680809881492",
  "bbob-biobj_f29_i03_d40 0.837222139669099",
  "bbob-biobj_f29_i04_d02 0.966909956807889",
  "bbob-biobj_f29_i04_d03 0.973842229090499",
  "bbob-biobj_f29_i04_d05 0.899730185027102",
  "bbob-biobj_f29_i04_d10 0.894932413687638",
  "bbob-biobj_f29_i04_d20 0.865286297173798",
  "bbob-biobj_f29_i04_d40 0.836656239480014",
  "bbob-biobj_f29_i05_d02 0.988529417699778",
  "bbob-biobj_f29_i05_d03 0.949159799844754",
  "bbob-biobj_f29_i05_d05 0.927476309493910",
  "bbob-biobj_f29_i05_d10 0.900624474319130",
  "bbob-biobj_f29_i05_d20 0.835998458242497",
  "bbob-biobj_f29_i05_d40 0.846002036747550",
  "bbob-biobj_f29_i06_d02 0.967561275916612",
  "bbob-biobj_f29_i06_d03 0.967381034072613",
  "bbob-biobj_f29_i06_d05 0.931360954695002",
  "bbob-biobj_f29_i06_d10 0.847570141723538",
  "bbob-biobj_f29_i06_d20 0.842328347108882",
  "bbob-biobj_f29_i06_d40 0.824828363278599",
  "bbob-biobj_f29_i07_d02 0.981144976196740",
  "bbob-biobj_f29_i07_d03 0.957221055688806",
  "bbob-biobj_f29_i07_d05 0.902497486504058",
  "bbob-biobj_f29_i07_d10 0.947016245190751",
  "bbob-biobj_f29_i07_d20 0.821568365741335",
  "bbob-biobj_f29_i07_d40 0.861935573164123",
  "bbob-biobj_f29_i08_d02 0.984747228463624",
  "bbob-biobj_f29_i08_d03 0.924610488247452",
  "bbob-biobj_f29_i08_d05 0.804302384537842",
  "bbob-biobj_f29_i08_d10 0.919025320584330",
  "bbob-biobj_f29_i08_d20 0.842093400939269",
  "bbob-biobj_f29_i08_d40 0.852737325066746",
  "bbob-biobj_f29_i09_d02 0.992967815171660",
  "bbob-biobj_f29_i09_d03 0.997316964918812",
  "bbob-biobj_f29_i09_d05 0.872403467217192",
  "bbob-biobj_f29_i09_d10 0.868612481971066",
  "bbob-biobj_f29_i09_d20 0.832062370273532",
  "bbob-biobj_f29_i09_d40 0.824499220933538",
  "bbob-biobj_f29_i10_d02 0.998399141958328",
  "bbob-biobj_f29_i10_d03 0.957728840680481",
  "bbob-biobj_f29_i10_d05 0.987548487098636",
  "bbob-biobj_f29_i10_d10 0.855764301444605",
  "bbob-biobj_f29_i10_d20 0.862220662567331",
  "bbob-biobj_f29_i10_d40 0.815090921214875",
  "bbob-biobj_f29_i11_d02 1.0",
  "bbob-biobj_f29_i11_d03 1.0",
  "bbob-biobj_f29_i11_d05 1.0",
  "bbob-biobj_f29_i11_d10 1.0",
  "bbob-biobj_f29_i11_d20 1.0",
  "bbob-biobj_f29_i11_d40 1.0",
  "bbob-biobj_f29_i12_d02 1.0",
  "bbob-biobj_f29_i12_d03 1.0",
  "bbob-biobj_f29_i12_d05 1.0",
  "bbob-biobj_f29_i12_d10 1.0",
  "bbob-biobj_f29_i12_d20 1.0",
  "bbob-biobj_f29_i12_d40 1.0",
  "bbob-biobj_f29_i13_d02 1.0",
  "bbob-biobj_f29_i13_d03 1.0",
  "bbob-biobj_f29_i13_d05 1.0",
  "bbob-biobj_f29_i13_d10 1.0",
  "bbob-biobj_f29_i13_d20 1.0",
  "bbob-biobj_f29_i13_d40 1.0",
  "bbob-biobj_f29_i14_d02 1.0",
  "bbob-biobj_f29_i14_d03 1.0",
  "bbob-biobj_f29_i14_d05 1.0",
  "bbob-biobj_f29_i14_d10 1.0",
  "bbob-biobj_f29_i14_d20 1.0",
  "bbob-biobj_f29_i14_d40 1.0",
  "bbob-biobj_f29_i15_d02 1.0",
  "bbob-biobj_f29_i15_d03 1.0",
  "bbob-biobj_f29_i15_d05 1.0",
  "bbob-biobj_f29_i15_d10 1.0",
  "bbob-biobj_f29_i15_d20 1.0",
  "bbob-biobj_f29_i15_d40 1.0",
  "bbob-biobj_f30_i01_d02 0.976212496500615",
  "bbob-biobj_f30_i01_d03 0.795600491238115",
  "bbob-biobj_f30_i01_d05 0.940212728170741",
  "bbob-biobj_f30_i01_d10 0.926791108055421",
  "bbob-biobj_f30_i01_d20 0.962016418104353",
  "bbob-biobj_f30_i01_d40 0.971355928747424",
  "bbob-biobj_f30_i02_d02 0.996222809065562",
  "bbob-biobj_f30_i02_d03 0.931861838448886",
  "bbob-biobj_f30_i02_d05 0.982392914075324",
  "bbob-biobj_f30_i02_d10 0.979277027146897",
  "bbob-biobj_f30_i02_d20 0.948080162530366",
  "bbob-biobj_f30_i02_d40 0.968917233282535",
  "bbob-biobj_f30_i03_d02 0.941596330281521",
  "bbob-biobj_f30_i03_d03 0.913955563877558",
  "bbob-biobj_f30_i03_d05 0.914687792830228",
  "bbob-biobj_f30_i03_d10 0.986335271790599",
  "bbob-biobj_f30_i03_d20 0.968017340368554",
  "bbob-biobj_f30_i03_d40 0.963762378676632",
  "bbob-biobj_f30_i04_d02 0.983030095609877",
  "bbob-biobj_f30_i04_d03 0.977328392211862",
  "bbob-biobj_f30_i04_d05 0.979633656408353",
  "bbob-biobj_f30_i04_d10 0.980904314822086",
  "bbob-biobj_f30_i04_d20 0.981008986766704",
  "bbob-biobj_f30_i04_d40 0.989250894160866",
  "bbob-biobj_f30_i05_d02 0.995608485934112",
  "bbob-biobj_f30_i05_d03 0.938514746776463",
  "bbob-biobj_f30_i05_d05 0.933174697747160",
  "bbob-biobj_f30_i05_d10 0.978395618518700",
  "bbob-biobj_f30_i05_d20 0.986572139258712",
  "bbob-biobj_f30_i05_d40 0.943187951117802",
  "bbob-biobj_f30_i06_d02 0.869044705496845",
  "bbob-biobj_f30_i06_d03 0.941160771663024",
  "bbob-biobj_f30_i06_d05 0.956684475241962",
  "bbob-biobj_f30_i06_d10 0.943060007508355",
  "bbob-biobj_f30_i06_d20 0.983039746676330",
  "bbob-biobj_f30_i06_d40 0.981497478183836",
  "bbob-biobj_f30_i07_d02 0.850960681023553",
  "bbob-biobj_f30_i07_d03 0.992467861821325",
  "bbob-biobj_f30_i07_d05 0.978309575157745",
  "bbob-biobj_f30_i07_d10 0.961499382394324",
  "bbob-biobj_f30_i07_d20 0.977389065119878",
  "bbob-biobj_f30_i07_d40 0.984027487033295",
  "bbob-biobj_f30_i08_d02 0.994801823711915",
  "bbob-biobj_f30_i08_d03 0.971842687259984",
  "bbob-biobj_f30_i08_d05 0.983779330291038",
  "bbob-biobj_f30_i08_d10 0.979515650534861",
  "bbob-biobj_f30_i08_d20 0.962484456332405",
  "bbob-biobj_f30_i08_d40 0.989696154928725",
  "bbob-biobj_f30_i09_d02 0.940370678583058",
  "bbob-biobj_f30_i09_d03 0.992737527012105",
  "bbob-biobj_f30_i09_d05 0.985290453392201",
  "bbob-biobj_f30_i09_d10 0.951809039137262",
  "bbob-biobj_f30_i09_d20 0.958857385033886",
  "bbob-biobj_f30_i09_d40 0.980920533375933",
  "bbob-biobj_f30_i10_d02 0.833084721027244",
  "bbob-biobj_f30_i10_d03 0.971038926173698",
  "bbob-biobj_f30_i10_d05 0.988040718239608",
  "bbob-biobj_f30_i10_d10 0.964535015378092",
  "bbob-biobj_f30_i10_d20 0.969259910717665",
  "bbob-biobj_f30_i10_d40 0.956033907644104",
  "bbob-biobj_f30_i11_d02 1.0",
  "bbob-biobj_f30_i11_d03 1.0",
  "bbob-biobj_f30_i11_d05 1.0",
  "bbob-biobj_f30_i11_d10 1.0",
  "bbob-biobj_f30_i11_d20 1.0",
  "bbob-biobj_f30_i11_d40 1.0",
  "bbob-biobj_f30_i12_d02 1.0",
  "bbob-biobj_f30_i12_d03 1.0",
  "bbob-biobj_f30_i12_d05 1.0",
  "bbob-biobj_f30_i12_d10 1.0",
  "bbob-biobj_f30_i12_d20 1.0",
  "bbob-biobj_f30_i12_d40 1.0",
  "bbob-biobj_f30_i13_d02 1.0",
  "bbob-biobj_f30_i13_d03 1.0",
  "bbob-biobj_f30_i13_d05 1.0",
  "bbob-biobj_f30_i13_d10 1.0",
  "bbob-biobj_f30_i13_d20 1.0",
  "bbob-biobj_f30_i13_d40 1.0",
  "bbob-biobj_f30_i14_d02 1.0",
  "bbob-biobj_f30_i14_d03 1.0",
  "bbob-biobj_f30_i14_d05 1.0",
  "bbob-biobj_f30_i14_d10 1.0",
  "bbob-biobj_f30_i14_d20 1.0",
  "bbob-biobj_f30_i14_d40 1.0",
  "bbob-biobj_f30_i15_d02 1.0",
  "bbob-biobj_f30_i15_d03 1.0",
  "bbob-biobj_f30_i15_d05 1.0",
  "bbob-biobj_f30_i15_d10 1.0",
  "bbob-biobj_f30_i15_d20 1.0",
  "bbob-biobj_f30_i15_d40 1.0",
  "bbob-biobj_f31_i01_d02 0.955984203318467",
  "bbob-biobj_f31_i01_d03 0.988038763109000",
  "bbob-biobj_f31_i01_d05 0.973310702562224",
  "bbob-biobj_f31_i01_d10 0.961863891126134",
  "bbob-biobj_f31_i01_d20 0.965891744541523",
  "bbob-biobj_f31_i01_d40 0.964799971076738",
  "bbob-biobj_f31_i02_d02 0.963697290724264",
  "bbob-biobj_f31_i02_d03 0.923775493019718",
  "bbob-biobj_f31_i02_d05 0.977110292317126",
  "bbob-biobj_f31_i02_d10 0.943832912280240",
  "bbob-biobj_f31_i02_d20 0.968205987196875",
  "bbob-biobj_f31_i02_d40 0.930030695053169",
  "bbob-biobj_f31_i03_d02 0.981722066734756",
  "bbob-biobj_f31_i03_d03 0.984776066232686",
  "bbob-biobj_f31_i03_d05 0.976057593637481",
  "bbob-biobj_f31_i03_d10 0.955101239990016",
  "bbob-biobj_f31_i03_d20 0.971423707814403",
  "bbob-biobj_f31_i03_d40 0.952189734459685",
  "bbob-biobj_f31_i04_d02 0.952958780406800",
  "bbob-biobj_f31_i04_d03 0.994016773366593",
  "bbob-biobj_f31_i04_d05 0.977998109456579",
  "bbob-biobj_f31_i04_d10 0.972504038682838",
  "bbob-biobj_f31_i04_d20 0.975249631453836",
  "bbob-biobj_f31_i04_d40 0.953391474750933",
  "bbob-biobj_f31_i05_d02 0.989988727984181",
  "bbob-biobj_f31_i05_d03 0.975602682258664",
  "bbob-biobj_f31_i05_d05 0.985516667669071",
  "bbob-biobj_f31_i05_d10 0.974740149273617",
  "bbob-biobj_f31_i05_d20 0.962503915381192",
  "bbob-biobj_f31_i05_d40 0.954674973148641",
  "bbob-biobj_f31_i06_d02 0.993357857224112",
  "bbob-biobj_f31_i06_d03 0.997778716569991",
  "bbob-biobj_f31_i06_d05 0.972233776312855",
  "bbob-biobj_f31_i06_d10 0.965556182517270",
  "bbob-biobj_f31_i06_d20 0.967195654811397",
  "bbob-biobj_f31_i06_d40 0.980987684830602",
  "bbob-biobj_f31_i07_d02 0.974743597398361",
  "bbob-biobj_f31_i07_d03 0.980529945532394",
  "bbob-biobj_f31_i07_d05 0.969699321870536",
  "bbob-biobj_f31_i07_d10 0.967682773907980",
  "bbob-biobj_f31_i07_d20 0.965103317096624",
  "bbob-biobj_f31_i07_d40 0.940523398304900",
  "bbob-biobj_f31_i08_d02 0.979779884141499",
  "bbob-biobj_f31_i08_d03 0.941945659669748",
  "bbob-biobj_f31_i08_d05 0.966924518837618",
  "bbob-biobj_f31_i08_d10 0.959246003469760",
  "bbob-biobj_f31_i08_d20 0.968526371519269",
  "bbob-biobj_f31_i08_d40 0.932858285108423",
  "bbob-biobj_f31_i09_d02 0.975510769784063",
  "bbob-biobj_f31_i09_d03 0.946197097176584",
  "bbob-biobj_f31_i09_d05 0.957823291215911",
  "bbob-biobj_f31_i09_d10 0.986319214118380",
  "bbob-biobj_f31_i09_d20 0.961832047996528",
  "bbob-biobj_f31_i09_d40 0.938909433905805",
  "bbob-biobj_f31_i10_d02 0.933842856787938",
  "bbob-biobj_f31_i10_d03 0.961940385993735",
  "bbob-biobj_f31_i10_d05 0.950563580668377",
  "bbob-biobj_f31_i10_d10 0.974970269649730",
  "bbob-biobj_f31_i10_d20 0.972029378489098",
  "bbob-biobj_f31_i10_d40 0.932931521286107",
  "bbob-biobj_f31_i11_d02 1.0",
  "bbob-biobj_f31_i11_d03 1.0",
  "bbob-biobj_f31_i11_d05 1.0",
  "bbob-biobj_f31_i11_d10 1.0",
  "bbob-biobj_f31_i11_d20 1.0",
  "bbob-biobj_f31_i11_d40 1.0",
  "bbob-biobj_f31_i12_d02 1.0",
  "bbob-biobj_f31_i12_d03 1.0",
  "bbob-biobj_f31_i12_d05 1.0",
  "bbob-biobj_f31_i12_d10 1.0",
  "bbob-biobj_f31_i12_d20 1.0",
  "bbob-biobj_f31_i12_d40 1.0",
  "bbob-biobj_f31_i13_d02 1.0",
  "bbob-biobj_f31_i13_d03 1.0",
  "bbob-biobj_f31_i13_d05 1.0",
  "bbob-biobj_f31_i13_d10 1.0",
  "bbob-biobj_f31_i13_d20 1.0",
  "bbob-biobj_f31_i13_d40 1.0",
  "bbob-biobj_f31_i14_d02 1.0",
  "bbob-biobj_f31_i14_d03 1.0",
  "bbob-biobj_f31_i14_d05 1.0",
  "bbob-biobj_f31_i14_d10 1.0",
  "bbob-biobj_f31_i14_d20 1.0",
  "bbob-biobj_f31_i14_d40 1.0",
  "bbob-biobj_f31_i15_d02 1.0",
  "bbob-biobj_f31_i15_d03 1.0",
  "bbob-biobj_f31_i15_d05 1.0",
  "bbob-biobj_f31_i15_d10 1.0",
  "bbob-biobj_f31_i15_d20 1.0",
  "bbob-biobj_f31_i15_d40 1.0",
  "bbob-biobj_f32_i01_d02 0.920330807593056",
  "bbob-biobj_f32_i01_d03 0.915347636063932",
  "bbob-biobj_f32_i01_d05 0.972868759339442",
  "bbob-biobj_f32_i01_d10 0.949751776226948",
  "bbob-biobj_f32_i01_d20 0.982261266068264",
  "bbob-biobj_f32_i01_d40 0.961314360293532",
  "bbob-biobj_f32_i02_d02 0.675234526590275",
  "bbob-biobj_f32_i02_d03 0.922157604765695",
  "bbob-biobj_f32_i02_d05 0.938926461979921",
  "bbob-biobj_f32_i02_d10 0.964591928399060",
  "bbob-biobj_f32_i02_d20 0.963820281610019",
  "bbob-biobj_f32_i02_d40 0.957708349242610",
  "bbob-biobj_f32_i03_d02 0.921767853591544",
  "bbob-biobj_f32_i03_d03 0.968199302056700",
  "bbob-biobj_f32_i03_d05 0.983208237869525",
  "bbob-biobj_f32_i03_d10 0.963910336376183",
  "bbob-biobj_f32_i03_d20 0.981327925011236",
  "bbob-biobj_f32_i03_d40 0.965118174109783",
  "bbob-biobj_f32_i04_d02 0.944003999651929",
  "bbob-biobj_f32_i04_d03 0.906091440313324",
  "bbob-biobj_f32_i04_d05 0.988063008394600",
  "bbob-biobj_f32_i04_d10 0.988153900557819",
  "bbob-biobj_f32_i04_d20 0.990484909433935",
  "bbob-biobj_f32_i04_d40 0.962618969844935",
  "bbob-biobj_f32_i05_d02 0.844964589477472",
  "bbob-biobj_f32_i05_d03 0.983689208376728",
  "bbob-biobj_f32_i05_d05 0.982677657147418",
  "bbob-biobj_f32_i05_d10 0.993430675481161",
  "bbob-biobj_f32_i05_d20 0.990033697332113",
  "bbob-biobj_f32_i05_d40 0.951484636110526",
  "bbob-biobj_f32_i06_d02 0.936204097698518",
  "bbob-biobj_f32_i06_d03 0.956619283725773",
  "bbob-biobj_f32_i06_d05 0.950112390592106",
  "bbob-biobj_f32_i06_d10 0.946943467616295",
  "bbob-biobj_f32_i06_d20 0.989994441779766",
  "bbob-biobj_f32_i06_d40 0.919737293782797",
  "bbob-biobj_f32_i07_d02 0.957436979340003",
  "bbob-biobj_f32_i07_d03 0.944961624486113",
  "bbob-biobj_f32_i07_d05 0.978229682653724",
  "bbob-biobj_f32_i07_d10 0.979327906952184",
  "bbob-biobj_f32_i07_d20 0.979526216532551",
  "bbob-biobj_f32_i07_d40 0.910205393765360",
  "bbob-biobj_f32_i08_d02 0.969123822394381",
  "bbob-biobj_f32_i08_d03 0.995708954595939",
  "bbob-biobj_f32_i08_d05 0.987304424437171",
  "bbob-biobj_f32_i08_d10 0.972058059457178",
  "bbob-biobj_f32_i08_d20 0.967219697949135",
  "bbob-biobj_f32_i08_d40 0.949088678292387",
  "bbob-biobj_f32_i09_d02 0.721600217882241",
  "bbob-biobj_f32_i09_d03 0.987357454724409",
  "bbob-biobj_f32_i09_d05 0.970303366825941",
  "bbob-biobj_f32_i09_d10 0.967120897293300",
  "bbob-biobj_f32_i09_d20 0.946781561034528",
  "bbob-biobj_f32_i09_d40 0.944323858726891",
  "bbob-biobj_f32_i10_d02 0.650841308813176",
  "bbob-biobj_f32_i10_d03 0.976666059973336",
  "bbob-biobj_f32_i10_d05 0.965946078228456",
  "bbob-biobj_f32_i10_d10 0.988498634332224",
  "bbob-biobj_f32_i10_d20 0.986184571295129",
  "bbob-biobj_f32_i10_d40 0.919965785330364",
  "bbob-biobj_f32_i11_d02 1.0",
  "bbob-biobj_f32_i11_d03 1.0",
  "bbob-biobj_f32_i11_d05 1.0",
  "bbob-biobj_f32_i11_d10 1.0",
  "bbob-biobj_f32_i11_d20 1.0",
  "bbob-biobj_f32_i11_d40 1.0",
  "bbob-biobj_f32_i12_d02 1.0",
  "bbob-biobj_f32_i12_d03 1.0",
  "bbob-biobj_f32_i12_d05 1.0",
  "bbob-biobj_f32_i12_d10 1.0",
  "bbob-biobj_f32_i12_d20 1.0",
  "bbob-biobj_f32_i12_d40 1.0",
  "bbob-biobj_f32_i13_d02 1.0",
  "bbob-biobj_f32_i13_d03 1.0",
  "bbob-biobj_f32_i13_d05 1.0",
  "bbob-biobj_f32_i13_d10 1.0",
  "bbob-biobj_f32_i13_d20 1.0",
  "bbob-biobj_f32_i13_d40 1.0",
  "bbob-biobj_f32_i14_d02 1.0",
  "bbob-biobj_f32_i14_d03 1.0",
  "bbob-biobj_f32_i14_d05 1.0",
  "bbob-biobj_f32_i14_d10 1.0",
  "bbob-biobj_f32_i14_d20 1.0",
  "bbob-biobj_f32_i14_d40 1.0",
  "bbob-biobj_f32_i15_d02 1.0",
  "bbob-biobj_f32_i15_d03 1.0",
  "bbob-biobj_f32_i15_d05 1.0",
  "bbob-biobj_f32_i15_d10 1.0",
  "bbob-biobj_f32_i15_d20 1.0",
  "bbob-biobj_f32_i15_d40 1.0",
  "bbob-biobj_f33_i01_d02 0.997888861007257",
  "bbob-biobj_f33_i01_d03 0.997624607259834",
  "bbob-biobj_f33_i01_d05 0.998342123785751",
  "bbob-biobj_f33_i01_d10 0.995514921361480",
  "bbob-biobj_f33_i01_d20 0.990229526419369",
  "bbob-biobj_f33_i01_d40 0.996701111011607",
  "bbob-biobj_f33_i02_d02 0.997492969261123",
  "bbob-biobj_f33_i02_d03 0.998822661203961",
  "bbob-biobj_f33_i02_d05 0.989647953655975",
  "bbob-biobj_f33_i02_d10 0.995932074985828",
  "bbob-biobj_f33_i02_d20 0.995150898775676",
  "bbob-biobj_f33_i02_d40 0.994052083498776",
  "bbob-biobj_f33_i03_d02 0.999823325771698",
  "bbob-biobj_f33_i03_d03 0.935833479827530",
  "bbob-biobj_f33_i03_d05 0.999018432915998",
  "bbob-biobj_f33_i03_d10 0.998456700267246",
  "bbob-biobj_f33_i03_d20 0.998381767176213",
  "bbob-biobj_f33_i03_d40 0.994147433157067",
  "bbob-biobj_f33_i04_d02 0.959892993583654",
  "bbob-biobj_f33_i04_d03 0.989064355606771",
  "bbob-biobj_f33_i04_d05 0.999673943582960",
  "bbob-biobj_f33_i04_d10 0.996077761850721",
  "bbob-biobj_f33_i04_d20 0.996916761370186",
  "bbob-biobj_f33_i04_d40 0.994789090637766",
  "bbob-biobj_f33_i05_d02 0.966110435471982",
  "bbob-biobj_f33_i05_d03 0.998861256315896",
  "bbob-biobj_f33_i05_d05 0.999857137978720",
  "bbob-biobj_f33_i05_d10 0.999704303631583",
  "bbob-biobj_f33_i05_d20 0.994312060281260",
  "bbob-biobj_f33_i05_d40 0.994376965244970",
  "bbob-biobj_f33_i06_d02 0.988009732046640",
  "bbob-biobj_f33_i06_d03 0.998480224248358",
  "bbob-biobj_f33_i06_d05 0.994232785423387",
  "bbob-biobj_f33_i06_d10 0.997422760448887",
  "bbob-biobj_f33_i06_d20 0.995665197211799",
  "bbob-biobj_f33_i06_d40 0.991042766384042",
  "bbob-biobj_f33_i07_d02 0.908430553337673",
  "bbob-biobj_f33_i07_d03 0.893739385154451",
  "bbob-biobj_f33_i07_d05 0.975829278284660",
  "bbob-biobj_f33_i07_d10 0.999470787751560",
  "bbob-biobj_f33_i07_d20 0.986509510317036",
  "bbob-biobj_f33_i07_d40 0.994583833048433",
  "bbob-biobj_f33_i08_d02 0.999902182898304",
  "bbob-biobj_f33_i08_d03 0.995374809986469",
  "bbob-biobj_f33_i08_d05 0.999239753141073",
  "bbob-biobj_f33_i08_d10 0.997624890172242",
  "bbob-biobj_f33_i08_d20 0.998073374826546",
  "bbob-biobj_f33_i08_d40 0.995614476200213",
  "bbob-biobj_f33_i09_d02 0.797570804275967",
  "bbob-biobj_f33_i09_d03 0.817567756795762",
  "bbob-biobj_f33_i09_d05 0.881506581668767",
  "bbob-biobj_f33_i09_d10 0.997480885377791",
  "bbob-biobj_f33_i09_d20 0.995999979147304",
  "bbob-biobj_f33_i09_d40 0.991403299118439",
  "bbob-biobj_f33_i10_d02 0.999798379887308",
  "bbob-biobj_f33_i10_d03 0.999774876932958",
  "bbob-biobj_f33_i10_d05 0.998698332765040",
  "bbob-biobj_f33_i10_d10 0.999978774834782",
  "bbob-biobj_f33_i10_d20 0.994436919909665",
  "bbob-biobj_f33_i10_d40 0.990882022342690",
  "bbob-biobj_f33_i11_d02 1.0",
  "bbob-biobj_f33_i11_d03 1.0",
  "bbob-biobj_f33_i11_d05 1.0",
  "bbob-biobj_f33_i11_d10 1.0",
  "bbob-biobj_f33_i11_d20 1.0",
  "bbob-biobj_f33_i11_d40 1.0",
  "bbob-biobj_f33_i12_d02 1.0",
  "bbob-biobj_f33_i12_d03 1.0",
  "bbob-biobj_f33_i12_d05 1.0",
  "bbob-biobj_f33_i12_d10 1.0",
  "bbob-biobj_f33_i12_d20 1.0",
  "bbob-biobj_f33_i12_d40 1.0",
  "bbob-biobj_f33_i13_d02 1.0",
  "bbob-biobj_f33_i13_d03 1.0",
  "bbob-biobj_f33_i13_d05 1.0",
  "bbob-biobj_f33_i13_d10 1.0",
  "bbob-biobj_f33_i13_d20 1.0",
  "bbob-biobj_f33_i13_d40 1.0",
  "bbob-biobj_f33_i14_d02 1.0",
  "bbob-biobj_f33_i14_d03 1.0",
  "bbob-biobj_f33_i14_d05 1.0",
  "bbob-biobj_f33_i14_d10 1.0",
  "bbob-biobj_f33_i14_d20 1.0",
  "bbob-biobj_f33_i14_d40 1.0",
  "bbob-biobj_f33_i15_d02 1.0",
  "bbob-biobj_f33_i15_d03 1.0",
  "bbob-biobj_f33_i15_d05 1.0",
  "bbob-biobj_f33_i15_d10 1.0",
  "bbob-biobj_f33_i15_d20 1.0",
  "bbob-biobj_f33_i15_d40 1.0",
  "bbob-biobj_f34_i01_d02 0.929758929322608",
  "bbob-biobj_f34_i01_d03 0.914925194601784",
  "bbob-biobj_f34_i01_d05 0.981337253946081",
  "bbob-biobj_f34_i01_d10 0.966135341522572",
  "bbob-biobj_f34_i01_d20 0.950027819314510",
  "bbob-biobj_f34_i01_d40 0.936086573703597",
  "bbob-biobj_f34_i02_d02 0.996825933655160",
  "bbob-biobj_f34_i02_d03 0.914730991843946",
  "bbob-biobj_f34_i02_d05 0.991099954828458",
  "bbob-biobj_f34_i02_d10 0.959859389596542",
  "bbob-biobj_f34_i02_d20 0.945916596486679",
  "bbob-biobj_f34_i02_d40 0.927594653284621",
  "bbob-biobj_f34_i03_d02 0.990964408471433",
  "bbob-biobj_f34_i03_d03 0.962894086641955",
  "bbob-biobj_f34_i03_d05 0.968963676660207",
  "bbob-biobj_f34_i03_d10 0.983124332815711",
  "bbob-biobj_f34_i03_d20 0.940566682476787",
  "bbob-biobj_f34_i03_d40 0.906946793901871",
  "bbob-biobj_f34_i04_d02 0.967251442667350",
  "bbob-biobj_f34_i04_d03 0.985996809599654",
  "bbob-biobj_f34_i04_d05 0.990433973580184",
  "bbob-biobj_f34_i04_d10 0.985663113470084",
  "bbob-biobj_f34_i04_d20 0.948754075971924",
  "bbob-biobj_f34_i04_d40 0.946951667926375",
  "bbob-biobj_f34_i05_d02 0.956107428609130",
  "bbob-biobj_f34_i05_d03 0.990692180330387",
  "bbob-biobj_f34_i05_d05 0.991916562797896",
  "bbob-biobj_f34_i05_d10 0.986752097164820",
  "bbob-biobj_f34_i05_d20 0.946958487980024",
  "bbob-biobj_f34_i05_d40 0.918444229830166",
  "bbob-biobj_f34_i06_d02 0.992429886449376",
  "bbob-biobj_f34_i06_d03 0.932399280827160",
  "bbob-biobj_f34_i06_d05 0.977893847715360",
  "bbob-biobj_f34_i06_d10 0.958563676459415",
  "bbob-biobj_f34_i06_d20 0.894811155382025",
  "bbob-biobj_f34_i06_d40 0.931703102080399",
  "bbob-biobj_f34_i07_d02 0.997954492490622",
  "bbob-biobj_f34_i07_d03 0.968805690123032",
  "bbob-biobj_f34_i07_d05 0.968367998531163",
  "bbob-biobj_f34_i07_d10 0.987731138382579",
  "bbob-biobj_f34_i07_d20 0.969075746932237",
  "bbob-biobj_f34_i07_d40 0.917332073668378",
  "bbob-biobj_f34_i08_d02 0.986443420424866",
  "bbob-biobj_f34_i08_d03 0.987754907741372",
  "bbob-biobj_f34_i08_d05 0.974500340117013",
  "bbob-biobj_f34_i08_d10 0.978729873615868",
  "bbob-biobj_f34_i08_d20 0.953820740741337",
  "bbob-biobj_f34_i08_d40 0.924674794407715",
  "bbob-biobj_f34_i09_d02 0.979374484880669",
  "bbob-biobj_f34_i09_d03 0.985062906638243",
  "bbob-biobj_f34_i09_d05 0.971686099090047",
  "bbob-biobj_f34_i09_d10 0.974275115903830",
  "bbob-biobj_f34_i09_d20 0.954195747728012",
  "bbob-biobj_f34_i09_d40 0.901234461994402",
  "bbob-biobj_f34_i10_d02 0.986288225521123",
  "bbob-biobj_f34_i10_d03 0.995621185099773",
  "bbob-biobj_f34_i10_d05 0.979832910022795",
  "bbob-biobj_f34_i10_d10 0.967225902546355",
  "bbob-biobj_f34_i10_d20 0.977569814442117",
  "bbob-biobj_f34_i10_d40 0.952895733599954",
  "bbob-biobj_f34_i11_d02 1.0",
  "bbob-biobj_f34_i11_d03 1.0",
  "bbob-biobj_f34_i11_d05 1.0",
  "bbob-biobj_f34_i11_d10 1.0",
  "bbob-biobj_f34_i11_d20 1.0",
  "bbob-biobj_f34_i11_d40 1.0",
  "bbob-biobj_f34_i12_d02 1.0",
  "bbob-biobj_f34_i12_d03 1.0",
  "bbob-biobj_f34_i12_d05 1.0",
  "bbob-biobj_f34_i12_d10 1.0",
  "bbob-biobj_f34_i12_d20 1.0",
  "bbob-biobj_f34_i12_d40 1.0",
  "bbob-biobj_f34_i13_d02 1.0",
  "bbob-biobj_f34_i13_d03 1.0",
  "bbob-biobj_f34_i13_d05 1.0",
  "bbob-biobj_f34_i13_d10 1.0",
  "bbob-biobj_f34_i13_d20 1.0",
  "bbob-biobj_f34_i13_d40 1.0",
  "bbob-biobj_f34_i14_d02 1.0",
  "bbob-biobj_f34_i14_d03 1.0",
  "bbob-biobj_f34_i14_d05 1.0",
  "bbob-biobj_f34_i14_d10 1.0",
  "bbob-biobj_f34_i14_d20 1.0",
  "bbob-biobj_f34_i14_d40 1.0",
  "bbob-biobj_f34_i15_d02 1.0",
  "bbob-biobj_f34_i15_d03 1.0",
  "bbob-biobj_f34_i15_d05 1.0",
  "bbob-biobj_f34_i15_d10 1.0",
  "bbob-biobj_f34_i15_d20 1.0",
  "bbob-biobj_f34_i15_d40 1.0",
  "bbob-biobj_f35_i01_d02 0.928165023377733",
  "bbob-biobj_f35_i01_d03 0.534055619821766",
  "bbob-biobj_f35_i01_d05 0.635785264847719",
  "bbob-biobj_f35_i01_d10 0.580893332905660",
  "bbob-biobj_f35_i01_d20 0.555009913932858",
  "bbob-biobj_f35_i01_d40 0.559291679999780",
  "bbob-biobj_f35_i02_d02 0.502442148070469",
  "bbob-biobj_f35_i02_d03 0.788715455127397",
  "bbob-biobj_f35_i02_d05 0.581829083029604",
  "bbob-biobj_f35_i02_d10 0.592322531696048",
  "bbob-biobj_f35_i02_d20 0.562948570954766",
  "bbob-biobj_f35_i02_d40 0.592245557480910",
  "bbob-biobj_f35_i03_d02 0.875499317155131",
  "bbob-biobj_f35_i03_d03 0.573982254861297",
  "bbob-biobj_f35_i03_d05 0.768271998392316",
  "bbob-biobj_f35_i03_d10 0.606855088657482",
  "bbob-biobj_f35_i03_d20 0.608162028996865",
  "bbob-biobj_f35_i03_d40 0.570890760357843",
  "bbob-biobj_f35_i04_d02 0.818254253194650",
  "bbob-biobj_f35_i04_d03 0.945470032951302",
  "bbob-biobj_f35_i04_d05 0.735474192489836",
  "bbob-biobj_f35_i04_d10 0.582620387060225",
  "bbob-biobj_f35_i04_d20 0.561822431849767",
  "bbob-biobj_f35_i04_d40 0.578381821406569",
  "bbob-biobj_f35_i05_d02 0.572776224251585",
  "bbob-biobj_f35_i05_d03 0.773381678533065",
  "bbob-biobj_f35_i05_d05 0.761556411704463",
  "bbob-biobj_f35_i05_d10 0.597858723534282",
  "bbob-biobj_f35_i05_d20 0.540863511855373",
  "bbob-biobj_f35_i05_d40 0.564959774511071",
  "bbob-biobj_f35_i06_d02 0.980545810362832",
  "bbob-biobj_f35_i06_d03 0.835515917249604",
  "bbob-biobj_f35_i06_d05 0.562675894859738",
  "bbob-biobj_f35_i06_d10 0.587998910768782",
  "bbob-biobj_f35_i06_d20 0.610158534435878",
  "bbob-biobj_f35_i06_d40 0.557990262720255",
  "bbob-biobj_f35_i07_d02 0.990109059705509",
  "bbob-biobj_f35_i07_d03 0.582852699773140",
  "bbob-biobj_f35_i07_d05 0.622869533970826",
  "bbob-biobj_f35_i07_d10 0.558675575323569",
  "bbob-biobj_f35_i07_d20 0.581910823709002",
  "bbob-biobj_f35_i07_d40 0.575992078136005",
  "bbob-biobj_f35_i08_d02 0.585094381837897",
  "bbob-biobj_f35_i08_d03 0.650187942772000",
  "bbob-biobj_f35_i08_d05 0.527195232798797",
  "bbob-biobj_f35_i08_d10 0.589619688257382",
  "bbob-biobj_f35_i08_d20 0.560957241658461",
  "bbob-biobj_f35_i08_d40 0.571976375123490",
  "bbob-biobj_f35_i09_d02 0.812412156635612",
  "bbob-biobj_f35_i09_d03 0.853294574665627",
  "bbob-biobj_f35_i09_d05 0.536664471012996",
  "bbob-biobj_f35_i09_d10 0.540732629900937",
  "bbob-biobj_f35_i09_d20 0.596217349423397",
  "bbob-biobj_f35_i09_d40 0.558307017123422",
  "bbob-biobj_f35_i10_d02 0.997531862926884",
  "bbob-biobj_f35_i10_d03 0.587724972246620",
  "bbob-biobj_f35_i10_d05 0.714494414954690",
  "bbob-biobj_f35_i10_d10 0.588778532125469",
  "bbob-biobj_f35_i10_d20 0.602194058692651",
  "bbob-biobj_f35_i10_d40 0.579319565668644",
  "bbob-biobj_f35_i11_d02 1.0",
  "bbob-biobj_f35_i11_d03 1.0",
  "bbob-biobj_f35_i11_d05 1.0",
  "bbob-biobj_f35_i11_d10 1.0",
  "bbob-biobj_f35_i11_d20 1.0",
  "bbob-biobj_f35_i11_d40 1.0",
  "bbob-biobj_f35_i12_d02 1.0",
  "bbob-biobj_f35_i12_d03 1.0",
  "bbob-biobj_f35_i12_d05 1.0",
  "bbob-biobj_f35_i12_d10 1.0",
  "bbob-biobj_f35_i12_d20 1.0",
  "bbob-biobj_f35_i12_d40 1.0",
  "bbob-biobj_f35_i13_d02 1.0",
  "bbob-biobj_f35_i13_d03 1.0",
  "bbob-biobj_f35_i13_d05 1.0",
  "bbob-biobj_f35_i13_d10 1.0",
  "bbob-biobj_f35_i13_d20 1.0",
  "bbob-biobj_f35_i13_d40 1.0",
  "bbob-biobj_f35_i14_d02 1.0",
  "bbob-biobj_f35_i14_d03 1.0",
  "bbob-biobj_f35_i14_d05 1.0",
  "bbob-biobj_f35_i14_d10 1.0",
  "bbob-biobj_f35_i14_d20 1.0",
  "bbob-biobj_f35_i14_d40 1.0",
  "bbob-biobj_f35_i15_d02 1.0",
  "bbob-biobj_f35_i15_d03 1.0",
  "bbob-biobj_f35_i15_d05 1.0",
  "bbob-biobj_f35_i15_d10 1.0",
  "bbob-biobj_f35_i15_d20 1.0",
  "bbob-biobj_f35_i15_d40 1.0",
  "bbob-biobj_f36_i01_d02 0.945914909650879",
  "bbob-biobj_f36_i01_d03 0.700109856462607",
  "bbob-biobj_f36_i01_d05 0.885952164445788",
  "bbob-biobj_f36_i01_d10 0.781262866659404",
  "bbob-biobj_f36_i01_d20 0.804662561948655",
  "bbob-biobj_f36_i01_d40 0.843727110022577",
  "bbob-biobj_f36_i02_d02 0.982339540243401",
  "bbob-biobj_f36_i02_d03 0.737153235635198",
  "bbob-biobj_f36_i02_d05 0.930168983454577",
  "bbob-biobj_f36_i02_d10 0.793783018152196",
  "bbob-biobj_f36_i02_d20 0.853227721845795",
  "bbob-biobj_f36_i02_d40 0.813629554178298",
  "bbob-biobj_f36_i03_d02 0.808775848047423",
  "bbob-biobj_f36_i03_d03 0.842944476275771",
  "bbob-biobj_f36_i03_d05 0.627315450736968",
  "bbob-biobj_f36_i03_d10 0.756222564706876",
  "bbob-biobj_f36_i03_d20 0.787428462209797",
  "bbob-biobj_f36_i03_d40 0.828040162624608",
  "bbob-biobj_f36_i04_d02 0.913204042095969",
  "bbob-biobj_f36_i04_d03 0.970232389126045",
  "bbob-biobj_f36_i04_d05 0.914724538410507",
  "bbob-biobj_f36_i04_d10 0.850577591819772",
  "bbob-biobj_f36_i04_d20 0.765304766819093",
  "bbob-biobj_f36_i04_d40 0.832480177648528",
  "bbob-biobj_f36_i05_d02 0.771462368593207",
  "bbob-biobj_f36_i05_d03 0.852784708249660",
  "bbob-biobj_f36_i05_d05 0.954700600300869",
  "bbob-biobj_f36_i05_d10 0.808824228788957",
  "bbob-biobj_f36_i05_d20 0.865957681426684",
  "bbob-biobj_f36_i05_d40 0.817384571058914",
  "bbob-biobj_f36_i06_d02 0.933757511624877",
  "bbob-biobj_f36_i06_d03 0.801490292261758",
  "bbob-biobj_f36_i06_d05 0.831167509137038",
  "bbob-biobj_f36_i06_d10 0.753171814797282",
  "bbob-biobj_f36_i06_d20 0.886786607453693",
  "bbob-biobj_f36_i06_d40 0.847003001535661",
  "bbob-biobj_f36_i07_d02 0.937778239862453",
  "bbob-biobj_f36_i07_d03 0.733737285335212",
  "bbob-biobj_f36_i07_d05 0.893219544430853",
  "bbob-biobj_f36_i07_d10 0.726223759918117",
  "bbob-biobj_f36_i07_d20 0.877639804021368",
  "bbob-biobj_f36_i07_d40 0.793394212282513",
  "bbob-biobj_f36_i08_d02 0.773830567033427",
  "bbob-biobj_f36_i08_d03 0.582247061486501",
  "bbob-biobj_f36_i08_d05 0.936003331558890",
  "bbob-biobj_f36_i08_d10 0.876721678494681",
  "bbob-biobj_f36_i08_d20 0.824199112721230",
  "bbob-biobj_f36_i08_d40 0.866349154828268",
  "bbob-biobj_f36_i09_d02 0.891599150476082",
  "bbob-biobj_f36_i09_d03 0.625247418888436",
  "bbob-biobj_f36_i09_d05 0.820474066076352",
  "bbob-biobj_f36_i09_d10 0.861572066804544",
  "bbob-biobj_f36_i09_d20 0.829598520946785",
  "bbob-biobj_f36_i09_d40 0.831002696373230",
  "bbob-biobj_f36_i10_d02 0.934505677966086",
  "bbob-biobj_f36_i10_d03 0.705534355336708",
  "bbob-biobj_f36_i10_d05 0.921117234905876",
  "bbob-biobj_f36_i10_d10 0.835625761352605",
  "bbob-biobj_f36_i10_d20 0.809940774549477",
  "bbob-biobj_f36_i10_d40 0.791407454276026",
  "bbob-biobj_f36_i11_d02 1.0",
  "bbob-biobj_f36_i11_d03 1.0",
  "bbob-biobj_f36_i11_d05 1.0",
  "bbob-biobj_f36_i11_d10 1.0",
  "bbob-biobj_f36_i11_d20 1.0",
  "bbob-biobj_f36_i11_d40 1.0",
  "bbob-biobj_f36_i12_d02 1.0",
  "bbob-biobj_f36_i12_d03 1.0",
  "bbob-biobj_f36_i12_d05 1.0",
  "bbob-biobj_f36_i12_d10 1.0",
  "bbob-biobj_f36_i12_d20 1.0",
  "bbob-biobj_f36_i12_d40 1.0",
  "bbob-biobj_f36_i13_d02 1.0",
  "bbob-biobj_f36_i13_d03 1.0",
  "bbob-biobj_f36_i13_d05 1.0",
  "bbob-biobj_f36_i13_d10 1.0",
  "bbob-biobj_f36_i13_d20 1.0",
  "bbob-biobj_f36_i13_d40 1.0",
  "bbob-biobj_f36_i14_d02 1.0",
  "bbob-biobj_f36_i14_d03 1.0",
  "bbob-biobj_f36_i14_d05 1.0",
  "bbob-biobj_f36_i14_d10 1.0",
  "bbob-biobj_f36_i14_d20 1.0",
  "bbob-biobj_f36_i14_d40 1.0",
  "bbob-biobj_f36_i15_d02 1.0",
  "bbob-biobj_f36_i15_d03 1.0",
  "bbob-biobj_f36_i15_d05 1.0",
  "bbob-biobj_f36_i15_d10 1.0",
  "bbob-biobj_f36_i15_d20 1.0",
  "bbob-biobj_f36_i15_d40 1.0",
  "bbob-biobj_f37_i01_d02 0.887327150785189",
  "bbob-biobj_f37_i01_d03 0.807659103583171",
  "bbob-biobj_f37_i01_d05 0.835169073581692",
  "bbob-biobj_f37_i01_d10 0.783383806133355",
  "bbob-biobj_f37_i01_d20 0.842538681321092",
  "bbob-biobj_f37_i01_d40 0.789266834004849",
  "bbob-biobj_f37_i02_d02 0.737269531615254",
  "bbob-biobj_f37_i02_d03 0.835012042630217",
  "bbob-biobj_f37_i02_d05 0.929695833351144",
  "bbob-biobj_f37_i02_d10 0.807367846432204",
  "bbob-biobj_f37_i02_d20 0.795982543055911",
  "bbob-biobj_f37_i02_d40 0.799926339032289",
  "bbob-biobj_f37_i03_d02 0.802436789939889",
  "bbob-biobj_f37_i03_d03 0.779817789428857",
  "bbob-biobj_f37_i03_d05 0.841136949398313",
  "bbob-biobj_f37_i03_d10 0.836194268914144",
  "bbob-biobj_f37_i03_d20 0.810391886715767",
  "bbob-biobj_f37_i03_d40 0.795436593770643",
  "bbob-biobj_f37_i04_d02 0.827630242603522",
  "bbob-biobj_f37_i04_d03 0.927472402212497",
  "bbob-biobj_f37_i04_d05 0.767439994597515",
  "bbob-biobj_f37_i04_d10 0.783346258032096",
  "bbob-biobj_f37_i04_d20 0.834870828901681",
  "bbob-biobj_f37_i04_d40 0.773457976069492",
  "bbob-biobj_f37_i05_d02 0.833851305921953",
  "bbob-biobj_f37_i05_d03 0.841497210688420",
  "bbob-biobj_f37_i05_d05 0.837678216567634",
  "bbob-biobj_f37_i05_d10 0.756617132180980",
  "bbob-biobj_f37_i05_d20 0.761274844547117",
  "bbob-biobj_f37_i05_d40 0.808772206948863",
  "bbob-biobj_f37_i06_d02 0.966420015253537",
  "bbob-biobj_f37_i06_d03 0.920849455423808",
  "bbob-biobj_f37_i06_d05 0.873507320873903",
  "bbob-biobj_f37_i06_d10 0.804471040357580",
  "bbob-biobj_f37_i06_d20 0.789492844708795",
  "bbob-biobj_f37_i06_d40 0.827597990486504",
  "bbob-biobj_f37_i07_d02 0.865979675631339",
  "bbob-biobj_f37_i07_d03 0.888456306531734",
  "bbob-biobj_f37_i07_d05 0.854814203846037",
  "bbob-biobj_f37_i07_d10 0.772690622496085",
  "bbob-biobj_f37_i07_d20 0.758857267001082",
  "bbob-biobj_f37_i07_d40 0.777389827183152",
  "bbob-biobj_f37_i08_d02 0.954408950578442",
  "bbob-biobj_f37_i08_d03 0.780283570455847",
  "bbob-biobj_f37_i08_d05 0.877427033200894",
  "bbob-biobj_f37_i08_d10 0.824052669824190",
  "bbob-biobj_f37_i08_d20 0.854319728694541",
  "bbob-biobj_f37_i08_d40 0.741015280728301",
  "bbob-biobj_f37_i09_d02 0.651230495436546",
  "bbob-biobj_f37_i09_d03 0.972084171390988",
  "bbob-biobj_f37_i09_d05 0.877591783597792",
  "bbob-biobj_f37_i09_d10 0.933689627469846",
  "bbob-biobj_f37_i09_d20 0.825557689086336",
  "bbob-biobj_f37_i09_d40 0.812287034624196",
  "bbob-biobj_f37_i10_d02 0.846357897009702",
  "bbob-biobj_f37_i10_d03 0.783860232975673",
  "bbob-biobj_f37_i10_d05 0.780136988735689",
  "bbob-biobj_f37_i10_d10 0.828012221896017",
  "bbob-biobj_f37_i10_d20 0.843542995824816",
  "bbob-biobj_f37_i10_d40 0.778211407123295",
  "bbob-biobj_f37_i11_d02 1.0",
  "bbob-biobj_f37_i11_d03 1.0",
  "bbob-biobj_f37_i11_d05 1.0",
  "bbob-biobj_f37_i11_d10 1.0",
  "bbob-biobj_f37_i11_d20 1.0",
  "bbob-biobj_f37_i11_d40 1.0",
  "bbob-biobj_f37_i12_d02 1.0",
  "bbob-biobj_f37_i12_d03 1.0",
  "bbob-biobj_f37_i12_d05 1.0",
  "bbob-biobj_f37_i12_d10 1.0",
  "bbob-biobj_f37_i12_d20 1.0",
  "bbob-biobj_f37_i12_d40 1.0",
  "bbob-biobj_f37_i13_d02 1.0",
  "bbob-biobj_f37_i13_d03 1.0",
  "bbob-biobj_f37_i13_d05 1.0",
  "bbob-biobj_f37_i13_d10 1.0",
  "bbob-biobj_f37_i13_d20 1.0",
  "bbob-biobj_f37_i13_d40 1.0",
  "bbob-biobj_f37_i14_d02 1.0",
  "bbob-biobj_f37_i14_d03 1.0",
  "bbob-biobj_f37_i14_d05 1.0",
  "bbob-biobj_f37_i14_d10 1.0",
  "bbob-biobj_f37_i14_d20 1.0",
  "bbob-biobj_f37_i14_d40 1.0",
  "bbob-biobj_f37_i15_d02 1.0",
  "bbob-biobj_f37_i15_d03 1.0",
  "bbob-biobj_f37_i15_d05 1.0",
  "bbob-biobj_f37_i15_d10 1.0",
  "bbob-biobj_f37_i15_d20 1.0",
  "bbob-biobj_f37_i15_d40 1.0",
  "bbob-biobj_f38_i01_d02 0.877156843506436",
  "bbob-biobj_f38_i01_d03 0.866660750237696",
  "bbob-biobj_f38_i01_d05 0.915603692393344",
  "bbob-biobj_f38_i01_d10 0.843099701921333",
  "bbob-biobj_f38_i01_d20 0.863967450160593",
  "bbob-biobj_f38_i01_d40 0.824400441198263",
  "bbob-biobj_f38_i02_d02 0.906613260227146",
  "bbob-biobj_f38_i02_d03 0.793052724368069",
  "bbob-biobj_f38_i02_d05 0.873182776085757",
  "bbob-biobj_f38_i02_d10 0.886942198063175",
  "bbob-biobj_f38_i02_d20 0.839510401884203",
  "bbob-biobj_f38_i02_d40 0.846825019783951",
  "bbob-biobj_f38_i03_d02 0.700820333532969",
  "bbob-biobj_f38_i03_d03 0.753316709210581",
  "bbob-biobj_f38_i03_d05 0.891034871722847",
  "bbob-biobj_f38_i03_d10 0.876129020211822",
  "bbob-biobj_f38_i03_d20 0.838138772336944",
  "bbob-biobj_f38_i03_d40 0.884522048799347",
  "bbob-biobj_f38_i04_d02 0.629304231237416",
  "bbob-biobj_f38_i04_d03 0.886856579988315",
  "bbob-biobj_f38_i04_d05 0.807520695902719",
  "bbob-biobj_f38_i04_d10 0.888020207269440",
  "bbob-biobj_f38_i04_d20 0.900010122202705",
  "bbob-biobj_f38_i04_d40 0.861091112317590",
  "bbob-biobj_f38_i05_d02 0.802625605762915",
  "bbob-biobj_f38_i05_d03 0.930559083395830",
  "bbob-biobj_f38_i05_d05 0.906957714667672",
  "bbob-biobj_f38_i05_d10 0.873582089195529",
  "bbob-biobj_f38_i05_d20 0.879901788772980",
  "bbob-biobj_f38_i05_d40 0.853941170248609",
  "bbob-biobj_f38_i06_d02 0.853438943384095",
  "bbob-biobj_f38_i06_d03 0.904605729077286",
  "bbob-biobj_f38_i06_d05 0.809818054476904",
  "bbob-biobj_f38_i06_d10 0.832582085732755",
  "bbob-biobj_f38_i06_d20 0.900769552940082",
  "bbob-biobj_f38_i06_d40 0.868311434245990",
  "bbob-biobj_f38_i07_d02 0.737929740773552",
  "bbob-biobj_f38_i07_d03 0.950678057069629",
  "bbob-biobj_f38_i07_d05 0.873532192074075",
  "bbob-biobj_f38_i07_d10 0.910023034171693",
  "bbob-biobj_f38_i07_d20 0.875502508826690",
  "bbob-biobj_f38_i07_d40 0.851028475183864",
  "bbob-biobj_f38_i08_d02 0.935955754649867",
  "bbob-biobj_f38_i08_d03 0.843236331688389",
  "bbob-biobj_f38_i08_d05 0.904238494052232",
  "bbob-biobj_f38_i08_d10 0.881791805051563",
  "bbob-biobj_f38_i08_d20 0.860665425596752",
  "bbob-biobj_f38_i08_d40 0.787794169083193",
  "bbob-biobj_f38_i09_d02 0.629808149332137",
  "bbob-biobj_f38_i09_d03 0.945175953384347",
  "bbob-biobj_f38_i09_d05 0.849833350256497",
  "bbob-biobj_f38_i09_d10 0.776135238303976",
  "bbob-biobj_f38_i09_d20 0.852273565562963",
  "bbob-biobj_f38_i09_d40 0.848647145774241",
  "bbob-biobj_f38_i10_d02 0.861762938112428",
  "bbob-biobj_f38_i10_d03 0.908446440795465",
  "bbob-biobj_f38_i10_d05 0.950441319887604",
  "bbob-biobj_f38_i10_d10 0.935265987677227",
  "bbob-biobj_f38_i10_d20 0.898876233462130",
  "bbob-biobj_f38_i10_d40 0.883444006163019",
  "bbob-biobj_f38_i11_d02 1.0",
  "bbob-biobj_f38_i11_d03 1.0",
  "bbob-biobj_f38_i11_d05 1.0",
  "bbob-biobj_f38_i11_d10 1.0",
  "bbob-biobj_f38_i11_d20 1.0",
  "bbob-biobj_f38_i11_d40 1.0",
  "bbob-biobj_f38_i12_d02 1.0",
  "bbob-biobj_f38_i12_d03 1.0",
  "bbob-biobj_f38_i12_d05 1.0",
  "bbob-biobj_f38_i12_d10 1.0",
  "bbob-biobj_f38_i12_d20 1.0",
  "bbob-biobj_f38_i12_d40 1.0",
  "bbob-biobj_f38_i13_d02 1.0",
  "bbob-biobj_f38_i13_d03 1.0",
  "bbob-biobj_f38_i13_d05 1.0",
  "bbob-biobj_f38_i13_d10 1.0",
  "bbob-biobj_f38_i13_d20 1.0",
  "bbob-biobj_f38_i13_d40 1.0",
  "bbob-biobj_f38_i14_d02 1.0",
  "bbob-biobj_f38_i14_d03 1.0",
  "bbob-biobj_f38_i14_d05 1.0",
  "bbob-biobj_f38_i14_d10 1.0",
  "bbob-biobj_f38_i14_d20 1.0",
  "bbob-biobj_f38_i14_d40 1.0",
  "bbob-biobj_f38_i15_d02 1.0",
  "bbob-biobj_f38_i15_d03 1.0",
  "bbob-biobj_f38_i15_d05 1.0",
  "bbob-biobj_f38_i15_d10 1.0",
  "bbob-biobj_f38_i15_d20 1.0",
  "bbob-biobj_f38_i15_d40 1.0",
  "bbob-biobj_f39_i01_d02 0.904226041231484",
  "bbob-biobj_f39_i01_d03 0.853573746699372",
  "bbob-biobj_f39_i01_d05 0.891207393990008",
  "bbob-biobj_f39_i01_d10 0.980211550827831",
  "bbob-biobj_f39_i01_d20 0.867119325562294",
  "bbob-biobj_f39_i01_d40 0.840147372120187",
  "bbob-biobj_f39_i02_d02 0.983637777721370",
  "bbob-biobj_f39_i02_d03 0.963324588533684",
  "bbob-biobj_f39_i02_d05 0.979070560169339",
  "bbob-biobj_f39_i02_d10 0.937097427270213",
  "bbob-biobj_f39_i02_d20 0.927055417557828",
  "bbob-biobj_f39_i02_d40 0.872981528838690",
  "bbob-biobj_f39_i03_d02 0.979201351160540",
  "bbob-biobj_f39_i03_d03 0.842277108051492",
  "bbob-biobj_f39_i03_d05 0.903051958458865",
  "bbob-biobj_f39_i03_d10 0.857805673867875",
  "bbob-biobj_f39_i03_d20 0.868170329189111",
  "bbob-biobj_f39_i03_d40 0.881480213422992",
  "bbob-biobj_f39_i04_d02 0.978307712731035",
  "bbob-biobj_f39_i04_d03 0.956663318916884",
  "bbob-biobj_f39_i04_d05 0.968094919360432",
  "bbob-biobj_f39_i04_d10 0.906017852170130",
  "bbob-biobj_f39_i04_d20 0.876867427446496",
  "bbob-biobj_f39_i04_d40 0.887145156397282",
  "bbob-biobj_f39_i05_d02 0.986275618889540",
  "bbob-biobj_f39_i05_d03 0.958680470699555",
  "bbob-biobj_f39_i05_d05 0.921040002155228",
  "bbob-biobj_f39_i05_d10 0.884172759606038",
  "bbob-biobj_f39_i05_d20 0.868533797925938",
  "bbob-biobj_f39_i05_d40 0.866860232364979",
  "bbob-biobj_f39_i06_d02 0.996952730131110",
  "bbob-biobj_f39_i06_d03 0.953947539293277",
  "bbob-biobj_f39_i06_d05 0.845030994798176",
  "bbob-biobj_f39_i06_d10 0.891285361180478",
  "bbob-biobj_f39_i06_d20 0.889771417560960",
  "bbob-biobj_f39_i06_d40 0.887168950613248",
  "bbob-biobj_f39_i07_d02 0.945548966985767",
  "bbob-biobj_f39_i07_d03 0.990685319479432",
  "bbob-biobj_f39_i07_d05 0.943155366283496",
  "bbob-biobj_f39_i07_d10 0.844355265733683",
  "bbob-biobj_f39_i07_d20 0.871126501732900",
  "bbob-biobj_f39_i07_d40 0.852384048442937",
  "bbob-biobj_f39_i08_d02 0.784535495200657",
  "bbob-biobj_f39_i08_d03 0.787172668018590",
  "bbob-biobj_f39_i08_d05 0.992843816462946",
  "bbob-biobj_f39_i08_d10 0.810579392397199",
  "bbob-biobj_f39_i08_d20 0.849764168665382",
  "bbob-biobj_f39_i08_d40 0.852499131000823",
  "bbob-biobj_f39_i09_d02 0.995638433562137",
  "bbob-biobj_f39_i09_d03 0.929891599701217",
  "bbob-biobj_f39_i09_d05 0.880536302208195",
  "bbob-biobj_f39_i09_d10 0.992998001826045",
  "bbob-biobj_f39_i09_d20 0.902078102558508",
  "bbob-biobj_f39_i09_d40 0.875004561638512",
  "bbob-biobj_f39_i10_d02 0.733472668213708",
  "bbob-biobj_f39_i10_d03 0.811074833803342",
  "bbob-biobj_f39_i10_d05 0.906479471448780",
  "bbob-biobj_f39_i10_d10 0.864345054451404",
  "bbob-biobj_f39_i10_d20 0.870539322571479",
  "bbob-biobj_f39_i10_d40 0.863108028491588",
  "bbob-biobj_f39_i11_d02 1.0",
  "bbob-biobj_f39_i11_d03 1.0",
  "bbob-biobj_f39_i11_d05 1.0",
  "bbob-biobj_f39_i11_d10 1.0",
  "bbob-biobj_f39_i11_d20 1.0",
  "bbob-biobj_f39_i11_d40 1.0",
  "bbob-biobj_f39_i12_d02 1.0",
  "bbob-biobj_f39_i12_d03 1.0",
  "bbob-biobj_f39_i12_d05 1.0",
  "bbob-biobj_f39_i12_d10 1.0",
  "bbob-biobj_f39_i12_d20 1.0",
  "bbob-biobj_f39_i12_d40 1.0",
  "bbob-biobj_f39_i13_d02 1.0",
  "bbob-biobj_f39_i13_d03 1.0",
  "bbob-biobj_f39_i13_d05 1.0",
  "bbob-biobj_f39_i13_d10 1.0",
  "bbob-biobj_f39_i13_d20 1.0",
  "bbob-biobj_f39_i13_d40 1.0",
  "bbob-biobj_f39_i14_d02 1.0",
  "bbob-biobj_f39_i14_d03 1.0",
  "bbob-biobj_f39_i14_d05 1.0",
  "bbob-biobj_f39_i14_d10 1.0",
  "bbob-biobj_f39_i14_d20 1.0",
  "bbob-biobj_f39_i14_d40 1.0",
  "bbob-biobj_f39_i15_d02 1.0",
  "bbob-biobj_f39_i15_d03 1.0",
  "bbob-biobj_f39_i15_d05 1.0",
  "bbob-biobj_f39_i15_d10 1.0",
  "bbob-biobj_f39_i15_d20 1.0",
  "bbob-biobj_f39_i15_d40 1.0",
  "bbob-biobj_f40_i01_d02 0.800161494536693",
  "bbob-biobj_f40_i01_d03 0.914340224327633",
  "bbob-biobj_f40_i01_d05 0.903892856481269",
  "bbob-biobj_f40_i01_d10 0.728473869257068",
  "bbob-biobj_f40_i01_d20 0.604365459168164",
  "bbob-biobj_f40_i01_d40 0.459307378153473",
  "bbob-biobj_f40_i02_d02 0.814422943992905",
  "bbob-biobj_f40_i02_d03 0.684385884079890",
  "bbob-biobj_f40_i02_d05 0.823853518802178",
  "bbob-biobj_f40_i02_d10 0.744541086997719",
  "bbob-biobj_f40_i02_d20 0.645764795297761",
  "bbob-biobj_f40_i02_d40 0.481198540236717",
  "bbob-biobj_f40_i03_d02 0.841354252700995",
  "bbob-biobj_f40_i03_d03 0.938507644904524",
  "bbob-biobj_f40_i03_d05 0.809182823776307",
  "bbob-biobj_f40_i03_d10 0.729687061912397",
  "bbob-biobj_f40_i03_d20 0.527197277062025",
  "bbob-biobj_f40_i03_d40 0.594846348813518",
  "bbob-biobj_f40_i04_d02 0.545132030408484",
  "bbob-biobj_f40_i04_d03 0.820237436915800",
  "bbob-biobj_f40_i04_d05 0.914046946523336",
  "bbob-biobj_f40_i04_d10 0.707855529463683",
  "bbob-biobj_f40_i04_d20 0.748188983914643",
  "bbob-biobj_f40_i04_d40 0.594662978172518",
  "bbob-biobj_f40_i05_d02 0.710070245743081",
  "bbob-biobj_f40_i05_d03 0.761269356172402",
  "bbob-biobj_f40_i05_d05 0.946428096409420",
  "bbob-biobj_f40_i05_d10 0.738294222442384",
  "bbob-biobj_f40_i05_d20 0.619037463430707",
  "bbob-biobj_f40_i05_d40 0.525261934519682",
  "bbob-biobj_f40_i06_d02 0.940602213917627",
  "bbob-biobj_f40_i06_d03 0.965600494584402",
  "bbob-biobj_f40_i06_d05 0.874266215663434",
  "bbob-biobj_f40_i06_d10 0.712568335159037",
  "bbob-biobj_f40_i06_d20 0.837228563460123",
  "bbob-biobj_f40_i06_d40 0.581922202344157",
  "bbob-biobj_f40_i07_d02 0.911307906444006",
  "bbob-biobj_f40_i07_d03 0.831573964105432",
  "bbob-biobj_f40_i07_d05 0.906195003329187",
  "bbob-biobj_f40_i07_d10 0.785546219384903",
  "bbob-biobj_f40_i07_d20 0.632231071888189",
  "bbob-biobj_f40_i07_d40 0.602091185443697",
  "bbob-biobj_f40_i08_d02 0.931109790953942",
  "bbob-biobj_f40_i08_d03 0.960523180912951",
  "bbob-biobj_f40_i08_d05 0.866076738417337",
  "bbob-biobj_f40_i08_d10 0.823347856120497",
  "bbob-biobj_f40_i08_d20 0.681358009427460",
  "bbob-biobj_f40_i08_d40 0.591788902137592",
  "bbob-biobj_f40_i09_d02 0.929776153108144",
  "bbob-biobj_f40_i09_d03 0.706659001418762",
  "bbob-biobj_f40_i09_d05 0.846713727987104",
  "bbob-biobj_f40_i09_d10 0.873474383102206",
  "bbob-biobj_f40_i09_d20 0.679130772298345",
  "bbob-biobj_f40_i09_d40 0.561415958634155",
  "bbob-biobj_f40_i10_d02 0.978837863487893",
  "bbob-biobj_f40_i10_d03 0.932807541759723",
  "bbob-biobj_f40_i10_d05 0.839724013939153",
  "bbob-biobj_f40_i10_d10 0.835128803151321",
  "bbob-biobj_f40_i10_d20 0.720869751773177",
  "bbob-biobj_f40_i10_d40 0.555111732383338",
  "bbob-biobj_f40_i11_d02 1.0",
  "bbob-biobj_f40_i11_d03 1.0",
  "bbob-biobj_f40_i11_d05 1.0",
  "bbob-biobj_f40_i11_d10 1.0",
  "bbob-biobj_f40_i11_d20 1.0",
  "bbob-biobj_f40_i11_d40 1.0",
  "bbob-biobj_f40_i12_d02 1.0",
  "bbob-biobj_f40_i12_d03 1.0",
  "bbob-biobj_f40_i12_d05 1.0",
  "bbob-biobj_f40_i12_d10 1.0",
  "bbob-biobj_f40_i12_d20 1.0",
  "bbob-biobj_f40_i12_d40 1.0",
  "bbob-biobj_f40_i13_d02 1.0",
  "bbob-biobj_f40_i13_d03 1.0",
  "bbob-biobj_f40_i13_d05 1.0",
  "bbob-biobj_f40_i13_d10 1.0",
  "bbob-biobj_f40_i13_d20 1.0",
  "bbob-biobj_f40_i13_d40 1.0",
  "bbob-biobj_f40_i14_d02 1.0",
  "bbob-biobj_f40_i14_d03 1.0",
  "bbob-biobj_f40_i14_d05 1.0",
  "bbob-biobj_f40_i14_d10 1.0",
  "bbob-biobj_f40_i14_d20 1.0",
  "bbob-biobj_f40_i14_d40 1.0",
  "bbob-biobj_f40_i15_d02 1.0",
  "bbob-biobj_f40_i15_d03 1.0",
  "bbob-biobj_f40_i15_d05 1.0",
  "bbob-biobj_f40_i15_d10 1.0",
  "bbob-biobj_f40_i15_d20 1.0",
  "bbob-biobj_f40_i15_d40 1.0",
  "bbob-biobj_f41_i01_d02 0.822033307817962",
  "bbob-biobj_f41_i01_d03 0.885738075689406",
  "bbob-biobj_f41_i01_d05 0.975610197116841",
  "bbob-biobj_f41_i01_d10 0.976594721097833",
  "bbob-biobj_f41_i01_d20 0.971528564301038",
  "bbob-biobj_f41_i01_d40 0.965160551572485",
  "bbob-biobj_f41_i02_d02 0.914732085156474",
  "bbob-biobj_f41_i02_d03 0.923052605100394",
  "bbob-biobj_f41_i02_d05 0.924070436138730",
  "bbob-biobj_f41_i02_d10 0.931074251640139",
  "bbob-biobj_f41_i02_d20 0.981609407937339",
  "bbob-biobj_f41_i02_d40 0.951243795129035",
  "bbob-biobj_f41_i03_d02 0.853587922588129",
  "bbob-biobj_f41_i03_d03 0.914428027190248",
  "bbob-biobj_f41_i03_d05 0.974030934514910",
  "bbob-biobj_f41_i03_d10 0.872008403001157",
  "bbob-biobj_f41_i03_d20 0.941022611913163",
  "bbob-biobj_f41_i03_d40 0.954750241529540",
  "bbob-biobj_f41_i04_d02 0.512373847991457",
  "bbob-biobj_f41_i04_d03 0.695238799108705",
  "bbob-biobj_f41_i04_d05 0.889989722498128",
  "bbob-biobj_f41_i04_d10 0.951237336242326",
  "bbob-biobj_f41_i04_d20 0.913071583840937",
  "bbob-biobj_f41_i04_d40 0.980678709328714",
  "bbob-biobj_f41_i05_d02 0.821558599298996",
  "bbob-biobj_f41_i05_d03 0.822872733697532",
  "bbob-biobj_f41_i05_d05 0.876473147087528",
  "bbob-biobj_f41_i05_d10 0.966128724260189",
  "bbob-biobj_f41_i05_d20 0.924564290390755",
  "bbob-biobj_f41_i05_d40 0.951503528251981",
  "bbob-biobj_f41_i06_d02 0.767877747901371",
  "bbob-biobj_f41_i06_d03 0.896826333117921",
  "bbob-biobj_f41_i06_d05 0.792741819695997",
  "bbob-biobj_f41_i06_d10 0.893687033801400",
  "bbob-biobj_f41_i06_d20 0.963777758984029",
  "bbob-biobj_f41_i06_d40 0.958296705553301",
  "bbob-biobj_f41_i07_d02 0.935919502696871",
  "bbob-biobj_f41_i07_d03 0.915129505092837",
  "bbob-biobj_f41_i07_d05 0.922633056935486",
  "bbob-biobj_f41_i07_d10 0.850434631393912",
  "bbob-biobj_f41_i07_d20 0.901424361629473",
  "bbob-biobj_f41_i07_d40 0.978178007181946",
  "bbob-biobj_f41_i08_d02 0.948612145014462",
  "bbob-biobj_f41_i08_d03 0.818796188016961",
  "bbob-biobj_f41_i08_d05 0.967973190423149",
  "bbob-biobj_f41_i08_d10 0.823557579779486",
  "bbob-biobj_f41_i08_d20 0.958607082914751",
  "bbob-biobj_f41_i08_d40 0.945546529248454",
  "bbob-biobj_f41_i09_d02 0.784344360812359",
  "bbob-biobj_f41_i09_d03 0.946504928270416",
  "bbob-biobj_f41_i09_d05 0.824379078684278",
  "bbob-biobj_f41_i09_d10 0.938668422809685",
  "bbob-biobj_f41_i09_d20 0.980249890151125",
  "bbob-biobj_f41_i09_d40 0.949698682084359",
  "bbob-biobj_f41_i10_d02 0.636649862585077",
  "bbob-biobj_f41_i10_d03 0.839207215249613",
  "bbob-biobj_f41_i10_d05 0.911512650028684",
  "bbob-biobj_f41_i10_d10 0.875925359818394",
  "bbob-biobj_f41_i10_d20 0.976035237647000",
  "bbob-biobj_f41_i10_d40 0.953358560472507",
  "bbob-biobj_f41_i11_d02 1.0",
  "bbob-biobj_f41_i11_d03 1.0",
  "bbob-biobj_f41_i11_d05 1.0",
  "bbob-biobj_f41_i11_d10 1.0",
  "bbob-biobj_f41_i11_d20 1.0",
  "bbob-biobj_f41_i11_d40 1.0",
  "bbob-biobj_f41_i12_d02 1.0",
  "bbob-biobj_f41_i12_d03 1.0",
  "bbob-biobj_f41_i12_d05 1.0",
  "bbob-biobj_f41_i12_d10 1.0",
  "bbob-biobj_f41_i12_d20 1.0",
  "bbob-biobj_f41_i12_d40 1.0",
  "bbob-biobj_f41_i13_d02 1.0",
  "bbob-biobj_f41_i13_d03 1.0",
  "bbob-biobj_f41_i13_d05 1.0",
  "bbob-biobj_f41_i13_d10 1.0",
  "bbob-biobj_f41_i13_d20 1.0",
  "bbob-biobj_f41_i13_d40 1.0",
  "bbob-biobj_f41_i14_d02 1.0",
  "bbob-biobj_f41_i14_d03 1.0",
  "bbob-biobj_f41_i14_d05 1.0",
  "bbob-biobj_f41_i14_d10 1.0",
  "bbob-biobj_f41_i14_d20 1.0",
  "bbob-biobj_f41_i14_d40 1.0",
  "bbob-biobj_f41_i15_d02 1.0",
  "bbob-biobj_f41_i15_d03 1.0",
  "bbob-biobj_f41_i15_d05 1.0",
  "bbob-biobj_f41_i15_d10 1.0",
  "bbob-biobj_f41_i15_d20 1.0",
  "bbob-biobj_f41_i15_d40 1.0",
  "bbob-biobj_f42_i01_d02 0.948464970307738",
  "bbob-biobj_f42_i01_d03 0.964394084499346",
  "bbob-biobj_f42_i01_d05 0.935365033796674",
  "bbob-biobj_f42_i01_d10 0.943128312717771",
  "bbob-biobj_f42_i01_d20 0.963643732031699",
  "bbob-biobj_f42_i01_d40 0.965657560445270",
  "bbob-biobj_f42_i02_d02 0.938890033676112",
  "bbob-biobj_f42_i02_d03 0.896774246071198",
  "bbob-biobj_f42_i02_d05 0.918221443136859",
  "bbob-biobj_f42_i02_d10 0.933845341832528",
  "bbob-biobj_f42_i02_d20 0.934284063910811",
  "bbob-biobj_f42_i02_d40 0.930178208511253",
  "bbob-biobj_f42_i03_d02 0.813087641534480",
  "bbob-biobj_f42_i03_d03 0.936776356752616",
  "bbob-biobj_f42_i03_d05 0.860970332418219",
  "bbob-biobj_f42_i03_d10 0.912972610994230",
  "bbob-biobj_f42_i03_d20 0.940879346362087",
  "bbob-biobj_f42_i03_d40 0.938604902807888",
  "bbob-biobj_f42_i04_d02 0.882606100873865",
  "bbob-biobj_f42_i04_d03 0.956287162433495",
  "bbob-biobj_f42_i04_d05 0.929527509865346",
  "bbob-biobj_f42_i04_d10 0.870792113484450",
  "bbob-biobj_f42_i04_d20 0.953784523505501",
  "bbob-biobj_f42_i04_d40 0.917639525057008",
  "bbob-biobj_f42_i05_d02 0.980207584330832",
  "bbob-biobj_f42_i05_d03 0.933372828275547",
  "bbob-biobj_f42_i05_d05 0.970898927457852",
  "bbob-biobj_f42_i05_d10 0.914627662372202",
  "bbob-biobj_f42_i05_d20 0.957373316635058",
  "bbob-biobj_f42_i05_d40 0.953427715660606",
  "bbob-biobj_f42_i06_d02 0.948846761282047",
  "bbob-biobj_f42_i06_d03 0.887124002044022",
  "bbob-biobj_f42_i06_d05 0.935908567200754",
  "bbob-biobj_f42_i06_d10 0.944780454333613",
  "bbob-biobj_f42_i06_d20 0.949590672140516",
  "bbob-biobj_f42_i06_d40 0.923157629370350",
  "bbob-biobj_f42_i07_d02 0.973656420019213",
  "bbob-biobj_f42_i07_d03 0.912513948568898",
  "bbob-biobj_f42_i07_d05 0.955518741476096",
  "bbob-biobj_f42_i07_d10 0.896366744021679",
  "bbob-biobj_f42_i07_d20 0.966419103159295",
  "bbob-biobj_f42_i07_d40 0.938747933877435",
  "bbob-biobj_f42_i08_d02 0.917270613774874",
  "bbob-biobj_f42_i08_d03 0.966076886108044",
  "bbob-biobj_f42_i08_d05 0.947996999410747",
  "bbob-biobj_f42_i08_d10 0.954771017540515",
  "bbob-biobj_f42_i08_d20 0.956942184799609",
  "bbob-biobj_f42_i08_d40 0.905355950827379",
  "bbob-biobj_f42_i09_d02 0.966954242919577",
  "bbob-biobj_f42_i09_d03 0.965253588182618",
  "bbob-biobj_f42_i09_d05 0.917762969301912",
  "bbob-biobj_f42_i09_d10 0.927523147148266",
  "bbob-biobj_f42_i09_d20 0.963016275684763",
  "bbob-biobj_f42_i09_d40 0.903749434997353",
  "bbob-biobj_f42_i10_d02 0.968069449243095",
  "bbob-biobj_f42_i10_d03 0.958413404226314",
  "bbob-biobj_f42_i10_d05 0.931477412814219",
  "bbob-biobj_f42_i10_d10 0.946346732258908",
  "bbob-biobj_f42_i10_d20 0.967523989495793",
  "bbob-biobj_f42_i10_d40 0.870374219537932",
  "bbob-biobj_f42_i11_d02 1.0",
  "bbob-biobj_f42_i11_d03 1.0",
  "bbob-biobj_f42_i11_d05 1.0",
  "bbob-biobj_f42_i11_d10 1.0",
  "bbob-biobj_f42_i11_d20 1.0",
  "bbob-biobj_f42_i11_d40 1.0",
  "bbob-biobj_f42_i12_d02 1.0",
  "bbob-biobj_f42_i12_d03 1.0",
  "bbob-biobj_f42_i12_d05 1.0",
  "bbob-biobj_f42_i12_d10 1.0",
  "bbob-biobj_f42_i12_d20 1.0",
  "bbob-biobj_f42_i12_d40 1.0",
  "bbob-biobj_f42_i13_d02 1.0",
  "bbob-biobj_f42_i13_d03 1.0",
  "bbob-biobj_f42_i13_d05 1.0",
  "bbob-biobj_f42_i13_d10 1.0",
  "bbob-biobj_f42_i13_d20 1.0",
  "bbob-biobj_f42_i13_d40 1.0",
  "bbob-biobj_f42_i14_d02 1.0",
  "bbob-biobj_f42_i14_d03 1.0",
  "bbob-biobj_f42_i14_d05 1.0",
  "bbob-biobj_f42_i14_d10 1.0",
  "bbob-biobj_f42_i14_d20 1.0",
  "bbob-biobj_f42_i14_d40 1.0",
  "bbob-biobj_f42_i15_d02 1.0",
  "bbob-biobj_f42_i15_d03 1.0",
  "bbob-biobj_f42_i15_d05 1.0",
  "bbob-biobj_f42_i15_d10 1.0",
  "bbob-biobj_f42_i15_d20 1.0",
  "bbob-biobj_f42_i15_d40 1.0",
  "bbob-biobj_f43_i01_d02 0.806235943768491",
  "bbob-biobj_f43_i01_d03 0.961033203638514",
  "bbob-biobj_f43_i01_d05 0.898217378671209",
  "bbob-biobj_f43_i01_d10 0.993264379743037",
  "bbob-biobj_f43_i01_d20 0.940948652738029",
  "bbob-biobj_f43_i01_d40 0.983857298283952",
  "bbob-biobj_f43_i02_d02 0.928981208319704",
  "bbob-biobj_f43_i02_d03 0.900499905679907",
  "bbob-biobj_f43_i02_d05 0.953668359506045",
  "bbob-biobj_f43_i02_d10 0.943994958453547",
  "bbob-biobj_f43_i02_d20 0.958781146064377",
  "bbob-biobj_f43_i02_d40 0.975597556762495",
  "bbob-biobj_f43_i03_d02 0.703325658552559",
  "bbob-biobj_f43_i03_d03 0.738244890811554",
  "bbob-biobj_f43_i03_d05 0.988642280121935",
  "bbob-biobj_f43_i03_d10 0.982688206261094",
  "bbob-biobj_f43_i03_d20 0.973220372898028",
  "bbob-biobj_f43_i03_d40 0.944180440428360",
  "bbob-biobj_f43_i04_d02 0.955511395205191",
  "bbob-biobj_f43_i04_d03 0.898306002923848",
  "bbob-biobj_f43_i04_d05 0.945720250592969",
  "bbob-biobj_f43_i04_d10 0.963172283120474",
  "bbob-biobj_f43_i04_d20 0.966626097868362",
  "bbob-biobj_f43_i04_d40 0.966946756102161",
  "bbob-biobj_f43_i05_d02 0.527808863109296",
  "bbob-biobj_f43_i05_d03 0.967872270114991",
  "bbob-biobj_f43_i05_d05 0.902672633183536",
  "bbob-biobj_f43_i05_d10 0.969216100402235",
  "bbob-biobj_f43_i05_d20 0.979575039404802",
  "bbob-biobj_f43_i05_d40 0.966070621705196",
  "bbob-biobj_f43_i06_d02 0.913935322722399",
  "bbob-biobj_f43_i06_d03 0.914558564989063",
  "bbob-biobj_f43_i06_d05 0.973356892013738",
  "bbob-biobj_f43_i06_d10 0.932387469095507",
  "bbob-biobj_f43_i06_d20 0.961581182505789",
  "bbob-biobj_f43_i06_d40 0.918320667845127",
  "bbob-biobj_f43_i07_d02 0.901301713558448",
  "bbob-biobj_f43_i07_d03 0.882128040824323",
  "bbob-biobj_f43_i07_d05 0.963346626771001",
  "bbob-biobj_f43_i07_d10 0.985079491142971",
  "bbob-biobj_f43_i07_d20 0.968594673624419",
  "bbob-biobj_f43_i07_d40 0.950843373521710",
  "bbob-biobj_f43_i08_d02 0.701553018378812",
  "bbob-biobj_f43_i08_d03 0.969205249181385",
  "bbob-biobj_f43_i08_d05 0.952748332982779",
  "bbob-biobj_f43_i08_d10 0.987278435011227",
  "bbob-biobj_f43_i08_d20 0.976546324177090",
  "bbob-biobj_f43_i08_d40 0.957306668204760",
  "bbob-biobj_f43_i09_d02 0.862739669961301",
  "bbob-biobj_f43_i09_d03 0.862473364865682",
  "bbob-biobj_f43_i09_d05 0.990618031218341",
  "bbob-biobj_f43_i09_d10 0.979466959858262",
  "bbob-biobj_f43_i09_d20 0.972941110626848",
  "bbob-biobj_f43_i09_d40 0.855701729006963",
  "bbob-biobj_f43_i10_d02 0.848752159713627",
  "bbob-biobj_f43_i10_d03 0.820261488990871",
  "bbob-biobj_f43_i10_d05 0.940942353330467",
  "bbob-biobj_f43_i10_d10 0.959250198964312",
  "bbob-biobj_f43_i10_d20 0.982486894377839",
  "bbob-biobj_f43_i10_d40 0.966505095865004",
  "bbob-biobj_f43_i11_d02 1.0",
  "bbob-biobj_f43_i11_d03 1.0",
  "bbob-biobj_f43_i11_d05 1.0",
  "bbob-biobj_f43_i11_d10 1.0",
  "bbob-biobj_f43_i11_d20 1.0",
  "bbob-biobj_f43_i11_d40 1.0",
  "bbob-biobj_f43_i12_d02 1.0",
  "bbob-biobj_f43_i12_d03 1.0",
  "bbob-biobj_f43_i12_d05 1.0",
  "bbob-biobj_f43_i12_d10 1.0",
  "bbob-biobj_f43_i12_d20 1.0",
  "bbob-biobj_f43_i12_d40 1.0",
  "bbob-biobj_f43_i13_d02 1.0",
  "bbob-biobj_f43_i13_d03 1.0",
  "bbob-biobj_f43_i13_d05 1.0",
  "bbob-biobj_f43_i13_d10 1.0",
  "bbob-biobj_f43_i13_d20 1.0",
  "bbob-biobj_f43_i13_d40 1.0",
  "bbob-biobj_f43_i14_d02 1.0",
  "bbob-biobj_f43_i14_d03 1.0",
  "bbob-biobj_f43_i14_d05 1.0",
  "bbob-biobj_f43_i14_d10 1.0",
  "bbob-biobj_f43_i14_d20 1.0",
  "bbob-biobj_f43_i14_d40 1.0",
  "bbob-biobj_f43_i15_d02 1.0",
  "bbob-biobj_f43_i15_d03 1.0",
  "bbob-biobj_f43_i15_d05 1.0",
  "bbob-biobj_f43_i15_d10 1.0",
  "bbob-biobj_f43_i15_d20 1.0",
  "bbob-biobj_f43_i15_d40 1.0",
  "bbob-biobj_f44_i01_d02 0.990000595892640",
  "bbob-biobj_f44_i01_d03 0.989550585423686",
  "bbob-biobj_f44_i01_d05 0.993557596127607",
  "bbob-biobj_f44_i01_d10 0.977883915649076",
  "bbob-biobj_f44_i01_d20 0.989486565699468",
  "bbob-biobj_f44_i01_d40 0.992108571259436",
  "bbob-biobj_f44_i02_d02 0.996682463157142",
  "bbob-biobj_f44_i02_d03 0.744916519935829",
  "bbob-biobj_f44_i02_d05 0.951282424534247",
  "bbob-biobj_f44_i02_d10 0.975039053298177",
  "bbob-biobj_f44_i02_d20 0.984828537903076",
  "bbob-biobj_f44_i02_d40 0.981720374250759",
  "bbob-biobj_f44_i03_d02 0.982516671080787",
  "bbob-biobj_f44_i03_d03 0.988830152101750",
  "bbob-biobj_f44_i03_d05 0.958340799855323",
  "bbob-biobj_f44_i03_d10 0.978364069931773",
  "bbob-biobj_f44_i03_d20 0.977809886620423",
  "bbob-biobj_f44_i03_d40 0.982023087875918",
  "bbob-biobj_f44_i04_d02 0.975372409782800",
  "bbob-biobj_f44_i04_d03 0.927814333720962",
  "bbob-biobj_f44_i04_d05 0.965365808549549",
  "bbob-biobj_f44_i04_d10 0.987132635647892",
  "bbob-biobj_f44_i04_d20 0.984400141874121",
  "bbob-biobj_f44_i04_d40 0.980545459880095",
  "bbob-biobj_f44_i05_d02 0.992233483129565",
  "bbob-biobj_f44_i05_d03 0.972161392354809",
  "bbob-biobj_f44_i05_d05 0.966253887504727",
  "bbob-biobj_f44_i05_d10 0.972827697321209",
  "bbob-biobj_f44_i05_d20 0.974162510436130",
  "bbob-biobj_f44_i05_d40 0.981283407181326",
  "bbob-biobj_f44_i06_d02 0.955968657960968",
  "bbob-biobj_f44_i06_d03 0.938348236403148",
  "bbob-biobj_f44_i06_d05 0.926885126651733",
  "bbob-biobj_f44_i06_d10 0.970964654710173",
  "bbob-biobj_f44_i06_d20 0.981304079255180",
  "bbob-biobj_f44_i06_d40 0.989117615203209",
  "bbob-biobj_f44_i07_d02 0.982852507134425",
  "bbob-biobj_f44_i07_d03 0.968227402387315",
  "bbob-biobj_f44_i07_d05 0.996067968789636",
  "bbob-biobj_f44_i07_d10 0.961750345752744",
  "bbob-biobj_f44_i07_d20 0.936897342317752",
  "bbob-biobj_f44_i07_d40 0.987908927174219",
  "bbob-biobj_f44_i08_d02 0.881906919128662",
  "bbob-biobj_f44_i08_d03 0.950801778125565",
  "bbob-biobj_f44_i08_d05 0.982836821697150",
  "bbob-biobj_f44_i08_d10 0.976118709915315",
  "bbob-biobj_f44_i08_d20 0.985350032800047",
  "bbob-biobj_f44_i08_d40 0.981973075253610",
  "bbob-biobj_f44_i09_d02 0.974727927786401",
  "bbob-biobj_f44_i09_d03 0.993236716308686",
  "bbob-biobj_f44_i09_d05 0.952963758145309",
  "bbob-biobj_f44_i09_d10 0.966612414910251",
  "bbob-biobj_f44_i09_d20 0.996512281009713",
  "bbob-biobj_f44_i09_d40 0.991905893016871",
  "bbob-biobj_f44_i10_d02 0.742777154829549",
  "bbob-biobj_f44_i10_d03 0.788223027934632",
  "bbob-biobj_f44_i10_d05 0.997556014152405",
  "bbob-biobj_f44_i10_d10 0.974890816409705",
  "bbob-biobj_f44_i10_d20 0.961984655728772",
  "bbob-biobj_f44_i10_d40 0.984830506106585",
  "bbob-biobj_f44_i11_d02 1.0",
  "bbob-biobj_f44_i11_d03 1.0",
  "bbob-biobj_f44_i11_d05 1.0",
  "bbob-biobj_f44_i11_d10 1.0",
  "bbob-biobj_f44_i11_d20 1.0",
  "bbob-biobj_f44_i11_d40 1.0",
  "bbob-biobj_f44_i12_d02 1.0",
  "bbob-biobj_f44_i12_d03 1.0",
  "bbob-biobj_f44_i12_d05 1.0",
  "bbob-biobj_f44_i12_d10 1.0",
  "bbob-biobj_f44_i12_d20 1.0",
  "bbob-biobj_f44_i12_d40 1.0",
  "bbob-biobj_f44_i13_d02 1.0",
  "bbob-biobj_f44_i13_d03 1.0",
  "bbob-biobj_f44_i13_d05 1.0",
  "bbob-biobj_f44_i13_d10 1.0",
  "bbob-biobj_f44_i13_d20 1.0",
  "bbob-biobj_f44_i13_d40 1.0",
  "bbob-biobj_f44_i14_d02 1.0",
  "bbob-biobj_f44_i14_d03 1.0",
  "bbob-biobj_f44_i14_d05 1.0",
  "bbob-biobj_f44_i14_d10 1.0",
  "bbob-biobj_f44_i14_d20 1.0",
  "bbob-biobj_f44_i14_d40 1.0",
  "bbob-biobj_f44_i15_d02 1.0",
  "bbob-biobj_f44_i15_d03 1.0",
  "bbob-biobj_f44_i15_d05 1.0",
  "bbob-biobj_f44_i15_d10 1.0",
  "bbob-biobj_f44_i15_d20 1.0",
  "bbob-biobj_f44_i15_d40 1.0",
  "bbob-biobj_f45_i01_d02 0.951720778563444",
  "bbob-biobj_f45_i01_d03 0.828188774139641",
  "bbob-biobj_f45_i01_d05 0.917829241994999",
  "bbob-biobj_f45_i01_d10 0.962469633903938",
  "bbob-biobj_f45_i01_d20 0.916304775084538",
  "bbob-biobj_f45_i01_d40 0.785675041892531",
  "bbob-biobj_f45_i02_d02 0.797328068544284",
  "bbob-biobj_f45_i02_d03 0.976528938897792",
  "bbob-biobj_f45_i02_d05 0.892698290420688",
  "bbob-biobj_f45_i02_d10 0.937581779101428",
  "bbob-biobj_f45_i02_d20 0.852142004666287",
  "bbob-biobj_f45_i02_d40 0.842088275967232",
  "bbob-biobj_f45_i03_d02 0.729950961654969",
  "bbob-biobj_f45_i03_d03 0.953106329190224",
  "bbob-biobj_f45_i03_d05 0.953232088542056",
  "bbob-biobj_f45_i03_d10 0.939036646431562",
  "bbob-biobj_f45_i03_d20 0.945642771814716",
  "bbob-biobj_f45_i03_d40 0.870390002108839",
  "bbob-biobj_f45_i04_d02 0.962996637246124",
  "bbob-biobj_f45_i04_d03 0.954354478584170",
  "bbob-biobj_f45_i04_d05 0.946648046778442",
  "bbob-biobj_f45_i04_d10 0.932574403452305",
  "bbob-biobj_f45_i04_d20 0.929155103399460",
  "bbob-biobj_f45_i04_d40 0.920627283625365",
  "bbob-biobj_f45_i05_d02 0.924715343675183",
  "bbob-biobj_f45_i05_d03 0.942411861147468",
  "bbob-biobj_f45_i05_d05 0.909309881321831",
  "bbob-biobj_f45_i05_d10 0.875483160778896",
  "bbob-biobj_f45_i05_d20 0.913250748248638",
  "bbob-biobj_f45_i05_d40 0.885145586412210",
  "bbob-biobj_f45_i06_d02 0.904151517470615",
  "bbob-biobj_f45_i06_d03 0.926797766252780",
  "bbob-biobj_f45_i06_d05 0.908617558889420",
  "bbob-biobj_f45_i06_d10 0.974109498807929",
  "bbob-biobj_f45_i06_d20 0.850302309800901",
  "bbob-biobj_f45_i06_d40 0.845186435780144",
  "bbob-biobj_f45_i07_d02 0.436539479783815",
  "bbob-biobj_f45_i07_d03 0.967414649977868",
  "bbob-biobj_f45_i07_d05 0.948236042645629",
  "bbob-biobj_f45_i07_d10 0.960582199925476",
  "bbob-biobj_f45_i07_d20 0.868675710134838",
  "bbob-biobj_f45_i07_d40 0.882136275227096",
  "bbob-biobj_f45_i08_d02 0.780758545153498",
  "bbob-biobj_f45_i08_d03 0.986355057252456",
  "bbob-biobj_f45_i08_d05 0.971723931694909",
  "bbob-biobj_f45_i08_d10 0.895507628923304",
  "bbob-biobj_f45_i08_d20 0.920745397390923",
  "bbob-biobj_f45_i08_d40 0.935690944028907",
  "bbob-biobj_f45_i09_d02 0.717458466488061",
  "bbob-biobj_f45_i09_d03 0.983845938391830",
  "bbob-biobj_f45_i09_d05 0.661805959381686",
  "bbob-biobj_f45_i09_d10 0.944316649954434",
  "bbob-biobj_f45_i09_d20 0.879514185323017",
  "bbob-biobj_f45_i09_d40 0.874909222653475",
  "bbob-biobj_f45_i10_d02 0.963315916430492",
  "bbob-biobj_f45_i10_d03 0.911574142441829",
  "bbob-biobj_f45_i10_d05 0.888738706562716",
  "bbob-biobj_f45_i10_d10 0.921140534269231",
  "bbob-biobj_f45_i10_d20 0.919843625609510",
  "bbob-biobj_f45_i10_d40 0.862108899490471",
  "bbob-biobj_f45_i11_d02 1.0",
  "bbob-biobj_f45_i11_d03 1.0",
  "bbob-biobj_f45_i11_d05 1.0",
  "bbob-biobj_f45_i11_d10 1.0",
  "bbob-biobj_f45_i11_d20 1.0",
  "bbob-biobj_f45_i11_d40 1.0",
  "bbob-biobj_f45_i12_d02 1.0",
  "bbob-biobj_f45_i12_d03 1.0",
  "bbob-biobj_f45_i12_d05 1.0",
  "bbob-biobj_f45_i12_d10 1.0",
  "bbob-biobj_f45_i12_d20 1.0",
  "bbob-biobj_f45_i12_d40 1.0",
  "bbob-biobj_f45_i13_d02 1.0",
  "bbob-biobj_f45_i13_d03 1.0",
  "bbob-biobj_f45_i13_d05 1.0",
  "bbob-biobj_f45_i13_d10 1.0",
  "bbob-biobj_f45_i13_d20 1.0",
  "bbob-biobj_f45_i13_d40 1.0",
  "bbob-biobj_f45_i14_d02 1.0",
  "bbob-biobj_f45_i14_d03 1.0",
  "bbob-biobj_f45_i14_d05 1.0",
  "bbob-biobj_f45_i14_d10 1.0",
  "bbob-biobj_f45_i14_d20 1.0",
  "bbob-biobj_f45_i14_d40 1.0",
  "bbob-biobj_f45_i15_d02 1.0",
  "bbob-biobj_f45_i15_d03 1.0",
  "bbob-biobj_f45_i15_d05 1.0",
  "bbob-biobj_f45_i15_d10 1.0",
  "bbob-biobj_f45_i15_d20 1.0",
  "bbob-biobj_f45_i15_d40 1.0",
  "bbob-biobj_f46_i01_d02 0.761066104646006",
  "bbob-biobj_f46_i01_d03 0.903455657283090",
  "bbob-biobj_f46_i01_d05 0.947155255052143",
  "bbob-biobj_f46_i01_d10 0.919954436328063",
  "bbob-biobj_f46_i01_d20 0.945598418627500",
  "bbob-biobj_f46_i01_d40 0.854396387596525",
  "bbob-biobj_f46_i02_d02 0.848797542826665",
  "bbob-biobj_f46_i02_d03 0.962407933268246",
  "bbob-biobj_f46_i02_d05 0.903214891429348",
  "bbob-biobj_f46_i02_d10 0.909587131133991",
  "bbob-biobj_f46_i02_d20 0.966860028053004",
  "bbob-biobj_f46_i02_d40 0.870595248224547",
  "bbob-biobj_f46_i03_d02 0.924046984238807",
  "bbob-biobj_f46_i03_d03 0.899715614728189",
  "bbob-biobj_f46_i03_d05 0.893336832760998",
  "bbob-biobj_f46_i03_d10 0.895450896546758",
  "bbob-biobj_f46_i03_d20 0.927204345279249",
  "bbob-biobj_f46_i03_d40 0.891228213110070",
  "bbob-biobj_f46_i04_d02 0.934267008284511",
  "bbob-biobj_f46_i04_d03 0.892788551956789",
  "bbob-biobj_f46_i04_d05 0.909706115828279",
  "bbob-biobj_f46_i04_d10 0.908484577426152",
  "bbob-biobj_f46_i04_d20 0.942407577748297",
  "bbob-biobj_f46_i04_d40 0.945168681141010",
  "bbob-biobj_f46_i05_d02 0.925925620986090",
  "bbob-biobj_f46_i05_d03 0.846193701948347",
  "bbob-biobj_f46_i05_d05 0.942645632857587",
  "bbob-biobj_f46_i05_d10 0.901215565547812",
  "bbob-biobj_f46_i05_d20 0.908885610056779",
  "bbob-biobj_f46_i05_d40 0.922071009546490",
  "bbob-biobj_f46_i06_d02 0.860734962124473",
  "bbob-biobj_f46_i06_d03 0.909512574555902",
  "bbob-biobj_f46_i06_d05 0.912395540418240",
  "bbob-biobj_f46_i06_d10 0.945879219145466",
  "bbob-biobj_f46_i06_d20 0.934417784485787",
  "bbob-biobj_f46_i06_d40 0.896824860378772",
  "bbob-biobj_f46_i07_d02 0.839634628439667",
  "bbob-biobj_f46_i07_d03 0.924631055190915",
  "bbob-biobj_f46_i07_d05 0.905744579684656",
  "bbob-biobj_f46_i07_d10 0.936001504422621",
  "bbob-biobj_f46_i07_d20 0.931713066453506",
  "bbob-biobj_f46_i07_d40 0.844508631054395",
  "bbob-biobj_f46_i08_d02 0.936215978446970",
  "bbob-biobj_f46_i08_d03 0.935361352163704",
  "bbob-biobj_f46_i08_d05 0.930770425017802",
  "bbob-biobj_f46_i08_d10 0.940363563270556",
  "bbob-biobj_f46_i08_d20 0.901716131910073",
  "bbob-biobj_f46_i08_d40 0.876036789920220",
  "bbob-biobj_f46_i09_d02 0.883558016602446",
  "bbob-biobj_f46_i09_d03 0.961217059863281",
  "bbob-biobj_f46_i09_d05 0.920512705062287",
  "bbob-biobj_f46_i09_d10 0.915637929973320",
  "bbob-biobj_f46_i09_d20 0.919588637937134",
  "bbob-biobj_f46_i09_d40 0.832647572651101",
  "bbob-biobj_f46_i10_d02 0.881302389490815",
  "bbob-biobj_f46_i10_d03 0.912389800404993",
  "bbob-biobj_f46_i10_d05 0.950846397090625",
  "bbob-biobj_f46_i10_d10 0.940838283358756",
  "bbob-biobj_f46_i10_d20 0.957739921106021",
  "bbob-biobj_f46_i10_d40 0.866216426397061",
  "bbob-biobj_f46_i11_d02 1.0",
  "bbob-biobj_f46_i11_d03 1.0",
  "bbob-biobj_f46_i11_d05 1.0",
  "bbob-biobj_f46_i11_d10 1.0",
  "bbob-biobj_f46_i11_d20 1.0",
  "bbob-biobj_f46_i11_d40 1.0",
  "bbob-biobj_f46_i12_d02 1.0",
  "bbob-biobj_f46_i12_d03 1.0",
  "bbob-biobj_f46_i12_d05 1.0",
  "bbob-biobj_f46_i12_d10 1.0",
  "bbob-biobj_f46_i12_d20 1.0",
  "bbob-biobj_f46_i12_d40 1.0",
  "bbob-biobj_f46_i13_d02 1.0",
  "bbob-biobj_f46_i13_d03 1.0",
  "bbob-biobj_f46_i13_d05 1.0",
  "bbob-biobj_f46_i13_d10 1.0",
  "bbob-biobj_f46_i13_d20 1.0",
  "bbob-biobj_f46_i13_d40 1.0",
  "bbob-biobj_f46_i14_d02 1.0",
  "bbob-biobj_f46_i14_d03 1.0",
  "bbob-biobj_f46_i14_d05 1.0",
  "bbob-biobj_f46_i14_d10 1.0",
  "bbob-biobj_f46_i14_d20 1.0",
  "bbob-biobj_f46_i14_d40 1.0",
  "bbob-biobj_f46_i15_d02 1.0",
  "bbob-biobj_f46_i15_d03 1.0",
  "bbob-biobj_f46_i15_d05 1.0",
  "bbob-biobj_f46_i15_d10 1.0",
  "bbob-biobj_f46_i15_d20 1.0",
  "bbob-biobj_f46_i15_d40 1.0",
  "bbob-biobj_f47_i01_d02 0.712272111791356",
  "bbob-biobj_f47_i01_d03 0.868966645319236",
  "bbob-biobj_f47_i01_d05 0.942803986114097",
  "bbob-biobj_f47_i01_d10 0.958595956960254",
  "bbob-biobj_f47_i01_d20 0.956311217436412",
  "bbob-biobj_f47_i01_d40 0.897754675221834",
  "bbob-biobj_f47_i02_d02 0.939177930209109",
  "bbob-biobj_f47_i02_d03 0.954707497948604",
  "bbob-biobj_f47_i02_d05 0.930188556236347",
  "bbob-biobj_f47_i02_d10 0.907412646523823",
  "bbob-biobj_f47_i02_d20 0.957628844998111",
  "bbob-biobj_f47_i02_d40 0.897070275922110",
  "bbob-biobj_f47_i03_d02 0.739793311026710",
  "bbob-biobj_f47_i03_d03 0.961193348929021",
  "bbob-biobj_f47_i03_d05 0.976761772334493",
  "bbob-biobj_f47_i03_d10 0.947746576774328",
  "bbob-biobj_f47_i03_d20 0.948952272573593",
  "bbob-biobj_f47_i03_d40 0.910206700506767",
  "bbob-biobj_f47_i04_d02 0.779551399583096",
  "bbob-biobj_f47_i04_d03 0.940080659208807",
  "bbob-biobj_f47_i04_d05 0.923138847386112",
  "bbob-biobj_f47_i04_d10 0.954607220664130",
  "bbob-biobj_f47_i04_d20 0.950790324701075",
  "bbob-biobj_f47_i04_d40 0.959029969929543",
  "bbob-biobj_f47_i05_d02 0.944459738086973",
  "bbob-biobj_f47_i05_d03 0.910114477537581",
  "bbob-biobj_f47_i05_d05 0.870342540144716",
  "bbob-biobj_f47_i05_d10 0.976767183928314",
  "bbob-biobj_f47_i05_d20 0.925275080789924",
  "bbob-biobj_f47_i05_d40 0.960071913752091",
  "bbob-biobj_f47_i06_d02 0.844227429918601",
  "bbob-biobj_f47_i06_d03 0.970327206332497",
  "bbob-biobj_f47_i06_d05 0.923220981670475",
  "bbob-biobj_f47_i06_d10 0.958667740503491",
  "bbob-biobj_f47_i06_d20 0.963271559272385",
  "bbob-biobj_f47_i06_d40 0.889244323840422",
  "bbob-biobj_f47_i07_d02 0.868593925185596",
  "bbob-biobj_f47_i07_d03 0.831195707749476",
  "bbob-biobj_f47_i07_d05 0.973478443591967",
  "bbob-biobj_f47_i07_d10 0.981014053828823",
  "bbob-biobj_f47_i07_d20 0.990613064893604",
  "bbob-biobj_f47_i07_d40 0.892267489922570",
  "bbob-biobj_f47_i08_d02 0.958237497140224",
  "bbob-biobj_f47_i08_d03 0.914357101153653",
  "bbob-biobj_f47_i08_d05 0.925329224941109",
  "bbob-biobj_f47_i08_d10 0.955671208532904",
  "bbob-biobj_f47_i08_d20 0.960990198339395",
  "bbob-biobj_f47_i08_d40 0.899984046495390",
  "bbob-biobj_f47_i09_d02 0.889918311037458",
  "bbob-biobj_f47_i09_d03 0.962481922443155",
  "bbob-biobj_f47_i09_d05 0.951487158898893",
  "bbob-biobj_f47_i09_d10 0.938769098398943",
  "bbob-biobj_f47_i09_d20 0.942997363149960",
  "bbob-biobj_f47_i09_d40 0.855410552220298",
  "bbob-biobj_f47_i10_d02 0.645459744382548",
  "bbob-biobj_f47_i10_d03 0.900594867338457",
  "bbob-biobj_f47_i10_d05 0.913296663358001",
  "bbob-biobj_f47_i10_d10 0.955661889534839",
  "bbob-biobj_f47_i10_d20 0.957137531350378",
  "bbob-biobj_f47_i10_d40 0.817284464911829",
  "bbob-biobj_f47_i11_d02 1.0",
  "bbob-biobj_f47_i11_d03 1.0",
  "bbob-biobj_f47_i11_d05 1.0",
  "bbob-biobj_f47_i11_d10 1.0",
  "bbob-biobj_f47_i11_d20 1.0",
  "bbob-biobj_f47_i11_d40 1.0",
  "bbob-biobj_f47_i12_d02 1.0",
  "bbob-biobj_f47_i12_d03 1.0",
  "bbob-biobj_f47_i12_d05 1.0",
  "bbob-biobj_f47_i12_d10 1.0",
  "bbob-biobj_f47_i12_d20 1.0",
  "bbob-biobj_f47_i12_d40 1.0",
  "bbob-biobj_f47_i13_d02 1.0",
  "bbob-biobj_f47_i13_d03 1.0",
  "bbob-biobj_f47_i13_d05 1.0",
  "bbob-biobj_f47_i13_d10 1.0",
  "bbob-biobj_f47_i13_d20 1.0",
  "bbob-biobj_f47_i13_d40 1.0",
  "bbob-biobj_f47_i14_d02 1.0",
  "bbob-biobj_f47_i14_d03 1.0",
  "bbob-biobj_f47_i14_d05 1.0",
  "bbob-biobj_f47_i14_d10 1.0",
  "bbob-biobj_f47_i14_d20 1.0",
  "bbob-biobj_f47_i14_d40 1.0",
  "bbob-biobj_f47_i15_d02 1.0",
  "bbob-biobj_f47_i15_d03 1.0",
  "bbob-biobj_f47_i15_d05 1.0",
  "bbob-biobj_f47_i15_d10 1.0",
  "bbob-biobj_f47_i15_d20 1.0",
  "bbob-biobj_f47_i15_d40 1.0",
  "bbob-biobj_f48_i01_d02 0.848026567036178",
  "bbob-biobj_f48_i01_d03 0.973370144857933",
  "bbob-biobj_f48_i01_d05 0.970800527122283",
  "bbob-biobj_f48_i01_d10 0.975895168240419",
  "bbob-biobj_f48_i01_d20 0.973375090620421",
  "bbob-biobj_f48_i01_d40 0.967214456244375",
  "bbob-biobj_f48_i02_d02 0.994793675128085",
  "bbob-biobj_f48_i02_d03 0.870173205353387",
  "bbob-biobj_f48_i02_d05 0.991213945773693",
  "bbob-biobj_f48_i02_d10 0.987508273428610",
  "bbob-biobj_f48_i02_d20 0.974128630034868",
  "bbob-biobj_f48_i02_d40 0.950354813207245",
  "bbob-biobj_f48_i03_d02 0.974655409279260",
  "bbob-biobj_f48_i03_d03 0.980825113451769",
  "bbob-biobj_f48_i03_d05 0.992968816240894",
  "bbob-biobj_f48_i03_d10 0.980232053391337",
  "bbob-biobj_f48_i03_d20 0.973440818791231",
  "bbob-biobj_f48_i03_d40 0.963798862128042",
  "bbob-biobj_f48_i04_d02 0.991892550874017",
  "bbob-biobj_f48_i04_d03 0.977081432021119",
  "bbob-biobj_f48_i04_d05 0.979439532610778",
  "bbob-biobj_f48_i04_d10 0.971231534195990",
  "bbob-biobj_f48_i04_d20 0.985547755790250",
  "bbob-biobj_f48_i04_d40 0.963752407198306",
  "bbob-biobj_f48_i05_d02 0.988654650933056",
  "bbob-biobj_f48_i05_d03 0.980478926766229",
  "bbob-biobj_f48_i05_d05 0.973958153574718",
  "bbob-biobj_f48_i05_d10 0.988012377114494",
  "bbob-biobj_f48_i05_d20 0.964751207182267",
  "bbob-biobj_f48_i05_d40 0.974037853059613",
  "bbob-biobj_f48_i06_d02 0.389161282124619",
  "bbob-biobj_f48_i06_d03 0.992882959391076",
  "bbob-biobj_f48_i06_d05 0.970629126722307",
  "bbob-biobj_f48_i06_d10 0.983341943001059",
  "bbob-biobj_f48_i06_d20 0.970668477893046",
  "bbob-biobj_f48_i06_d40 0.939575270632468",
  "bbob-biobj_f48_i07_d02 0.986578674413247",
  "bbob-biobj_f48_i07_d03 0.985043874797828",
  "bbob-biobj_f48_i07_d05 0.988805264493350",
  "bbob-biobj_f48_i07_d10 0.982119946475616",
  "bbob-biobj_f48_i07_d20 0.993810678267889",
  "bbob-biobj_f48_i07_d40 0.923761463863783",
  "bbob-biobj_f48_i08_d02 0.954772791083423",
  "bbob-biobj_f48_i08_d03 0.954650939645044",
  "bbob-biobj_f48_i08_d05 0.972928647840613",
  "bbob-biobj_f48_i08_d10 0.974914493338424",
  "bbob-biobj_f48_i08_d20 0.987509150847552",
  "bbob-biobj_f48_i08_d40 0.934297273342588",
  "bbob-biobj_f48_i09_d02 0.957382149250395",
  "bbob-biobj_f48_i09_d03 0.953072541289578",
  "bbob-biobj_f48_i09_d05 0.976017541500116",
  "bbob-biobj_f48_i09_d10 0.992404633270568",
  "bbob-biobj_f48_i09_d20 0.980543548849746",
  "bbob-biobj_f48_i09_d40 0.925438387242997",
  "bbob-biobj_f48_i10_d02 0.993959990229703",
  "bbob-biobj_f48_i10_d03 0.983284115742582",
  "bbob-biobj_f48_i10_d05 0.974352914287965",
  "bbob-biobj_f48_i10_d10 0.994718446676883",
  "bbob-biobj_f48_i10_d20 0.988265396177716",
  "bbob-biobj_f48_i10_d40 0.914954240869837",
  "bbob-biobj_f48_i11_d02 1.0",
  "bbob-biobj_f48_i11_d03 1.0",
  "bbob-biobj_f48_i11_d05 1.0",
  "bbob-biobj_f48_i11_d10 1.0",
  "bbob-biobj_f48_i11_d20 1.0",
  "bbob-biobj_f48_i11_d40 1.0",
  "bbob-biobj_f48_i12_d02 1.0",
  "bbob-biobj_f48_i12_d03 1.0",
  "bbob-biobj_f48_i12_d05 1.0",
  "bbob-biobj_f48_i12_d10 1.0",
  "bbob-biobj_f48_i12_d20 1.0",
  "bbob-biobj_f48_i12_d40 1.0",
  "bbob-biobj_f48_i13_d02 1.0",
  "bbob-biobj_f48_i13_d03 1.0",
  "bbob-biobj_f48_i13_d05 1.0",
  "bbob-biobj_f48_i13_d10 1.0",
  "bbob-biobj_f48_i13_d20 1.0",
  "bbob-biobj_f48_i13_d40 1.0",
  "bbob-biobj_f48_i14_d02 1.0",
  "bbob-biobj_f48_i14_d03 1.0",
  "bbob-biobj_f48_i14_d05 1.0",
  "bbob-biobj_f48_i14_d10 1.0",
  "bbob-biobj_f48_i14_d20 1.0",
  "bbob-biobj_f48_i14_d40 1.0",
  "bbob-biobj_f48_i15_d02 1.0",
  "bbob-biobj_f48_i15_d03 1.0",
  "bbob-biobj_f48_i15_d05 1.0",
  "bbob-biobj_f48_i15_d10 1.0",
  "bbob-biobj_f48_i15_d20 1.0",
  "bbob-biobj_f48_i15_d40 1.0",
  "bbob-biobj_f49_i01_d02 0.926653583269295",
  "bbob-biobj_f49_i01_d03 0.957627451209422",
  "bbob-biobj_f49_i01_d05 0.956464400283289",
  "bbob-biobj_f49_i01_d10 0.924691143308051",
  "bbob-biobj_f49_i01_d20 0.883001271777902",
  "bbob-biobj_f49_i01_d40 0.808010446209920",
  "bbob-biobj_f49_i02_d02 0.961467454994650",
  "bbob-biobj_f49_i02_d03 0.946679568656762",
  "bbob-biobj_f49_i02_d05 0.955517449441158",
  "bbob-biobj_f49_i02_d10 0.850133695702020",
  "bbob-biobj_f49_i02_d20 0.840080814232969",
  "bbob-biobj_f49_i02_d40 0.774650848356241",
  "bbob-biobj_f49_i03_d02 0.848123581192731",
  "bbob-biobj_f49_i03_d03 0.942236304247361",
  "bbob-biobj_f49_i03_d05 0.964450544000933",
  "bbob-biobj_f49_i03_d10 0.928249510306635",
  "bbob-biobj_f49_i03_d20 0.870765936863144",
  "bbob-biobj_f49_i03_d40 0.774359285613625",
  "bbob-biobj_f49_i04_d02 0.934365886903383",
  "bbob-biobj_f49_i04_d03 0.927010836573489",
  "bbob-biobj_f49_i04_d05 0.968264078437842",
  "bbob-biobj_f49_i04_d10 0.879063574520805",
  "bbob-biobj_f49_i04_d20 0.874796797376654",
  "bbob-biobj_f49_i04_d40 0.904891343924901",
  "bbob-biobj_f49_i05_d02 0.928529428713596",
  "bbob-biobj_f49_i05_d03 0.899348277744186",
  "bbob-biobj_f49_i05_d05 0.951111018439651",
  "bbob-biobj_f49_i05_d10 0.876306137236896",
  "bbob-biobj_f49_i05_d20 0.841627007998341",
  "bbob-biobj_f49_i05_d40 0.866802118424087",
  "bbob-biobj_f49_i06_d02 0.786973926020565",
  "bbob-biobj_f49_i06_d03 0.938858420224315",
  "bbob-biobj_f49_i06_d05 0.892745715408364",
  "bbob-biobj_f49_i06_d10 0.960146156492849",
  "bbob-biobj_f49_i06_d20 0.921534008454093",
  "bbob-biobj_f49_i06_d40 0.794969577213544",
  "bbob-biobj_f49_i07_d02 0.725230278357155",
  "bbob-biobj_f49_i07_d03 0.976798654205709",
  "bbob-biobj_f49_i07_d05 0.959532844172694",
  "bbob-biobj_f49_i07_d10 0.949618692671281",
  "bbob-biobj_f49_i07_d20 0.859873584525347",
  "bbob-biobj_f49_i07_d40 0.764664632091040",
  "bbob-biobj_f49_i08_d02 0.959123907095592",
  "bbob-biobj_f49_i08_d03 0.979610246558130",
  "bbob-biobj_f49_i08_d05 0.887922490868379",
  "bbob-biobj_f49_i08_d10 0.951943512943179",
  "bbob-biobj_f49_i08_d20 0.924566368254306",
  "bbob-biobj_f49_i08_d40 0.765412859882910",
  "bbob-biobj_f49_i09_d02 0.970413096029000",
  "bbob-biobj_f49_i09_d03 0.934579800752181",
  "bbob-biobj_f49_i09_d05 0.984332151398107",
  "bbob-biobj_f49_i09_d10 0.919846544331273",
  "bbob-biobj_f49_i09_d20 0.817439482831461",
  "bbob-biobj_f49_i09_d40 0.781086100934854",
  "bbob-biobj_f49_i10_d02 0.914459126016654",
  "bbob-biobj_f49_i10_d03 0.956448691116473",
  "bbob-biobj_f49_i10_d05 0.983357299901989",
  "bbob-biobj_f49_i10_d10 0.971101207563038",
  "bbob-biobj_f49_i10_d20 0.908009824573513",
  "bbob-biobj_f49_i10_d40 0.853708880678775",
  "bbob-biobj_f49_i11_d02 1.0",
  "bbob-biobj_f49_i11_d03 1.0",
  "bbob-biobj_f49_i11_d05 1.0",
  "bbob-biobj_f49_i11_d10 1.0",
  "bbob-biobj_f49_i11_d20 1.0",
  "bbob-biobj_f49_i11_d40 1.0",
  "bbob-biobj_f49_i12_d02 1.0",
  "bbob-biobj_f49_i12_d03 1.0",
  "bbob-biobj_f49_i12_d05 1.0",
  "bbob-biobj_f49_i12_d10 1.0",
  "bbob-biobj_f49_i12_d20 1.0",
  "bbob-biobj_f49_i12_d40 1.0",
  "bbob-biobj_f49_i13_d02 1.0",
  "bbob-biobj_f49_i13_d03 1.0",
  "bbob-biobj_f49_i13_d05 1.0",
  "bbob-biobj_f49_i13_d10 1.0",
  "bbob-biobj_f49_i13_d20 1.0",
  "bbob-biobj_f49_i13_d40 1.0",
  "bbob-biobj_f49_i14_d02 1.0",
  "bbob-biobj_f49_i14_d03 1.0",
  "bbob-biobj_f49_i14_d05 1.0",
  "bbob-biobj_f49_i14_d10 1.0",
  "bbob-biobj_f49_i14_d20 1.0",
  "bbob-biobj_f49_i14_d40 1.0",
  "bbob-biobj_f49_i15_d02 1.0",
  "bbob-biobj_f49_i15_d03 1.0",
  "bbob-biobj_f49_i15_d05 1.0",
  "bbob-biobj_f49_i15_d10 1.0",
  "bbob-biobj_f49_i15_d20 1.0",
  "bbob-biobj_f49_i15_d40 1.0",
  "bbob-biobj_f50_i01_d02 0.908153019029111",
  "bbob-biobj_f50_i01_d03 0.969143802394247",
  "bbob-biobj_f50_i01_d05 0.971221008283031",
  "bbob-biobj_f50_i01_d10 0.937806045408397",
  "bbob-biobj_f50_i01_d20 0.970534096699938",
  "bbob-biobj_f50_i01_d40 0.963208776860388",
  "bbob-biobj_f50_i02_d02 0.951055982970115",
  "bbob-biobj_f50_i02_d03 0.967318967089656",
  "bbob-biobj_f50_i02_d05 0.954449745524030",
  "bbob-biobj_f50_i02_d10 0.962522683964437",
  "bbob-biobj_f50_i02_d20 0.972725400959617",
  "bbob-biobj_f50_i02_d40 0.961091047677661",
  "bbob-biobj_f50_i03_d02 0.921885288089289",
  "bbob-biobj_f50_i03_d03 0.913001104267832",
  "bbob-biobj_f50_i03_d05 0.961402283754169",
  "bbob-biobj_f50_i03_d10 0.935786175448639",
  "bbob-biobj_f50_i03_d20 0.961360669884832",
  "bbob-biobj_f50_i03_d40 0.930380508112140",
  "bbob-biobj_f50_i04_d02 0.908169184637656",
  "bbob-biobj_f50_i04_d03 0.826472492158909",
  "bbob-biobj_f50_i04_d05 0.972864780921446",
  "bbob-biobj_f50_i04_d10 0.980278685964988",
  "bbob-biobj_f50_i04_d20 0.988771672486027",
  "bbob-biobj_f50_i04_d40 0.948589732443901",
  "bbob-biobj_f50_i05_d02 0.808869771430175",
  "bbob-biobj_f50_i05_d03 0.990888217555794",
  "bbob-biobj_f50_i05_d05 0.994314364726843",
  "bbob-biobj_f50_i05_d10 0.975063208666217",
  "bbob-biobj_f50_i05_d20 0.984258303473839",
  "bbob-biobj_f50_i05_d40 0.974082920338684",
  "bbob-biobj_f50_i06_d02 0.917893652037625",
  "bbob-biobj_f50_i06_d03 0.982435397284171",
  "bbob-biobj_f50_i06_d05 0.960919636000301",
  "bbob-biobj_f50_i06_d10 0.932682528025179",
  "bbob-biobj_f50_i06_d20 0.987220010500114",
  "bbob-biobj_f50_i06_d40 0.949611215723517",
  "bbob-biobj_f50_i07_d02 0.900109012077019",
  "bbob-biobj_f50_i07_d03 0.920306601245777",
  "bbob-biobj_f50_i07_d05 0.964454455379369",
  "bbob-biobj_f50_i07_d10 0.920054372340101",
  "bbob-biobj_f50_i07_d20 0.969312542880872",
  "bbob-biobj_f50_i07_d40 0.909610744875217",
  "bbob-biobj_f50_i08_d02 0.947066859501796",
  "bbob-biobj_f50_i08_d03 0.907539940520492",
  "bbob-biobj_f50_i08_d05 0.934424379801040",
  "bbob-biobj_f50_i08_d10 0.955482538306266",
  "bbob-biobj_f50_i08_d20 0.967561661901138",
  "bbob-biobj_f50_i08_d40 0.920378712617924",
  "bbob-biobj_f50_i09_d02 0.864750386477023",
  "bbob-biobj_f50_i09_d03 0.838936880954842",
  "bbob-biobj_f50_i09_d05 0.974769010237717",
  "bbob-biobj_f50_i09_d10 0.959771707855121",
  "bbob-biobj_f50_i09_d20 0.978814760423101",
  "bbob-biobj_f50_i09_d40 0.919391253272068",
  "bbob-biobj_f50_i10_d02 0.901004616658922",
  "bbob-biobj_f50_i10_d03 0.792672193879899",
  "bbob-biobj_f50_i10_d05 0.861738251567046",
  "bbob-biobj_f50_i10_d10 0.967553940320211",
  "bbob-biobj_f50_i10_d20 0.979927098989679",
  "bbob-biobj_f50_i10_d40 0.921438436191370",
  "bbob-biobj_f50_i11_d02 1.0",
  "bbob-biobj_f50_i11_d03 1.0",
  "bbob-biobj_f50_i11_d05 1.0",
  "bbob-biobj_f50_i11_d10 1.0",
  "bbob-biobj_f50_i11_d20 1.0",
  "bbob-biobj_f50_i11_d40 1.0",
  "bbob-biobj_f50_i12_d02 1.0",
  "bbob-biobj_f50_i12_d03 1.0",
  "bbob-biobj_f50_i12_d05 1.0",
  "bbob-biobj_f50_i12_d10 1.0",
  "bbob-biobj_f50_i12_d20 1.0",
  "bbob-biobj_f50_i12_d40 1.0",
  "bbob-biobj_f50_i13_d02 1.0",
  "bbob-biobj_f50_i13_d03 1.0",
  "bbob-biobj_f50_i13_d05 1.0",
  "bbob-biobj_f50_i13_d10 1.0",
  "bbob-biobj_f50_i13_d20 1.0",
  "bbob-biobj_f50_i13_d40 1.0",
  "bbob-biobj_f50_i14_d02 1.0",
  "bbob-biobj_f50_i14_d03 1.0",
  "bbob-biobj_f50_i14_d05 1.0",
  "bbob-biobj_f50_i14_d10 1.0",
  "bbob-biobj_f50_i14_d20 1.0",
  "bbob-biobj_f50_i14_d40 1.0",
  "bbob-biobj_f50_i15_d02 1.0",
  "bbob-biobj_f50_i15_d03 1.0",
  "bbob-biobj_f50_i15_d05 1.0",
  "bbob-biobj_f50_i15_d10 1.0",
  "bbob-biobj_f50_i15_d20 1.0",
  "bbob-biobj_f50_i15_d40 1.0",
  "bbob-biobj_f51_i01_d02 0.940860622956402",
  "bbob-biobj_f51_i01_d03 0.868132937449447",
  "bbob-biobj_f51_i01_d05 0.986104585596155",
  "bbob-biobj_f51_i01_d10 0.980590654398549",
  "bbob-biobj_f51_i01_d20 0.988870039487499",
  "bbob-biobj_f51_i01_d40 0.962072401299433",
  "bbob-biobj_f51_i02_d02 0.909955920096103",
  "bbob-biobj_f51_i02_d03 0.990190457639888",
  "bbob-biobj_f51_i02_d05 0.980597486676055",
  "bbob-biobj_f51_i02_d10 0.991120502560612",
  "bbob-biobj_f51_i02_d20 0.993638471205306",
  "bbob-biobj_f51_i02_d40 0.971053668423443",
  "bbob-biobj_f51_i03_d02 0.941593321951149",
  "bbob-biobj_f51_i03_d03 0.933223460442475",
  "bbob-biobj_f51_i03_d05 0.972313234080961",
  "bbob-biobj_f51_i03_d10 0.968916597600458",
  "bbob-biobj_f51_i03_d20 0.984620917109957",
  "bbob-biobj_f51_i03_d40 0.976831958247220",
  "bbob-biobj_f51_i04_d02 0.986038980461909",
  "bbob-biobj_f51_i04_d03 0.947539307416669",
  "bbob-biobj_f51_i04_d05 0.983531663993893",
  "bbob-biobj_f51_i04_d10 0.997696597033701",
  "bbob-biobj_f51_i04_d20 0.989275960210909",
  "bbob-biobj_f51_i04_d40 0.987757024704872",
  "bbob-biobj_f51_i05_d02 0.971128897412135",
  "bbob-biobj_f51_i05_d03 0.983405022796850",
  "bbob-biobj_f51_i05_d05 0.994450760664070",
  "bbob-biobj_f51_i05_d10 0.965345335670043",
  "bbob-biobj_f51_i05_d20 0.985153015165469",
  "bbob-biobj_f51_i05_d40 0.980794637268118",
  "bbob-biobj_f51_i06_d02 0.911416519677454",
  "bbob-biobj_f51_i06_d03 0.973249576387185",
  "bbob-biobj_f51_i06_d05 0.983513317510326",
  "bbob-biobj_f51_i06_d10 0.982481654524230",
  "bbob-biobj_f51_i06_d20 0.981270294614853",
  "bbob-biobj_f51_i06_d40 0.976389464264435",
  "bbob-biobj_f51_i07_d02 0.993373695494302",
  "bbob-biobj_f51_i07_d03 0.993098771393319",
  "bbob-biobj_f51_i07_d05 0.986024836022638",
  "bbob-biobj_f51_i07_d10 0.990500668757965",
  "bbob-biobj_f51_i07_d20 0.981966525510357",
  "bbob-biobj_f51_i07_d40 0.966246144584898",
  "bbob-biobj_f51_i08_d02 0.985466839182422",
  "bbob-biobj_f51_i08_d03 0.761177260258873",
  "bbob-biobj_f51_i08_d05 0.964401210532718",
  "bbob-biobj_f51_i08_d10 0.995237997246082",
  "bbob-biobj_f51_i08_d20 0.985624637157006",
  "bbob-biobj_f51_i08_d40 0.962822382253301",
  "bbob-biobj_f51_i09_d02 0.942349371111133",
  "bbob-biobj_f51_i09_d03 0.994401740148738",
  "bbob-biobj_f51_i09_d05 0.981355916126111",
  "bbob-biobj_f51_i09_d10 0.982895323802829",
  "bbob-biobj_f51_i09_d20 0.992431408415566",
  "bbob-biobj_f51_i09_d40 0.904488520450946",
  "bbob-biobj_f51_i10_d02 0.969188689957788",
  "bbob-biobj_f51_i10_d03 0.982914020236925",
  "bbob-biobj_f51_i10_d05 0.961917917181196",
  "bbob-biobj_f51_i10_d10 0.976536065224792",
  "bbob-biobj_f51_i10_d20 0.982313386455097",
  "bbob-biobj_f51_i10_d40 0.972730879274001",
  "bbob-biobj_f51_i11_d02 1.0",
  "bbob-biobj_f51_i11_d03 1.0",
  "bbob-biobj_f51_i11_d05 1.0",
  "bbob-biobj_f51_i11_d10 1.0",
  "bbob-biobj_f51_i11_d20 1.0",
  "bbob-biobj_f51_i11_d40 1.0",
  "bbob-biobj_f51_i12_d02 1.0",
  "bbob-biobj_f51_i12_d03 1.0",
  "bbob-biobj_f51_i12_d05 1.0",
  "bbob-biobj_f51_i12_d10 1.0",
  "bbob-biobj_f51_i12_d20 1.0",
  "bbob-biobj_f51_i12_d40 1.0",
  "bbob-biobj_f51_i13_d02 1.0",
  "bbob-biobj_f51_i13_d03 1.0",
  "bbob-biobj_f51_i13_d05 1.0",
  "bbob-biobj_f51_i13_d10 1.0",
  "bbob-biobj_f51_i13_d20 1.0",
  "bbob-biobj_f51_i13_d40 1.0",
  "bbob-biobj_f51_i14_d02 1.0",
  "bbob-biobj_f51_i14_d03 1.0",
  "bbob-biobj_f51_i14_d05 1.0",
  "bbob-biobj_f51_i14_d10 1.0",
  "bbob-biobj_f51_i14_d20 1.0",
  "bbob-biobj_f51_i14_d40 1.0",
  "bbob-biobj_f51_i15_d02 1.0",
  "bbob-biobj_f51_i15_d03 1.0",
  "bbob-biobj_f51_i15_d05 1.0",
  "bbob-biobj_f51_i15_d10 1.0",
  "bbob-biobj_f51_i15_d20 1.0",
  "bbob-biobj_f51_i15_d40 1.0",
  "bbob-biobj_f52_i01_d02 0.947011565973429",
  "bbob-biobj_f52_i01_d03 0.887655083094057",
  "bbob-biobj_f52_i01_d05 0.961544365098409",
  "bbob-biobj_f52_i01_d10 0.880609852012585",
  "bbob-biobj_f52_i01_d20 0.951850617726202",
  "bbob-biobj_f52_i01_d40 0.959665804760387",
  "bbob-biobj_f52_i02_d02 0.786879929743507",
  "bbob-biobj_f52_i02_d03 0.975827723704229",
  "bbob-biobj_f52_i02_d05 0.979002884357261",
  "bbob-biobj_f52_i02_d10 0.908877396177977",
  "bbob-biobj_f52_i02_d20 0.952795318395061",
  "bbob-biobj_f52_i02_d40 0.763031400172234",
  "bbob-biobj_f52_i03_d02 0.931104872267525",
  "bbob-biobj_f52_i03_d03 0.944892661248204",
  "bbob-biobj_f52_i03_d05 0.981095090201299",
  "bbob-biobj_f52_i03_d10 0.928681714337413",
  "bbob-biobj_f52_i03_d20 0.898681720929998",
  "bbob-biobj_f52_i03_d40 0.930998941682665",
  "bbob-biobj_f52_i04_d02 0.719116774075094",
  "bbob-biobj_f52_i04_d03 0.963898316853337",
  "bbob-biobj_f52_i04_d05 0.990702143798999",
  "bbob-biobj_f52_i04_d10 0.991866341435212",
  "bbob-biobj_f52_i04_d20 0.922398943813953",
  "bbob-biobj_f52_i04_d40 0.850048158211609",
  "bbob-biobj_f52_i05_d02 0.781598890865292",
  "bbob-biobj_f52_i05_d03 0.970416804183748",
  "bbob-biobj_f52_i05_d05 0.986024903943445",
  "bbob-biobj_f52_i05_d10 0.882103650889895",
  "bbob-biobj_f52_i05_d20 0.937532910793425",
  "bbob-biobj_f52_i05_d40 0.915203342619471",
  "bbob-biobj_f52_i06_d02 0.642310315034435",
  "bbob-biobj_f52_i06_d03 0.891675389086588",
  "bbob-biobj_f52_i06_d05 0.973023404031696",
  "bbob-biobj_f52_i06_d10 0.800202443940251",
  "bbob-biobj_f52_i06_d20 0.868836337974470",
  "bbob-biobj_f52_i06_d40 0.869567853696131",
  "bbob-biobj_f52_i07_d02 0.921082784944154",
  "bbob-biobj_f52_i07_d03 0.981614450878430",
  "bbob-biobj_f52_i07_d05 0.944431471766394",
  "bbob-biobj_f52_i07_d10 0.956950187414469",
  "bbob-biobj_f52_i07_d20 0.943960851479189",
  "bbob-biobj_f52_i07_d40 0.872863936698003",
  "bbob-biobj_f52_i08_d02 0.993175903549896",
  "bbob-biobj_f52_i08_d03 0.936550893756555",
  "bbob-biobj_f52_i08_d05 0.974796495933045",
  "bbob-biobj_f52_i08_d10 0.913646661161340",
  "bbob-biobj_f52_i08_d20 0.900335395151447",
  "bbob-biobj_f52_i08_d40 0.926013943945283",
  "bbob-biobj_f52_i09_d02 0.975729533922839",
  "bbob-biobj_f52_i09_d03 0.946974860900458",
  "bbob-biobj_f52_i09_d05 0.973454345970633",
  "bbob-biobj_f52_i09_d10 0.937299524235731",
  "bbob-biobj_f52_i09_d20 0.987128376438317",
  "bbob-biobj_f52_i09_d40 0.849349703458808",
  "bbob-biobj_f52_i10_d02 0.873618189290749",
  "bbob-biobj_f52_i10_d03 0.976858150689502",
  "bbob-biobj_f52_i10_d05 0.968338375271559",
  "bbob-biobj_f52_i10_d10 0.948528475089375",
  "bbob-biobj_f52_i10_d20 0.926756784862981",
  "bbob-biobj_f52_i10_d40 0.857193764938067",
  "bbob-biobj_f52_i11_d02 1.0",
  "bbob-biobj_f52_i11_d03 1.0",
  "bbob-biobj_f52_i11_d05 1.0",
  "bbob-biobj_f52_i11_d10 1.0",
  "bbob-biobj_f52_i11_d20 1.0",
  "bbob-biobj_f52_i11_d40 1.0",
  "bbob-biobj_f52_i12_d02 1.0",
  "bbob-biobj_f52_i12_d03 1.0",
  "bbob-biobj_f52_i12_d05 1.0",
  "bbob-biobj_f52_i12_d10 1.0",
  "bbob-biobj_f52_i12_d20 1.0",
  "bbob-biobj_f52_i12_d40 1.0",
  "bbob-biobj_f52_i13_d02 1.0",
  "bbob-biobj_f52_i13_d03 1.0",
  "bbob-biobj_f52_i13_d05 1.0",
  "bbob-biobj_f52_i13_d10 1.0",
  "bbob-biobj_f52_i13_d20 1.0",
  "bbob-biobj_f52_i13_d40 1.0",
  "bbob-biobj_f52_i14_d02 1.0",
  "bbob-biobj_f52_i14_d03 1.0",
  "bbob-biobj_f52_i14_d05 1.0",
  "bbob-biobj_f52_i14_d10 1.0",
  "bbob-biobj_f52_i14_d20 1.0",
  "bbob-biobj_f52_i14_d40 1.0",
  "bbob-biobj_f52_i15_d02 1.0",
  "bbob-biobj_f52_i15_d03 1.0",
  "bbob-biobj_f52_i15_d05 1.0",
  "bbob-biobj_f52_i15_d10 1.0",
  "bbob-biobj_f52_i15_d20 1.0",
  "bbob-biobj_f52_i15_d40 1.0",
  "bbob-biobj_f53_i01_d02 0.997875692921781",
  "bbob-biobj_f53_i01_d03 0.999784714181698",
  "bbob-biobj_f53_i01_d05 0.999101073133201",
  "bbob-biobj_f53_i01_d10 0.997770935634617",
  "bbob-biobj_f53_i01_d20 0.993620588032558",
  "bbob-biobj_f53_i01_d40 0.997008253936025",
  "bbob-biobj_f53_i02_d02 0.975730767360763",
  "bbob-biobj_f53_i02_d03 0.981829133734374",
  "bbob-biobj_f53_i02_d05 0.998940051716482",
  "bbob-biobj_f53_i02_d10 0.995012930374049",
  "bbob-biobj_f53_i02_d20 0.987972894242487",
  "bbob-biobj_f53_i02_d40 0.997893943682911",
  "bbob-biobj_f53_i03_d02 0.975730768524520",
  "bbob-biobj_f53_i03_d03 0.981829137688355",
  "bbob-biobj_f53_i03_d05 0.997632245909664",
  "bbob-biobj_f53_i03_d10 0.999758330110187",
  "bbob-biobj_f53_i03_d20 0.998226457785466",
  "bbob-biobj_f53_i03_d40 0.999617870488956",
  "bbob-biobj_f53_i04_d02 0.997875695023842",
  "bbob-biobj_f53_i04_d03 0.999784714340619",
  "bbob-biobj_f53_i04_d05 0.999956640160364",
  "bbob-biobj_f53_i04_d10 0.998086949083842",
  "bbob-biobj_f53_i04_d20 0.998921075119552",
  "bbob-biobj_f53_i04_d40 0.998673414327444",
  "bbob-biobj_f53_i05_d02 0.706101956086970",
  "bbob-biobj_f53_i05_d03 0.998471514383848",
  "bbob-biobj_f53_i05_d05 0.998401788287043",
  "bbob-biobj_f53_i05_d10 0.999964506688086",
  "bbob-biobj_f53_i05_d20 0.995342078634822",
  "bbob-biobj_f53_i05_d40 0.992053521228669",
  "bbob-biobj_f53_i06_d02 0.975730736637401",
  "bbob-biobj_f53_i06_d03 0.981829088690173",
  "bbob-biobj_f53_i06_d05 0.999948601090440",
  "bbob-biobj_f53_i06_d10 0.999359368456460",
  "bbob-biobj_f53_i06_d20 0.991810210157473",
  "bbob-biobj_f53_i06_d40 0.994170782028727",
  "bbob-biobj_f53_i07_d02 0.975730732746261",
  "bbob-biobj_f53_i07_d03 0.981829051670767",
  "bbob-biobj_f53_i07_d05 0.997626866918283",
  "bbob-biobj_f53_i07_d10 0.991073860517323",
  "bbob-biobj_f53_i07_d20 0.994946657102409",
  "bbob-biobj_f53_i07_d40 0.997128393161806",
  "bbob-biobj_f53_i08_d02 0.975730731124355",
  "bbob-biobj_f53_i08_d03 0.981829129285381",
  "bbob-biobj_f53_i08_d05 0.986891604919335",
  "bbob-biobj_f53_i08_d10 0.992859639681137",
  "bbob-biobj_f53_i08_d20 0.991326253228450",
  "bbob-biobj_f53_i08_d40 0.993144813419714",
  "bbob-biobj_f53_i09_d02 0.997875685000665",
  "bbob-biobj_f53_i09_d03 0.999784714335332",
  "bbob-biobj_f53_i09_d05 0.999955671721642",
  "bbob-biobj_f53_i09_d10 0.999954725874065",
  "bbob-biobj_f53_i09_d20 0.997813381121439",
  "bbob-biobj_f53_i09_d40 0.989629515726637",
  "bbob-biobj_f53_i10_d02 0.706101787141003",
  "bbob-biobj_f53_i10_d03 0.998471495513510",
  "bbob-biobj_f53_i10_d05 0.996114532965430",
  "bbob-biobj_f53_i10_d10 0.999940309390393",
  "bbob-biobj_f53_i10_d20 0.998288727005705",
  "bbob-biobj_f53_i10_d40 0.996969229118885",
  "bbob-biobj_f53_i11_d02 1.0",
  "bbob-biobj_f53_i11_d03 1.0",
  "bbob-biobj_f53_i11_d05 1.0",
  "bbob-biobj_f53_i11_d10 1.0",
  "bbob-biobj_f53_i11_d20 1.0",
  "bbob-biobj_f53_i11_d40 1.0",
  "bbob-biobj_f53_i12_d02 1.0",
  "bbob-biobj_f53_i12_d03 1.0",
  "bbob-biobj_f53_i12_d05 1.0",
  "bbob-biobj_f53_i12_d10 1.0",
  "bbob-biobj_f53_i12_d20 1.0",
  "bbob-biobj_f53_i12_d40 1.0",
  "bbob-biobj_f53_i13_d02 1.0",
  "bbob-biobj_f53_i13_d03 1.0",
  "bbob-biobj_f53_i13_d05 1.0",
  "bbob-biobj_f53_i13_d10 1.0",
  "bbob-biobj_f53_i13_d20 1.0",
  "bbob-biobj_f53_i13_d40 1.0",
  "bbob-biobj_f53_i14_d02 1.0",
  "bbob-biobj_f53_i14_d03 1.0",
  "bbob-biobj_f53_i14_d05 1.0",
  "bbob-biobj_f53_i14_d10 1.0",
  "bbob-biobj_f53_i14_d20 1.0",
  "bbob-biobj_f53_i14_d40 1.0",
  "bbob-biobj_f53_i15_d02 1.0",
  "bbob-biobj_f53_i15_d03 1.0",
  "bbob-biobj_f53_i15_d05 1.0",
  "bbob-biobj_f53_i15_d10 1.0",
  "bbob-biobj_f53_i15_d20 1.0",
  "bbob-biobj_f53_i15_d40 1.0",
  "bbob-biobj_f54_i01_d02 0.943563947345365",
  "bbob-biobj_f54_i01_d03 0.975179483828660",
  "bbob-biobj_f54_i01_d05 0.986735978557260",
  "bbob-biobj_f54_i01_d10 0.971020975989233",
  "bbob-biobj_f54_i01_d20 0.975754562298456",
  "bbob-biobj_f54_i01_d40 0.940619701836704",
  "bbob-biobj_f54_i02_d02 0.929274474829962",
  "bbob-biobj_f54_i02_d03 0.971505490328074",
  "bbob-biobj_f54_i02_d05 0.988596502264416",
  "bbob-biobj_f54_i02_d10 0.979018161955299",
  "bbob-biobj_f54_i02_d20 0.978503935730430",
  "bbob-biobj_f54_i02_d40 0.944836572146846",
  "bbob-biobj_f54_i03_d02 0.991716135247573",
  "bbob-biobj_f54_i03_d03 0.991021577879334",
  "bbob-biobj_f54_i03_d05 0.990017103886950",
  "bbob-biobj_f54_i03_d10 0.971485743914789",
  "bbob-biobj_f54_i03_d20 0.985298830810207",
  "bbob-biobj_f54_i03_d40 0.956655441082847",
  "bbob-biobj_f54_i04_d02 0.984075359489210",
  "bbob-biobj_f54_i04_d03 0.985246735720961",
  "bbob-biobj_f54_i04_d05 0.992621802368755",
  "bbob-biobj_f54_i04_d10 0.987233984371954",
  "bbob-biobj_f54_i04_d20 0.980209807661969",
  "bbob-biobj_f54_i04_d40 0.913196115921382",
  "bbob-biobj_f54_i05_d02 0.977343826732106",
  "bbob-biobj_f54_i05_d03 0.956277057377107",
  "bbob-biobj_f54_i05_d05 0.991382525490389",
  "bbob-biobj_f54_i05_d10 0.963574327834559",
  "bbob-biobj_f54_i05_d20 0.984351904270309",
  "bbob-biobj_f54_i05_d40 0.912702276221518",
  "bbob-biobj_f54_i06_d02 0.737530413924196",
  "bbob-biobj_f54_i06_d03 0.999165572581585",
  "bbob-biobj_f54_i06_d05 0.989460331214994",
  "bbob-biobj_f54_i06_d10 0.978486755665419",
  "bbob-biobj_f54_i06_d20 0.975547602300677",
  "bbob-biobj_f54_i06_d40 0.972391142773130",
  "bbob-biobj_f54_i07_d02 0.905800915025149",
  "bbob-biobj_f54_i07_d03 0.998327608576897",
  "bbob-biobj_f54_i07_d05 0.978766814862955",
  "bbob-biobj_f54_i07_d10 0.967201101453933",
  "bbob-biobj_f54_i07_d20 0.965349683834782",
  "bbob-biobj_f54_i07_d40 0.952199451989289",
  "bbob-biobj_f54_i08_d02 0.989659979524769",
  "bbob-biobj_f54_i08_d03 0.980826071667587",
  "bbob-biobj_f54_i08_d05 0.978332540639450",
  "bbob-biobj_f54_i08_d10 0.986066276830726",
  "bbob-biobj_f54_i08_d20 0.988117414569477",
  "bbob-biobj_f54_i08_d40 0.965650240352066",
  "bbob-biobj_f54_i09_d02 0.948189832749533",
  "bbob-biobj_f54_i09_d03 0.982166030217899",
  "bbob-biobj_f54_i09_d05 0.994880188200191",
  "bbob-biobj_f54_i09_d10 0.983274688085683",
  "bbob-biobj_f54_i09_d20 0.977318477762187",
  "bbob-biobj_f54_i09_d40 0.961157141306381",
  "bbob-biobj_f54_i10_d02 0.974466009143839",
  "bbob-biobj_f54_i10_d03 0.993510177330670",
  "bbob-biobj_f54_i10_d05 0.979811064365768",
  "bbob-biobj_f54_i10_d10 0.980122926489943",
  "bbob-biobj_f54_i10_d20 0.973981891477217",
  "bbob-biobj_f54_i10_d40 0.808176202164876",
  "bbob-biobj_f54_i11_d02 1.0",
  "bbob-biobj_f54_i11_d03 1.0",
  "bbob-biobj_f54_i11_d05 1.0",
  "bbob-biobj_f54_i11_d10 1.0",
  "bbob-biobj_f54_i11_d20 1.0",
  "bbob-biobj_f54_i11_d40 1.0",
  "bbob-biobj_f54_i12_d02 1.0",
  "bbob-biobj_f54_i12_d03 1.0",
  "bbob-biobj_f54_i12_d05 1.0",
  "bbob-biobj_f54_i12_d10 1.0",
  "bbob-biobj_f54_i12_d20 1.0",
  "bbob-biobj_f54_i12_d40 1.0",
  "bbob-biobj_f54_i13_d02 1.0",
  "bbob-biobj_f54_i13_d03 1.0",
  "bbob-biobj_f54_i13_d05 1.0",
  "bbob-biobj_f54_i13_d10 1.0",
  "bbob-biobj_f54_i13_d20 1.0",
  "bbob-biobj_f54_i13_d40 1.0",
  "bbob-biobj_f54_i14_d02 1.0",
  "bbob-biobj_f54_i14_d03 1.0",
  "bbob-biobj_f54_i14_d05 1.0",
  "bbob-biobj_f54_i14_d10 1.0",
  "bbob-biobj_f54_i14_d20 1.0",
  "bbob-biobj_f54_i14_d40 1.0",
  "bbob-biobj_f54_i15_d02 1.0",
  "bbob-biobj_f54_i15_d03 1.0",
  "bbob-biobj_f54_i15_d05 1.0",
  "bbob-biobj_f54_i15_d10 1.0",
  "bbob-biobj_f54_i15_d20 1.0",
  "bbob-biobj_f54_i15_d40 1.0",
  "bbob-biobj_f55_i01_d02 0.994912013563836",
  "bbob-biobj_f55_i01_d03 0.969816640299804",
  "bbob-biobj_f55_i01_d05 0.966515517568895",
  "bbob-biobj_f55_i01_d10 0.911775730837247",
  "bbob-biobj_f55_i01_d20 0.625511450683819",
  "bbob-biobj_f55_i01_d40 0.718241618381943",
  "bbob-biobj_f55_i02_d02 0.936711803357930",
  "bbob-biobj_f55_i02_d03 0.932687634023037",
  "bbob-biobj_f55_i02_d05 0.982279478571098",
  "bbob-biobj_f55_i02_d10 0.933516109255220",
  "bbob-biobj_f55_i02_d20 0.739370942673084",
  "bbob-biobj_f55_i02_d40 0.523868454410108",
  "bbob-biobj_f55_i03_d02 0.979743710493495",
  "bbob-biobj_f55_i03_d03 0.960031091125810",
  "bbob-biobj_f55_i03_d05 0.964544646626252",
  "bbob-biobj_f55_i03_d10 0.953346430698596",
  "bbob-biobj_f55_i03_d20 0.669096157082512",
  "bbob-biobj_f55_i03_d40 0.275426248650253",
  "bbob-biobj_f55_i04_d02 0.931560158530144",
  "bbob-biobj_f55_i04_d03 0.978791413401360",
  "bbob-biobj_f55_i04_d05 0.987527825777800",
  "bbob-biobj_f55_i04_d10 0.853420086888775",
  "bbob-biobj_f55_i04_d20 0.673468643988903",
  "bbob-biobj_f55_i04_d40 0.395054772003883",
  "bbob-biobj_f55_i05_d02 0.890941864072125",
  "bbob-biobj_f55_i05_d03 0.890297839156039",
  "bbob-biobj_f55_i05_d05 0.972349521936232",
  "bbob-biobj_f55_i05_d10 0.932483593169201",
  "bbob-biobj_f55_i05_d20 0.831137046489323",
  "bbob-biobj_f55_i05_d40 0.613317764576754",
  "bbob-biobj_f55_i06_d02 0.878456416911427",
  "bbob-biobj_f55_i06_d03 0.907837070000508",
  "bbob-biobj_f55_i06_d05 0.960630894628764",
  "bbob-biobj_f55_i06_d10 0.872164591569205",
  "bbob-biobj_f55_i06_d20 0.788122550862569",
  "bbob-biobj_f55_i06_d40 0.319534927517951",
  "bbob-biobj_f55_i07_d02 0.938782814424875",
  "bbob-biobj_f55_i07_d03 0.971714983311250",
  "bbob-biobj_f55_i07_d05 0.986550279789914",
  "bbob-biobj_f55_i07_d10 0.938174903148621",
  "bbob-biobj_f55_i07_d20 0.737311436988115",
  "bbob-biobj_f55_i07_d40 0.491484808717019",
  "bbob-biobj_f55_i08_d02 0.901150986249256",
  "bbob-biobj_f55_i08_d03 0.986755213840288",
  "bbob-biobj_f55_i08_d05 0.986851570930107",
  "bbob-biobj_f55_i08_d10 0.863147731496326",
  "bbob-biobj_f55_i08_d20 0.696497298536161",
  "bbob-biobj_f55_i08_d40 0.396202187649523",
  "bbob-biobj_f55_i09_d02 0.869253762734078",
  "bbob-biobj_f55_i09_d03 0.938483554432844",
  "bbob-biobj_f55_i09_d05 0.960424634210952",
  "bbob-biobj_f55_i09_d10 0.910655216684625",
  "bbob-biobj_f55_i09_d20 0.782795793522196",
  "bbob-biobj_f55_i09_d40 0.444870008526301",
  "bbob-biobj_f55_i10_d02 0.851309741230642",
  "bbob-biobj_f55_i10_d03 0.983937286062800",
  "bbob-biobj_f55_i10_d05 0.960687748293214",
  "bbob-biobj_f55_i10_d10 0.892965931330228",
  "bbob-biobj_f55_i10_d20 0.753936183639365",
  "bbob-biobj_f55_i10_d40 0.320849661893284",
  "bbob-biobj_f55_i11_d02 1.0",
  "bbob-biobj_f55_i11_d03 1.0",
  "bbob-biobj_f55_i11_d05 1.0",
  "bbob-biobj_f55_i11_d10 1.0",
  "bbob-biobj_f55_i11_d20 1.0",
  "bbob-biobj_f55_i11_d40 1.0",
  "bbob-biobj_f55_i12_d02 1.0",
  "bbob-biobj_f55_i12_d03 1.0",
  "bbob-biobj_f55_i12_d05 1.0",
  "bbob-biobj_f55_i12_d10 1.0",
  "bbob-biobj_f55_i12_d20 1.0",
  "bbob-biobj_f55_i12_d40 1.0",
  "bbob-biobj_f55_i13_d02 1.0",
  "bbob-biobj_f55_i13_d03 1.0",
  "bbob-biobj_f55_i13_d05 1.0",
  "bbob-biobj_f55_i13_d10 1.0",
  "bbob-biobj_f55_i13_d20 1.0",
  "bbob-biobj_f55_i13_d40 1.0",
  "bbob-biobj_f55_i14_d02 1.0",
  "bbob-biobj_f55_i14_d03 1.0",
  "bbob-biobj_f55_i14_d05 1.0",
  "bbob-biobj_f55_i14_d10 1.0",
  "bbob-biobj_f55_i14_d20 1.0",
  "bbob-biobj_f55_i14_d40 1.0",
  "bbob-biobj_f55_i15_d02 1.0",
  "bbob-biobj_f55_i15_d03 1.0",
  "bbob-biobj_f55_i15_d05 1.0",
  "bbob-biobj_f55_i15_d10 1.0",
  "bbob-biobj_f55_i15_d20 1.0",
  "bbob-biobj_f55_i15_d40 1.0",
  "bbob-biobj_f56_i01_d02 1.0",
  "bbob-biobj_f56_i01_d03 1.0",
  "bbob-biobj_f56_i01_d05 1.0",
  "bbob-biobj_f56_i01_d10 1.0",
  "bbob-biobj_f56_i01_d20 1.0",
  "bbob-biobj_f56_i01_d40 1.0",
  "bbob-biobj_f56_i02_d02 1.0",
  "bbob-biobj_f56_i02_d03 1.0",
  "bbob-biobj_f56_i02_d05 1.0",
  "bbob-biobj_f56_i02_d10 1.0",
  "bbob-biobj_f56_i02_d20 1.0",
  "bbob-biobj_f56_i02_d40 1.0",
  "bbob-biobj_f56_i03_d02 1.0",
  "bbob-biobj_f56_i03_d03 1.0",
  "bbob-biobj_f56_i03_d05 1.0",
  "bbob-biobj_f56_i03_d10 1.0",
  "bbob-biobj_f56_i03_d20 1.0",
  "bbob-biobj_f56_i03_d40 1.0",
  "bbob-biobj_f56_i04_d02 1.0",
  "bbob-biobj_f56_i04_d03 1.0",
  "bbob-biobj_f56_i04_d05 1.0",
  "bbob-biobj_f56_i04_d10 1.0",
  "bbob-biobj_f56_i04_d20 1.0",
  "bbob-biobj_f56_i04_d40 1.0",
  "bbob-biobj_f56_i05_d02 1.0",
  "bbob-biobj_f56_i05_d03 1.0",
  "bbob-biobj_f56_i05_d05 1.0",
  "bbob-biobj_f56_i05_d10 1.0",
  "bbob-biobj_f56_i05_d20 1.0",
  "bbob-biobj_f56_i05_d40 1.0",
  "bbob-biobj_f56_i06_d02 1.0",
  "bbob-biobj_f56_i06_d03 1.0",
  "bbob-biobj_f56_i06_d05 1.0",
  "bbob-biobj_f56_i06_d10 1.0",
  "bbob-biobj_f56_i06_d20 1.0",
  "bbob-biobj_f56_i06_d40 1.0",
  "bbob-biobj_f56_i07_d02 1.0",
  "bbob-biobj_f56_i07_d03 1.0",
  "bbob-biobj_f56_i07_d05 1.0",
  "bbob-biobj_f56_i07_d10 1.0",
  "bbob-biobj_f56_i07_d20 1.0",
  "bbob-biobj_f56_i07_d40 1.0",
  "bbob-biobj_f56_i08_d02 1.0",
  "bbob-biobj_f56_i08_d03 1.0",
  "bbob-biobj_f56_i08_d05 1.0",
  "bbob-biobj_f56_i08_d10 1.0",
  "bbob-biobj_f56_i08_d20 1.0",
  "bbob-biobj_f56_i08_d40 1.0",
  "bbob-biobj_f56_i09_d02 1.0",
  "bbob-biobj_f56_i09_d03 1.0",
  "bbob-biobj_f56_i09_d05 1.0",
  "bbob-biobj_f56_i09_d10 1.0",
  "bbob-biobj_f56_i09_d20 1.0",
  "bbob-biobj_f56_i09_d40 1.0",
  "bbob-biobj_f56_i10_d02 1.0",
  "bbob-biobj_f56_i10_d03 1.0",
  "bbob-biobj_f56_i10_d05 1.0",
  "bbob-biobj_f56_i10_d10 1.0",
  "bbob-biobj_f56_i10_d20 1.0",
  "bbob-biobj_f56_i10_d40 1.0",
  "bbob-biobj_f56_i11_d02 1.0",
  "bbob-biobj_f56_i11_d03 1.0",
  "bbob-biobj_f56_i11_d05 1.0",
  "bbob-biobj_f56_i11_d10 1.0",
  "bbob-biobj_f56_i11_d20 1.0",
  "bbob-biobj_f56_i11_d40 1.0",
  "bbob-biobj_f56_i12_d02 1.0",
  "bbob-biobj_f56_i12_d03 1.0",
  "bbob-biobj_f56_i12_d05 1.0",
  "bbob-biobj_f56_i12_d10 1.0",
  "bbob-biobj_f56_i12_d20 1.0",
  "bbob-biobj_f56_i12_d40 1.0",
  "bbob-biobj_f56_i13_d02 1.0",
  "bbob-biobj_f56_i13_d03 1.0",
  "bbob-biobj_f56_i13_d05 1.0",
  "bbob-biobj_f56_i13_d10 1.0",
  "bbob-biobj_f56_i13_d20 1.0",
  "bbob-biobj_f56_i13_d40 1.0",
  "bbob-biobj_f56_i14_d02 1.0",
  "bbob-biobj_f56_i14_d03 1.0",
  "bbob-biobj_f56_i14_d05 1.0",
  "bbob-biobj_f56_i14_d10 1.0",
  "bbob-biobj_f56_i14_d20 1.0",
  "bbob-biobj_f56_i14_d40 1.0",
  "bbob-biobj_f56_i15_d02 1.0",
  "bbob-biobj_f56_i15_d03 1.0",
  "bbob-biobj_f56_i15_d05 1.0",
  "bbob-biobj_f56_i15_d10 1.0",
  "bbob-biobj_f56_i15_d20 1.0",
  "bbob-biobj_f56_i15_d40 1.0",
  "bbob-biobj_f57_i01_d02 1.0",
  "bbob-biobj_f57_i01_d03 1.0",
  "bbob-biobj_f57_i01_d05 1.0",
  "bbob-biobj_f57_i01_d10 1.0",
  "bbob-biobj_f57_i01_d20 1.0",
  "bbob-biobj_f57_i01_d40 1.0",
  "bbob-biobj_f57_i02_d02 1.0",
  "bbob-biobj_f57_i02_d03 1.0",
  "bbob-biobj_f57_i02_d05 1.0",
  "bbob-biobj_f57_i02_d10 1.0",
  "bbob-biobj_f57_i02_d20 1.0",
  "bbob-biobj_f57_i02_d40 1.0",
  "bbob-biobj_f57_i03_d02 1.0",
  "bbob-biobj_f57_i03_d03 1.0",
  "bbob-biobj_f57_i03_d05 1.0",
  "bbob-biobj_f57_i03_d10 1.0",
  "bbob-biobj_f57_i03_d20 1.0",
  "bbob-biobj_f57_i03_d40 1.0",
  "bbob-biobj_f57_i04_d02 1.0",
  "bbob-biobj_f57_i04_d03 1.0",
  "bbob-biobj_f57_i04_d05 1.0",
  "bbob-biobj_f57_i04_d10 1.0",
  "bbob-biobj_f57_i04_d20 1.0",
  "bbob-biobj_f57_i04_d40 1.0",
  "bbob-biobj_f57_i05_d02 1.0",
  "bbob-biobj_f57_i05_d03 1.0",
  "bbob-biobj_f57_i05_d05 1.0",
  "bbob-biobj_f57_i05_d10 1.0",
  "bbob-biobj_f57_i05_d20 1.0",
  "bbob-biobj_f57_i05_d40 1.0",
  "bbob-biobj_f57_i06_d02 1.0",
  "bbob-biobj_f57_i06_d03 1.0",
  "bbob-biobj_f57_i06_d05 1.0",
  "bbob-biobj_f57_i06_d10 1.0",
  "bbob-biobj_f57_i06_d20 1.0",
  "bbob-biobj_f57_i06_d40 1.0",
  "bbob-biobj_f57_i07_d02 1.0",
  "bbob-biobj_f57_i07_d03 1.0",
  "bbob-biobj_f57_i07_d05 1.0",
  "bbob-biobj_f57_i07_d10 1.0",
  "bbob-biobj_f57_i07_d20 1.0",
  "bbob-biobj_f57_i07_d40 1.0",
  "bbob-biobj_f57_i08_d02 1.0",
  "bbob-biobj_f57_i08_d03 1.0",
  "bbob-biobj_f57_i08_d05 1.0",
  "bbob-biobj_f57_i08_d10 1.0",
  "bbob-biobj_f57_i08_d20 1.0",
  "bbob-biobj_f57_i08_d40 1.0",
  "bbob-biobj_f57_i09_d02 1.0",
  "bbob-biobj_f57_i09_d03 1.0",
  "bbob-biobj_f57_i09_d05 1.0",
  "bbob-biobj_f57_i09_d10 1.0",
  "bbob-biobj_f57_i09_d20 1.0",
  "bbob-biobj_f57_i09_d40 1.0",
  "bbob-biobj_f57_i10_d02 1.0",
  "bbob-biobj_f57_i10_d03 1.0",
  "bbob-biobj_f57_i10_d05 1.0",
  "bbob-biobj_f57_i10_d10 1.0",
  "bbob-biobj_f57_i10_d20 1.0",
  "bbob-biobj_f57_i10_d40 1.0",
  "bbob-biobj_f57_i11_d02 1.0",
  "bbob-biobj_f57_i11_d03 1.0",
  "bbob-biobj_f57_i11_d05 1.0",
  "bbob-biobj_f57_i11_d10 1.0",
  "bbob-biobj_f57_i11_d20 1.0",
  "bbob-biobj_f57_i11_d40 1.0",
  "bbob-biobj_f57_i12_d02 1.0",
  "bbob-biobj_f57_i12_d03 1.0",
  "bbob-biobj_f57_i12_d05 1.0",
  "bbob-biobj_f57_i12_d10 1.0",
  "bbob-biobj_f57_i12_d20 1.0",
  "bbob-biobj_f57_i12_d40 1.0",
  "bbob-biobj_f57_i13_d02 1.0",
  "bbob-biobj_f57_i13_d03 1.0",
  "bbob-biobj_f57_i13_d05 1.0",
  "bbob-biobj_f57_i13_d10 1.0",
  "bbob-biobj_f57_i13_d20 1.0",
  "bbob-biobj_f57_i13_d40 1.0",
  "bbob-biobj_f57_i14_d02 1.0",
  "bbob-biobj_f57_i14_d03 1.0",
  "bbob-biobj_f57_i14_d05 1.0",
  "bbob-biobj_f57_i14_d10 1.0",
  "bbob-biobj_f57_i14_d20 1.0",
  "bbob-biobj_f57_i14_d40 1.0",
  "bbob-biobj_f57_i15_d02 1.0",
  "bbob-biobj_f57_i15_d03 1.0",
  "bbob-biobj_f57_i15_d05 1.0",
  "bbob-biobj_f57_i15_d10 1.0",
  "bbob-biobj_f57_i15_d20 1.0",
  "bbob-biobj_f57_i15_d40 1.0",
  "bbob-biobj_f58_i01_d02 1.0",
  "bbob-biobj_f58_i01_d03 1.0",
  "bbob-biobj_f58_i01_d05 1.0",
  "bbob-biobj_f58_i01_d10 1.0",
  "bbob-biobj_f58_i01_d20 1.0",
  "bbob-biobj_f58_i01_d40 1.0",
  "bbob-biobj_f58_i02_d02 1.0",
  "bbob-biobj_f58_i02_d03 1.0",
  "bbob-biobj_f58_i02_d05 1.0",
  "bbob-biobj_f58_i02_d10 1.0",
  "bbob-biobj_f58_i02_d20 1.0",
  "bbob-biobj_f58_i02_d40 1.0",
  "bbob-biobj_f58_i03_d02 1.0",
  "bbob-biobj_f58_i03_d03 1.0",
  "bbob-biobj_f58_i03_d05 1.0",
  "bbob-biobj_f58_i03_d10 1.0",
  "bbob-biobj_f58_i03_d20 1.0",
  "bbob-biobj_f58_i03_d40 1.0",
  "bbob-biobj_f58_i04_d02 1.0",
  "bbob-biobj_f58_i04_d03 1.0",
  "bbob-biobj_f58_i04_d05 1.0",
  "bbob-biobj_f58_i04_d10 1.0",
  "bbob-biobj_f58_i04_d20 1.0",
  "bbob-biobj_f58_i04_d40 1.0",
  "bbob-biobj_f58_i05_d02 1.0",
  "bbob-biobj_f58_i05_d03 1.0",
  "bbob-biobj_f58_i05_d05 1.0",
  "bbob-biobj_f58_i05_d10 1.0",
  "bbob-biobj_f58_i05_d20 1.0",
  "bbob-biobj_f58_i05_d40 1.0",
  "bbob-biobj_f58_i06_d02 1.0",
  "bbob-biobj_f58_i06_d03 1.0",
  "bbob-biobj_f58_i06_d05 1.0",
  "bbob-biobj_f58_i06_d10 1.0",
  "bbob-biobj_f58_i06_d20 1.0",
  "bbob-biobj_f58_i06_d40 1.0",
  "bbob-biobj_f58_i07_d02 1.0",
  "bbob-biobj_f58_i07_d03 1.0",
  "bbob-biobj_f58_i07_d05 1.0",
  "bbob-biobj_f58_i07_d10 1.0",
  "bbob-biobj_f58_i07_d20 1.0",
  "bbob-biobj_f58_i07_d40 1.0",
  "bbob-biobj_f58_i08_d02 1.0",
  "bbob-biobj_f58_i08_d03 1.0",
  "bbob-biobj_f58_i08_d05 1.0",
  "bbob-biobj_f58_i08_d10 1.0",
  "bbob-biobj_f58_i08_d20 1.0",
  "bbob-biobj_f58_i08_d40 1.0",
  "bbob-biobj_f58_i09_d02 1.0",
  "bbob-biobj_f58_i09_d03 1.0",
  "bbob-biobj_f58_i09_d05 1.0",
  "bbob-biobj_f58_i09_d10 1.0",
  "bbob-biobj_f58_i09_d20 1.0",
  "bbob-biobj_f58_i09_d40 1.0",
  "bbob-biobj_f58_i10_d02 1.0",
  "bbob-biobj_f58_i10_d03 1.0",
  "bbob-biobj_f58_i10_d05 1.0",
  "bbob-biobj_f58_i10_d10 1.0",
  "bbob-biobj_f58_i10_d20 1.0",
  "bbob-biobj_f58_i10_d40 1.0",
  "bbob-biobj_f58_i11_d02 1.0",
  "bbob-biobj_f58_i11_d03 1.0",
  "bbob-biobj_f58_i11_d05 1.0",
  "bbob-biobj_f58_i11_d10 1.0",
  "bbob-biobj_f58_i11_d20 1.0",
  "bbob-biobj_f58_i11_d40 1.0",
  "bbob-biobj_f58_i12_d02 1.0",
  "bbob-biobj_f58_i12_d03 1.0",
  "bbob-biobj_f58_i12_d05 1.0",
  "bbob-biobj_f58_i12_d10 1.0",
  "bbob-biobj_f58_i12_d20 1.0",
  "bbob-biobj_f58_i12_d40 1.0",
  "bbob-biobj_f58_i13_d02 1.0",
  "bbob-biobj_f58_i13_d03 1.0",
  "bbob-biobj_f58_i13_d05 1.0",
  "bbob-biobj_f58_i13_d10 1.0",
  "bbob-biobj_f58_i13_d20 1.0",
  "bbob-biobj_f58_i13_d40 1.0",
  "bbob-biobj_f58_i14_d02 1.0",
  "bbob-biobj_f58_i14_d03 1.0",
  "bbob-biobj_f58_i14_d05 1.0",
  "bbob-biobj_f58_i14_d10 1.0",
  "bbob-biobj_f58_i14_d20 1.0",
  "bbob-biobj_f58_i14_d40 1.0",
  "bbob-biobj_f58_i15_d02 1.0",
  "bbob-biobj_f58_i15_d03 1.0",
  "bbob-biobj_f58_i15_d05 1.0",
  "bbob-biobj_f58_i15_d10 1.0",
  "bbob-biobj_f58_i15_d20 1.0",
  "bbob-biobj_f58_i15_d40 1.0",
  "bbob-biobj_f59_i01_d02 1.0",
  "bbob-biobj_f59_i01_d03 1.0",
  "bbob-biobj_f59_i01_d05 1.0",
  "bbob-biobj_f59_i01_d10 1.0",
  "bbob-biobj_f59_i01_d20 1.0",
  "bbob-biobj_f59_i01_d40 1.0",
  "bbob-biobj_f59_i02_d02 1.0",
  "bbob-biobj_f59_i02_d03 1.0",
  "bbob-biobj_f59_i02_d05 1.0",
  "bbob-biobj_f59_i02_d10 1.0",
  "bbob-biobj_f59_i02_d20 1.0",
  "bbob-biobj_f59_i02_d40 1.0",
  "bbob-biobj_f59_i03_d02 1.0",
  "bbob-biobj_f59_i03_d03 1.0",
  "bbob-biobj_f59_i03_d05 1.0",
  "bbob-biobj_f59_i03_d10 1.0",
  "bbob-biobj_f59_i03_d20 1.0",
  "bbob-biobj_f59_i03_d40 1.0",
  "bbob-biobj_f59_i04_d02 1.0",
  "bbob-biobj_f59_i04_d03 1.0",
  "bbob-biobj_f59_i04_d05 1.0",
  "bbob-biobj_f59_i04_d10 1.0",
  "bbob-biobj_f59_i04_d20 1.0",
  "bbob-biobj_f59_i04_d40 1.0",
  "bbob-biobj_f59_i05_d02 1.0",
  "bbob-biobj_f59_i05_d03 1.0",
  "bbob-biobj_f59_i05_d05 1.0",
  "bbob-biobj_f59_i05_d10 1.0",
  "bbob-biobj_f59_i05_d20 1.0",
  "bbob-biobj_f59_i05_d40 1.0",
  "bbob-biobj_f59_i06_d02 1.0",
  "bbob-biobj_f59_i06_d03 1.0",
  "bbob-biobj_f59_i06_d05 1.0",
  "bbob-biobj_f59_i06_d10 1.0",
  "bbob-biobj_f59_i06_d20 1.0",
  "bbob-biobj_f59_i06_d40 1.0",
  "bbob-biobj_f59_i07_d02 1.0",
  "bbob-biobj_f59_i07_d03 1.0",
  "bbob-biobj_f59_i07_d05 1.0",
  "bbob-biobj_f59_i07_d10 1.0",
  "bbob-biobj_f59_i07_d20 1.0",
  "bbob-biobj_f59_i07_d40 1.0",
  "bbob-biobj_f59_i08_d02 1.0",
  "bbob-biobj_f59_i08_d03 1.0",
  "bbob-biobj_f59_i08_d05 1.0",
  "bbob-biobj_f59_i08_d10 1.0",
  "bbob-biobj_f59_i08_d20 1.0",
  "bbob-biobj_f59_i08_d40 1.0",
  "bbob-biobj_f59_i09_d02 1.0",
  "bbob-biobj_f59_i09_d03 1.0",
  "bbob-biobj_f59_i09_d05 1.0",
  "bbob-biobj_f59_i09_d10 1.0",
  "bbob-biobj_f59_i09_d20 1.0",
  "bbob-biobj_f59_i09_d40 1.0",
  "bbob-biobj_f59_i10_d02 1.0",
  "bbob-biobj_f59_i10_d03 1.0",
  "bbob-biobj_f59_i10_d05 1.0",
  "bbob-biobj_f59_i10_d10 1.0",
  "bbob-biobj_f59_i10_d20 1.0",
  "bbob-biobj_f59_i10_d40 1.0",
  "bbob-biobj_f59_i11_d02 1.0",
  "bbob-biobj_f59_i11_d03 1.0",
  "bbob-biobj_f59_i11_d05 1.0",
  "bbob-biobj_f59_i11_d10 1.0",
  "bbob-biobj_f59_i11_d20 1.0",
  "bbob-biobj_f59_i11_d40 1.0",
  "bbob-biobj_f59_i12_d02 1.0",
  "bbob-biobj_f59_i12_d03 1.0",
  "bbob-biobj_f59_i12_d05 1.0",
  "bbob-biobj_f59_i12_d10 1.0",
  "bbob-biobj_f59_i12_d20 1.0",
  "bbob-biobj_f59_i12_d40 1.0",
  "bbob-biobj_f59_i13_d02 1.0",
  "bbob-biobj_f59_i13_d03 1.0",
  "bbob-biobj_f59_i13_d05 1.0",
  "bbob-biobj_f59_i13_d10 1.0",
  "bbob-biobj_f59_i13_d20 1.0",
  "bbob-biobj_f59_i13_d40 1.0",
  "bbob-biobj_f59_i14_d02 1.0",
  "bbob-biobj_f59_i14_d03 1.0",
  "bbob-biobj_f59_i14_d05 1.0",
  "bbob-biobj_f59_i14_d10 1.0",
  "bbob-biobj_f59_i14_d20 1.0",
  "bbob-biobj_f59_i14_d40 1.0",
  "bbob-biobj_f59_i15_d02 1.0",
  "bbob-biobj_f59_i15_d03 1.0",
  "bbob-biobj_f59_i15_d05 1.0",
  "bbob-biobj_f59_i15_d10 1.0",
  "bbob-biobj_f59_i15_d20 1.0",
  "bbob-biobj_f59_i15_d40 1.0",
  "bbob-biobj_f60_i01_d02 1.0",
  "bbob-biobj_f60_i01_d03 1.0",
  "bbob-biobj_f60_i01_d05 1.0",
  "bbob-biobj_f60_i01_d10 1.0",
  "bbob-biobj_f60_i01_d20 1.0",
  "bbob-biobj_f60_i01_d40 1.0",
  "bbob-biobj_f60_i02_d02 1.0",
  "bbob-biobj_f60_i02_d03 1.0",
  "bbob-biobj_f60_i02_d05 1.0",
  "bbob-biobj_f60_i02_d10 1.0",
  "bbob-biobj_f60_i02_d20 1.0",
  "bbob-biobj_f60_i02_d40 1.0",
  "bbob-biobj_f60_i03_d02 1.0",
  "bbob-biobj_f60_i03_d03 1.0",
  "bbob-biobj_f60_i03_d05 1.0",
  "bbob-biobj_f60_i03_d10 1.0",
  "bbob-biobj_f60_i03_d20 1.0",
  "bbob-biobj_f60_i03_d40 1.0",
  "bbob-biobj_f60_i04_d02 1.0",
  "bbob-biobj_f60_i04_d03 1.0",
  "bbob-biobj_f60_i04_d05 1.0",
  "bbob-biobj_f60_i04_d10 1.0",
  "bbob-biobj_f60_i04_d20 1.0",
  "bbob-biobj_f60_i04_d40 1.0",
  "bbob-biobj_f60_i05_d02 1.0",
  "bbob-biobj_f60_i05_d03 1.0",
  "bbob-biobj_f60_i05_d05 1.0",
  "bbob-biobj_f60_i05_d10 1.0",
  "bbob-biobj_f60_i05_d20 1.0",
  "bbob-biobj_f60_i05_d40 1.0",
  "bbob-biobj_f60_i06_d02 1.0",
  "bbob-biobj_f60_i06_d03 1.0",
  "bbob-biobj_f60_i06_d05 1.0",
  "bbob-biobj_f60_i06_d10 1.0",
  "bbob-biobj_f60_i06_d20 1.0",
  "bbob-biobj_f60_i06_d40 1.0",
  "bbob-biobj_f60_i07_d02 1.0",
  "bbob-biobj_f60_i07_d03 1.0",
  "bbob-biobj_f60_i07_d05 1.0",
  "bbob-biobj_f60_i07_d10 1.0",
  "bbob-biobj_f60_i07_d20 1.0",
  "bbob-biobj_f60_i07_d40 1.0",
  "bbob-biobj_f60_i08_d02 1.0",
  "bbob-biobj_f60_i08_d03 1.0",
  "bbob-biobj_f60_i08_d05 1.0",
  "bbob-biobj_f60_i08_d10 1.0",
  "bbob-biobj_f60_i08_d20 1.0",
  "bbob-biobj_f60_i08_d40 1.0",
  "bbob-biobj_f60_i09_d02 1.0",
  "bbob-biobj_f60_i09_d03 1.0",
  "bbob-biobj_f60_i09_d05 1.0",
  "bbob-biobj_f60_i09_d10 1.0",
  "bbob-biobj_f60_i09_d20 1.0",
  "bbob-biobj_f60_i09_d40 1.0",
  "bbob-biobj_f60_i10_d02 1.0",
  "bbob-biobj_f60_i10_d03 1.0",
  "bbob-biobj_f60_i10_d05 1.0",
  "bbob-biobj_f60_i10_d10 1.0",
  "bbob-biobj_f60_i10_d20 1.0",
  "bbob-biobj_f60_i10_d40 1.0",
  "bbob-biobj_f60_i11_d02 1.0",
  "bbob-biobj_f60_i11_d03 1.0",
  "bbob-biobj_f60_i11_d05 1.0",
  "bbob-biobj_f60_i11_d10 1.0",
  "bbob-biobj_f60_i11_d20 1.0",
  "bbob-biobj_f60_i11_d40 1.0",
  "bbob-biobj_f60_i12_d02 1.0",
  "bbob-biobj_f60_i12_d03 1.0",
  "bbob-biobj_f60_i12_d05 1.0",
  "bbob-biobj_f60_i12_d10 1.0",
  "bbob-biobj_f60_i12_d20 1.0",
  "bbob-biobj_f60_i12_d40 1.0",
  "bbob-biobj_f60_i13_d02 1.0",
  "bbob-biobj_f60_i13_d03 1.0",
  "bbob-biobj_f60_i13_d05 1.0",
  "bbob-biobj_f60_i13_d10 1.0",
  "bbob-biobj_f60_i13_d20 1.0",
  "bbob-biobj_f60_i13_d40 1.0",
  "bbob-biobj_f60_i14_d02 1.0",
  "bbob-biobj_f60_i14_d03 1.0",
  "bbob-biobj_f60_i14_d05 1.0",
  "bbob-biobj_f60_i14_d10 1.0",
  "bbob-biobj_f60_i14_d20 1.0",
  "bbob-biobj_f60_i14_d40 1.0",
  "bbob-biobj_f60_i15_d02 1.0",
  "bbob-biobj_f60_i15_d03 1.0",
  "bbob-biobj_f60_i15_d05 1.0",
  "bbob-biobj_f60_i15_d10 1.0",
  "bbob-biobj_f60_i15_d20 1.0",
  "bbob-biobj_f60_i15_d40 1.0",
  "bbob-biobj_f61_i01_d02 1.0",
  "bbob-biobj_f61_i01_d03 1.0",
  "bbob-biobj_f61_i01_d05 1.0",
  "bbob-biobj_f61_i01_d10 1.0",
  "bbob-biobj_f61_i01_d20 1.0",
  "bbob-biobj_f61_i01_d40 1.0",
  "bbob-biobj_f61_i02_d02 1.0",
  "bbob-biobj_f61_i02_d03 1.0",
  "bbob-biobj_f61_i02_d05 1.0",
  "bbob-biobj_f61_i02_d10 1.0",
  "bbob-biobj_f61_i02_d20 1.0",
  "bbob-biobj_f61_i02_d40 1.0",
  "bbob-biobj_f61_i03_d02 1.0",
  "bbob-biobj_f61_i03_d03 1.0",
  "bbob-biobj_f61_i03_d05 1.0",
  "bbob-biobj_f61_i03_d10 1.0",
  "bbob-biobj_f61_i03_d20 1.0",
  "bbob-biobj_f61_i03_d40 1.0",
  "bbob-biobj_f61_i04_d02 1.0",
  "bbob-biobj_f61_i04_d03 1.0",
  "bbob-biobj_f61_i04_d05 1.0",
  "bbob-biobj_f61_i04_d10 1.0",
  "bbob-biobj_f61_i04_d20 1.0",
  "bbob-biobj_f61_i04_d40 1.0",
  "bbob-biobj_f61_i05_d02 1.0",
  "bbob-biobj_f61_i05_d03 1.0",
  "bbob-biobj_f61_i05_d05 1.0",
  "bbob-biobj_f61_i05_d10 1.0",
  "bbob-biobj_f61_i05_d20 1.0",
  "bbob-biobj_f61_i05_d40 1.0",
  "bbob-biobj_f61_i06_d02 1.0",
  "bbob-biobj_f61_i06_d03 1.0",
  "bbob-biobj_f61_i06_d05 1.0",
  "bbob-biobj_f61_i06_d10 1.0",
  "bbob-biobj_f61_i06_d20 1.0",
  "bbob-biobj_f61_i06_d40 1.0",
  "bbob-biobj_f61_i07_d02 1.0",
  "bbob-biobj_f61_i07_d03 1.0",
  "bbob-biobj_f61_i07_d05 1.0",
  "bbob-biobj_f61_i07_d10 1.0",
  "bbob-biobj_f61_i07_d20 1.0",
  "bbob-biobj_f61_i07_d40 1.0",
  "bbob-biobj_f61_i08_d02 1.0",
  "bbob-biobj_f61_i08_d03 1.0",
  "bbob-biobj_f61_i08_d05 1.0",
  "bbob-biobj_f61_i08_d10 1.0",
  "bbob-biobj_f61_i08_d20 1.0",
  "bbob-biobj_f61_i08_d40 1.0",
  "bbob-biobj_f61_i09_d02 1.0",
  "bbob-biobj_f61_i09_d03 1.0",
  "bbob-biobj_f61_i09_d05 1.0",
  "bbob-biobj_f61_i09_d10 1.0",
  "bbob-biobj_f61_i09_d20 1.0",
  "bbob-biobj_f61_i09_d40 1.0",
  "bbob-biobj_f61_i10_d02 1.0",
  "bbob-biobj_f61_i10_d03 1.0",
  "bbob-biobj_f61_i10_d05 1.0",
  "bbob-biobj_f61_i10_d10 1.0",
  "bbob-biobj_f61_i10_d20 1.0",
  "bbob-biobj_f61_i10_d40 1.0",
  "bbob-biobj_f61_i11_d02 1.0",
  "bbob-biobj_f61_i11_d03 1.0",
  "bbob-biobj_f61_i11_d05 1.0",
  "bbob-biobj_f61_i11_d10 1.0",
  "bbob-biobj_f61_i11_d20 1.0",
  "bbob-biobj_f61_i11_d40 1.0",
  "bbob-biobj_f61_i12_d02 1.0",
  "bbob-biobj_f61_i12_d03 1.0",
  "bbob-biobj_f61_i12_d05 1.0",
  "bbob-biobj_f61_i12_d10 1.0",
  "bbob-biobj_f61_i12_d20 1.0",
  "bbob-biobj_f61_i12_d40 1.0",
  "bbob-biobj_f61_i13_d02 1.0",
  "bbob-biobj_f61_i13_d03 1.0",
  "bbob-biobj_f61_i13_d05 1.0",
  "bbob-biobj_f61_i13_d10 1.0",
  "bbob-biobj_f61_i13_d20 1.0",
  "bbob-biobj_f61_i13_d40 1.0",
  "bbob-biobj_f61_i14_d02 1.0",
  "bbob-biobj_f61_i14_d03 1.0",
  "bbob-biobj_f61_i14_d05 1.0",
  "bbob-biobj_f61_i14_d10 1.0",
  "bbob-biobj_f61_i14_d20 1.0",
  "bbob-biobj_f61_i14_d40 1.0",
  "bbob-biobj_f61_i15_d02 1.0",
  "bbob-biobj_f61_i15_d03 1.0",
  "bbob-biobj_f61_i15_d05 1.0",
  "bbob-biobj_f61_i15_d10 1.0",
  "bbob-biobj_f61_i15_d20 1.0",
  "bbob-biobj_f61_i15_d40 1.0",
  "bbob-biobj_f62_i01_d02 1.0",
  "bbob-biobj_f62_i01_d03 1.0",
  "bbob-biobj_f62_i01_d05 1.0",
  "bbob-biobj_f62_i01_d10 1.0",
  "bbob-biobj_f62_i01_d20 1.0",
  "bbob-biobj_f62_i01_d40 1.0",
  "bbob-biobj_f62_i02_d02 1.0",
  "bbob-biobj_f62_i02_d03 1.0",
  "bbob-biobj_f62_i02_d05 1.0",
  "bbob-biobj_f62_i02_d10 1.0",
  "bbob-biobj_f62_i02_d20 1.0",
  "bbob-biobj_f62_i02_d40 1.0",
  "bbob-biobj_f62_i03_d02 1.0",
  "bbob-biobj_f62_i03_d03 1.0",
  "bbob-biobj_f62_i03_d05 1.0",
  "bbob-biobj_f62_i03_d10 1.0",
  "bbob-biobj_f62_i03_d20 1.0",
  "bbob-biobj_f62_i03_d40 1.0",
  "bbob-biobj_f62_i04_d02 1.0",
  "bbob-biobj_f62_i04_d03 1.0",
  "bbob-biobj_f62_i04_d05 1.0",
  "bbob-biobj_f62_i04_d10 1.0",
  "bbob-biobj_f62_i04_d20 1.0",
  "bbob-biobj_f62_i04_d40 1.0",
  "bbob-biobj_f62_i05_d02 1.0",
  "bbob-biobj_f62_i05_d03 1.0",
  "bbob-biobj_f62_i05_d05 1.0",
  "bbob-biobj_f62_i05_d10 1.0",
  "bbob-biobj_f62_i05_d20 1.0",
  "bbob-biobj_f62_i05_d40 1.0",
  "bbob-biobj_f62_i06_d02 1.0",
  "bbob-biobj_f62_i06_d03 1.0",
  "bbob-biobj_f62_i06_d05 1.0",
  "bbob-biobj_f62_i06_d10 1.0",
  "bbob-biobj_f62_i06_d20 1.0",
  "bbob-biobj_f62_i06_d40 1.0",
  "bbob-biobj_f62_i07_d02 1.0",
  "bbob-biobj_f62_i07_d03 1.0",
  "bbob-biobj_f62_i07_d05 1.0",
  "bbob-biobj_f62_i07_d10 1.0",
  "bbob-biobj_f62_i07_d20 1.0",
  "bbob-biobj_f62_i07_d40 1.0",
  "bbob-biobj_f62_i08_d02 1.0",
  "bbob-biobj_f62_i08_d03 1.0",
  "bbob-biobj_f62_i08_d05 1.0",
  "bbob-biobj_f62_i08_d10 1.0",
  "bbob-biobj_f62_i08_d20 1.0",
  "bbob-biobj_f62_i08_d40 1.0",
  "bbob-biobj_f62_i09_d02 1.0",
  "bbob-biobj_f62_i09_d03 1.0",
  "bbob-biobj_f62_i09_d05 1.0",
  "bbob-biobj_f62_i09_d10 1.0",
  "bbob-biobj_f62_i09_d20 1.0",
  "bbob-biobj_f62_i09_d40 1.0",
  "bbob-biobj_f62_i10_d02 1.0",
  "bbob-biobj_f62_i10_d03 1.0",
  "bbob-biobj_f62_i10_d05 1.0",
  "bbob-biobj_f62_i10_d10 1.0",
  "bbob-biobj_f62_i10_d20 1.0",
  "bbob-biobj_f62_i10_d40 1.0",
  "bbob-biobj_f62_i11_d02 1.0",
  "bbob-biobj_f62_i11_d03 1.0",
  "bbob-biobj_f62_i11_d05 1.0",
  "bbob-biobj_f62_i11_d10 1.0",
  "bbob-biobj_f62_i11_d20 1.0",
  "bbob-biobj_f62_i11_d40 1.0",
  "bbob-biobj_f62_i12_d02 1.0",
  "bbob-biobj_f62_i12_d03 1.0",
  "bbob-biobj_f62_i12_d05 1.0",
  "bbob-biobj_f62_i12_d10 1.0",
  "bbob-biobj_f62_i12_d20 1.0",
  "bbob-biobj_f62_i12_d40 1.0",
  "bbob-biobj_f62_i13_d02 1.0",
  "bbob-biobj_f62_i13_d03 1.0",
  "bbob-biobj_f62_i13_d05 1.0",
  "bbob-biobj_f62_i13_d10 1.0",
  "bbob-biobj_f62_i13_d20 1.0",
  "bbob-biobj_f62_i13_d40 1.0",
  "bbob-biobj_f62_i14_d02 1.0",
  "bbob-biobj_f62_i14_d03 1.0",
  "bbob-biobj_f62_i14_d05 1.0",
  "bbob-biobj_f62_i14_d10 1.0",
  "bbob-biobj_f62_i14_d20 1.0",
  "bbob-biobj_f62_i14_d40 1.0",
  "bbob-biobj_f62_i15_d02 1.0",
  "bbob-biobj_f62_i15_d03 1.0",
  "bbob-biobj_f62_i15_d05 1.0",
  "bbob-biobj_f62_i15_d10 1.0",
  "bbob-biobj_f62_i15_d20 1.0",
  "bbob-biobj_f62_i15_d40 1.0",
  "bbob-biobj_f63_i01_d02 1.0",
  "bbob-biobj_f63_i01_d03 1.0",
  "bbob-biobj_f63_i01_d05 1.0",
  "bbob-biobj_f63_i01_d10 1.0",
  "bbob-biobj_f63_i01_d20 1.0",
  "bbob-biobj_f63_i01_d40 1.0",
  "bbob-biobj_f63_i02_d02 1.0",
  "bbob-biobj_f63_i02_d03 1.0",
  "bbob-biobj_f63_i02_d05 1.0",
  "bbob-biobj_f63_i02_d10 1.0",
  "bbob-biobj_f63_i02_d20 1.0",
  "bbob-biobj_f63_i02_d40 1.0",
  "bbob-biobj_f63_i03_d02 1.0",
  "bbob-biobj_f63_i03_d03 1.0",
  "bbob-biobj_f63_i03_d05 1.0",
  "bbob-biobj_f63_i03_d10 1.0",
  "bbob-biobj_f63_i03_d20 1.0",
  "bbob-biobj_f63_i03_d40 1.0",
  "bbob-biobj_f63_i04_d02 1.0",
  "bbob-biobj_f63_i04_d03 1.0",
  "bbob-biobj_f63_i04_d05 1.0",
  "bbob-biobj_f63_i04_d10 1.0",
  "bbob-biobj_f63_i04_d20 1.0",
  "bbob-biobj_f63_i04_d40 1.0",
  "bbob-biobj_f63_i05_d02 1.0",
  "bbob-biobj_f63_i05_d03 1.0",
  "bbob-biobj_f63_i05_d05 1.0",
  "bbob-biobj_f63_i05_d10 1.0",
  "bbob-biobj_f63_i05_d20 1.0",
  "bbob-biobj_f63_i05_d40 1.0",
  "bbob-biobj_f63_i06_d02 1.0",
  "bbob-biobj_f63_i06_d03 1.0",
  "bbob-biobj_f63_i06_d05 1.0",
  "bbob-biobj_f63_i06_d10 1.0",
  "bbob-biobj_f63_i06_d20 1.0",
  "bbob-biobj_f63_i06_d40 1.0",
  "bbob-biobj_f63_i07_d02 1.0",
  "bbob-biobj_f63_i07_d03 1.0",
  "bbob-biobj_f63_i07_d05 1.0",
  "bbob-biobj_f63_i07_d10 1.0",
  "bbob-biobj_f63_i07_d20 1.0",
  "bbob-biobj_f63_i07_d40 1.0",
  "bbob-biobj_f63_i08_d02 1.0",
  "bbob-biobj_f63_i08_d03 1.0",
  "bbob-biobj_f63_i08_d05 1.0",
  "bbob-biobj_f63_i08_d10 1.0",
  "bbob-biobj_f63_i08_d20 1.0",
  "bbob-biobj_f63_i08_d40 1.0",
  "bbob-biobj_f63_i09_d02 1.0",
  "bbob-biobj_f63_i09_d03 1.0",
  "bbob-biobj_f63_i09_d05 1.0",
  "bbob-biobj_f63_i09_d10 1.0",
  "bbob-biobj_f63_i09_d20 1.0",
  "bbob-biobj_f63_i09_d40 1.0",
  "bbob-biobj_f63_i10_d02 1.0",
  "bbob-biobj_f63_i10_d03 1.0",
  "bbob-biobj_f63_i10_d05 1.0",
  "bbob-biobj_f63_i10_d10 1.0",
  "bbob-biobj_f63_i10_d20 1.0",
  "bbob-biobj_f63_i10_d40 1.0",
  "bbob-biobj_f63_i11_d02 1.0",
  "bbob-biobj_f63_i11_d03 1.0",
  "bbob-biobj_f63_i11_d05 1.0",
  "bbob-biobj_f63_i11_d10 1.0",
  "bbob-biobj_f63_i11_d20 1.0",
  "bbob-biobj_f63_i11_d40 1.0",
  "bbob-biobj_f63_i12_d02 1.0",
  "bbob-biobj_f63_i12_d03 1.0",
  "bbob-biobj_f63_i12_d05 1.0",
  "bbob-biobj_f63_i12_d10 1.0",
  "bbob-biobj_f63_i12_d20 1.0",
  "bbob-biobj_f63_i12_d40 1.0",
  "bbob-biobj_f63_i13_d02 1.0",
  "bbob-biobj_f63_i13_d03 1.0",
  "bbob-biobj_f63_i13_d05 1.0",
  "bbob-biobj_f63_i13_d10 1.0",
  "bbob-biobj_f63_i13_d20 1.0",
  "bbob-biobj_f63_i13_d40 1.0",
  "bbob-biobj_f63_i14_d02 1.0",
  "bbob-biobj_f63_i14_d03 1.0",
  "bbob-biobj_f63_i14_d05 1.0",
  "bbob-biobj_f63_i14_d10 1.0",
  "bbob-biobj_f63_i14_d20 1.0",
  "bbob-biobj_f63_i14_d40 1.0",
  "bbob-biobj_f63_i15_d02 1.0",
  "bbob-biobj_f63_i15_d03 1.0",
  "bbob-biobj_f63_i15_d05 1.0",
  "bbob-biobj_f63_i15_d10 1.0",
  "bbob-biobj_f63_i15_d20 1.0",
  "bbob-biobj_f63_i15_d40 1.0",
  "bbob-biobj_f64_i01_d02 1.0",
  "bbob-biobj_f64_i01_d03 1.0",
  "bbob-biobj_f64_i01_d05 1.0",
  "bbob-biobj_f64_i01_d10 1.0",
  "bbob-biobj_f64_i01_d20 1.0",
  "bbob-biobj_f64_i01_d40 1.0",
  "bbob-biobj_f64_i02_d02 1.0",
  "bbob-biobj_f64_i02_d03 1.0",
  "bbob-biobj_f64_i02_d05 1.0",
  "bbob-biobj_f64_i02_d10 1.0",
  "bbob-biobj_f64_i02_d20 1.0",
  "bbob-biobj_f64_i02_d40 1.0",
  "bbob-biobj_f64_i03_d02 1.0",
  "bbob-biobj_f64_i03_d03 1.0",
  "bbob-biobj_f64_i03_d05 1.0",
  "bbob-biobj_f64_i03_d10 1.0",
  "bbob-biobj_f64_i03_d20 1.0",
  "bbob-biobj_f64_i03_d40 1.0",
  "bbob-biobj_f64_i04_d02 1.0",
  "bbob-biobj_f64_i04_d03 1.0",
  "bbob-biobj_f64_i04_d05 1.0",
  "bbob-biobj_f64_i04_d10 1.0",
  "bbob-biobj_f64_i04_d20 1.0",
  "bbob-biobj_f64_i04_d40 1.0",
  "bbob-biobj_f64_i05_d02 1.0",
  "bbob-biobj_f64_i05_d03 1.0",
  "bbob-biobj_f64_i05_d05 1.0",
  "bbob-biobj_f64_i05_d10 1.0",
  "bbob-biobj_f64_i05_d20 1.0",
  "bbob-biobj_f64_i05_d40 1.0",
  "bbob-biobj_f64_i06_d02 1.0",
  "bbob-biobj_f64_i06_d03 1.0",
  "bbob-biobj_f64_i06_d05 1.0",
  "bbob-biobj_f64_i06_d10 1.0",
  "bbob-biobj_f64_i06_d20 1.0",
  "bbob-biobj_f64_i06_d40 1.0",
  "bbob-biobj_f64_i07_d02 1.0",
  "bbob-biobj_f64_i07_d03 1.0",
  "bbob-biobj_f64_i07_d05 1.0",
  "bbob-biobj_f64_i07_d10 1.0",
  "bbob-biobj_f64_i07_d20 1.0",
  "bbob-biobj_f64_i07_d40 1.0",
  "bbob-biobj_f64_i08_d02 1.0",
  "bbob-biobj_f64_i08_d03 1.0",
  "bbob-biobj_f64_i08_d05 1.0",
  "bbob-biobj_f64_i08_d10 1.0",
  "bbob-biobj_f64_i08_d20 1.0",
  "bbob-biobj_f64_i08_d40 1.0",
  "bbob-biobj_f64_i09_d02 1.0",
  "bbob-biobj_f64_i09_d03 1.0",
  "bbob-biobj_f64_i09_d05 1.0",
  "bbob-biobj_f64_i09_d10 1.0",
  "bbob-biobj_f64_i09_d20 1.0",
  "bbob-biobj_f64_i09_d40 1.0",
  "bbob-biobj_f64_i10_d02 1.0",
  "bbob-biobj_f64_i10_d03 1.0",
  "bbob-biobj_f64_i10_d05 1.0",
  "bbob-biobj_f64_i10_d10 1.0",
  "bbob-biobj_f64_i10_d20 1.0",
  "bbob-biobj_f64_i10_d40 1.0",
  "bbob-biobj_f64_i11_d02 1.0",
  "bbob-biobj_f64_i11_d03 1.0",
  "bbob-biobj_f64_i11_d05 1.0",
  "bbob-biobj_f64_i11_d10 1.0",
  "bbob-biobj_f64_i11_d20 1.0",
  "bbob-biobj_f64_i11_d40 1.0",
  "bbob-biobj_f64_i12_d02 1.0",
  "bbob-biobj_f64_i12_d03 1.0",
  "bbob-biobj_f64_i12_d05 1.0",
  "bbob-biobj_f64_i12_d10 1.0",
  "bbob-biobj_f64_i12_d20 1.0",
  "bbob-biobj_f64_i12_d40 1.0",
  "bbob-biobj_f64_i13_d02 1.0",
  "bbob-biobj_f64_i13_d03 1.0",
  "bbob-biobj_f64_i13_d05 1.0",
  "bbob-biobj_f64_i13_d10 1.0",
  "bbob-biobj_f64_i13_d20 1.0",
  "bbob-biobj_f64_i13_d40 1.0",
  "bbob-biobj_f64_i14_d02 1.0",
  "bbob-biobj_f64_i14_d03 1.0",
  "bbob-biobj_f64_i14_d05 1.0",
  "bbob-biobj_f64_i14_d10 1.0",
  "bbob-biobj_f64_i14_d20 1.0",
  "bbob-biobj_f64_i14_d40 1.0",
  "bbob-biobj_f64_i15_d02 1.0",
  "bbob-biobj_f64_i15_d03 1.0",
  "bbob-biobj_f64_i15_d05 1.0",
  "bbob-biobj_f64_i15_d10 1.0",
  "bbob-biobj_f64_i15_d20 1.0",
  "bbob-biobj_f64_i15_d40 1.0",
  "bbob-biobj_f65_i01_d02 1.0",
  "bbob-biobj_f65_i01_d03 1.0",
  "bbob-biobj_f65_i01_d05 1.0",
  "bbob-biobj_f65_i01_d10 1.0",
  "bbob-biobj_f65_i01_d20 1.0",
  "bbob-biobj_f65_i01_d40 1.0",
  "bbob-biobj_f65_i02_d02 1.0",
  "bbob-biobj_f65_i02_d03 1.0",
  "bbob-biobj_f65_i02_d05 1.0",
  "bbob-biobj_f65_i02_d10 1.0",
  "bbob-biobj_f65_i02_d20 1.0",
  "bbob-biobj_f65_i02_d40 1.0",
  "bbob-biobj_f65_i03_d02 1.0",
  "bbob-biobj_f65_i03_d03 1.0",
  "bbob-biobj_f65_i03_d05 1.0",
  "bbob-biobj_f65_i03_d10 1.0",
  "bbob-biobj_f65_i03_d20 1.0",
  "bbob-biobj_f65_i03_d40 1.0",
  "bbob-biobj_f65_i04_d02 1.0",
  "bbob-biobj_f65_i04_d03 1.0",
  "bbob-biobj_f65_i04_d05 1.0",
  "bbob-biobj_f65_i04_d10 1.0",
  "bbob-biobj_f65_i04_d20 1.0",
  "bbob-biobj_f65_i04_d40 1.0",
  "bbob-biobj_f65_i05_d02 1.0",
  "bbob-biobj_f65_i05_d03 1.0",
  "bbob-biobj_f65_i05_d05 1.0",
  "bbob-biobj_f65_i05_d10 1.0",
  "bbob-biobj_f65_i05_d20 1.0",
  "bbob-biobj_f65_i05_d40 1.0",
  "bbob-biobj_f65_i06_d02 1.0",
  "bbob-biobj_f65_i06_d03 1.0",
  "bbob-biobj_f65_i06_d05 1.0",
  "bbob-biobj_f65_i06_d10 1.0",
  "bbob-biobj_f65_i06_d20 1.0",
  "bbob-biobj_f65_i06_d40 1.0",
  "bbob-biobj_f65_i07_d02 1.0",
  "bbob-biobj_f65_i07_d03 1.0",
  "bbob-biobj_f65_i07_d05 1.0",
  "bbob-biobj_f65_i07_d10 1.0",
  "bbob-biobj_f65_i07_d20 1.0",
  "bbob-biobj_f65_i07_d40 1.0",
  "bbob-biobj_f65_i08_d02 1.0",
  "bbob-biobj_f65_i08_d03 1.0",
  "bbob-biobj_f65_i08_d05 1.0",
  "bbob-biobj_f65_i08_d10 1.0",
  "bbob-biobj_f65_i08_d20 1.0",
  "bbob-biobj_f65_i08_d40 1.0",
  "bbob-biobj_f65_i09_d02 1.0",
  "bbob-biobj_f65_i09_d03 1.0",
  "bbob-biobj_f65_i09_d05 1.0",
  "bbob-biobj_f65_i09_d10 1.0",
  "bbob-biobj_f65_i09_d20 1.0",
  "bbob-biobj_f65_i09_d40 1.0",
  "bbob-biobj_f65_i10_d02 1.0",
  "bbob-biobj_f65_i10_d03 1.0",
  "bbob-biobj_f65_i10_d05 1.0",
  "bbob-biobj_f65_i10_d10 1.0",
  "bbob-biobj_f65_i10_d20 1.0",
  "bbob-biobj_f65_i10_d40 1.0",
  "bbob-biobj_f65_i11_d02 1.0",
  "bbob-biobj_f65_i11_d03 1.0",
  "bbob-biobj_f65_i11_d05 1.0",
  "bbob-biobj_f65_i11_d10 1.0",
  "bbob-biobj_f65_i11_d20 1.0",
  "bbob-biobj_f65_i11_d40 1.0",
  "bbob-biobj_f65_i12_d02 1.0",
  "bbob-biobj_f65_i12_d03 1.0",
  "bbob-biobj_f65_i12_d05 1.0",
  "bbob-biobj_f65_i12_d10 1.0",
  "bbob-biobj_f65_i12_d20 1.0",
  "bbob-biobj_f65_i12_d40 1.0",
  "bbob-biobj_f65_i13_d02 1.0",
  "bbob-biobj_f65_i13_d03 1.0",
  "bbob-biobj_f65_i13_d05 1.0",
  "bbob-biobj_f65_i13_d10 1.0",
  "bbob-biobj_f65_i13_d20 1.0",
  "bbob-biobj_f65_i13_d40 1.0",
  "bbob-biobj_f65_i14_d02 1.0",
  "bbob-biobj_f65_i14_d03 1.0",
  "bbob-biobj_f65_i14_d05 1.0",
  "bbob-biobj_f65_i14_d10 1.0",
  "bbob-biobj_f65_i14_d20 1.0",
  "bbob-biobj_f65_i14_d40 1.0",
  "bbob-biobj_f65_i15_d02 1.0",
  "bbob-biobj_f65_i15_d03 1.0",
  "bbob-biobj_f65_i15_d05 1.0",
  "bbob-biobj_f65_i15_d10 1.0",
  "bbob-biobj_f65_i15_d20 1.0",
  "bbob-biobj_f65_i15_d40 1.0",
  "bbob-biobj_f66_i01_d02 1.0",
  "bbob-biobj_f66_i01_d03 1.0",
  "bbob-biobj_f66_i01_d05 1.0",
  "bbob-biobj_f66_i01_d10 1.0",
  "bbob-biobj_f66_i01_d20 1.0",
  "bbob-biobj_f66_i01_d40 1.0",
  "bbob-biobj_f66_i02_d02 1.0",
  "bbob-biobj_f66_i02_d03 1.0",
  "bbob-biobj_f66_i02_d05 1.0",
  "bbob-biobj_f66_i02_d10 1.0",
  "bbob-biobj_f66_i02_d20 1.0",
  "bbob-biobj_f66_i02_d40 1.0",
  "bbob-biobj_f66_i03_d02 1.0",
  "bbob-biobj_f66_i03_d03 1.0",
  "bbob-biobj_f66_i03_d05 1.0",
  "bbob-biobj_f66_i03_d10 1.0",
  "bbob-biobj_f66_i03_d20 1.0",
  "bbob-biobj_f66_i03_d40 1.0",
  "bbob-biobj_f66_i04_d02 1.0",
  "bbob-biobj_f66_i04_d03 1.0",
  "bbob-biobj_f66_i04_d05 1.0",
  "bbob-biobj_f66_i04_d10 1.0",
  "bbob-biobj_f66_i04_d20 1.0",
  "bbob-biobj_f66_i04_d40 1.0",
  "bbob-biobj_f66_i05_d02 1.0",
  "bbob-biobj_f66_i05_d03 1.0",
  "bbob-biobj_f66_i05_d05 1.0",
  "bbob-biobj_f66_i05_d10 1.0",
  "bbob-biobj_f66_i05_d20 1.0",
  "bbob-biobj_f66_i05_d40 1.0",
  "bbob-biobj_f66_i06_d02 1.0",
  "bbob-biobj_f66_i06_d03 1.0",
  "bbob-biobj_f66_i06_d05 1.0",
  "bbob-biobj_f66_i06_d10 1.0",
  "bbob-biobj_f66_i06_d20 1.0",
  "bbob-biobj_f66_i06_d40 1.0",
  "bbob-biobj_f66_i07_d02 1.0",
  "bbob-biobj_f66_i07_d03 1.0",
  "bbob-biobj_f66_i07_d05 1.0",
  "bbob-biobj_f66_i07_d10 1.0",
  "bbob-biobj_f66_i07_d20 1.0",
  "bbob-biobj_f66_i07_d40 1.0",
  "bbob-biobj_f66_i08_d02 1.0",
  "bbob-biobj_f66_i08_d03 1.0",
  "bbob-biobj_f66_i08_d05 1.0",
  "bbob-biobj_f66_i08_d10 1.0",
  "bbob-biobj_f66_i08_d20 1.0",
  "bbob-biobj_f66_i08_d40 1.0",
  "bbob-biobj_f66_i09_d02 1.0",
  "bbob-biobj_f66_i09_d03 1.0",
  "bbob-biobj_f66_i09_d05 1.0",
  "bbob-biobj_f66_i09_d10 1.0",
  "bbob-biobj_f66_i09_d20 1.0",
  "bbob-biobj_f66_i09_d40 1.0",
  "bbob-biobj_f66_i10_d02 1.0",
  "bbob-biobj_f66_i10_d03 1.0",
  "bbob-biobj_f66_i10_d05 1.0",
  "bbob-biobj_f66_i10_d10 1.0",
  "bbob-biobj_f66_i10_d20 1.0",
  "bbob-biobj_f66_i10_d40 1.0",
  "bbob-biobj_f66_i11_d02 1.0",
  "bbob-biobj_f66_i11_d03 1.0",
  "bbob-biobj_f66_i11_d05 1.0",
  "bbob-biobj_f66_i11_d10 1.0",
  "bbob-biobj_f66_i11_d20 1.0",
  "bbob-biobj_f66_i11_d40 1.0",
  "bbob-biobj_f66_i12_d02 1.0",
  "bbob-biobj_f66_i12_d03 1.0",
  "bbob-biobj_f66_i12_d05 1.0",
  "bbob-biobj_f66_i12_d10 1.0",
  "bbob-biobj_f66_i12_d20 1.0",
  "bbob-biobj_f66_i12_d40 1.0",
  "bbob-biobj_f66_i13_d02 1.0",
  "bbob-biobj_f66_i13_d03 1.0",
  "bbob-biobj_f66_i13_d05 1.0",
  "bbob-biobj_f66_i13_d10 1.0",
  "bbob-biobj_f66_i13_d20 1.0",
  "bbob-biobj_f66_i13_d40 1.0",
  "bbob-biobj_f66_i14_d02 1.0",
  "bbob-biobj_f66_i14_d03 1.0",
  "bbob-biobj_f66_i14_d05 1.0",
  "bbob-biobj_f66_i14_d10 1.0",
  "bbob-biobj_f66_i14_d20 1.0",
  "bbob-biobj_f66_i14_d40 1.0",
  "bbob-biobj_f66_i15_d02 1.0",
  "bbob-biobj_f66_i15_d03 1.0",
  "bbob-biobj_f66_i15_d05 1.0",
  "bbob-biobj_f66_i15_d10 1.0",
  "bbob-biobj_f66_i15_d20 1.0",
  "bbob-biobj_f66_i15_d40 1.0",
  "bbob-biobj_f67_i01_d02 1.0",
  "bbob-biobj_f67_i01_d03 1.0",
  "bbob-biobj_f67_i01_d05 1.0",
  "bbob-biobj_f67_i01_d10 1.0",
  "bbob-biobj_f67_i01_d20 1.0",
  "bbob-biobj_f67_i01_d40 1.0",
  "bbob-biobj_f67_i02_d02 1.0",
  "bbob-biobj_f67_i02_d03 1.0",
  "bbob-biobj_f67_i02_d05 1.0",
  "bbob-biobj_f67_i02_d10 1.0",
  "bbob-biobj_f67_i02_d20 1.0",
  "bbob-biobj_f67_i02_d40 1.0",
  "bbob-biobj_f67_i03_d02 1.0",
  "bbob-biobj_f67_i03_d03 1.0",
  "bbob-biobj_f67_i03_d05 1.0",
  "bbob-biobj_f67_i03_d10 1.0",
  "bbob-biobj_f67_i03_d20 1.0",
  "bbob-biobj_f67_i03_d40 1.0",
  "bbob-biobj_f67_i04_d02 1.0",
  "bbob-biobj_f67_i04_d03 1.0",
  "bbob-biobj_f67_i04_d05 1.0",
  "bbob-biobj_f67_i04_d10 1.0",
  "bbob-biobj_f67_i04_d20 1.0",
  "bbob-biobj_f67_i04_d40 1.0",
  "bbob-biobj_f67_i05_d02 1.0",
  "bbob-biobj_f67_i05_d03 1.0",
  "bbob-biobj_f67_i05_d05 1.0",
  "bbob-biobj_f67_i05_d10 1.0",
  "bbob-biobj_f67_i05_d20 1.0",
  "bbob-biobj_f67_i05_d40 1.0",
  "bbob-biobj_f67_i06_d02 1.0",
  "bbob-biobj_f67_i06_d03 1.0",
  "bbob-biobj_f67_i06_d05 1.0",
  "bbob-biobj_f67_i06_d10 1.0",
  "bbob-biobj_f67_i06_d20 1.0",
  "bbob-biobj_f67_i06_d40 1.0",
  "bbob-biobj_f67_i07_d02 1.0",
  "bbob-biobj_f67_i07_d03 1.0",
  "bbob-biobj_f67_i07_d05 1.0",
  "bbob-biobj_f67_i07_d10 1.0",
  "bbob-biobj_f67_i07_d20 1.0",
  "bbob-biobj_f67_i07_d40 1.0",
  "bbob-biobj_f67_i08_d02 1.0",
  "bbob-biobj_f67_i08_d03 1.0",
  "bbob-biobj_f67_i08_d05 1.0",
  "bbob-biobj_f67_i08_d10 1.0",
  "bbob-biobj_f67_i08_d20 1.0",
  "bbob-biobj_f67_i08_d40 1.0",
  "bbob-biobj_f67_i09_d02 1.0",
  "bbob-biobj_f67_i09_d03 1.0",
  "bbob-biobj_f67_i09_d05 1.0",
  "bbob-biobj_f67_i09_d10 1.0",
  "bbob-biobj_f67_i09_d20 1.0",
  "bbob-biobj_f67_i09_d40 1.0",
  "bbob-biobj_f67_i10_d02 1.0",
  "bbob-biobj_f67_i10_d03 1.0",
  "bbob-biobj_f67_i10_d05 1.0",
  "bbob-biobj_f67_i10_d10 1.0",
  "bbob-biobj_f67_i10_d20 1.0",
  "bbob-biobj_f67_i10_d40 1.0",
  "bbob-biobj_f67_i11_d02 1.0",
  "bbob-biobj_f67_i11_d03 1.0",
  "bbob-biobj_f67_i11_d05 1.0",
  "bbob-biobj_f67_i11_d10 1.0",
  "bbob-biobj_f67_i11_d20 1.0",
  "bbob-biobj_f67_i11_d40 1.0",
  "bbob-biobj_f67_i12_d02 1.0",
  "bbob-biobj_f67_i12_d03 1.0",
  "bbob-biobj_f67_i12_d05 1.0",
  "bbob-biobj_f67_i12_d10 1.0",
  "bbob-biobj_f67_i12_d20 1.0",
  "bbob-biobj_f67_i12_d40 1.0",
  "bbob-biobj_f67_i13_d02 1.0",
  "bbob-biobj_f67_i13_d03 1.0",
  "bbob-biobj_f67_i13_d05 1.0",
  "bbob-biobj_f67_i13_d10 1.0",
  "bbob-biobj_f67_i13_d20 1.0",
  "bbob-biobj_f67_i13_d40 1.0",
  "bbob-biobj_f67_i14_d02 1.0",
  "bbob-biobj_f67_i14_d03 1.0",
  "bbob-biobj_f67_i14_d05 1.0",
  "bbob-biobj_f67_i14_d10 1.0",
  "bbob-biobj_f67_i14_d20 1.0",
  "bbob-biobj_f67_i14_d40 1.0",
  "bbob-biobj_f67_i15_d02 1.0",
  "bbob-biobj_f67_i15_d03 1.0",
  "bbob-biobj_f67_i15_d05 1.0",
  "bbob-biobj_f67_i15_d10 1.0",
  "bbob-biobj_f67_i15_d20 1.0",
  "bbob-biobj_f67_i15_d40 1.0",
  "bbob-biobj_f68_i01_d02 1.0",
  "bbob-biobj_f68_i01_d03 1.0",
  "bbob-biobj_f68_i01_d05 1.0",
  "bbob-biobj_f68_i01_d10 1.0",
  "bbob-biobj_f68_i01_d20 1.0",
  "bbob-biobj_f68_i01_d40 1.0",
  "bbob-biobj_f68_i02_d02 1.0",
  "bbob-biobj_f68_i02_d03 1.0",
  "bbob-biobj_f68_i02_d05 1.0",
  "bbob-biobj_f68_i02_d10 1.0",
  "bbob-biobj_f68_i02_d20 1.0",
  "bbob-biobj_f68_i02_d40 1.0",
  "bbob-biobj_f68_i03_d02 1.0",
  "bbob-biobj_f68_i03_d03 1.0",
  "bbob-biobj_f68_i03_d05 1.0",
  "bbob-biobj_f68_i03_d10 1.0",
  "bbob-biobj_f68_i03_d20 1.0",
  "bbob-biobj_f68_i03_d40 1.0",
  "bbob-biobj_f68_i04_d02 1.0",
  "bbob-biobj_f68_i04_d03 1.0",
  "bbob-biobj_f68_i04_d05 1.0",
  "bbob-biobj_f68_i04_d10 1.0",
  "bbob-biobj_f68_i04_d20 1.0",
  "bbob-biobj_f68_i04_d40 1.0",
  "bbob-biobj_f68_i05_d02 1.0",
  "bbob-biobj_f68_i05_d03 1.0",
  "bbob-biobj_f68_i05_d05 1.0",
  "bbob-biobj_f68_i05_d10 1.0",
  "bbob-biobj_f68_i05_d20 1.0",
  "bbob-biobj_f68_i05_d40 1.0",
  "bbob-biobj_f68_i06_d02 1.0",
  "bbob-biobj_f68_i06_d03 1.0",
  "bbob-biobj_f68_i06_d05 1.0",
  "bbob-biobj_f68_i06_d10 1.0",
  "bbob-biobj_f68_i06_d20 1.0",
  "bbob-biobj_f68_i06_d40 1.0",
  "bbob-biobj_f68_i07_d02 1.0",
  "bbob-biobj_f68_i07_d03 1.0",
  "bbob-biobj_f68_i07_d05 1.0",
  "bbob-biobj_f68_i07_d10 1.0",
  "bbob-biobj_f68_i07_d20 1.0",
  "bbob-biobj_f68_i07_d40 1.0",
  "bbob-biobj_f68_i08_d02 1.0",
  "bbob-biobj_f68_i08_d03 1.0",
  "bbob-biobj_f68_i08_d05 1.0",
  "bbob-biobj_f68_i08_d10 1.0",
  "bbob-biobj_f68_i08_d20 1.0",
  "bbob-biobj_f68_i08_d40 1.0",
  "bbob-biobj_f68_i09_d02 1.0",
  "bbob-biobj_f68_i09_d03 1.0",
  "bbob-biobj_f68_i09_d05 1.0",
  "bbob-biobj_f68_i09_d10 1.0",
  "bbob-biobj_f68_i09_d20 1.0",
  "bbob-biobj_f68_i09_d40 1.0",
  "bbob-biobj_f68_i10_d02 1.0",
  "bbob-biobj_f68_i10_d03 1.0",
  "bbob-biobj_f68_i10_d05 1.0",
  "bbob-biobj_f68_i10_d10 1.0",
  "bbob-biobj_f68_i10_d20 1.0",
  "bbob-biobj_f68_i10_d40 1.0",
  "bbob-biobj_f68_i11_d02 1.0",
  "bbob-biobj_f68_i11_d03 1.0",
  "bbob-biobj_f68_i11_d05 1.0",
  "bbob-biobj_f68_i11_d10 1.0",
  "bbob-biobj_f68_i11_d20 1.0",
  "bbob-biobj_f68_i11_d40 1.0",
  "bbob-biobj_f68_i12_d02 1.0",
  "bbob-biobj_f68_i12_d03 1.0",
  "bbob-biobj_f68_i12_d05 1.0",
  "bbob-biobj_f68_i12_d10 1.0",
  "bbob-biobj_f68_i12_d20 1.0",
  "bbob-biobj_f68_i12_d40 1.0",
  "bbob-biobj_f68_i13_d02 1.0",
  "bbob-biobj_f68_i13_d03 1.0",
  "bbob-biobj_f68_i13_d05 1.0",
  "bbob-biobj_f68_i13_d10 1.0",
  "bbob-biobj_f68_i13_d20 1.0",
  "bbob-biobj_f68_i13_d40 1.0",
  "bbob-biobj_f68_i14_d02 1.0",
  "bbob-biobj_f68_i14_d03 1.0",
  "bbob-biobj_f68_i14_d05 1.0",
  "bbob-biobj_f68_i14_d10 1.0",
  "bbob-biobj_f68_i14_d20 1.0",
  "bbob-biobj_f68_i14_d40 1.0",
  "bbob-biobj_f68_i15_d02 1.0",
  "bbob-biobj_f68_i15_d03 1.0",
  "bbob-biobj_f68_i15_d05 1.0",
  "bbob-biobj_f68_i15_d10 1.0",
  "bbob-biobj_f68_i15_d20 1.0",
  "bbob-biobj_f68_i15_d40 1.0",
  "bbob-biobj_f69_i01_d02 1.0",
  "bbob-biobj_f69_i01_d03 1.0",
  "bbob-biobj_f69_i01_d05 1.0",
  "bbob-biobj_f69_i01_d10 1.0",
  "bbob-biobj_f69_i01_d20 1.0",
  "bbob-biobj_f69_i01_d40 1.0",
  "bbob-biobj_f69_i02_d02 1.0",
  "bbob-biobj_f69_i02_d03 1.0",
  "bbob-biobj_f69_i02_d05 1.0",
  "bbob-biobj_f69_i02_d10 1.0",
  "bbob-biobj_f69_i02_d20 1.0",
  "bbob-biobj_f69_i02_d40 1.0",
  "bbob-biobj_f69_i03_d02 1.0",
  "bbob-biobj_f69_i03_d03 1.0",
  "bbob-biobj_f69_i03_d05 1.0",
  "bbob-biobj_f69_i03_d10 1.0",
  "bbob-biobj_f69_i03_d20 1.0",
  "bbob-biobj_f69_i03_d40 1.0",
  "bbob-biobj_f69_i04_d02 1.0",
  "bbob-biobj_f69_i04_d03 1.0",
  "bbob-biobj_f69_i04_d05 1.0",
  "bbob-biobj_f69_i04_d10 1.0",
  "bbob-biobj_f69_i04_d20 1.0",
  "bbob-biobj_f69_i04_d40 1.0",
  "bbob-biobj_f69_i05_d02 1.0",
  "bbob-biobj_f69_i05_d03 1.0",
  "bbob-biobj_f69_i05_d05 1.0",
  "bbob-biobj_f69_i05_d10 1.0",
  "bbob-biobj_f69_i05_d20 1.0",
  "bbob-biobj_f69_i05_d40 1.0",
  "bbob-biobj_f69_i06_d02 1.0",
  "bbob-biobj_f69_i06_d03 1.0",
  "bbob-biobj_f69_i06_d05 1.0",
  "bbob-biobj_f69_i06_d10 1.0",
  "bbob-biobj_f69_i06_d20 1.0",
  "bbob-biobj_f69_i06_d40 1.0",
  "bbob-biobj_f69_i07_d02 1.0",
  "bbob-biobj_f69_i07_d03 1.0",
  "bbob-biobj_f69_i07_d05 1.0",
  "bbob-biobj_f69_i07_d10 1.0",
  "bbob-biobj_f69_i07_d20 1.0",
  "bbob-biobj_f69_i07_d40 1.0",
  "bbob-biobj_f69_i08_d02 1.0",
  "bbob-biobj_f69_i08_d03 1.0",
  "bbob-biobj_f69_i08_d05 1.0",
  "bbob-biobj_f69_i08_d10 1.0",
  "bbob-biobj_f69_i08_d20 1.0",
  "bbob-biobj_f69_i08_d40 1.0",
  "bbob-biobj_f69_i09_d02 1.0",
  "bbob-biobj_f69_i09_d03 1.0",
  "bbob-biobj_f69_i09_d05 1.0",
  "bbob-biobj_f69_i09_d10 1.0",
  "bbob-biobj_f69_i09_d20 1.0",
  "bbob-biobj_f69_i09_d40 1.0",
  "bbob-biobj_f69_i10_d02 1.0",
  "bbob-biobj_f69_i10_d03 1.0",
  "bbob-biobj_f69_i10_d05 1.0",
  "bbob-biobj_f69_i10_d10 1.0",
  "bbob-biobj_f69_i10_d20 1.0",
  "bbob-biobj_f69_i10_d40 1.0",
  "bbob-biobj_f69_i11_d02 1.0",
  "bbob-biobj_f69_i11_d03 1.0",
  "bbob-biobj_f69_i11_d05 1.0",
  "bbob-biobj_f69_i11_d10 1.0",
  "bbob-biobj_f69_i11_d20 1.0",
  "bbob-biobj_f69_i11_d40 1.0",
  "bbob-biobj_f69_i12_d02 1.0",
  "bbob-biobj_f69_i12_d03 1.0",
  "bbob-biobj_f69_i12_d05 1.0",
  "bbob-biobj_f69_i12_d10 1.0",
  "bbob-biobj_f69_i12_d20 1.0",
  "bbob-biobj_f69_i12_d40 1.0",
  "bbob-biobj_f69_i13_d02 1.0",
  "bbob-biobj_f69_i13_d03 1.0",
  "bbob-biobj_f69_i13_d05 1.0",
  "bbob-biobj_f69_i13_d10 1.0",
  "bbob-biobj_f69_i13_d20 1.0",
  "bbob-biobj_f69_i13_d40 1.0",
  "bbob-biobj_f69_i14_d02 1.0",
  "bbob-biobj_f69_i14_d03 1.0",
  "bbob-biobj_f69_i14_d05 1.0",
  "bbob-biobj_f69_i14_d10 1.0",
  "bbob-biobj_f69_i14_d20 1.0",
  "bbob-biobj_f69_i14_d40 1.0",
  "bbob-biobj_f69_i15_d02 1.0",
  "bbob-biobj_f69_i15_d03 1.0",
  "bbob-biobj_f69_i15_d05 1.0",
  "bbob-biobj_f69_i15_d10 1.0",
  "bbob-biobj_f69_i15_d20 1.0",
  "bbob-biobj_f69_i15_d40 1.0",
  "bbob-biobj_f70_i01_d02 1.0",
  "bbob-biobj_f70_i01_d03 1.0",
  "bbob-biobj_f70_i01_d05 1.0",
  "bbob-biobj_f70_i01_d10 1.0",
  "bbob-biobj_f70_i01_d20 1.0",
  "bbob-biobj_f70_i01_d40 1.0",
  "bbob-biobj_f70_i02_d02 1.0",
  "bbob-biobj_f70_i02_d03 1.0",
  "bbob-biobj_f70_i02_d05 1.0",
  "bbob-biobj_f70_i02_d10 1.0",
  "bbob-biobj_f70_i02_d20 1.0",
  "bbob-biobj_f70_i02_d40 1.0",
  "bbob-biobj_f70_i03_d02 1.0",
  "bbob-biobj_f70_i03_d03 1.0",
  "bbob-biobj_f70_i03_d05 1.0",
  "bbob-biobj_f70_i03_d10 1.0",
  "bbob-biobj_f70_i03_d20 1.0",
  "bbob-biobj_f70_i03_d40 1.0",
  "bbob-biobj_f70_i04_d02 1.0",
  "bbob-biobj_f70_i04_d03 1.0",
  "bbob-biobj_f70_i04_d05 1.0",
  "bbob-biobj_f70_i04_d10 1.0",
  "bbob-biobj_f70_i04_d20 1.0",
  "bbob-biobj_f70_i04_d40 1.0",
  "bbob-biobj_f70_i05_d02 1.0",
  "bbob-biobj_f70_i05_d03 1.0",
  "bbob-biobj_f70_i05_d05 1.0",
  "bbob-biobj_f70_i05_d10 1.0",
  "bbob-biobj_f70_i05_d20 1.0",
  "bbob-biobj_f70_i05_d40 1.0",
  "bbob-biobj_f70_i06_d02 1.0",
  "bbob-biobj_f70_i06_d03 1.0",
  "bbob-biobj_f70_i06_d05 1.0",
  "bbob-biobj_f70_i06_d10 1.0",
  "bbob-biobj_f70_i06_d20 1.0",
  "bbob-biobj_f70_i06_d40 1.0",
  "bbob-biobj_f70_i07_d02 1.0",
  "bbob-biobj_f70_i07_d03 1.0",
  "bbob-biobj_f70_i07_d05 1.0",
  "bbob-biobj_f70_i07_d10 1.0",
  "bbob-biobj_f70_i07_d20 1.0",
  "bbob-biobj_f70_i07_d40 1.0",
  "bbob-biobj_f70_i08_d02 1.0",
  "bbob-biobj_f70_i08_d03 1.0",
  "bbob-biobj_f70_i08_d05 1.0",
  "bbob-biobj_f70_i08_d10 1.0",
  "bbob-biobj_f70_i08_d20 1.0",
  "bbob-biobj_f70_i08_d40 1.0",
  "bbob-biobj_f70_i09_d02 1.0",
  "bbob-biobj_f70_i09_d03 1.0",
  "bbob-biobj_f70_i09_d05 1.0",
  "bbob-biobj_f70_i09_d10 1.0",
  "bbob-biobj_f70_i09_d20 1.0",
  "bbob-biobj_f70_i09_d40 1.0",
  "bbob-biobj_f70_i10_d02 1.0",
  "bbob-biobj_f70_i10_d03 1.0",
  "bbob-biobj_f70_i10_d05 1.0",
  "bbob-biobj_f70_i10_d10 1.0",
  "bbob-biobj_f70_i10_d20 1.0",
  "bbob-biobj_f70_i10_d40 1.0",
  "bbob-biobj_f70_i11_d02 1.0",
  "bbob-biobj_f70_i11_d03 1.0",
  "bbob-biobj_f70_i11_d05 1.0",
  "bbob-biobj_f70_i11_d10 1.0",
  "bbob-biobj_f70_i11_d20 1.0",
  "bbob-biobj_f70_i11_d40 1.0",
  "bbob-biobj_f70_i12_d02 1.0",
  "bbob-biobj_f70_i12_d03 1.0",
  "bbob-biobj_f70_i12_d05 1.0",
  "bbob-biobj_f70_i12_d10 1.0",
  "bbob-biobj_f70_i12_d20 1.0",
  "bbob-biobj_f70_i12_d40 1.0",
  "bbob-biobj_f70_i13_d02 1.0",
  "bbob-biobj_f70_i13_d03 1.0",
  "bbob-biobj_f70_i13_d05 1.0",
  "bbob-biobj_f70_i13_d10 1.0",
  "bbob-biobj_f70_i13_d20 1.0",
  "bbob-biobj_f70_i13_d40 1.0",
  "bbob-biobj_f70_i14_d02 1.0",
  "bbob-biobj_f70_i14_d03 1.0",
  "bbob-biobj_f70_i14_d05 1.0",
  "bbob-biobj_f70_i14_d10 1.0",
  "bbob-biobj_f70_i14_d20 1.0",
  "bbob-biobj_f70_i14_d40 1.0",
  "bbob-biobj_f70_i15_d02 1.0",
  "bbob-biobj_f70_i15_d03 1.0",
  "bbob-biobj_f70_i15_d05 1.0",
  "bbob-biobj_f70_i15_d10 1.0",
  "bbob-biobj_f70_i15_d20 1.0",
  "bbob-biobj_f70_i15_d40 1.0",
  "bbob-biobj_f71_i01_d02 1.0",
  "bbob-biobj_f71_i01_d03 1.0",
  "bbob-biobj_f71_i01_d05 1.0",
  "bbob-biobj_f71_i01_d10 1.0",
  "bbob-biobj_f71_i01_d20 1.0",
  "bbob-biobj_f71_i01_d40 1.0",
  "bbob-biobj_f71_i02_d02 1.0",
  "bbob-biobj_f71_i02_d03 1.0",
  "bbob-biobj_f71_i02_d05 1.0",
  "bbob-biobj_f71_i02_d10 1.0",
  "bbob-biobj_f71_i02_d20 1.0",
  "bbob-biobj_f71_i02_d40 1.0",
  "bbob-biobj_f71_i03_d02 1.0",
  "bbob-biobj_f71_i03_d03 1.0",
  "bbob-biobj_f71_i03_d05 1.0",
  "bbob-biobj_f71_i03_d10 1.0",
  "bbob-biobj_f71_i03_d20 1.0",
  "bbob-biobj_f71_i03_d40 1.0",
  "bbob-biobj_f71_i04_d02 1.0",
  "bbob-biobj_f71_i04_d03 1.0",
  "bbob-biobj_f71_i04_d05 1.0",
  "bbob-biobj_f71_i04_d10 1.0",
  "bbob-biobj_f71_i04_d20 1.0",
  "bbob-biobj_f71_i04_d40 1.0",
  "bbob-biobj_f71_i05_d02 1.0",
  "bbob-biobj_f71_i05_d03 1.0",
  "bbob-biobj_f71_i05_d05 1.0",
  "bbob-biobj_f71_i05_d10 1.0",
  "bbob-biobj_f71_i05_d20 1.0",
  "bbob-biobj_f71_i05_d40 1.0",
  "bbob-biobj_f71_i06_d02 1.0",
  "bbob-biobj_f71_i06_d03 1.0",
  "bbob-biobj_f71_i06_d05 1.0",
  "bbob-biobj_f71_i06_d10 1.0",
  "bbob-biobj_f71_i06_d20 1.0",
  "bbob-biobj_f71_i06_d40 1.0",
  "bbob-biobj_f71_i07_d02 1.0",
  "bbob-biobj_f71_i07_d03 1.0",
  "bbob-biobj_f71_i07_d05 1.0",
  "bbob-biobj_f71_i07_d10 1.0",
  "bbob-biobj_f71_i07_d20 1.0",
  "bbob-biobj_f71_i07_d40 1.0",
  "bbob-biobj_f71_i08_d02 1.0",
  "bbob-biobj_f71_i08_d03 1.0",
  "bbob-biobj_f71_i08_d05 1.0",
  "bbob-biobj_f71_i08_d10 1.0",
  "bbob-biobj_f71_i08_d20 1.0",
  "bbob-biobj_f71_i08_d40 1.0",
  "bbob-biobj_f71_i09_d02 1.0",
  "bbob-biobj_f71_i09_d03 1.0",
  "bbob-biobj_f71_i09_d05 1.0",
  "bbob-biobj_f71_i09_d10 1.0",
  "bbob-biobj_f71_i09_d20 1.0",
  "bbob-biobj_f71_i09_d40 1.0",
  "bbob-biobj_f71_i10_d02 1.0",
  "bbob-biobj_f71_i10_d03 1.0",
  "bbob-biobj_f71_i10_d05 1.0",
  "bbob-biobj_f71_i10_d10 1.0",
  "bbob-biobj_f71_i10_d20 1.0",
  "bbob-biobj_f71_i10_d40 1.0",
  "bbob-biobj_f71_i11_d02 1.0",
  "bbob-biobj_f71_i11_d03 1.0",
  "bbob-biobj_f71_i11_d05 1.0",
  "bbob-biobj_f71_i11_d10 1.0",
  "bbob-biobj_f71_i11_d20 1.0",
  "bbob-biobj_f71_i11_d40 1.0",
  "bbob-biobj_f71_i12_d02 1.0",
  "bbob-biobj_f71_i12_d03 1.0",
  "bbob-biobj_f71_i12_d05 1.0",
  "bbob-biobj_f71_i12_d10 1.0",
  "bbob-biobj_f71_i12_d20 1.0",
  "bbob-biobj_f71_i12_d40 1.0",
  "bbob-biobj_f71_i13_d02 1.0",
  "bbob-biobj_f71_i13_d03 1.0",
  "bbob-biobj_f71_i13_d05 1.0",
  "bbob-biobj_f71_i13_d10 1.0",
  "bbob-biobj_f71_i13_d20 1.0",
  "bbob-biobj_f71_i13_d40 1.0",
  "bbob-biobj_f71_i14_d02 1.0",
  "bbob-biobj_f71_i14_d03 1.0",
  "bbob-biobj_f71_i14_d05 1.0",
  "bbob-biobj_f71_i14_d10 1.0",
  "bbob-biobj_f71_i14_d20 1.0",
  "bbob-biobj_f71_i14_d40 1.0",
  "bbob-biobj_f71_i15_d02 1.0",
  "bbob-biobj_f71_i15_d03 1.0",
  "bbob-biobj_f71_i15_d05 1.0",
  "bbob-biobj_f71_i15_d10 1.0",
  "bbob-biobj_f71_i15_d20 1.0",
  "bbob-biobj_f71_i15_d40 1.0",
  "bbob-biobj_f72_i01_d02 1.0",
  "bbob-biobj_f72_i01_d03 1.0",
  "bbob-biobj_f72_i01_d05 1.0",
  "bbob-biobj_f72_i01_d10 1.0",
  "bbob-biobj_f72_i01_d20 1.0",
  "bbob-biobj_f72_i01_d40 1.0",
  "bbob-biobj_f72_i02_d02 1.0",
  "bbob-biobj_f72_i02_d03 1.0",
  "bbob-biobj_f72_i02_d05 1.0",
  "bbob-biobj_f72_i02_d10 1.0",
  "bbob-biobj_f72_i02_d20 1.0",
  "bbob-biobj_f72_i02_d40 1.0",
  "bbob-biobj_f72_i03_d02 1.0",
  "bbob-biobj_f72_i03_d03 1.0",
  "bbob-biobj_f72_i03_d05 1.0",
  "bbob-biobj_f72_i03_d10 1.0",
  "bbob-biobj_f72_i03_d20 1.0",
  "bbob-biobj_f72_i03_d40 1.0",
  "bbob-biobj_f72_i04_d02 1.0",
  "bbob-biobj_f72_i04_d03 1.0",
  "bbob-biobj_f72_i04_d05 1.0",
  "bbob-biobj_f72_i04_d10 1.0",
  "bbob-biobj_f72_i04_d20 1.0",
  "bbob-biobj_f72_i04_d40 1.0",
  "bbob-biobj_f72_i05_d02 1.0",
  "bbob-biobj_f72_i05_d03 1.0",
  "bbob-biobj_f72_i05_d05 1.0",
  "bbob-biobj_f72_i05_d10 1.0",
  "bbob-biobj_f72_i05_d20 1.0",
  "bbob-biobj_f72_i05_d40 1.0",
  "bbob-biobj_f72_i06_d02 1.0",
  "bbob-biobj_f72_i06_d03 1.0",
  "bbob-biobj_f72_i06_d05 1.0",
  "bbob-biobj_f72_i06_d10 1.0",
  "bbob-biobj_f72_i06_d20 1.0",
  "bbob-biobj_f72_i06_d40 1.0",
  "bbob-biobj_f72_i07_d02 1.0",
  "bbob-biobj_f72_i07_d03 1.0",
  "bbob-biobj_f72_i07_d05 1.0",
  "bbob-biobj_f72_i07_d10 1.0",
  "bbob-biobj_f72_i07_d20 1.0",
  "bbob-biobj_f72_i07_d40 1.0",
  "bbob-biobj_f72_i08_d02 1.0",
  "bbob-biobj_f72_i08_d03 1.0",
  "bbob-biobj_f72_i08_d05 1.0",
  "bbob-biobj_f72_i08_d10 1.0",
  "bbob-biobj_f72_i08_d20 1.0",
  "bbob-biobj_f72_i08_d40 1.0",
  "bbob-biobj_f72_i09_d02 1.0",
  "bbob-biobj_f72_i09_d03 1.0",
  "bbob-biobj_f72_i09_d05 1.0",
  "bbob-biobj_f72_i09_d10 1.0",
  "bbob-biobj_f72_i09_d20 1.0",
  "bbob-biobj_f72_i09_d40 1.0",
  "bbob-biobj_f72_i10_d02 1.0",
  "bbob-biobj_f72_i10_d03 1.0",
  "bbob-biobj_f72_i10_d05 1.0",
  "bbob-biobj_f72_i10_d10 1.0",
  "bbob-biobj_f72_i10_d20 1.0",
  "bbob-biobj_f72_i10_d40 1.0",
  "bbob-biobj_f72_i11_d02 1.0",
  "bbob-biobj_f72_i11_d03 1.0",
  "bbob-biobj_f72_i11_d05 1.0",
  "bbob-biobj_f72_i11_d10 1.0",
  "bbob-biobj_f72_i11_d20 1.0",
  "bbob-biobj_f72_i11_d40 1.0",
  "bbob-biobj_f72_i12_d02 1.0",
  "bbob-biobj_f72_i12_d03 1.0",
  "bbob-biobj_f72_i12_d05 1.0",
  "bbob-biobj_f72_i12_d10 1.0",
  "bbob-biobj_f72_i12_d20 1.0",
  "bbob-biobj_f72_i12_d40 1.0",
  "bbob-biobj_f72_i13_d02 1.0",
  "bbob-biobj_f72_i13_d03 1.0",
  "bbob-biobj_f72_i13_d05 1.0",
  "bbob-biobj_f72_i13_d10 1.0",
  "bbob-biobj_f72_i13_d20 1.0",
  "bbob-biobj_f72_i13_d40 1.0",
  "bbob-biobj_f72_i14_d02 1.0",
  "bbob-biobj_f72_i14_d03 1.0",
  "bbob-biobj_f72_i14_d05 1.0",
  "bbob-biobj_f72_i14_d10 1.0",
  "bbob-biobj_f72_i14_d20 1.0",
  "bbob-biobj_f72_i14_d40 1.0",
  "bbob-biobj_f72_i15_d02 1.0",
  "bbob-biobj_f72_i15_d03 1.0",
  "bbob-biobj_f72_i15_d05 1.0",
  "bbob-biobj_f72_i15_d10 1.0",
  "bbob-biobj_f72_i15_d20 1.0",
  "bbob-biobj_f72_i15_d40 1.0",
  "bbob-biobj_f73_i01_d02 1.0",
  "bbob-biobj_f73_i01_d03 1.0",
  "bbob-biobj_f73_i01_d05 1.0",
  "bbob-biobj_f73_i01_d10 1.0",
  "bbob-biobj_f73_i01_d20 1.0",
  "bbob-biobj_f73_i01_d40 1.0",
  "bbob-biobj_f73_i02_d02 1.0",
  "bbob-biobj_f73_i02_d03 1.0",
  "bbob-biobj_f73_i02_d05 1.0",
  "bbob-biobj_f73_i02_d10 1.0",
  "bbob-biobj_f73_i02_d20 1.0",
  "bbob-biobj_f73_i02_d40 1.0",
  "bbob-biobj_f73_i03_d02 1.0",
  "bbob-biobj_f73_i03_d03 1.0",
  "bbob-biobj_f73_i03_d05 1.0",
  "bbob-biobj_f73_i03_d10 1.0",
  "bbob-biobj_f73_i03_d20 1.0",
  "bbob-biobj_f73_i03_d40 1.0",
  "bbob-biobj_f73_i04_d02 1.0",
  "bbob-biobj_f73_i04_d03 1.0",
  "bbob-biobj_f73_i04_d05 1.0",
  "bbob-biobj_f73_i04_d10 1.0",
  "bbob-biobj_f73_i04_d20 1.0",
  "bbob-biobj_f73_i04_d40 1.0",
  "bbob-biobj_f73_i05_d02 1.0",
  "bbob-biobj_f73_i05_d03 1.0",
  "bbob-biobj_f73_i05_d05 1.0",
  "bbob-biobj_f73_i05_d10 1.0",
  "bbob-biobj_f73_i05_d20 1.0",
  "bbob-biobj_f73_i05_d40 1.0",
  "bbob-biobj_f73_i06_d02 1.0",
  "bbob-biobj_f73_i06_d03 1.0",
  "bbob-biobj_f73_i06_d05 1.0",
  "bbob-biobj_f73_i06_d10 1.0",
  "bbob-biobj_f73_i06_d20 1.0",
  "bbob-biobj_f73_i06_d40 1.0",
  "bbob-biobj_f73_i07_d02 1.0",
  "bbob-biobj_f73_i07_d03 1.0",
  "bbob-biobj_f73_i07_d05 1.0",
  "bbob-biobj_f73_i07_d10 1.0",
  "bbob-biobj_f73_i07_d20 1.0",
  "bbob-biobj_f73_i07_d40 1.0",
  "bbob-biobj_f73_i08_d02 1.0",
  "bbob-biobj_f73_i08_d03 1.0",
  "bbob-biobj_f73_i08_d05 1.0",
  "bbob-biobj_f73_i08_d10 1.0",
  "bbob-biobj_f73_i08_d20 1.0",
  "bbob-biobj_f73_i08_d40 1.0",
  "bbob-biobj_f73_i09_d02 1.0",
  "bbob-biobj_f73_i09_d03 1.0",
  "bbob-biobj_f73_i09_d05 1.0",
  "bbob-biobj_f73_i09_d10 1.0",
  "bbob-biobj_f73_i09_d20 1.0",
  "bbob-biobj_f73_i09_d40 1.0",
  "bbob-biobj_f73_i10_d02 1.0",
  "bbob-biobj_f73_i10_d03 1.0",
  "bbob-biobj_f73_i10_d05 1.0",
  "bbob-biobj_f73_i10_d10 1.0",
  "bbob-biobj_f73_i10_d20 1.0",
  "bbob-biobj_f73_i10_d40 1.0",
  "bbob-biobj_f73_i11_d02 1.0",
  "bbob-biobj_f73_i11_d03 1.0",
  "bbob-biobj_f73_i11_d05 1.0",
  "bbob-biobj_f73_i11_d10 1.0",
  "bbob-biobj_f73_i11_d20 1.0",
  "bbob-biobj_f73_i11_d40 1.0",
  "bbob-biobj_f73_i12_d02 1.0",
  "bbob-biobj_f73_i12_d03 1.0",
  "bbob-biobj_f73_i12_d05 1.0",
  "bbob-biobj_f73_i12_d10 1.0",
  "bbob-biobj_f73_i12_d20 1.0",
  "bbob-biobj_f73_i12_d40 1.0",
  "bbob-biobj_f73_i13_d02 1.0",
  "bbob-biobj_f73_i13_d03 1.0",
  "bbob-biobj_f73_i13_d05 1.0",
  "bbob-biobj_f73_i13_d10 1.0",
  "bbob-biobj_f73_i13_d20 1.0",
  "bbob-biobj_f73_i13_d40 1.0",
  "bbob-biobj_f73_i14_d02 1.0",
  "bbob-biobj_f73_i14_d03 1.0",
  "bbob-biobj_f73_i14_d05 1.0",
  "bbob-biobj_f73_i14_d10 1.0",
  "bbob-biobj_f73_i14_d20 1.0",
  "bbob-biobj_f73_i14_d40 1.0",
  "bbob-biobj_f73_i15_d02 1.0",
  "bbob-biobj_f73_i15_d03 1.0",
  "bbob-biobj_f73_i15_d05 1.0",
  "bbob-biobj_f73_i15_d10 1.0",
  "bbob-biobj_f73_i15_d20 1.0",
  "bbob-biobj_f73_i15_d40 1.0",
  "bbob-biobj_f74_i01_d02 1.0",
  "bbob-biobj_f74_i01_d03 1.0",
  "bbob-biobj_f74_i01_d05 1.0",
  "bbob-biobj_f74_i01_d10 1.0",
  "bbob-biobj_f74_i01_d20 1.0",
  "bbob-biobj_f74_i01_d40 1.0",
  "bbob-biobj_f74_i02_d02 1.0",
  "bbob-biobj_f74_i02_d03 1.0",
  "bbob-biobj_f74_i02_d05 1.0",
  "bbob-biobj_f74_i02_d10 1.0",
  "bbob-biobj_f74_i02_d20 1.0",
  "bbob-biobj_f74_i02_d40 1.0",
  "bbob-biobj_f74_i03_d02 1.0",
  "bbob-biobj_f74_i03_d03 1.0",
  "bbob-biobj_f74_i03_d05 1.0",
  "bbob-biobj_f74_i03_d10 1.0",
  "bbob-biobj_f74_i03_d20 1.0",
  "bbob-biobj_f74_i03_d40 1.0",
  "bbob-biobj_f74_i04_d02 1.0",
  "bbob-biobj_f74_i04_d03 1.0",
  "bbob-biobj_f74_i04_d05 1.0",
  "bbob-biobj_f74_i04_d10 1.0",
  "bbob-biobj_f74_i04_d20 1.0",
  "bbob-biobj_f74_i04_d40 1.0",
  "bbob-biobj_f74_i05_d02 1.0",
  "bbob-biobj_f74_i05_d03 1.0",
  "bbob-biobj_f74_i05_d05 1.0",
  "bbob-biobj_f74_i05_d10 1.0",
  "bbob-biobj_f74_i05_d20 1.0",
  "bbob-biobj_f74_i05_d40 1.0",
  "bbob-biobj_f74_i06_d02 1.0",
  "bbob-biobj_f74_i06_d03 1.0",
  "bbob-biobj_f74_i06_d05 1.0",
  "bbob-biobj_f74_i06_d10 1.0",
  "bbob-biobj_f74_i06_d20 1.0",
  "bbob-biobj_f74_i06_d40 1.0",
  "bbob-biobj_f74_i07_d02 1.0",
  "bbob-biobj_f74_i07_d03 1.0",
  "bbob-biobj_f74_i07_d05 1.0",
  "bbob-biobj_f74_i07_d10 1.0",
  "bbob-biobj_f74_i07_d20 1.0",
  "bbob-biobj_f74_i07_d40 1.0",
  "bbob-biobj_f74_i08_d02 1.0",
  "bbob-biobj_f74_i08_d03 1.0",
  "bbob-biobj_f74_i08_d05 1.0",
  "bbob-biobj_f74_i08_d10 1.0",
  "bbob-biobj_f74_i08_d20 1.0",
  "bbob-biobj_f74_i08_d40 1.0",
  "bbob-biobj_f74_i09_d02 1.0",
  "bbob-biobj_f74_i09_d03 1.0",
  "bbob-biobj_f74_i09_d05 1.0",
  "bbob-biobj_f74_i09_d10 1.0",
  "bbob-biobj_f74_i09_d20 1.0",
  "bbob-biobj_f74_i09_d40 1.0",
  "bbob-biobj_f74_i10_d02 1.0",
  "bbob-biobj_f74_i10_d03 1.0",
  "bbob-biobj_f74_i10_d05 1.0",
  "bbob-biobj_f74_i10_d10 1.0",
  "bbob-biobj_f74_i10_d20 1.0",
  "bbob-biobj_f74_i10_d40 1.0",
  "bbob-biobj_f74_i11_d02 1.0",
  "bbob-biobj_f74_i11_d03 1.0",
  "bbob-biobj_f74_i11_d05 1.0",
  "bbob-biobj_f74_i11_d10 1.0",
  "bbob-biobj_f74_i11_d20 1.0",
  "bbob-biobj_f74_i11_d40 1.0",
  "bbob-biobj_f74_i12_d02 1.0",
  "bbob-biobj_f74_i12_d03 1.0",
  "bbob-biobj_f74_i12_d05 1.0",
  "bbob-biobj_f74_i12_d10 1.0",
  "bbob-biobj_f74_i12_d20 1.0",
  "bbob-biobj_f74_i12_d40 1.0",
  "bbob-biobj_f74_i13_d02 1.0",
  "bbob-biobj_f74_i13_d03 1.0",
  "bbob-biobj_f74_i13_d05 1.0",
  "bbob-biobj_f74_i13_d10 1.0",
  "bbob-biobj_f74_i13_d20 1.0",
  "bbob-biobj_f74_i13_d40 1.0",
  "bbob-biobj_f74_i14_d02 1.0",
  "bbob-biobj_f74_i14_d03 1.0",
  "bbob-biobj_f74_i14_d05 1.0",
  "bbob-biobj_f74_i14_d10 1.0",
  "bbob-biobj_f74_i14_d20 1.0",
  "bbob-biobj_f74_i14_d40 1.0",
  "bbob-biobj_f74_i15_d02 1.0",
  "bbob-biobj_f74_i15_d03 1.0",
  "bbob-biobj_f74_i15_d05 1.0",
  "bbob-biobj_f74_i15_d10 1.0",
  "bbob-biobj_f74_i15_d20 1.0",
  "bbob-biobj_f74_i15_d40 1.0",
  "bbob-biobj_f75_i01_d02 1.0",
  "bbob-biobj_f75_i01_d03 1.0",
  "bbob-biobj_f75_i01_d05 1.0",
  "bbob-biobj_f75_i01_d10 1.0",
  "bbob-biobj_f75_i01_d20 1.0",
  "bbob-biobj_f75_i01_d40 1.0",
  "bbob-biobj_f75_i02_d02 1.0",
  "bbob-biobj_f75_i02_d03 1.0",
  "bbob-biobj_f75_i02_d05 1.0",
  "bbob-biobj_f75_i02_d10 1.0",
  "bbob-biobj_f75_i02_d20 1.0",
  "bbob-biobj_f75_i02_d40 1.0",
  "bbob-biobj_f75_i03_d02 1.0",
  "bbob-biobj_f75_i03_d03 1.0",
  "bbob-biobj_f75_i03_d05 1.0",
  "bbob-biobj_f75_i03_d10 1.0",
  "bbob-biobj_f75_i03_d20 1.0",
  "bbob-biobj_f75_i03_d40 1.0",
  "bbob-biobj_f75_i04_d02 1.0",
  "bbob-biobj_f75_i04_d03 1.0",
  "bbob-biobj_f75_i04_d05 1.0",
  "bbob-biobj_f75_i04_d10 1.0",
  "bbob-biobj_f75_i04_d20 1.0",
  "bbob-biobj_f75_i04_d40 1.0",
  "bbob-biobj_f75_i05_d02 1.0",
  "bbob-biobj_f75_i05_d03 1.0",
  "bbob-biobj_f75_i05_d05 1.0",
  "bbob-biobj_f75_i05_d10 1.0",
  "bbob-biobj_f75_i05_d20 1.0",
  "bbob-biobj_f75_i05_d40 1.0",
  "bbob-biobj_f75_i06_d02 1.0",
  "bbob-biobj_f75_i06_d03 1.0",
  "bbob-biobj_f75_i06_d05 1.0",
  "bbob-biobj_f75_i06_d10 1.0",
  "bbob-biobj_f75_i06_d20 1.0",
  "bbob-biobj_f75_i06_d40 1.0",
  "bbob-biobj_f75_i07_d02 1.0",
  "bbob-biobj_f75_i07_d03 1.0",
  "bbob-biobj_f75_i07_d05 1.0",
  "bbob-biobj_f75_i07_d10 1.0",
  "bbob-biobj_f75_i07_d20 1.0",
  "bbob-biobj_f75_i07_d40 1.0",
  "bbob-biobj_f75_i08_d02 1.0",
  "bbob-biobj_f75_i08_d03 1.0",
  "bbob-biobj_f75_i08_d05 1.0",
  "bbob-biobj_f75_i08_d10 1.0",
  "bbob-biobj_f75_i08_d20 1.0",
  "bbob-biobj_f75_i08_d40 1.0",
  "bbob-biobj_f75_i09_d02 1.0",
  "bbob-biobj_f75_i09_d03 1.0",
  "bbob-biobj_f75_i09_d05 1.0",
  "bbob-biobj_f75_i09_d10 1.0",
  "bbob-biobj_f75_i09_d20 1.0",
  "bbob-biobj_f75_i09_d40 1.0",
  "bbob-biobj_f75_i10_d02 1.0",
  "bbob-biobj_f75_i10_d03 1.0",
  "bbob-biobj_f75_i10_d05 1.0",
  "bbob-biobj_f75_i10_d10 1.0",
  "bbob-biobj_f75_i10_d20 1.0",
  "bbob-biobj_f75_i10_d40 1.0",
  "bbob-biobj_f75_i11_d02 1.0",
  "bbob-biobj_f75_i11_d03 1.0",
  "bbob-biobj_f75_i11_d05 1.0",
  "bbob-biobj_f75_i11_d10 1.0",
  "bbob-biobj_f75_i11_d20 1.0",
  "bbob-biobj_f75_i11_d40 1.0",
  "bbob-biobj_f75_i12_d02 1.0",
  "bbob-biobj_f75_i12_d03 1.0",
  "bbob-biobj_f75_i12_d05 1.0",
  "bbob-biobj_f75_i12_d10 1.0",
  "bbob-biobj_f75_i12_d20 1.0",
  "bbob-biobj_f75_i12_d40 1.0",
  "bbob-biobj_f75_i13_d02 1.0",
  "bbob-biobj_f75_i13_d03 1.0",
  "bbob-biobj_f75_i13_d05 1.0",
  "bbob-biobj_f75_i13_d10 1.0",
  "bbob-biobj_f75_i13_d20 1.0",
  "bbob-biobj_f75_i13_d40 1.0",
  "bbob-biobj_f75_i14_d02 1.0",
  "bbob-biobj_f75_i14_d03 1.0",
  "bbob-biobj_f75_i14_d05 1.0",
  "bbob-biobj_f75_i14_d10 1.0",
  "bbob-biobj_f75_i14_d20 1.0",
  "bbob-biobj_f75_i14_d40 1.0",
  "bbob-biobj_f75_i15_d02 1.0",
  "bbob-biobj_f75_i15_d03 1.0",
  "bbob-biobj_f75_i15_d05 1.0",
  "bbob-biobj_f75_i15_d10 1.0",
  "bbob-biobj_f75_i15_d20 1.0",
  "bbob-biobj_f75_i15_d40 1.0",
  "bbob-biobj_f76_i01_d02 1.0",
  "bbob-biobj_f76_i01_d03 1.0",
  "bbob-biobj_f76_i01_d05 1.0",
  "bbob-biobj_f76_i01_d10 1.0",
  "bbob-biobj_f76_i01_d20 1.0",
  "bbob-biobj_f76_i01_d40 1.0",
  "bbob-biobj_f76_i02_d02 1.0",
  "bbob-biobj_f76_i02_d03 1.0",
  "bbob-biobj_f76_i02_d05 1.0",
  "bbob-biobj_f76_i02_d10 1.0",
  "bbob-biobj_f76_i02_d20 1.0",
  "bbob-biobj_f76_i02_d40 1.0",
  "bbob-biobj_f76_i03_d02 1.0",
  "bbob-biobj_f76_i03_d03 1.0",
  "bbob-biobj_f76_i03_d05 1.0",
  "bbob-biobj_f76_i03_d10 1.0",
  "bbob-biobj_f76_i03_d20 1.0",
  "bbob-biobj_f76_i03_d40 1.0",
  "bbob-biobj_f76_i04_d02 1.0",
  "bbob-biobj_f76_i04_d03 1.0",
  "bbob-biobj_f76_i04_d05 1.0",
  "bbob-biobj_f76_i04_d10 1.0",
  "bbob-biobj_f76_i04_d20 1.0",
  "bbob-biobj_f76_i04_d40 1.0",
  "bbob-biobj_f76_i05_d02 1.0",
  "bbob-biobj_f76_i05_d03 1.0",
  "bbob-biobj_f76_i05_d05 1.0",
  "bbob-biobj_f76_i05_d10 1.0",
  "bbob-biobj_f76_i05_d20 1.0",
  "bbob-biobj_f76_i05_d40 1.0",
  "bbob-biobj_f76_i06_d02 1.0",
  "bbob-biobj_f76_i06_d03 1.0",
  "bbob-biobj_f76_i06_d05 1.0",
  "bbob-biobj_f76_i06_d10 1.0",
  "bbob-biobj_f76_i06_d20 1.0",
  "bbob-biobj_f76_i06_d40 1.0",
  "bbob-biobj_f76_i07_d02 1.0",
  "bbob-biobj_f76_i07_d03 1.0",
  "bbob-biobj_f76_i07_d05 1.0",
  "bbob-biobj_f76_i07_d10 1.0",
  "bbob-biobj_f76_i07_d20 1.0",
  "bbob-biobj_f76_i07_d40 1.0",
  "bbob-biobj_f76_i08_d02 1.0",
  "bbob-biobj_f76_i08_d03 1.0",
  "bbob-biobj_f76_i08_d05 1.0",
  "bbob-biobj_f76_i08_d10 1.0",
  "bbob-biobj_f76_i08_d20 1.0",
  "bbob-biobj_f76_i08_d40 1.0",
  "bbob-biobj_f76_i09_d02 1.0",
  "bbob-biobj_f76_i09_d03 1.0",
  "bbob-biobj_f76_i09_d05 1.0",
  "bbob-biobj_f76_i09_d10 1.0",
  "bbob-biobj_f76_i09_d20 1.0",
  "bbob-biobj_f76_i09_d40 1.0",
  "bbob-biobj_f76_i10_d02 1.0",
  "bbob-biobj_f76_i10_d03 1.0",
  "bbob-biobj_f76_i10_d05 1.0",
  "bbob-biobj_f76_i10_d10 1.0",
  "bbob-biobj_f76_i10_d20 1.0",
  "bbob-biobj_f76_i10_d40 1.0",
  "bbob-biobj_f76_i11_d02 1.0",
  "bbob-biobj_f76_i11_d03 1.0",
  "bbob-biobj_f76_i11_d05 1.0",
  "bbob-biobj_f76_i11_d10 1.0",
  "bbob-biobj_f76_i11_d20 1.0",
  "bbob-biobj_f76_i11_d40 1.0",
  "bbob-biobj_f76_i12_d02 1.0",
  "bbob-biobj_f76_i12_d03 1.0",
  "bbob-biobj_f76_i12_d05 1.0",
  "bbob-biobj_f76_i12_d10 1.0",
  "bbob-biobj_f76_i12_d20 1.0",
  "bbob-biobj_f76_i12_d40 1.0",
  "bbob-biobj_f76_i13_d02 1.0",
  "bbob-biobj_f76_i13_d03 1.0",
  "bbob-biobj_f76_i13_d05 1.0",
  "bbob-biobj_f76_i13_d10 1.0",
  "bbob-biobj_f76_i13_d20 1.0",
  "bbob-biobj_f76_i13_d40 1.0",
  "bbob-biobj_f76_i14_d02 1.0",
  "bbob-biobj_f76_i14_d03 1.0",
  "bbob-biobj_f76_i14_d05 1.0",
  "bbob-biobj_f76_i14_d10 1.0",
  "bbob-biobj_f76_i14_d20 1.0",
  "bbob-biobj_f76_i14_d40 1.0",
  "bbob-biobj_f76_i15_d02 1.0",
  "bbob-biobj_f76_i15_d03 1.0",
  "bbob-biobj_f76_i15_d05 1.0",
  "bbob-biobj_f76_i15_d10 1.0",
  "bbob-biobj_f76_i15_d20 1.0",
  "bbob-biobj_f76_i15_d40 1.0",
  "bbob-biobj_f77_i01_d02 1.0",
  "bbob-biobj_f77_i01_d03 1.0",
  "bbob-biobj_f77_i01_d05 1.0",
  "bbob-biobj_f77_i01_d10 1.0",
  "bbob-biobj_f77_i01_d20 1.0",
  "bbob-biobj_f77_i01_d40 1.0",
  "bbob-biobj_f77_i02_d02 1.0",
  "bbob-biobj_f77_i02_d03 1.0",
  "bbob-biobj_f77_i02_d05 1.0",
  "bbob-biobj_f77_i02_d10 1.0",
  "bbob-biobj_f77_i02_d20 1.0",
  "bbob-biobj_f77_i02_d40 1.0",
  "bbob-biobj_f77_i03_d02 1.0",
  "bbob-biobj_f77_i03_d03 1.0",
  "bbob-biobj_f77_i03_d05 1.0",
  "bbob-biobj_f77_i03_d10 1.0",
  "bbob-biobj_f77_i03_d20 1.0",
  "bbob-biobj_f77_i03_d40 1.0",
  "bbob-biobj_f77_i04_d02 1.0",
  "bbob-biobj_f77_i04_d03 1.0",
  "bbob-biobj_f77_i04_d05 1.0",
  "bbob-biobj_f77_i04_d10 1.0",
  "bbob-biobj_f77_i04_d20 1.0",
  "bbob-biobj_f77_i04_d40 1.0",
  "bbob-biobj_f77_i05_d02 1.0",
  "bbob-biobj_f77_i05_d03 1.0",
  "bbob-biobj_f77_i05_d05 1.0",
  "bbob-biobj_f77_i05_d10 1.0",
  "bbob-biobj_f77_i05_d20 1.0",
  "bbob-biobj_f77_i05_d40 1.0",
  "bbob-biobj_f77_i06_d02 1.0",
  "bbob-biobj_f77_i06_d03 1.0",
  "bbob-biobj_f77_i06_d05 1.0",
  "bbob-biobj_f77_i06_d10 1.0",
  "bbob-biobj_f77_i06_d20 1.0",
  "bbob-biobj_f77_i06_d40 1.0",
  "bbob-biobj_f77_i07_d02 1.0",
  "bbob-biobj_f77_i07_d03 1.0",
  "bbob-biobj_f77_i07_d05 1.0",
  "bbob-biobj_f77_i07_d10 1.0",
  "bbob-biobj_f77_i07_d20 1.0",
  "bbob-biobj_f77_i07_d40 1.0",
  "bbob-biobj_f77_i08_d02 1.0",
  "bbob-biobj_f77_i08_d03 1.0",
  "bbob-biobj_f77_i08_d05 1.0",
  "bbob-biobj_f77_i08_d10 1.0",
  "bbob-biobj_f77_i08_d20 1.0",
  "bbob-biobj_f77_i08_d40 1.0",
  "bbob-biobj_f77_i09_d02 1.0",
  "bbob-biobj_f77_i09_d03 1.0",
  "bbob-biobj_f77_i09_d05 1.0",
  "bbob-biobj_f77_i09_d10 1.0",
  "bbob-biobj_f77_i09_d20 1.0",
  "bbob-biobj_f77_i09_d40 1.0",
  "bbob-biobj_f77_i10_d02 1.0",
  "bbob-biobj_f77_i10_d03 1.0",
  "bbob-biobj_f77_i10_d05 1.0",
  "bbob-biobj_f77_i10_d10 1.0",
  "bbob-biobj_f77_i10_d20 1.0",
  "bbob-biobj_f77_i10_d40 1.0",
  "bbob-biobj_f77_i11_d02 1.0",
  "bbob-biobj_f77_i11_d03 1.0",
  "bbob-biobj_f77_i11_d05 1.0",
  "bbob-biobj_f77_i11_d10 1.0",
  "bbob-biobj_f77_i11_d20 1.0",
  "bbob-biobj_f77_i11_d40 1.0",
  "bbob-biobj_f77_i12_d02 1.0",
  "bbob-biobj_f77_i12_d03 1.0",
  "bbob-biobj_f77_i12_d05 1.0",
  "bbob-biobj_f77_i12_d10 1.0",
  "bbob-biobj_f77_i12_d20 1.0",
  "bbob-biobj_f77_i12_d40 1.0",
  "bbob-biobj_f77_i13_d02 1.0",
  "bbob-biobj_f77_i13_d03 1.0",
  "bbob-biobj_f77_i13_d05 1.0",
  "bbob-biobj_f77_i13_d10 1.0",
  "bbob-biobj_f77_i13_d20 1.0",
  "bbob-biobj_f77_i13_d40 1.0",
  "bbob-biobj_f77_i14_d02 1.0",
  "bbob-biobj_f77_i14_d03 1.0",
  "bbob-biobj_f77_i14_d05 1.0",
  "bbob-biobj_f77_i14_d10 1.0",
  "bbob-biobj_f77_i14_d20 1.0",
  "bbob-biobj_f77_i14_d40 1.0",
  "bbob-biobj_f77_i15_d02 1.0",
  "bbob-biobj_f77_i15_d03 1.0",
  "bbob-biobj_f77_i15_d05 1.0",
  "bbob-biobj_f77_i15_d10 1.0",
  "bbob-biobj_f77_i15_d20 1.0",
  "bbob-biobj_f77_i15_d40 1.0",
  "bbob-biobj_f78_i01_d02 1.0",
  "bbob-biobj_f78_i01_d03 1.0",
  "bbob-biobj_f78_i01_d05 1.0",
  "bbob-biobj_f78_i01_d10 1.0",
  "bbob-biobj_f78_i01_d20 1.0",
  "bbob-biobj_f78_i01_d40 1.0",
  "bbob-biobj_f78_i02_d02 1.0",
  "bbob-biobj_f78_i02_d03 1.0",
  "bbob-biobj_f78_i02_d05 1.0",
  "bbob-biobj_f78_i02_d10 1.0",
  "bbob-biobj_f78_i02_d20 1.0",
  "bbob-biobj_f78_i02_d40 1.0",
  "bbob-biobj_f78_i03_d02 1.0",
  "bbob-biobj_f78_i03_d03 1.0",
  "bbob-biobj_f78_i03_d05 1.0",
  "bbob-biobj_f78_i03_d10 1.0",
  "bbob-biobj_f78_i03_d20 1.0",
  "bbob-biobj_f78_i03_d40 1.0",
  "bbob-biobj_f78_i04_d02 1.0",
  "bbob-biobj_f78_i04_d03 1.0",
  "bbob-biobj_f78_i04_d05 1.0",
  "bbob-biobj_f78_i04_d10 1.0",
  "bbob-biobj_f78_i04_d20 1.0",
  "bbob-biobj_f78_i04_d40 1.0",
  "bbob-biobj_f78_i05_d02 1.0",
  "bbob-biobj_f78_i05_d03 1.0",
  "bbob-biobj_f78_i05_d05 1.0",
  "bbob-biobj_f78_i05_d10 1.0",
  "bbob-biobj_f78_i05_d20 1.0",
  "bbob-biobj_f78_i05_d40 1.0",
  "bbob-biobj_f78_i06_d02 1.0",
  "bbob-biobj_f78_i06_d03 1.0",
  "bbob-biobj_f78_i06_d05 1.0",
  "bbob-biobj_f78_i06_d10 1.0",
  "bbob-biobj_f78_i06_d20 1.0",
  "bbob-biobj_f78_i06_d40 1.0",
  "bbob-biobj_f78_i07_d02 1.0",
  "bbob-biobj_f78_i07_d03 1.0",
  "bbob-biobj_f78_i07_d05 1.0",
  "bbob-biobj_f78_i07_d10 1.0",
  "bbob-biobj_f78_i07_d20 1.0",
  "bbob-biobj_f78_i07_d40 1.0",
  "bbob-biobj_f78_i08_d02 1.0",
  "bbob-biobj_f78_i08_d03 1.0",
  "bbob-biobj_f78_i08_d05 1.0",
  "bbob-biobj_f78_i08_d10 1.0",
  "bbob-biobj_f78_i08_d20 1.0",
  "bbob-biobj_f78_i08_d40 1.0",
  "bbob-biobj_f78_i09_d02 1.0",
  "bbob-biobj_f78_i09_d03 1.0",
  "bbob-biobj_f78_i09_d05 1.0",
  "bbob-biobj_f78_i09_d10 1.0",
  "bbob-biobj_f78_i09_d20 1.0",
  "bbob-biobj_f78_i09_d40 1.0",
  "bbob-biobj_f78_i10_d02 1.0",
  "bbob-biobj_f78_i10_d03 1.0",
  "bbob-biobj_f78_i10_d05 1.0",
  "bbob-biobj_f78_i10_d10 1.0",
  "bbob-biobj_f78_i10_d20 1.0",
  "bbob-biobj_f78_i10_d40 1.0",
  "bbob-biobj_f78_i11_d02 1.0",
  "bbob-biobj_f78_i11_d03 1.0",
  "bbob-biobj_f78_i11_d05 1.0",
  "bbob-biobj_f78_i11_d10 1.0",
  "bbob-biobj_f78_i11_d20 1.0",
  "bbob-biobj_f78_i11_d40 1.0",
  "bbob-biobj_f78_i12_d02 1.0",
  "bbob-biobj_f78_i12_d03 1.0",
  "bbob-biobj_f78_i12_d05 1.0",
  "bbob-biobj_f78_i12_d10 1.0",
  "bbob-biobj_f78_i12_d20 1.0",
  "bbob-biobj_f78_i12_d40 1.0",
  "bbob-biobj_f78_i13_d02 1.0",
  "bbob-biobj_f78_i13_d03 1.0",
  "bbob-biobj_f78_i13_d05 1.0",
  "bbob-biobj_f78_i13_d10 1.0",
  "bbob-biobj_f78_i13_d20 1.0",
  "bbob-biobj_f78_i13_d40 1.0",
  "bbob-biobj_f78_i14_d02 1.0",
  "bbob-biobj_f78_i14_d03 1.0",
  "bbob-biobj_f78_i14_d05 1.0",
  "bbob-biobj_f78_i14_d10 1.0",
  "bbob-biobj_f78_i14_d20 1.0",
  "bbob-biobj_f78_i14_d40 1.0",
  "bbob-biobj_f78_i15_d02 1.0",
  "bbob-biobj_f78_i15_d03 1.0",
  "bbob-biobj_f78_i15_d05 1.0",
  "bbob-biobj_f78_i15_d10 1.0",
  "bbob-biobj_f78_i15_d20 1.0",
  "bbob-biobj_f78_i15_d40 1.0",
  "bbob-biobj_f79_i01_d02 1.0",
  "bbob-biobj_f79_i01_d03 1.0",
  "bbob-biobj_f79_i01_d05 1.0",
  "bbob-biobj_f79_i01_d10 1.0",
  "bbob-biobj_f79_i01_d20 1.0",
  "bbob-biobj_f79_i01_d40 1.0",
  "bbob-biobj_f79_i02_d02 1.0",
  "bbob-biobj_f79_i02_d03 1.0",
  "bbob-biobj_f79_i02_d05 1.0",
  "bbob-biobj_f79_i02_d10 1.0",
  "bbob-biobj_f79_i02_d20 1.0",
  "bbob-biobj_f79_i02_d40 1.0",
  "bbob-biobj_f79_i03_d02 1.0",
  "bbob-biobj_f79_i03_d03 1.0",
  "bbob-biobj_f79_i03_d05 1.0",
  "bbob-biobj_f79_i03_d10 1.0",
  "bbob-biobj_f79_i03_d20 1.0",
  "bbob-biobj_f79_i03_d40 1.0",
  "bbob-biobj_f79_i04_d02 1.0",
  "bbob-biobj_f79_i04_d03 1.0",
  "bbob-biobj_f79_i04_d05 1.0",
  "bbob-biobj_f79_i04_d10 1.0",
  "bbob-biobj_f79_i04_d20 1.0",
  "bbob-biobj_f79_i04_d40 1.0",
  "bbob-biobj_f79_i05_d02 1.0",
  "bbob-biobj_f79_i05_d03 1.0",
  "bbob-biobj_f79_i05_d05 1.0",
  "bbob-biobj_f79_i05_d10 1.0",
  "bbob-biobj_f79_i05_d20 1.0",
  "bbob-biobj_f79_i05_d40 1.0",
  "bbob-biobj_f79_i06_d02 1.0",
  "bbob-biobj_f79_i06_d03 1.0",
  "bbob-biobj_f79_i06_d05 1.0",
  "bbob-biobj_f79_i06_d10 1.0",
  "bbob-biobj_f79_i06_d20 1.0",
  "bbob-biobj_f79_i06_d40 1.0",
  "bbob-biobj_f79_i07_d02 1.0",
  "bbob-biobj_f79_i07_d03 1.0",
  "bbob-biobj_f79_i07_d05 1.0",
  "bbob-biobj_f79_i07_d10 1.0",
  "bbob-biobj_f79_i07_d20 1.0",
  "bbob-biobj_f79_i07_d40 1.0",
  "bbob-biobj_f79_i08_d02 1.0",
  "bbob-biobj_f79_i08_d03 1.0",
  "bbob-biobj_f79_i08_d05 1.0",
  "bbob-biobj_f79_i08_d10 1.0",
  "bbob-biobj_f79_i08_d20 1.0",
  "bbob-biobj_f79_i08_d40 1.0",
  "bbob-biobj_f79_i09_d02 1.0",
  "bbob-biobj_f79_i09_d03 1.0",
  "bbob-biobj_f79_i09_d05 1.0",
  "bbob-biobj_f79_i09_d10 1.0",
  "bbob-biobj_f79_i09_d20 1.0",
  "bbob-biobj_f79_i09_d40 1.0",
  "bbob-biobj_f79_i10_d02 1.0",
  "bbob-biobj_f79_i10_d03 1.0",
  "bbob-biobj_f79_i10_d05 1.0",
  "bbob-biobj_f79_i10_d10 1.0",
  "bbob-biobj_f79_i10_d20 1.0",
  "bbob-biobj_f79_i10_d40 1.0",
  "bbob-biobj_f79_i11_d02 1.0",
  "bbob-biobj_f79_i11_d03 1.0",
  "bbob-biobj_f79_i11_d05 1.0",
  "bbob-biobj_f79_i11_d10 1.0",
  "bbob-biobj_f79_i11_d20 1.0",
  "bbob-biobj_f79_i11_d40 1.0",
  "bbob-biobj_f79_i12_d02 1.0",
  "bbob-biobj_f79_i12_d03 1.0",
  "bbob-biobj_f79_i12_d05 1.0",
  "bbob-biobj_f79_i12_d10 1.0",
  "bbob-biobj_f79_i12_d20 1.0",
  "bbob-biobj_f79_i12_d40 1.0",
  "bbob-biobj_f79_i13_d02 1.0",
  "bbob-biobj_f79_i13_d03 1.0",
  "bbob-biobj_f79_i13_d05 1.0",
  "bbob-biobj_f79_i13_d10 1.0",
  "bbob-biobj_f79_i13_d20 1.0",
  "bbob-biobj_f79_i13_d40 1.0",
  "bbob-biobj_f79_i14_d02 1.0",
  "bbob-biobj_f79_i14_d03 1.0",
  "bbob-biobj_f79_i14_d05 1.0",
  "bbob-biobj_f79_i14_d10 1.0",
  "bbob-biobj_f79_i14_d20 1.0",
  "bbob-biobj_f79_i14_d40 1.0",
  "bbob-biobj_f79_i15_d02 1.0",
  "bbob-biobj_f79_i15_d03 1.0",
  "bbob-biobj_f79_i15_d05 1.0",
  "bbob-biobj_f79_i15_d10 1.0",
  "bbob-biobj_f79_i15_d20 1.0",
  "bbob-biobj_f79_i15_d40 1.0",
  "bbob-biobj_f80_i01_d02 1.0",
  "bbob-biobj_f80_i01_d03 1.0",
  "bbob-biobj_f80_i01_d05 1.0",
  "bbob-biobj_f80_i01_d10 1.0",
  "bbob-biobj_f80_i01_d20 1.0",
  "bbob-biobj_f80_i01_d40 1.0",
  "bbob-biobj_f80_i02_d02 1.0",
  "bbob-biobj_f80_i02_d03 1.0",
  "bbob-biobj_f80_i02_d05 1.0",
  "bbob-biobj_f80_i02_d10 1.0",
  "bbob-biobj_f80_i02_d20 1.0",
  "bbob-biobj_f80_i02_d40 1.0",
  "bbob-biobj_f80_i03_d02 1.0",
  "bbob-biobj_f80_i03_d03 1.0",
  "bbob-biobj_f80_i03_d05 1.0",
  "bbob-biobj_f80_i03_d10 1.0",
  "bbob-biobj_f80_i03_d20 1.0",
  "bbob-biobj_f80_i03_d40 1.0",
  "bbob-biobj_f80_i04_d02 1.0",
  "bbob-biobj_f80_i04_d03 1.0",
  "bbob-biobj_f80_i04_d05 1.0",
  "bbob-biobj_f80_i04_d10 1.0",
  "bbob-biobj_f80_i04_d20 1.0",
  "bbob-biobj_f80_i04_d40 1.0",
  "bbob-biobj_f80_i05_d02 1.0",
  "bbob-biobj_f80_i05_d03 1.0",
  "bbob-biobj_f80_i05_d05 1.0",
  "bbob-biobj_f80_i05_d10 1.0",
  "bbob-biobj_f80_i05_d20 1.0",
  "bbob-biobj_f80_i05_d40 1.0",
  "bbob-biobj_f80_i06_d02 1.0",
  "bbob-biobj_f80_i06_d03 1.0",
  "bbob-biobj_f80_i06_d05 1.0",
  "bbob-biobj_f80_i06_d10 1.0",
  "bbob-biobj_f80_i06_d20 1.0",
  "bbob-biobj_f80_i06_d40 1.0",
  "bbob-biobj_f80_i07_d02 1.0",
  "bbob-biobj_f80_i07_d03 1.0",
  "bbob-biobj_f80_i07_d05 1.0",
  "bbob-biobj_f80_i07_d10 1.0",
  "bbob-biobj_f80_i07_d20 1.0",
  "bbob-biobj_f80_i07_d40 1.0",
  "bbob-biobj_f80_i08_d02 1.0",
  "bbob-biobj_f80_i08_d03 1.0",
  "bbob-biobj_f80_i08_d05 1.0",
  "bbob-biobj_f80_i08_d10 1.0",
  "bbob-biobj_f80_i08_d20 1.0",
  "bbob-biobj_f80_i08_d40 1.0",
  "bbob-biobj_f80_i09_d02 1.0",
  "bbob-biobj_f80_i09_d03 1.0",
  "bbob-biobj_f80_i09_d05 1.0",
  "bbob-biobj_f80_i09_d10 1.0",
  "bbob-biobj_f80_i09_d20 1.0",
  "bbob-biobj_f80_i09_d40 1.0",
  "bbob-biobj_f80_i10_d02 1.0",
  "bbob-biobj_f80_i10_d03 1.0",
  "bbob-biobj_f80_i10_d05 1.0",
  "bbob-biobj_f80_i10_d10 1.0",
  "bbob-biobj_f80_i10_d20 1.0",
  "bbob-biobj_f80_i10_d40 1.0",
  "bbob-biobj_f80_i11_d02 1.0",
  "bbob-biobj_f80_i11_d03 1.0",
  "bbob-biobj_f80_i11_d05 1.0",
  "bbob-biobj_f80_i11_d10 1.0",
  "bbob-biobj_f80_i11_d20 1.0",
  "bbob-biobj_f80_i11_d40 1.0",
  "bbob-biobj_f80_i12_d02 1.0",
  "bbob-biobj_f80_i12_d03 1.0",
  "bbob-biobj_f80_i12_d05 1.0",
  "bbob-biobj_f80_i12_d10 1.0",
  "bbob-biobj_f80_i12_d20 1.0",
  "bbob-biobj_f80_i12_d40 1.0",
  "bbob-biobj_f80_i13_d02 1.0",
  "bbob-biobj_f80_i13_d03 1.0",
  "bbob-biobj_f80_i13_d05 1.0",
  "bbob-biobj_f80_i13_d10 1.0",
  "bbob-biobj_f80_i13_d20 1.0",
  "bbob-biobj_f80_i13_d40 1.0",
  "bbob-biobj_f80_i14_d02 1.0",
  "bbob-biobj_f80_i14_d03 1.0",
  "bbob-biobj_f80_i14_d05 1.0",
  "bbob-biobj_f80_i14_d10 1.0",
  "bbob-biobj_f80_i14_d20 1.0",
  "bbob-biobj_f80_i14_d40 1.0",
  "bbob-biobj_f80_i15_d02 1.0",
  "bbob-biobj_f80_i15_d03 1.0",
  "bbob-biobj_f80_i15_d05 1.0",
  "bbob-biobj_f80_i15_d10 1.0",
  "bbob-biobj_f80_i15_d20 1.0",
  "bbob-biobj_f80_i15_d40 1.0",
  "bbob-biobj_f81_i01_d02 1.0",
  "bbob-biobj_f81_i01_d03 1.0",
  "bbob-biobj_f81_i01_d05 1.0",
  "bbob-biobj_f81_i01_d10 1.0",
  "bbob-biobj_f81_i01_d20 1.0",
  "bbob-biobj_f81_i01_d40 1.0",
  "bbob-biobj_f81_i02_d02 1.0",
  "bbob-biobj_f81_i02_d03 1.0",
  "bbob-biobj_f81_i02_d05 1.0",
  "bbob-biobj_f81_i02_d10 1.0",
  "bbob-biobj_f81_i02_d20 1.0",
  "bbob-biobj_f81_i02_d40 1.0",
  "bbob-biobj_f81_i03_d02 1.0",
  "bbob-biobj_f81_i03_d03 1.0",
  "bbob-biobj_f81_i03_d05 1.0",
  "bbob-biobj_f81_i03_d10 1.0",
  "bbob-biobj_f81_i03_d20 1.0",
  "bbob-biobj_f81_i03_d40 1.0",
  "bbob-biobj_f81_i04_d02 1.0",
  "bbob-biobj_f81_i04_d03 1.0",
  "bbob-biobj_f81_i04_d05 1.0",
  "bbob-biobj_f81_i04_d10 1.0",
  "bbob-biobj_f81_i04_d20 1.0",
  "bbob-biobj_f81_i04_d40 1.0",
  "bbob-biobj_f81_i05_d02 1.0",
  "bbob-biobj_f81_i05_d03 1.0",
  "bbob-biobj_f81_i05_d05 1.0",
  "bbob-biobj_f81_i05_d10 1.0",
  "bbob-biobj_f81_i05_d20 1.0",
  "bbob-biobj_f81_i05_d40 1.0",
  "bbob-biobj_f81_i06_d02 1.0",
  "bbob-biobj_f81_i06_d03 1.0",
  "bbob-biobj_f81_i06_d05 1.0",
  "bbob-biobj_f81_i06_d10 1.0",
  "bbob-biobj_f81_i06_d20 1.0",
  "bbob-biobj_f81_i06_d40 1.0",
  "bbob-biobj_f81_i07_d02 1.0",
  "bbob-biobj_f81_i07_d03 1.0",
  "bbob-biobj_f81_i07_d05 1.0",
  "bbob-biobj_f81_i07_d10 1.0",
  "bbob-biobj_f81_i07_d20 1.0",
  "bbob-biobj_f81_i07_d40 1.0",
  "bbob-biobj_f81_i08_d02 1.0",
  "bbob-biobj_f81_i08_d03 1.0",
  "bbob-biobj_f81_i08_d05 1.0",
  "bbob-biobj_f81_i08_d10 1.0",
  "bbob-biobj_f81_i08_d20 1.0",
  "bbob-biobj_f81_i08_d40 1.0",
  "bbob-biobj_f81_i09_d02 1.0",
  "bbob-biobj_f81_i09_d03 1.0",
  "bbob-biobj_f81_i09_d05 1.0",
  "bbob-biobj_f81_i09_d10 1.0",
  "bbob-biobj_f81_i09_d20 1.0",
  "bbob-biobj_f81_i09_d40 1.0",
  "bbob-biobj_f81_i10_d02 1.0",
  "bbob-biobj_f81_i10_d03 1.0",
  "bbob-biobj_f81_i10_d05 1.0",
  "bbob-biobj_f81_i10_d10 1.0",
  "bbob-biobj_f81_i10_d20 1.0",
  "bbob-biobj_f81_i10_d40 1.0",
  "bbob-biobj_f81_i11_d02 1.0",
  "bbob-biobj_f81_i11_d03 1.0",
  "bbob-biobj_f81_i11_d05 1.0",
  "bbob-biobj_f81_i11_d10 1.0",
  "bbob-biobj_f81_i11_d20 1.0",
  "bbob-biobj_f81_i11_d40 1.0",
  "bbob-biobj_f81_i12_d02 1.0",
  "bbob-biobj_f81_i12_d03 1.0",
  "bbob-biobj_f81_i12_d05 1.0",
  "bbob-biobj_f81_i12_d10 1.0",
  "bbob-biobj_f81_i12_d20 1.0",
  "bbob-biobj_f81_i12_d40 1.0",
  "bbob-biobj_f81_i13_d02 1.0",
  "bbob-biobj_f81_i13_d03 1.0",
  "bbob-biobj_f81_i13_d05 1.0",
  "bbob-biobj_f81_i13_d10 1.0",
  "bbob-biobj_f81_i13_d20 1.0",
  "bbob-biobj_f81_i13_d40 1.0",
  "bbob-biobj_f81_i14_d02 1.0",
  "bbob-biobj_f81_i14_d03 1.0",
  "bbob-biobj_f81_i14_d05 1.0",
  "bbob-biobj_f81_i14_d10 1.0",
  "bbob-biobj_f81_i14_d20 1.0",
  "bbob-biobj_f81_i14_d40 1.0",
  "bbob-biobj_f81_i15_d02 1.0",
  "bbob-biobj_f81_i15_d03 1.0",
  "bbob-biobj_f81_i15_d05 1.0",
  "bbob-biobj_f81_i15_d10 1.0",
  "bbob-biobj_f81_i15_d20 1.0",
  "bbob-biobj_f81_i15_d40 1.0",
  "bbob-biobj_f82_i01_d02 1.0",
  "bbob-biobj_f82_i01_d03 1.0",
  "bbob-biobj_f82_i01_d05 1.0",
  "bbob-biobj_f82_i01_d10 1.0",
  "bbob-biobj_f82_i01_d20 1.0",
  "bbob-biobj_f82_i01_d40 1.0",
  "bbob-biobj_f82_i02_d02 1.0",
  "bbob-biobj_f82_i02_d03 1.0",
  "bbob-biobj_f82_i02_d05 1.0",
  "bbob-biobj_f82_i02_d10 1.0",
  "bbob-biobj_f82_i02_d20 1.0",
  "bbob-biobj_f82_i02_d40 1.0",
  "bbob-biobj_f82_i03_d02 1.0",
  "bbob-biobj_f82_i03_d03 1.0",
  "bbob-biobj_f82_i03_d05 1.0",
  "bbob-biobj_f82_i03_d10 1.0",
  "bbob-biobj_f82_i03_d20 1.0",
  "bbob-biobj_f82_i03_d40 1.0",
  "bbob-biobj_f82_i04_d02 1.0",
  "bbob-biobj_f82_i04_d03 1.0",
  "bbob-biobj_f82_i04_d05 1.0",
  "bbob-biobj_f82_i04_d10 1.0",
  "bbob-biobj_f82_i04_d20 1.0",
  "bbob-biobj_f82_i04_d40 1.0",
  "bbob-biobj_f82_i05_d02 1.0",
  "bbob-biobj_f82_i05_d03 1.0",
  "bbob-biobj_f82_i05_d05 1.0",
  "bbob-biobj_f82_i05_d10 1.0",
  "bbob-biobj_f82_i05_d20 1.0",
  "bbob-biobj_f82_i05_d40 1.0",
  "bbob-biobj_f82_i06_d02 1.0",
  "bbob-biobj_f82_i06_d03 1.0",
  "bbob-biobj_f82_i06_d05 1.0",
  "bbob-biobj_f82_i06_d10 1.0",
  "bbob-biobj_f82_i06_d20 1.0",
  "bbob-biobj_f82_i06_d40 1.0",
  "bbob-biobj_f82_i07_d02 1.0",
  "bbob-biobj_f82_i07_d03 1.0",
  "bbob-biobj_f82_i07_d05 1.0",
  "bbob-biobj_f82_i07_d10 1.0",
  "bbob-biobj_f82_i07_d20 1.0",
  "bbob-biobj_f82_i07_d40 1.0",
  "bbob-biobj_f82_i08_d02 1.0",
  "bbob-biobj_f82_i08_d03 1.0",
  "bbob-biobj_f82_i08_d05 1.0",
  "bbob-biobj_f82_i08_d10 1.0",
  "bbob-biobj_f82_i08_d20 1.0",
  "bbob-biobj_f82_i08_d40 1.0",
  "bbob-biobj_f82_i09_d02 1.0",
  "bbob-biobj_f82_i09_d03 1.0",
  "bbob-biobj_f82_i09_d05 1.0",
  "bbob-biobj_f82_i09_d10 1.0",
  "bbob-biobj_f82_i09_d20 1.0",
  "bbob-biobj_f82_i09_d40 1.0",
  "bbob-biobj_f82_i10_d02 1.0",
  "bbob-biobj_f82_i10_d03 1.0",
  "bbob-biobj_f82_i10_d05 1.0",
  "bbob-biobj_f82_i10_d10 1.0",
  "bbob-biobj_f82_i10_d20 1.0",
  "bbob-biobj_f82_i10_d40 1.0",
  "bbob-biobj_f82_i11_d02 1.0",
  "bbob-biobj_f82_i11_d03 1.0",
  "bbob-biobj_f82_i11_d05 1.0",
  "bbob-biobj_f82_i11_d10 1.0",
  "bbob-biobj_f82_i11_d20 1.0",
  "bbob-biobj_f82_i11_d40 1.0",
  "bbob-biobj_f82_i12_d02 1.0",
  "bbob-biobj_f82_i12_d03 1.0",
  "bbob-biobj_f82_i12_d05 1.0",
  "bbob-biobj_f82_i12_d10 1.0",
  "bbob-biobj_f82_i12_d20 1.0",
  "bbob-biobj_f82_i12_d40 1.0",
  "bbob-biobj_f82_i13_d02 1.0",
  "bbob-biobj_f82_i13_d03 1.0",
  "bbob-biobj_f82_i13_d05 1.0",
  "bbob-biobj_f82_i13_d10 1.0",
  "bbob-biobj_f82_i13_d20 1.0",
  "bbob-biobj_f82_i13_d40 1.0",
  "bbob-biobj_f82_i14_d02 1.0",
  "bbob-biobj_f82_i14_d03 1.0",
  "bbob-biobj_f82_i14_d05 1.0",
  "bbob-biobj_f82_i14_d10 1.0",
  "bbob-biobj_f82_i14_d20 1.0",
  "bbob-biobj_f82_i14_d40 1.0",
  "bbob-biobj_f82_i15_d02 1.0",
  "bbob-biobj_f82_i15_d03 1.0",
  "bbob-biobj_f82_i15_d05 1.0",
  "bbob-biobj_f82_i15_d10 1.0",
  "bbob-biobj_f82_i15_d20 1.0",
  "bbob-biobj_f82_i15_d40 1.0",
  "bbob-biobj_f83_i01_d02 1.0",
  "bbob-biobj_f83_i01_d03 1.0",
  "bbob-biobj_f83_i01_d05 1.0",
  "bbob-biobj_f83_i01_d10 1.0",
  "bbob-biobj_f83_i01_d20 1.0",
  "bbob-biobj_f83_i01_d40 1.0",
  "bbob-biobj_f83_i02_d02 1.0",
  "bbob-biobj_f83_i02_d03 1.0",
  "bbob-biobj_f83_i02_d05 1.0",
  "bbob-biobj_f83_i02_d10 1.0",
  "bbob-biobj_f83_i02_d20 1.0",
  "bbob-biobj_f83_i02_d40 1.0",
  "bbob-biobj_f83_i03_d02 1.0",
  "bbob-biobj_f83_i03_d03 1.0",
  "bbob-biobj_f83_i03_d05 1.0",
  "bbob-biobj_f83_i03_d10 1.0",
  "bbob-biobj_f83_i03_d20 1.0",
  "bbob-biobj_f83_i03_d40 1.0",
  "bbob-biobj_f83_i04_d02 1.0",
  "bbob-biobj_f83_i04_d03 1.0",
  "bbob-biobj_f83_i04_d05 1.0",
  "bbob-biobj_f83_i04_d10 1.0",
  "bbob-biobj_f83_i04_d20 1.0",
  "bbob-biobj_f83_i04_d40 1.0",
  "bbob-biobj_f83_i05_d02 1.0",
  "bbob-biobj_f83_i05_d03 1.0",
  "bbob-biobj_f83_i05_d05 1.0",
  "bbob-biobj_f83_i05_d10 1.0",
  "bbob-biobj_f83_i05_d20 1.0",
  "bbob-biobj_f83_i05_d40 1.0",
  "bbob-biobj_f83_i06_d02 1.0",
  "bbob-biobj_f83_i06_d03 1.0",
  "bbob-biobj_f83_i06_d05 1.0",
  "bbob-biobj_f83_i06_d10 1.0",
  "bbob-biobj_f83_i06_d20 1.0",
  "bbob-biobj_f83_i06_d40 1.0",
  "bbob-biobj_f83_i07_d02 1.0",
  "bbob-biobj_f83_i07_d03 1.0",
  "bbob-biobj_f83_i07_d05 1.0",
  "bbob-biobj_f83_i07_d10 1.0",
  "bbob-biobj_f83_i07_d20 1.0",
  "bbob-biobj_f83_i07_d40 1.0",
  "bbob-biobj_f83_i08_d02 1.0",
  "bbob-biobj_f83_i08_d03 1.0",
  "bbob-biobj_f83_i08_d05 1.0",
  "bbob-biobj_f83_i08_d10 1.0",
  "bbob-biobj_f83_i08_d20 1.0",
  "bbob-biobj_f83_i08_d40 1.0",
  "bbob-biobj_f83_i09_d02 1.0",
  "bbob-biobj_f83_i09_d03 1.0",
  "bbob-biobj_f83_i09_d05 1.0",
  "bbob-biobj_f83_i09_d10 1.0",
  "bbob-biobj_f83_i09_d20 1.0",
  "bbob-biobj_f83_i09_d40 1.0",
  "bbob-biobj_f83_i10_d02 1.0",
  "bbob-biobj_f83_i10_d03 1.0",
  "bbob-biobj_f83_i10_d05 1.0",
  "bbob-biobj_f83_i10_d10 1.0",
  "bbob-biobj_f83_i10_d20 1.0",
  "bbob-biobj_f83_i10_d40 1.0",
  "bbob-biobj_f83_i11_d02 1.0",
  "bbob-biobj_f83_i11_d03 1.0",
  "bbob-biobj_f83_i11_d05 1.0",
  "bbob-biobj_f83_i11_d10 1.0",
  "bbob-biobj_f83_i11_d20 1.0",
  "bbob-biobj_f83_i11_d40 1.0",
  "bbob-biobj_f83_i12_d02 1.0",
  "bbob-biobj_f83_i12_d03 1.0",
  "bbob-biobj_f83_i12_d05 1.0",
  "bbob-biobj_f83_i12_d10 1.0",
  "bbob-biobj_f83_i12_d20 1.0",
  "bbob-biobj_f83_i12_d40 1.0",
  "bbob-biobj_f83_i13_d02 1.0",
  "bbob-biobj_f83_i13_d03 1.0",
  "bbob-biobj_f83_i13_d05 1.0",
  "bbob-biobj_f83_i13_d10 1.0",
  "bbob-biobj_f83_i13_d20 1.0",
  "bbob-biobj_f83_i13_d40 1.0",
  "bbob-biobj_f83_i14_d02 1.0",
  "bbob-biobj_f83_i14_d03 1.0",
  "bbob-biobj_f83_i14_d05 1.0",
  "bbob-biobj_f83_i14_d10 1.0",
  "bbob-biobj_f83_i14_d20 1.0",
  "bbob-biobj_f83_i14_d40 1.0",
  "bbob-biobj_f83_i15_d02 1.0",
  "bbob-biobj_f83_i15_d03 1.0",
  "bbob-biobj_f83_i15_d05 1.0",
  "bbob-biobj_f83_i15_d10 1.0",
  "bbob-biobj_f83_i15_d20 1.0",
  "bbob-biobj_f83_i15_d40 1.0",
  "bbob-biobj_f84_i01_d02 1.0",
  "bbob-biobj_f84_i01_d03 1.0",
  "bbob-biobj_f84_i01_d05 1.0",
  "bbob-biobj_f84_i01_d10 1.0",
  "bbob-biobj_f84_i01_d20 1.0",
  "bbob-biobj_f84_i01_d40 1.0",
  "bbob-biobj_f84_i02_d02 1.0",
  "bbob-biobj_f84_i02_d03 1.0",
  "bbob-biobj_f84_i02_d05 1.0",
  "bbob-biobj_f84_i02_d10 1.0",
  "bbob-biobj_f84_i02_d20 1.0",
  "bbob-biobj_f84_i02_d40 1.0",
  "bbob-biobj_f84_i03_d02 1.0",
  "bbob-biobj_f84_i03_d03 1.0",
  "bbob-biobj_f84_i03_d05 1.0",
  "bbob-biobj_f84_i03_d10 1.0",
  "bbob-biobj_f84_i03_d20 1.0",
  "bbob-biobj_f84_i03_d40 1.0",
  "bbob-biobj_f84_i04_d02 1.0",
  "bbob-biobj_f84_i04_d03 1.0",
  "bbob-biobj_f84_i04_d05 1.0",
  "bbob-biobj_f84_i04_d10 1.0",
  "bbob-biobj_f84_i04_d20 1.0",
  "bbob-biobj_f84_i04_d40 1.0",
  "bbob-biobj_f84_i05_d02 1.0",
  "bbob-biobj_f84_i05_d03 1.0",
  "bbob-biobj_f84_i05_d05 1.0",
  "bbob-biobj_f84_i05_d10 1.0",
  "bbob-biobj_f84_i05_d20 1.0",
  "bbob-biobj_f84_i05_d40 1.0",
  "bbob-biobj_f84_i06_d02 1.0",
  "bbob-biobj_f84_i06_d03 1.0",
  "bbob-biobj_f84_i06_d05 1.0",
  "bbob-biobj_f84_i06_d10 1.0",
  "bbob-biobj_f84_i06_d20 1.0",
  "bbob-biobj_f84_i06_d40 1.0",
  "bbob-biobj_f84_i07_d02 1.0",
  "bbob-biobj_f84_i07_d03 1.0",
  "bbob-biobj_f84_i07_d05 1.0",
  "bbob-biobj_f84_i07_d10 1.0",
  "bbob-biobj_f84_i07_d20 1.0",
  "bbob-biobj_f84_i07_d40 1.0",
  "bbob-biobj_f84_i08_d02 1.0",
  "bbob-biobj_f84_i08_d03 1.0",
  "bbob-biobj_f84_i08_d05 1.0",
  "bbob-biobj_f84_i08_d10 1.0",
  "bbob-biobj_f84_i08_d20 1.0",
  "bbob-biobj_f84_i08_d40 1.0",
  "bbob-biobj_f84_i09_d02 1.0",
  "bbob-biobj_f84_i09_d03 1.0",
  "bbob-biobj_f84_i09_d05 1.0",
  "bbob-biobj_f84_i09_d10 1.0",
  "bbob-biobj_f84_i09_d20 1.0",
  "bbob-biobj_f84_i09_d40 1.0",
  "bbob-biobj_f84_i10_d02 1.0",
  "bbob-biobj_f84_i10_d03 1.0",
  "bbob-biobj_f84_i10_d05 1.0",
  "bbob-biobj_f84_i10_d10 1.0",
  "bbob-biobj_f84_i10_d20 1.0",
  "bbob-biobj_f84_i10_d40 1.0",
  "bbob-biobj_f84_i11_d02 1.0",
  "bbob-biobj_f84_i11_d03 1.0",
  "bbob-biobj_f84_i11_d05 1.0",
  "bbob-biobj_f84_i11_d10 1.0",
  "bbob-biobj_f84_i11_d20 1.0",
  "bbob-biobj_f84_i11_d40 1.0",
  "bbob-biobj_f84_i12_d02 1.0",
  "bbob-biobj_f84_i12_d03 1.0",
  "bbob-biobj_f84_i12_d05 1.0",
  "bbob-biobj_f84_i12_d10 1.0",
  "bbob-biobj_f84_i12_d20 1.0",
  "bbob-biobj_f84_i12_d40 1.0",
  "bbob-biobj_f84_i13_d02 1.0",
  "bbob-biobj_f84_i13_d03 1.0",
  "bbob-biobj_f84_i13_d05 1.0",
  "bbob-biobj_f84_i13_d10 1.0",
  "bbob-biobj_f84_i13_d20 1.0",
  "bbob-biobj_f84_i13_d40 1.0",
  "bbob-biobj_f84_i14_d02 1.0",
  "bbob-biobj_f84_i14_d03 1.0",
  "bbob-biobj_f84_i14_d05 1.0",
  "bbob-biobj_f84_i14_d10 1.0",
  "bbob-biobj_f84_i14_d20 1.0",
  "bbob-biobj_f84_i14_d40 1.0",
  "bbob-biobj_f84_i15_d02 1.0",
  "bbob-biobj_f84_i15_d03 1.0",
  "bbob-biobj_f84_i15_d05 1.0",
  "bbob-biobj_f84_i15_d10 1.0",
  "bbob-biobj_f84_i15_d20 1.0",
  "bbob-biobj_f84_i15_d40 1.0",
  "bbob-biobj_f85_i01_d02 1.0",
  "bbob-biobj_f85_i01_d03 1.0",
  "bbob-biobj_f85_i01_d05 1.0",
  "bbob-biobj_f85_i01_d10 1.0",
  "bbob-biobj_f85_i01_d20 1.0",
  "bbob-biobj_f85_i01_d40 1.0",
  "bbob-biobj_f85_i02_d02 1.0",
  "bbob-biobj_f85_i02_d03 1.0",
  "bbob-biobj_f85_i02_d05 1.0",
  "bbob-biobj_f85_i02_d10 1.0",
  "bbob-biobj_f85_i02_d20 1.0",
  "bbob-biobj_f85_i02_d40 1.0",
  "bbob-biobj_f85_i03_d02 1.0",
  "bbob-biobj_f85_i03_d03 1.0",
  "bbob-biobj_f85_i03_d05 1.0",
  "bbob-biobj_f85_i03_d10 1.0",
  "bbob-biobj_f85_i03_d20 1.0",
  "bbob-biobj_f85_i03_d40 1.0",
  "bbob-biobj_f85_i04_d02 1.0",
  "bbob-biobj_f85_i04_d03 1.0",
  "bbob-biobj_f85_i04_d05 1.0",
  "bbob-biobj_f85_i04_d10 1.0",
  "bbob-biobj_f85_i04_d20 1.0",
  "bbob-biobj_f85_i04_d40 1.0",
  "bbob-biobj_f85_i05_d02 1.0",
  "bbob-biobj_f85_i05_d03 1.0",
  "bbob-biobj_f85_i05_d05 1.0",
  "bbob-biobj_f85_i05_d10 1.0",
  "bbob-biobj_f85_i05_d20 1.0",
  "bbob-biobj_f85_i05_d40 1.0",
  "bbob-biobj_f85_i06_d02 1.0",
  "bbob-biobj_f85_i06_d03 1.0",
  "bbob-biobj_f85_i06_d05 1.0",
  "bbob-biobj_f85_i06_d10 1.0",
  "bbob-biobj_f85_i06_d20 1.0",
  "bbob-biobj_f85_i06_d40 1.0",
  "bbob-biobj_f85_i07_d02 1.0",
  "bbob-biobj_f85_i07_d03 1.0",
  "bbob-biobj_f85_i07_d05 1.0",
  "bbob-biobj_f85_i07_d10 1.0",
  "bbob-biobj_f85_i07_d20 1.0",
  "bbob-biobj_f85_i07_d40 1.0",
  "bbob-biobj_f85_i08_d02 1.0",
  "bbob-biobj_f85_i08_d03 1.0",
  "bbob-biobj_f85_i08_d05 1.0",
  "bbob-biobj_f85_i08_d10 1.0",
  "bbob-biobj_f85_i08_d20 1.0",
  "bbob-biobj_f85_i08_d40 1.0",
  "bbob-biobj_f85_i09_d02 1.0",
  "bbob-biobj_f85_i09_d03 1.0",
  "bbob-biobj_f85_i09_d05 1.0",
  "bbob-biobj_f85_i09_d10 1.0",
  "bbob-biobj_f85_i09_d20 1.0",
  "bbob-biobj_f85_i09_d40 1.0",
  "bbob-biobj_f85_i10_d02 1.0",
  "bbob-biobj_f85_i10_d03 1.0",
  "bbob-biobj_f85_i10_d05 1.0",
  "bbob-biobj_f85_i10_d10 1.0",
  "bbob-biobj_f85_i10_d20 1.0",
  "bbob-biobj_f85_i10_d40 1.0",
  "bbob-biobj_f85_i11_d02 1.0",
  "bbob-biobj_f85_i11_d03 1.0",
  "bbob-biobj_f85_i11_d05 1.0",
  "bbob-biobj_f85_i11_d10 1.0",
  "bbob-biobj_f85_i11_d20 1.0",
  "bbob-biobj_f85_i11_d40 1.0",
  "bbob-biobj_f85_i12_d02 1.0",
  "bbob-biobj_f85_i12_d03 1.0",
  "bbob-biobj_f85_i12_d05 1.0",
  "bbob-biobj_f85_i12_d10 1.0",
  "bbob-biobj_f85_i12_d20 1.0",
  "bbob-biobj_f85_i12_d40 1.0",
  "bbob-biobj_f85_i13_d02 1.0",
  "bbob-biobj_f85_i13_d03 1.0",
  "bbob-biobj_f85_i13_d05 1.0",
  "bbob-biobj_f85_i13_d10 1.0",
  "bbob-biobj_f85_i13_d20 1.0",
  "bbob-biobj_f85_i13_d40 1.0",
  "bbob-biobj_f85_i14_d02 1.0",
  "bbob-biobj_f85_i14_d03 1.0",
  "bbob-biobj_f85_i14_d05 1.0",
  "bbob-biobj_f85_i14_d10 1.0",
  "bbob-biobj_f85_i14_d20 1.0",
  "bbob-biobj_f85_i14_d40 1.0",
  "bbob-biobj_f85_i15_d02 1.0",
  "bbob-biobj_f85_i15_d03 1.0",
  "bbob-biobj_f85_i15_d05 1.0",
  "bbob-biobj_f85_i15_d10 1.0",
  "bbob-biobj_f85_i15_d20 1.0",
  "bbob-biobj_f85_i15_d40 1.0",
  "bbob-biobj_f86_i01_d02 1.0",
  "bbob-biobj_f86_i01_d03 1.0",
  "bbob-biobj_f86_i01_d05 1.0",
  "bbob-biobj_f86_i01_d10 1.0",
  "bbob-biobj_f86_i01_d20 1.0",
  "bbob-biobj_f86_i01_d40 1.0",
  "bbob-biobj_f86_i02_d02 1.0",
  "bbob-biobj_f86_i02_d03 1.0",
  "bbob-biobj_f86_i02_d05 1.0",
  "bbob-biobj_f86_i02_d10 1.0",
  "bbob-biobj_f86_i02_d20 1.0",
  "bbob-biobj_f86_i02_d40 1.0",
  "bbob-biobj_f86_i03_d02 1.0",
  "bbob-biobj_f86_i03_d03 1.0",
  "bbob-biobj_f86_i03_d05 1.0",
  "bbob-biobj_f86_i03_d10 1.0",
  "bbob-biobj_f86_i03_d20 1.0",
  "bbob-biobj_f86_i03_d40 1.0",
  "bbob-biobj_f86_i04_d02 1.0",
  "bbob-biobj_f86_i04_d03 1.0",
  "bbob-biobj_f86_i04_d05 1.0",
  "bbob-biobj_f86_i04_d10 1.0",
  "bbob-biobj_f86_i04_d20 1.0",
  "bbob-biobj_f86_i04_d40 1.0",
  "bbob-biobj_f86_i05_d02 1.0",
  "bbob-biobj_f86_i05_d03 1.0",
  "bbob-biobj_f86_i05_d05 1.0",
  "bbob-biobj_f86_i05_d10 1.0",
  "bbob-biobj_f86_i05_d20 1.0",
  "bbob-biobj_f86_i05_d40 1.0",
  "bbob-biobj_f86_i06_d02 1.0",
  "bbob-biobj_f86_i06_d03 1.0",
  "bbob-biobj_f86_i06_d05 1.0",
  "bbob-biobj_f86_i06_d10 1.0",
  "bbob-biobj_f86_i06_d20 1.0",
  "bbob-biobj_f86_i06_d40 1.0",
  "bbob-biobj_f86_i07_d02 1.0",
  "bbob-biobj_f86_i07_d03 1.0",
  "bbob-biobj_f86_i07_d05 1.0",
  "bbob-biobj_f86_i07_d10 1.0",
  "bbob-biobj_f86_i07_d20 1.0",
  "bbob-biobj_f86_i07_d40 1.0",
  "bbob-biobj_f86_i08_d02 1.0",
  "bbob-biobj_f86_i08_d03 1.0",
  "bbob-biobj_f86_i08_d05 1.0",
  "bbob-biobj_f86_i08_d10 1.0",
  "bbob-biobj_f86_i08_d20 1.0",
  "bbob-biobj_f86_i08_d40 1.0",
  "bbob-biobj_f86_i09_d02 1.0",
  "bbob-biobj_f86_i09_d03 1.0",
  "bbob-biobj_f86_i09_d05 1.0",
  "bbob-biobj_f86_i09_d10 1.0",
  "bbob-biobj_f86_i09_d20 1.0",
  "bbob-biobj_f86_i09_d40 1.0",
  "bbob-biobj_f86_i10_d02 1.0",
  "bbob-biobj_f86_i10_d03 1.0",
  "bbob-biobj_f86_i10_d05 1.0",
  "bbob-biobj_f86_i10_d10 1.0",
  "bbob-biobj_f86_i10_d20 1.0",
  "bbob-biobj_f86_i10_d40 1.0",
  "bbob-biobj_f86_i11_d02 1.0",
  "bbob-biobj_f86_i11_d03 1.0",
  "bbob-biobj_f86_i11_d05 1.0",
  "bbob-biobj_f86_i11_d10 1.0",
  "bbob-biobj_f86_i11_d20 1.0",
  "bbob-biobj_f86_i11_d40 1.0",
  "bbob-biobj_f86_i12_d02 1.0",
  "bbob-biobj_f86_i12_d03 1.0",
  "bbob-biobj_f86_i12_d05 1.0",
  "bbob-biobj_f86_i12_d10 1.0",
  "bbob-biobj_f86_i12_d20 1.0",
  "bbob-biobj_f86_i12_d40 1.0",
  "bbob-biobj_f86_i13_d02 1.0",
  "bbob-biobj_f86_i13_d03 1.0",
  "bbob-biobj_f86_i13_d05 1.0",
  "bbob-biobj_f86_i13_d10 1.0",
  "bbob-biobj_f86_i13_d20 1.0",
  "bbob-biobj_f86_i13_d40 1.0",
  "bbob-biobj_f86_i14_d02 1.0",
  "bbob-biobj_f86_i14_d03 1.0",
  "bbob-biobj_f86_i14_d05 1.0",
  "bbob-biobj_f86_i14_d10 1.0",
  "bbob-biobj_f86_i14_d20 1.0",
  "bbob-biobj_f86_i14_d40 1.0",
  "bbob-biobj_f86_i15_d02 1.0",
  "bbob-biobj_f86_i15_d03 1.0",
  "bbob-biobj_f86_i15_d05 1.0",
  "bbob-biobj_f86_i15_d10 1.0",
  "bbob-biobj_f86_i15_d20 1.0",
  "bbob-biobj_f86_i15_d40 1.0",
  "bbob-biobj_f87_i01_d02 1.0",
  "bbob-biobj_f87_i01_d03 1.0",
  "bbob-biobj_f87_i01_d05 1.0",
  "bbob-biobj_f87_i01_d10 1.0",
  "bbob-biobj_f87_i01_d20 1.0",
  "bbob-biobj_f87_i01_d40 1.0",
  "bbob-biobj_f87_i02_d02 1.0",
  "bbob-biobj_f87_i02_d03 1.0",
  "bbob-biobj_f87_i02_d05 1.0",
  "bbob-biobj_f87_i02_d10 1.0",
  "bbob-biobj_f87_i02_d20 1.0",
  "bbob-biobj_f87_i02_d40 1.0",
  "bbob-biobj_f87_i03_d02 1.0",
  "bbob-biobj_f87_i03_d03 1.0",
  "bbob-biobj_f87_i03_d05 1.0",
  "bbob-biobj_f87_i03_d10 1.0",
  "bbob-biobj_f87_i03_d20 1.0",
  "bbob-biobj_f87_i03_d40 1.0",
  "bbob-biobj_f87_i04_d02 1.0",
  "bbob-biobj_f87_i04_d03 1.0",
  "bbob-biobj_f87_i04_d05 1.0",
  "bbob-biobj_f87_i04_d10 1.0",
  "bbob-biobj_f87_i04_d20 1.0",
  "bbob-biobj_f87_i04_d40 1.0",
  "bbob-biobj_f87_i05_d02 1.0",
  "bbob-biobj_f87_i05_d03 1.0",
  "bbob-biobj_f87_i05_d05 1.0",
  "bbob-biobj_f87_i05_d10 1.0",
  "bbob-biobj_f87_i05_d20 1.0",
  "bbob-biobj_f87_i05_d40 1.0",
  "bbob-biobj_f87_i06_d02 1.0",
  "bbob-biobj_f87_i06_d03 1.0",
  "bbob-biobj_f87_i06_d05 1.0",
  "bbob-biobj_f87_i06_d10 1.0",
  "bbob-biobj_f87_i06_d20 1.0",
  "bbob-biobj_f87_i06_d40 1.0",
  "bbob-biobj_f87_i07_d02 1.0",
  "bbob-biobj_f87_i07_d03 1.0",
  "bbob-biobj_f87_i07_d05 1.0",
  "bbob-biobj_f87_i07_d10 1.0",
  "bbob-biobj_f87_i07_d20 1.0",
  "bbob-biobj_f87_i07_d40 1.0",
  "bbob-biobj_f87_i08_d02 1.0",
  "bbob-biobj_f87_i08_d03 1.0",
  "bbob-biobj_f87_i08_d05 1.0",
  "bbob-biobj_f87_i08_d10 1.0",
  "bbob-biobj_f87_i08_d20 1.0",
  "bbob-biobj_f87_i08_d40 1.0",
  "bbob-biobj_f87_i09_d02 1.0",
  "bbob-biobj_f87_i09_d03 1.0",
  "bbob-biobj_f87_i09_d05 1.0",
  "bbob-biobj_f87_i09_d10 1.0",
  "bbob-biobj_f87_i09_d20 1.0",
  "bbob-biobj_f87_i09_d40 1.0",
  "bbob-biobj_f87_i10_d02 1.0",
  "bbob-biobj_f87_i10_d03 1.0",
  "bbob-biobj_f87_i10_d05 1.0",
  "bbob-biobj_f87_i10_d10 1.0",
  "bbob-biobj_f87_i10_d20 1.0",
  "bbob-biobj_f87_i10_d40 1.0",
  "bbob-biobj_f87_i11_d02 1.0",
  "bbob-biobj_f87_i11_d03 1.0",
  "bbob-biobj_f87_i11_d05 1.0",
  "bbob-biobj_f87_i11_d10 1.0",
  "bbob-biobj_f87_i11_d20 1.0",
  "bbob-biobj_f87_i11_d40 1.0",
  "bbob-biobj_f87_i12_d02 1.0",
  "bbob-biobj_f87_i12_d03 1.0",
  "bbob-biobj_f87_i12_d05 1.0",
  "bbob-biobj_f87_i12_d10 1.0",
  "bbob-biobj_f87_i12_d20 1.0",
  "bbob-biobj_f87_i12_d40 1.0",
  "bbob-biobj_f87_i13_d02 1.0",
  "bbob-biobj_f87_i13_d03 1.0",
  "bbob-biobj_f87_i13_d05 1.0",
  "bbob-biobj_f87_i13_d10 1.0",
  "bbob-biobj_f87_i13_d20 1.0",
  "bbob-biobj_f87_i13_d40 1.0",
  "bbob-biobj_f87_i14_d02 1.0",
  "bbob-biobj_f87_i14_d03 1.0",
  "bbob-biobj_f87_i14_d05 1.0",
  "bbob-biobj_f87_i14_d10 1.0",
  "bbob-biobj_f87_i14_d20 1.0",
  "bbob-biobj_f87_i14_d40 1.0",
  "bbob-biobj_f87_i15_d02 1.0",
  "bbob-biobj_f87_i15_d03 1.0",
  "bbob-biobj_f87_i15_d05 1.0",
  "bbob-biobj_f87_i15_d10 1.0",
  "bbob-biobj_f87_i15_d20 1.0",
  "bbob-biobj_f87_i15_d40 1.0",
  "bbob-biobj_f88_i01_d02 1.0",
  "bbob-biobj_f88_i01_d03 1.0",
  "bbob-biobj_f88_i01_d05 1.0",
  "bbob-biobj_f88_i01_d10 1.0",
  "bbob-biobj_f88_i01_d20 1.0",
  "bbob-biobj_f88_i01_d40 1.0",
  "bbob-biobj_f88_i02_d02 1.0",
  "bbob-biobj_f88_i02_d03 1.0",
  "bbob-biobj_f88_i02_d05 1.0",
  "bbob-biobj_f88_i02_d10 1.0",
  "bbob-biobj_f88_i02_d20 1.0",
  "bbob-biobj_f88_i02_d40 1.0",
  "bbob-biobj_f88_i03_d02 1.0",
  "bbob-biobj_f88_i03_d03 1.0",
  "bbob-biobj_f88_i03_d05 1.0",
  "bbob-biobj_f88_i03_d10 1.0",
  "bbob-biobj_f88_i03_d20 1.0",
  "bbob-biobj_f88_i03_d40 1.0",
  "bbob-biobj_f88_i04_d02 1.0",
  "bbob-biobj_f88_i04_d03 1.0",
  "bbob-biobj_f88_i04_d05 1.0",
  "bbob-biobj_f88_i04_d10 1.0",
  "bbob-biobj_f88_i04_d20 1.0",
  "bbob-biobj_f88_i04_d40 1.0",
  "bbob-biobj_f88_i05_d02 1.0",
  "bbob-biobj_f88_i05_d03 1.0",
  "bbob-biobj_f88_i05_d05 1.0",
  "bbob-biobj_f88_i05_d10 1.0",
  "bbob-biobj_f88_i05_d20 1.0",
  "bbob-biobj_f88_i05_d40 1.0",
  "bbob-biobj_f88_i06_d02 1.0",
  "bbob-biobj_f88_i06_d03 1.0",
  "bbob-biobj_f88_i06_d05 1.0",
  "bbob-biobj_f88_i06_d10 1.0",
  "bbob-biobj_f88_i06_d20 1.0",
  "bbob-biobj_f88_i06_d40 1.0",
  "bbob-biobj_f88_i07_d02 1.0",
  "bbob-biobj_f88_i07_d03 1.0",
  "bbob-biobj_f88_i07_d05 1.0",
  "bbob-biobj_f88_i07_d10 1.0",
  "bbob-biobj_f88_i07_d20 1.0",
  "bbob-biobj_f88_i07_d40 1.0",
  "bbob-biobj_f88_i08_d02 1.0",
  "bbob-biobj_f88_i08_d03 1.0",
  "bbob-biobj_f88_i08_d05 1.0",
  "bbob-biobj_f88_i08_d10 1.0",
  "bbob-biobj_f88_i08_d20 1.0",
  "bbob-biobj_f88_i08_d40 1.0",
  "bbob-biobj_f88_i09_d02 1.0",
  "bbob-biobj_f88_i09_d03 1.0",
  "bbob-biobj_f88_i09_d05 1.0",
  "bbob-biobj_f88_i09_d10 1.0",
  "bbob-biobj_f88_i09_d20 1.0",
  "bbob-biobj_f88_i09_d40 1.0",
  "bbob-biobj_f88_i10_d02 1.0",
  "bbob-biobj_f88_i10_d03 1.0",
  "bbob-biobj_f88_i10_d05 1.0",
  "bbob-biobj_f88_i10_d10 1.0",
  "bbob-biobj_f88_i10_d20 1.0",
  "bbob-biobj_f88_i10_d40 1.0",
  "bbob-biobj_f88_i11_d02 1.0",
  "bbob-biobj_f88_i11_d03 1.0",
  "bbob-biobj_f88_i11_d05 1.0",
  "bbob-biobj_f88_i11_d10 1.0",
  "bbob-biobj_f88_i11_d20 1.0",
  "bbob-biobj_f88_i11_d40 1.0",
  "bbob-biobj_f88_i12_d02 1.0",
  "bbob-biobj_f88_i12_d03 1.0",
  "bbob-biobj_f88_i12_d05 1.0",
  "bbob-biobj_f88_i12_d10 1.0",
  "bbob-biobj_f88_i12_d20 1.0",
  "bbob-biobj_f88_i12_d40 1.0",
  "bbob-biobj_f88_i13_d02 1.0",
  "bbob-biobj_f88_i13_d03 1.0",
  "bbob-biobj_f88_i13_d05 1.0",
  "bbob-biobj_f88_i13_d10 1.0",
  "bbob-biobj_f88_i13_d20 1.0",
  "bbob-biobj_f88_i13_d40 1.0",
  "bbob-biobj_f88_i14_d02 1.0",
  "bbob-biobj_f88_i14_d03 1.0",
  "bbob-biobj_f88_i14_d05 1.0",
  "bbob-biobj_f88_i14_d10 1.0",
  "bbob-biobj_f88_i14_d20 1.0",
  "bbob-biobj_f88_i14_d40 1.0",
  "bbob-biobj_f88_i15_d02 1.0",
  "bbob-biobj_f88_i15_d03 1.0",
  "bbob-biobj_f88_i15_d05 1.0",
  "bbob-biobj_f88_i15_d10 1.0",
  "bbob-biobj_f88_i15_d20 1.0",
  "bbob-biobj_f88_i15_d40 1.0",
  "bbob-biobj_f89_i01_d02 1.0",
  "bbob-biobj_f89_i01_d03 1.0",
  "bbob-biobj_f89_i01_d05 1.0",
  "bbob-biobj_f89_i01_d10 1.0",
  "bbob-biobj_f89_i01_d20 1.0",
  "bbob-biobj_f89_i01_d40 1.0",
  "bbob-biobj_f89_i02_d02 1.0",
  "bbob-biobj_f89_i02_d03 1.0",
  "bbob-biobj_f89_i02_d05 1.0",
  "bbob-biobj_f89_i02_d10 1.0",
  "bbob-biobj_f89_i02_d20 1.0",
  "bbob-biobj_f89_i02_d40 1.0",
  "bbob-biobj_f89_i03_d02 1.0",
  "bbob-biobj_f89_i03_d03 1.0",
  "bbob-biobj_f89_i03_d05 1.0",
  "bbob-biobj_f89_i03_d10 1.0",
  "bbob-biobj_f89_i03_d20 1.0",
  "bbob-biobj_f89_i03_d40 1.0",
  "bbob-biobj_f89_i04_d02 1.0",
  "bbob-biobj_f89_i04_d03 1.0",
  "bbob-biobj_f89_i04_d05 1.0",
  "bbob-biobj_f89_i04_d10 1.0",
  "bbob-biobj_f89_i04_d20 1.0",
  "bbob-biobj_f89_i04_d40 1.0",
  "bbob-biobj_f89_i05_d02 1.0",
  "bbob-biobj_f89_i05_d03 1.0",
  "bbob-biobj_f89_i05_d05 1.0",
  "bbob-biobj_f89_i05_d10 1.0",
  "bbob-biobj_f89_i05_d20 1.0",
  "bbob-biobj_f89_i05_d40 1.0",
  "bbob-biobj_f89_i06_d02 1.0",
  "bbob-biobj_f89_i06_d03 1.0",
  "bbob-biobj_f89_i06_d05 1.0",
  "bbob-biobj_f89_i06_d10 1.0",
  "bbob-biobj_f89_i06_d20 1.0",
  "bbob-biobj_f89_i06_d40 1.0",
  "bbob-biobj_f89_i07_d02 1.0",
  "bbob-biobj_f89_i07_d03 1.0",
  "bbob-biobj_f89_i07_d05 1.0",
  "bbob-biobj_f89_i07_d10 1.0",
  "bbob-biobj_f89_i07_d20 1.0",
  "bbob-biobj_f89_i07_d40 1.0",
  "bbob-biobj_f89_i08_d02 1.0",
  "bbob-biobj_f89_i08_d03 1.0",
  "bbob-biobj_f89_i08_d05 1.0",
  "bbob-biobj_f89_i08_d10 1.0",
  "bbob-biobj_f89_i08_d20 1.0",
  "bbob-biobj_f89_i08_d40 1.0",
  "bbob-biobj_f89_i09_d02 1.0",
  "bbob-biobj_f89_i09_d03 1.0",
  "bbob-biobj_f89_i09_d05 1.0",
  "bbob-biobj_f89_i09_d10 1.0",
  "bbob-biobj_f89_i09_d20 1.0",
  "bbob-biobj_f89_i09_d40 1.0",
  "bbob-biobj_f89_i10_d02 1.0",
  "bbob-biobj_f89_i10_d03 1.0",
  "bbob-biobj_f89_i10_d05 1.0",
  "bbob-biobj_f89_i10_d10 1.0",
  "bbob-biobj_f89_i10_d20 1.0",
  "bbob-biobj_f89_i10_d40 1.0",
  "bbob-biobj_f89_i11_d02 1.0",
  "bbob-biobj_f89_i11_d03 1.0",
  "bbob-biobj_f89_i11_d05 1.0",
  "bbob-biobj_f89_i11_d10 1.0",
  "bbob-biobj_f89_i11_d20 1.0",
  "bbob-biobj_f89_i11_d40 1.0",
  "bbob-biobj_f89_i12_d02 1.0",
  "bbob-biobj_f89_i12_d03 1.0",
  "bbob-biobj_f89_i12_d05 1.0",
  "bbob-biobj_f89_i12_d10 1.0",
  "bbob-biobj_f89_i12_d20 1.0",
  "bbob-biobj_f89_i12_d40 1.0",
  "bbob-biobj_f89_i13_d02 1.0",
  "bbob-biobj_f89_i13_d03 1.0",
  "bbob-biobj_f89_i13_d05 1.0",
  "bbob-biobj_f89_i13_d10 1.0",
  "bbob-biobj_f89_i13_d20 1.0",
  "bbob-biobj_f89_i13_d40 1.0",
  "bbob-biobj_f89_i14_d02 1.0",
  "bbob-biobj_f89_i14_d03 1.0",
  "bbob-biobj_f89_i14_d05 1.0",
  "bbob-biobj_f89_i14_d10 1.0",
  "bbob-biobj_f89_i14_d20 1.0",
  "bbob-biobj_f89_i14_d40 1.0",
  "bbob-biobj_f89_i15_d02 1.0",
  "bbob-biobj_f89_i15_d03 1.0",
  "bbob-biobj_f89_i15_d05 1.0",
  "bbob-biobj_f89_i15_d10 1.0",
  "bbob-biobj_f89_i15_d20 1.0",
  "bbob-biobj_f89_i15_d40 1.0",
  "bbob-biobj_f90_i01_d02 1.0",
  "bbob-biobj_f90_i01_d03 1.0",
  "bbob-biobj_f90_i01_d05 1.0",
  "bbob-biobj_f90_i01_d10 1.0",
  "bbob-biobj_f90_i01_d20 1.0",
  "bbob-biobj_f90_i01_d40 1.0",
  "bbob-biobj_f90_i02_d02 1.0",
  "bbob-biobj_f90_i02_d03 1.0",
  "bbob-biobj_f90_i02_d05 1.0",
  "bbob-biobj_f90_i02_d10 1.0",
  "bbob-biobj_f90_i02_d20 1.0",
  "bbob-biobj_f90_i02_d40 1.0",
  "bbob-biobj_f90_i03_d02 1.0",
  "bbob-biobj_f90_i03_d03 1.0",
  "bbob-biobj_f90_i03_d05 1.0",
  "bbob-biobj_f90_i03_d10 1.0",
  "bbob-biobj_f90_i03_d20 1.0",
  "bbob-biobj_f90_i03_d40 1.0",
  "bbob-biobj_f90_i04_d02 1.0",
  "bbob-biobj_f90_i04_d03 1.0",
  "bbob-biobj_f90_i04_d05 1.0",
  "bbob-biobj_f90_i04_d10 1.0",
  "bbob-biobj_f90_i04_d20 1.0",
  "bbob-biobj_f90_i04_d40 1.0",
  "bbob-biobj_f90_i05_d02 1.0",
  "bbob-biobj_f90_i05_d03 1.0",
  "bbob-biobj_f90_i05_d05 1.0",
  "bbob-biobj_f90_i05_d10 1.0",
  "bbob-biobj_f90_i05_d20 1.0",
  "bbob-biobj_f90_i05_d40 1.0",
  "bbob-biobj_f90_i06_d02 1.0",
  "bbob-biobj_f90_i06_d03 1.0",
  "bbob-biobj_f90_i06_d05 1.0",
  "bbob-biobj_f90_i06_d10 1.0",
  "bbob-biobj_f90_i06_d20 1.0",
  "bbob-biobj_f90_i06_d40 1.0",
  "bbob-biobj_f90_i07_d02 1.0",
  "bbob-biobj_f90_i07_d03 1.0",
  "bbob-biobj_f90_i07_d05 1.0",
  "bbob-biobj_f90_i07_d10 1.0",
  "bbob-biobj_f90_i07_d20 1.0",
  "bbob-biobj_f90_i07_d40 1.0",
  "bbob-biobj_f90_i08_d02 1.0",
  "bbob-biobj_f90_i08_d03 1.0",
  "bbob-biobj_f90_i08_d05 1.0",
  "bbob-biobj_f90_i08_d10 1.0",
  "bbob-biobj_f90_i08_d20 1.0",
  "bbob-biobj_f90_i08_d40 1.0",
  "bbob-biobj_f90_i09_d02 1.0",
  "bbob-biobj_f90_i09_d03 1.0",
  "bbob-biobj_f90_i09_d05 1.0",
  "bbob-biobj_f90_i09_d10 1.0",
  "bbob-biobj_f90_i09_d20 1.0",
  "bbob-biobj_f90_i09_d40 1.0",
  "bbob-biobj_f90_i10_d02 1.0",
  "bbob-biobj_f90_i10_d03 1.0",
  "bbob-biobj_f90_i10_d05 1.0",
  "bbob-biobj_f90_i10_d10 1.0",
  "bbob-biobj_f90_i10_d20 1.0",
  "bbob-biobj_f90_i10_d40 1.0",
  "bbob-biobj_f90_i11_d02 1.0",
  "bbob-biobj_f90_i11_d03 1.0",
  "bbob-biobj_f90_i11_d05 1.0",
  "bbob-biobj_f90_i11_d10 1.0",
  "bbob-biobj_f90_i11_d20 1.0",
  "bbob-biobj_f90_i11_d40 1.0",
  "bbob-biobj_f90_i12_d02 1.0",
  "bbob-biobj_f90_i12_d03 1.0",
  "bbob-biobj_f90_i12_d05 1.0",
  "bbob-biobj_f90_i12_d10 1.0",
  "bbob-biobj_f90_i12_d20 1.0",
  "bbob-biobj_f90_i12_d40 1.0",
  "bbob-biobj_f90_i13_d02 1.0",
  "bbob-biobj_f90_i13_d03 1.0",
  "bbob-biobj_f90_i13_d05 1.0",
  "bbob-biobj_f90_i13_d10 1.0",
  "bbob-biobj_f90_i13_d20 1.0",
  "bbob-biobj_f90_i13_d40 1.0",
  "bbob-biobj_f90_i14_d02 1.0",
  "bbob-biobj_f90_i14_d03 1.0",
  "bbob-biobj_f90_i14_d05 1.0",
  "bbob-biobj_f90_i14_d10 1.0",
  "bbob-biobj_f90_i14_d20 1.0",
  "bbob-biobj_f90_i14_d40 1.0",
  "bbob-biobj_f90_i15_d02 1.0",
  "bbob-biobj_f90_i15_d03 1.0",
  "bbob-biobj_f90_i15_d05 1.0",
  "bbob-biobj_f90_i15_d10 1.0",
  "bbob-biobj_f90_i15_d20 1.0",
  "bbob-biobj_f90_i15_d40 1.0",
  "bbob-biobj_f91_i01_d02 1.0",
  "bbob-biobj_f91_i01_d03 1.0",
  "bbob-biobj_f91_i01_d05 1.0",
  "bbob-biobj_f91_i01_d10 1.0",
  "bbob-biobj_f91_i01_d20 1.0",
  "bbob-biobj_f91_i01_d40 1.0",
  "bbob-biobj_f91_i02_d02 1.0",
  "bbob-biobj_f91_i02_d03 1.0",
  "bbob-biobj_f91_i02_d05 1.0",
  "bbob-biobj_f91_i02_d10 1.0",
  "bbob-biobj_f91_i02_d20 1.0",
  "bbob-biobj_f91_i02_d40 1.0",
  "bbob-biobj_f91_i03_d02 1.0",
  "bbob-biobj_f91_i03_d03 1.0",
  "bbob-biobj_f91_i03_d05 1.0",
  "bbob-biobj_f91_i03_d10 1.0",
  "bbob-biobj_f91_i03_d20 1.0",
  "bbob-biobj_f91_i03_d40 1.0",
  "bbob-biobj_f91_i04_d02 1.0",
  "bbob-biobj_f91_i04_d03 1.0",
  "bbob-biobj_f91_i04_d05 1.0",
  "bbob-biobj_f91_i04_d10 1.0",
  "bbob-biobj_f91_i04_d20 1.0",
  "bbob-biobj_f91_i04_d40 1.0",
  "bbob-biobj_f91_i05_d02 1.0",
  "bbob-biobj_f91_i05_d03 1.0",
  "bbob-biobj_f91_i05_d05 1.0",
  "bbob-biobj_f91_i05_d10 1.0",
  "bbob-biobj_f91_i05_d20 1.0",
  "bbob-biobj_f91_i05_d40 1.0",
  "bbob-biobj_f91_i06_d02 1.0",
  "bbob-biobj_f91_i06_d03 1.0",
  "bbob-biobj_f91_i06_d05 1.0",
  "bbob-biobj_f91_i06_d10 1.0",
  "bbob-biobj_f91_i06_d20 1.0",
  "bbob-biobj_f91_i06_d40 1.0",
  "bbob-biobj_f91_i07_d02 1.0",
  "bbob-biobj_f91_i07_d03 1.0",
  "bbob-biobj_f91_i07_d05 1.0",
  "bbob-biobj_f91_i07_d10 1.0",
  "bbob-biobj_f91_i07_d20 1.0",
  "bbob-biobj_f91_i07_d40 1.0",
  "bbob-biobj_f91_i08_d02 1.0",
  "bbob-biobj_f91_i08_d03 1.0",
  "bbob-biobj_f91_i08_d05 1.0",
  "bbob-biobj_f91_i08_d10 1.0",
  "bbob-biobj_f91_i08_d20 1.0",
  "bbob-biobj_f91_i08_d40 1.0",
  "bbob-biobj_f91_i09_d02 1.0",
  "bbob-biobj_f91_i09_d03 1.0",
  "bbob-biobj_f91_i09_d05 1.0",
  "bbob-biobj_f91_i09_d10 1.0",
  "bbob-biobj_f91_i09_d20 1.0",
  "bbob-biobj_f91_i09_d40 1.0",
  "bbob-biobj_f91_i10_d02 1.0",
  "bbob-biobj_f91_i10_d03 1.0",
  "bbob-biobj_f91_i10_d05 1.0",
  "bbob-biobj_f91_i10_d10 1.0",
  "bbob-biobj_f91_i10_d20 1.0",
  "bbob-biobj_f91_i10_d40 1.0",
  "bbob-biobj_f91_i11_d02 1.0",
  "bbob-biobj_f91_i11_d03 1.0",
  "bbob-biobj_f91_i11_d05 1.0",
  "bbob-biobj_f91_i11_d10 1.0",
  "bbob-biobj_f91_i11_d20 1.0",
  "bbob-biobj_f91_i11_d40 1.0",
  "bbob-biobj_f91_i12_d02 1.0",
  "bbob-biobj_f91_i12_d03 1.0",
  "bbob-biobj_f91_i12_d05 1.0",
  "bbob-biobj_f91_i12_d10 1.0",
  "bbob-biobj_f91_i12_d20 1.0",
  "bbob-biobj_f91_i12_d40 1.0",
  "bbob-biobj_f91_i13_d02 1.0",
  "bbob-biobj_f91_i13_d03 1.0",
  "bbob-biobj_f91_i13_d05 1.0",
  "bbob-biobj_f91_i13_d10 1.0",
  "bbob-biobj_f91_i13_d20 1.0",
  "bbob-biobj_f91_i13_d40 1.0",
  "bbob-biobj_f91_i14_d02 1.0",
  "bbob-biobj_f91_i14_d03 1.0",
  "bbob-biobj_f91_i14_d05 1.0",
  "bbob-biobj_f91_i14_d10 1.0",
  "bbob-biobj_f91_i14_d20 1.0",
  "bbob-biobj_f91_i14_d40 1.0",
  "bbob-biobj_f91_i15_d02 1.0",
  "bbob-biobj_f91_i15_d03 1.0",
  "bbob-biobj_f91_i15_d05 1.0",
  "bbob-biobj_f91_i15_d10 1.0",
  "bbob-biobj_f91_i15_d20 1.0",
  "bbob-biobj_f91_i15_d40 1.0",
  "bbob-biobj_f92_i01_d02 1.0",
  "bbob-biobj_f92_i01_d03 1.0",
  "bbob-biobj_f92_i01_d05 1.0",
  "bbob-biobj_f92_i01_d10 1.0",
  "bbob-biobj_f92_i01_d20 1.0",
  "bbob-biobj_f92_i01_d40 1.0",
  "bbob-biobj_f92_i02_d02 1.0",
  "bbob-biobj_f92_i02_d03 1.0",
  "bbob-biobj_f92_i02_d05 1.0",
  "bbob-biobj_f92_i02_d10 1.0",
  "bbob-biobj_f92_i02_d20 1.0",
  "bbob-biobj_f92_i02_d40 1.0",
  "bbob-biobj_f92_i03_d02 1.0",
  "bbob-biobj_f92_i03_d03 1.0",
  "bbob-biobj_f92_i03_d05 1.0",
  "bbob-biobj_f92_i03_d10 1.0",
  "bbob-biobj_f92_i03_d20 1.0",
  "bbob-biobj_f92_i03_d40 1.0",
  "bbob-biobj_f92_i04_d02 1.0",
  "bbob-biobj_f92_i04_d03 1.0",
  "bbob-biobj_f92_i04_d05 1.0",
  "bbob-biobj_f92_i04_d10 1.0",
  "bbob-biobj_f92_i04_d20 1.0",
  "bbob-biobj_f92_i04_d40 1.0",
  "bbob-biobj_f92_i05_d02 1.0",
  "bbob-biobj_f92_i05_d03 1.0",
  "bbob-biobj_f92_i05_d05 1.0",
  "bbob-biobj_f92_i05_d10 1.0",
  "bbob-biobj_f92_i05_d20 1.0",
  "bbob-biobj_f92_i05_d40 1.0",
  "bbob-biobj_f92_i06_d02 1.0",
  "bbob-biobj_f92_i06_d03 1.0",
  "bbob-biobj_f92_i06_d05 1.0",
  "bbob-biobj_f92_i06_d10 1.0",
  "bbob-biobj_f92_i06_d20 1.0",
  "bbob-biobj_f92_i06_d40 1.0",
  "bbob-biobj_f92_i07_d02 1.0",
  "bbob-biobj_f92_i07_d03 1.0",
  "bbob-biobj_f92_i07_d05 1.0",
  "bbob-biobj_f92_i07_d10 1.0",
  "bbob-biobj_f92_i07_d20 1.0",
  "bbob-biobj_f92_i07_d40 1.0",
  "bbob-biobj_f92_i08_d02 1.0",
  "bbob-biobj_f92_i08_d03 1.0",
  "bbob-biobj_f92_i08_d05 1.0",
  "bbob-biobj_f92_i08_d10 1.0",
  "bbob-biobj_f92_i08_d20 1.0",
  "bbob-biobj_f92_i08_d40 1.0",
  "bbob-biobj_f92_i09_d02 1.0",
  "bbob-biobj_f92_i09_d03 1.0",
  "bbob-biobj_f92_i09_d05 1.0",
  "bbob-biobj_f92_i09_d10 1.0",
  "bbob-biobj_f92_i09_d20 1.0",
  "bbob-biobj_f92_i09_d40 1.0",
  "bbob-biobj_f92_i10_d02 1.0",
  "bbob-biobj_f92_i10_d03 1.0",
  "bbob-biobj_f92_i10_d05 1.0",
  "bbob-biobj_f92_i10_d10 1.0",
  "bbob-biobj_f92_i10_d20 1.0",
  "bbob-biobj_f92_i10_d40 1.0",
  "bbob-biobj_f92_i11_d02 1.0",
  "bbob-biobj_f92_i11_d03 1.0",
  "bbob-biobj_f92_i11_d05 1.0",
  "bbob-biobj_f92_i11_d10 1.0",
  "bbob-biobj_f92_i11_d20 1.0",
  "bbob-biobj_f92_i11_d40 1.0",
  "bbob-biobj_f92_i12_d02 1.0",
  "bbob-biobj_f92_i12_d03 1.0",
  "bbob-biobj_f92_i12_d05 1.0",
  "bbob-biobj_f92_i12_d10 1.0",
  "bbob-biobj_f92_i12_d20 1.0",
  "bbob-biobj_f92_i12_d40 1.0",
  "bbob-biobj_f92_i13_d02 1.0",
  "bbob-biobj_f92_i13_d03 1.0",
  "bbob-biobj_f92_i13_d05 1.0",
  "bbob-biobj_f92_i13_d10 1.0",
  "bbob-biobj_f92_i13_d20 1.0",
  "bbob-biobj_f92_i13_d40 1.0",
  "bbob-biobj_f92_i14_d02 1.0",
  "bbob-biobj_f92_i14_d03 1.0",
  "bbob-biobj_f92_i14_d05 1.0",
  "bbob-biobj_f92_i14_d10 1.0",
  "bbob-biobj_f92_i14_d20 1.0",
  "bbob-biobj_f92_i14_d40 1.0",
  "bbob-biobj_f92_i15_d02 1.0",
  "bbob-biobj_f92_i15_d03 1.0",
  "bbob-biobj_f92_i15_d05 1.0",
  "bbob-biobj_f92_i15_d10 1.0",
  "bbob-biobj_f92_i15_d20 1.0",
  "bbob-biobj_f92_i15_d40 1.0"
};
#line 15 "code-experiments/src/suite_biobj_utilities.c"

/**
 * @brief The array of triples biobj_instance - problem1_instance - problem2_instance connecting bi-objective
 * suite instances to the instances of the bbob suite.
 *
 * It should be updated with new instances when/if they are chosen.
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
    { 10, 21, 22 },
    { 11, 23, 24 },
    { 12, 25, 26 },
    { 13, 27, 28 },
    { 14, 29, 30 },
    { 15, 31, 34 }
}; 
 
/**
 * @brief A structure containing information about the new instances.
 */
typedef struct {

  size_t **new_instances;    /**< @brief A matrix of new instances (equal in form to suite_biobj_instances)
                                   that needs to be used only when an instance that is not in
                                   suite_biobj_instances is being invoked. */

  size_t max_new_instances;  /**< @brief The maximal number of new instances. */

} suite_biobj_new_inst_t;

/**
 * @brief  Frees the memory of the given suite_biobj_new_inst_t object.
 */
static void suite_biobj_new_inst_free(void *stuff) {

  suite_biobj_new_inst_t *data;
  size_t i;

  assert(stuff != NULL);
  data = (suite_biobj_new_inst_t *) stuff;

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
 * @brief  Performs a few checks and returns whether the two given problem instances should break the search
 * for new instances in suite_biobj_get_new_instance().
 */
static int suite_biobj_check_inst_consistency(const size_t dimension,
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
        "suite_biobj_check_inst_consistency(): The ideal and nadir points of %s are too close in the objective space",
        problem->problem_id);
    coco_debug("norm = %e, ideal = %e\t%e, nadir = %e\t%e", norm, problem->best_value[0],
        problem->best_value[1], problem->nadir_value[0], problem->nadir_value[1]);
    break_search = 1;
  }

  /* Check whether the extreme optimal points are too close in the decision space */
  norm = mo_get_norm(problem1->best_parameter, problem2->best_parameter, problem->number_of_variables);
  if (norm < apart_enough) {
    coco_debug(
        "suite_biobj_check_inst_consistency(): The extreme points of %s are too close in the decision space",
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
 * coco_warning, so that the user can see it and eventually manually add it to suite_biobj_instances.
 */
static size_t suite_biobj_get_new_instance(suite_biobj_new_inst_t *new_inst_data,
                                           const size_t instance,
                                           const size_t instance1,
                                           const size_t *bbob_functions,
                                           const size_t num_bbob_functions,
                                           const size_t *sel_bbob_functions,
                                           const size_t num_sel_bbob_functions,
                                           const size_t *dimensions,
                                           const size_t num_dimensions) {

  size_t instance2 = 0;
  size_t num_tries = 0;
  const size_t max_tries = 1000;
  int appropriate_instance_found = 0, break_search, warning_produced = 0;
  size_t d, f1, f2, i;
  size_t function1, function2, dimension;

  while ((!appropriate_instance_found) && (num_tries < max_tries)) {
    num_tries++;
    instance2 = instance1 + num_tries;
    break_search = 0;

    /* An instance is "appropriate" if the ideal and nadir points in the objective space and the two
     * extreme optimal points in the decisions space are apart enough for all problems (all dimensions
     * and function combinations); therefore iterate over all dimensions and function combinations */

    for (f1 = 0; (f1 < num_bbob_functions-1) && !break_search; f1++) {
      function1 = bbob_functions[f1];
      for (f2 = f1+1; (f2 < num_bbob_functions) && !break_search; f2++) {
        function2 = bbob_functions[f2];
        for (d = 0; (d < num_dimensions) && !break_search; d++) {
          dimension = dimensions[d];

          if (dimension == 0) {
            if (!warning_produced)
              coco_warning("suite_biobj_get_new_instance(): remove filtering of dimensions to get generally acceptable instances!");
            warning_produced = 1;
            continue;
          }

          break_search = suite_biobj_check_inst_consistency(dimension, function1, instance1, function2, instance2);
        }
      }
    }

    /* Finally, check all functions (f,f) with f in {f1, f2, f6, f8, f13, f14, f15, f17, f20, f21}: */
    for (f1 = 0; (f1 < num_sel_bbob_functions) && !break_search; f1++) {
      function1 = sel_bbob_functions[f1];
      function2 = sel_bbob_functions[f1];
      for (d = 0; (d < num_dimensions) && !break_search; d++) {
        dimension = dimensions[d];

        if (dimension == 0) {
            if (!warning_produced)
              coco_warning("suite_biobj_get_new_instance(): remove filtering of dimensions to get generally acceptable instances!");
            warning_produced = 1;
            continue;
          }

        break_search = suite_biobj_check_inst_consistency(dimension, function1, instance1, function2, instance2);
      }
    }

    if (break_search) {
      /* The search was broken, continue with next instance2 */
      continue;
    } else {
      /* An appropriate instance was found */
      appropriate_instance_found = 1;
      coco_info("suite_biobj_get_new_instance(): Instance %lu created from instances %lu and %lu",
          (unsigned long) instance, (unsigned long) instance1, (unsigned long) instance2);

      /* Save the instance to new_instances */
      for (i = 0; i < new_inst_data->max_new_instances; i++) {
        if (new_inst_data->new_instances[i][0] == 0) {
          new_inst_data->new_instances[i][0] = instance;
          new_inst_data->new_instances[i][1] = instance1;
          new_inst_data->new_instances[i][2] = instance2;
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
 * @brief Creates and returns a bi-objective problem without needing a suite.
 *
 * Useful for creating suites based on the bi-objective problems.
 *
 * Creates the bi-objective problem by constructing it from two single-objective problems. If the
 * invoked instance number is not in suite_biobj_instances, the function uses the following formula
 * to construct a new appropriate instance:
 *   problem1_instance = 2 * biobj_instance + 1
 *   problem2_instance = problem1_instance + 1
 *
 * If needed, problem2_instance is increased (see also the explanation in suite_biobj_get_new_instance).
 *
 * @param function Function
 * @param dimension Dimension
 * @param instance Instance
 * @param coco_get_problem_function The function that is used to access the single-objective problem.
 * @param new_inst_data Structure containing information on new instance data.
 * @param num_new_instances The number of new instances.
 * @param dimensions An array of dimensions to take into account when creating new instances.
 * @param num_dimensions The number of dimensions to take into account when creating new instances.
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *coco_get_biobj_problem(const size_t function,
                                              const size_t dimension,
                                              const size_t instance,
                                              const coco_get_problem_function_t coco_get_problem_function,
                                              suite_biobj_new_inst_t **new_inst_data,
                                              const size_t num_new_instances,
                                              const size_t *dimensions,
                                              const size_t num_dimensions) {
  
  /* Selected functions from the bbob suite that are used to construct the original bbob-biobj suite. */
  const size_t sel_bbob_functions[] = { 1, 2, 6, 8, 13, 14, 15, 17, 20, 21 };
  const size_t num_sel_bbob_functions = 10;
  /* All functions from the bbob suite for later use during instance generation. */
  const size_t all_bbob_functions[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
  const size_t num_all_bbob_functions = 24;
  
  coco_problem_t *problem1 = NULL, *problem2 = NULL, *problem = NULL;
  size_t instance1 = 0, instance2 = 0;
  size_t function1_idx, function2_idx;
  const size_t function_idx = function - 1;

  size_t i, j;
  const size_t num_existing_instances = sizeof(suite_biobj_instances) / sizeof(suite_biobj_instances[0]);
  int instance_found = 0;

  double *smallest_values_of_interest = coco_allocate_vector_with_value(dimension, -100);
  double *largest_values_of_interest = coco_allocate_vector_with_value(dimension, 100);
  
  /* Determine the corresponding single-objective function indices */
  if (function_idx < 55) {
    /* A "magic" formula to compute the BBOB function index from the bi-objective function index */
    function1_idx = num_sel_bbob_functions
        - coco_double_to_size_t(
            floor(-0.5 + sqrt(0.25 + 2.0 * (double) (55 - function_idx - 1)))) - 1;
    function2_idx = function_idx - (function1_idx * num_sel_bbob_functions) +
        (function1_idx * (function1_idx + 1)) / 2;
        
  } else {
    /* There is not a simple "magic" formula for functions >= 55 */
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
      
  /* Determine the instances */

  /* First search for the instance in suite_biobj_instances */
  for (i = 0; i < num_existing_instances; i++) {
    if (suite_biobj_instances[i][0] == instance) {
      /* The instance has been found in suite_biobj_instances */
      instance1 = suite_biobj_instances[i][1];
      instance2 = suite_biobj_instances[i][2];
      instance_found = 1;
      break;
    }
  }

  if ((!instance_found) && ((*new_inst_data) != NULL)) {
    /* Next, search for instance in new_instances */
    for (i = 0; i < (*new_inst_data)->max_new_instances; i++) {
      if ((*new_inst_data)->new_instances[i][0] == 0)
        break;
      if ((*new_inst_data)->new_instances[i][0] == instance) {
        /* The instance has been found in new_instances */
        instance1 = (*new_inst_data)->new_instances[i][1];
        instance2 = (*new_inst_data)->new_instances[i][2];
        instance_found = 1;
        break;
      }
    }
  }

  if (!instance_found) {
    /* Finally, if the instance is not found, create a new one */

    if ((*new_inst_data) == NULL) {
      /* Allocate space needed for saving new instances */
      (*new_inst_data) = (suite_biobj_new_inst_t *) coco_allocate_memory(sizeof(**new_inst_data));

      /* Most often the actual number of new instances will be lower than max_new_instances, because
       * some of them are already in suite_biobj_instances. However, in order to avoid iterating over
       * suite_biobj_new_inst_t, the allocation uses max_new_instances. */
      (*new_inst_data)->max_new_instances = num_new_instances;

      (*new_inst_data)->new_instances = (size_t **) coco_allocate_memory((*new_inst_data)->max_new_instances * sizeof(size_t *));
      for (i = 0; i < (*new_inst_data)->max_new_instances; i++) {
        (*new_inst_data)->new_instances[i] = (size_t *) malloc(3 * sizeof(size_t));
        for (j = 0; j < 3; j++) {
          (*new_inst_data)->new_instances[i][j] = 0;
        }
      }
    }

    /* A simple formula to set the first instance */
    instance1 = 2 * instance + 1;
    instance2 = suite_biobj_get_new_instance((*new_inst_data), instance, instance1, all_bbob_functions,
        num_all_bbob_functions, sel_bbob_functions, num_sel_bbob_functions, dimensions, num_dimensions);
  }
  
  /* Construct the problem based on the function index and dimension */
  if (function_idx < 55) {
    problem1 = coco_get_problem_function(sel_bbob_functions[function1_idx], dimension, instance1);
    problem2 = coco_get_problem_function(sel_bbob_functions[function2_idx], dimension, instance2);

  } else {
    problem1 = coco_get_problem_function(all_bbob_functions[function1_idx], dimension, instance1);
    problem2 = coco_get_problem_function(all_bbob_functions[function2_idx], dimension, instance2);
  }
  
  problem = coco_problem_stacked_allocate(problem1, problem2, smallest_values_of_interest, largest_values_of_interest);

  /* Use the standard stacked problem_id as problem_name and construct a new problem_id */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "bbob-biobj_f%03lu_i%02lu_d%02lu", (unsigned long) function,
      (unsigned long) instance, (unsigned long) dimension);

  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem1->problem_type, problem2->problem_type);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
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

    curr_key = coco_allocate_string(COCO_PATH_MAX + 1);
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
#line 25 "code-experiments/src/suite_biobj.c"
#line 26 "code-experiments/src/suite_biobj.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj suites.
 */
static coco_suite_t *suite_biobj_initialize(const char *suite_name) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  if (strcmp(suite_name, "bbob-biobj") == 0) {
    suite = coco_suite_allocate("bbob-biobj", 55, 6, dimensions, "instances: 1-15");
  } else if (strcmp(suite_name, "bbob-biobj-ext") == 0) {
    suite = coco_suite_allocate("bbob-biobj-ext", 55+37, 6, dimensions, "instances: 1-15");
  } else {
    coco_error("suite_biobj_initialize(): unknown problem suite");
    return NULL;
  }

  suite->data_free_function = suite_biobj_new_inst_free;

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj suites.
 *
 * @note The instances of the bi-objective suites generally do not changes with years.
 */
static const char *suite_biobj_get_instances_by_year(const int year) {

  if ((year == 2016) || (year == 0000)) { /* test case */
    return "1-10";
  }
  else
    return "1-15";
}

/**
 * @brief Returns the problem from the bbob-biobj suite that corresponds to the given parameters.
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

  coco_problem_t *problem = NULL;
  suite_biobj_new_inst_t *new_inst_data = (suite_biobj_new_inst_t *) suite->data;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = coco_get_biobj_problem(function, dimension, instance, coco_get_bbob_problem, &new_inst_data,
      suite->number_of_instances, suite->dimensions, suite->number_of_dimensions);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}

#line 21 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/suite_biobj_mixint.c"
/**
 * @file suite_biobj_mixint.c
 * @brief Implementation of a bi-objective mixed-integer suite. The functions are the same as those
 * in the bbob-biobj-ext suite with 92 functions, but the large-scale implementations of the
 * functions are used instead of the original ones.
 */

#line 9 "code-experiments/src/suite_biobj_mixint.c"
#line 10 "code-experiments/src/suite_biobj_mixint.c"
#line 11 "code-experiments/src/suite_biobj_mixint.c"
#line 12 "code-experiments/src/suite_biobj_mixint.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob-biobj-mixint suite.
 */
static coco_suite_t *suite_biobj_mixint_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 5, 10, 20, 40, 80, 160 };

  /* TODO: Use also dimensions 80 and 160 (change the 4 below into a 6) */
  suite = coco_suite_allocate("bbob-biobj-mixint", 92, 4, dimensions, "instances: 1-15");
  suite->data_free_function = suite_biobj_new_inst_free;

  return suite;
}

/**
 * @brief Sets the instances associated with years for the bbob-biobj-mixint suites.
 *
 * @note The instances of the bi-objective suites generally do not changes with years.
 */
static const char *suite_biobj_mixint_get_instances_by_year(const int year) {

  if ((year == 2016) || (year == 0000)) { /* test case */
    return "1-10";
  }
  else
    return "1-15";
}

/**
 * @brief Creates and returns a mixed-integer bi-objective bbob problem without needing the actual
 * bbob-mixint suite.
 *
 * The problem is constructed by first finding the underlying single-objective continuous problems,
 * then discretizing the problems and finally stacking them to get a bi-objective mixed-integer problem.
 *
 * @param function Function
 * @param dimension Dimension
 * @param instance Instance
 * @param coco_get_problem_function The function that is used to access the single-objective problem.
 * @param new_inst_data Structure containing information on new instance data.
 * @param num_new_instances The number of new instances.
 * @param dimensions An array of dimensions to take into account when creating new instances.
 * @param num_dimensions The number of dimensions to take into account when creating new instances.
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *coco_get_biobj_mixint_problem(const size_t function,
                                                     const size_t dimension,
                                                     const size_t instance,
                                                     const coco_get_problem_function_t coco_get_problem_function,
                                                     suite_biobj_new_inst_t **new_inst_data,
                                                     const size_t num_new_instances,
                                                     const size_t *dimensions,
                                                     const size_t num_dimensions) {

  coco_problem_t *problem_cont = NULL, *problem = NULL;
  coco_problem_t *problem1, *problem2;
  coco_problem_t *problem1_mixint, *problem2_mixint;
  coco_problem_t *problem1_cont, *problem2_cont;

  double *smallest_values_of_interest = coco_allocate_vector(dimension);
  double *largest_values_of_interest = coco_allocate_vector(dimension);

  size_t i, j;
  size_t num_integer = dimension;
  /* TODO: Use the correct cardinality!
   * The cardinality of variables (0 = continuous variables should always come last) */
  const size_t variable_cardinality[] = { 2, 4, 8, 16, 0 };

  if (dimension % 5 != 0)
    coco_error("coco_get_biobj_mixint_problem(): dimension %lu not supported for suite_bbob_mixint", dimension);

  /* Sets the ROI according to the given cardinality of variables */
  for (i = 0; i < dimension; i++) {
    j = i / (dimension / 5);
    if (variable_cardinality[j] == 0) {
      smallest_values_of_interest[i] = -5;
      largest_values_of_interest[i] = 5;
      if (num_integer == dimension)
        num_integer = i;
    }
    else {
      smallest_values_of_interest[i] = 0;
      largest_values_of_interest[i] = (double)variable_cardinality[j] - 1;
    }
  }

  /* First, find the underlying single-objective continuous problems */
  problem_cont = coco_get_biobj_problem(function, dimension, instance, coco_get_problem_function, new_inst_data,
      num_new_instances, dimensions, num_dimensions);
  assert(problem_cont != NULL);
  problem1_cont = ((coco_problem_stacked_data_t *) problem_cont->data)->problem1;
  problem2_cont = ((coco_problem_stacked_data_t *) problem_cont->data)->problem2;
  problem1 = coco_problem_duplicate(problem1_cont);
  problem2 = coco_problem_duplicate(problem2_cont);
  assert(problem1);
  assert(problem2);
  /* Copy also the data of the underlying problems and set all pointers in such a way that
   * problem_cont can be safely freed */
  problem1->data = problem1_cont->data;
  problem2->data = problem2_cont->data;
  problem1_cont->data = NULL;
  problem2_cont->data = NULL;
  problem1_cont->problem_free_function = NULL;
  problem2_cont->problem_free_function = NULL;
  coco_problem_free(problem_cont);

  /* Second, discretize the single-objective problems */
  problem1_mixint = transform_vars_discretize(problem1, smallest_values_of_interest,
      largest_values_of_interest, num_integer);
  problem2_mixint = transform_vars_discretize(problem2, smallest_values_of_interest,
      largest_values_of_interest, num_integer);

  /* Third, combine the problems in a bi-objective mixed-integer problem */
  problem = coco_problem_stacked_allocate(problem1_mixint, problem2_mixint, smallest_values_of_interest,
      largest_values_of_interest);

  /* Use the standard stacked problem_id as problem_name and construct a new problem_id */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "bbob-biobj-mixint_f%03lu_i%02lu_d%02lu", (unsigned long) function,
      (unsigned long) instance, (unsigned long) dimension);

  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem1->problem_type, problem2->problem_type);

  coco_free_memory(smallest_values_of_interest);
  coco_free_memory(largest_values_of_interest);

  return problem;
}
/**
 * @brief Returns the problem from the bbob-biobj-mixint suite that corresponds to the given parameters.
 *
 * Uses large-scale bbob functions if dimension is equal or larger than the hard-coded dim_large_scale
 * value (50).
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_biobj_mixint_get_problem(coco_suite_t *suite,
                                                      const size_t function_idx,
                                                      const size_t dimension_idx,
                                                      const size_t instance_idx) {

  coco_problem_t *problem = NULL;
  suite_biobj_new_inst_t *new_inst_data = (suite_biobj_new_inst_t *) suite->data;
  const size_t dim_large_scale = 50; /* Switch to large-scale functions for dimensions over 50 */

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  if (dimension < dim_large_scale)
    problem = coco_get_biobj_mixint_problem(function, dimension, instance, coco_get_bbob_problem,
        &new_inst_data, suite->number_of_instances, suite->dimensions, suite->number_of_dimensions);
  else
    problem = coco_get_biobj_mixint_problem(function, dimension, instance, mock_coco_get_largescale_problem,
        &new_inst_data, suite->number_of_instances, suite->dimensions, suite->number_of_dimensions);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}

#line 22 "code-experiments/src/coco_suite.c"
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
    coco_error("suite_toy_get_problem(): function %lu does not exist in this suite", (unsigned long) function);
    return NULL; /* Never reached */
  }

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
#line 23 "code-experiments/src/coco_suite.c"
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
    coco_error("coco_get_largescale_problem(): cannot retrieve problem f%lu instance %lu in %luD",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension);
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
#line 24 "code-experiments/src/coco_suite.c"
#line 1 "code-experiments/src/suite_cons_bbob.c"
/**
 * @file  suite_cons_bbob.c
 * @brief Implementation of the constrained bbob suite containing 
 *        48 constrained problems in 6 dimensions. See comments in
 *        "suite_cons_bbob_problems.c" for more details.
 */

#line 9 "code-experiments/src/suite_cons_bbob.c"
#line 1 "code-experiments/src/suite_cons_bbob_problems.c"
/**
 * @file  suite_cons_bbob_problems.c
 * @brief Implementation of the problems in the constrained BBOB suite.
 * 
 * This suite contains 48 constrained functions in continuous domain 
 * which are derived from combining 8 single-objective functions of the
 * noiseless bbob test suite with randomly-generated 
 * linear constraints perturbed by nonlinear transformations.
 * Each one of the 8 functions is combined with 6 different numbers of 
 * constraints: 1, 2, 6, 6+n/2, 6+n and 6+3n.
 * 
 * We consider constrained optimization problems of the form
 * 
 *     min f(x) s.t. g_i(x) <= 0, i=1,...,l.
 * 
 * The constrained functions are built in a way such that the 
 * KKT conditions are satisied at the origin, which is the initial
 * optimal solution. We then translate the constrained function
 * by a random vector in order to move the optimal solution away
 * from the origin. 
 * 
 * ***************************************************
 * General idea for building the constrained functions
 * ***************************************************
 * 
 * 1. Choose a bbob function f whose raw version is pseudoconvex
 *    to be the objective function. (see note below)
 * 2. Remove possible nonlinear transformations from f. (see note below)
 * 3. In order to make sure that the feasible set is not empty,
 *    we choose a direction p that should be feasible.
 *    Define p as the gradient of f at the origin.
 * 4. Define the first constraint g_1(x)=a_1'*x by setting its gradient
 *    to a_1 = -p. By definition, g_1(p) < 0. Thus p is feasible
 *    for g_1.
 * 5. Generate the other constraints randomly while making sure that 
 *    p remains feasible for each one.
 * 6. Apply to the whole constrained function the nonlinear transformations 
 *    that were removed from the objective function in Step 2. (see note below)
 * 7. Choose a random point xopt and move the optimum away from the 
 *    origin to xopt by translating the constrained function by -xopt.
 * 
 * @note The pseudoconvexity in Step 1 guarantees that the KKT conditions 
 *       are sufficient for optimality.
 * 
 * @note The removal of possible nonlinear transformations in Step 2
 *       and posterior application in Step 6 are necessary to make
 *       sure that the KKT conditions are satisfied in the optimum
 *       - until then the origin. As explained in the documentation, 
 *       the application of the nonlinear transformations in Step 6 
 *       does not affect the location of the optimum.
 * 
 * @note a_1 is set to -p, i.e. the negative gradient of f at the origin, 
 *       in order to have the KKT conditions easily satisfied.
 *
 * @note Steps 1 and 2 are done within the 'allocate' function of the
 *       objective function, e.g. f_ellipsoid_cons_bbob_problem_allocate().
 * 
 * @note Steps 4 and 5 are done within c_linear_cons_bbob_problem_allocate().
 * 
 * @note The constrained Rastrigin function's construction differs a bit
 *       from the steps above. Since it is a multimodal function with
 *       well distributed local optima, we choose one of its local optima
 *       to be the global constrained optimum by adding constraints 
 *       that pass through that point.
 * 
 * @note The testbed provides the user an initial solution which is given
 *       by the feasible direction p scaled by some constant.
 * 
 * *************************************************
 * COCO data structure for the constrained functions
 * *************************************************
 * 
 * First, we create a coco_problem_t object for the objective function.
 * Then, coco_problem_t objects are created for each constraint and
 * stacked together into a single coco_problem_t object (see "c_linear.c"). 
 * Finally, the coco_problem_t object containing the constraints and 
 * the coco_problem_t object containing the objective function are 
 * stacked together to form the constrained function.
 * 
 */

#include <math.h>

#line 85 "code-experiments/src/suite_cons_bbob_problems.c"
#line 86 "code-experiments/src/suite_cons_bbob_problems.c"
#line 87 "code-experiments/src/suite_cons_bbob_problems.c"
#line 1 "code-experiments/src/c_linear.c"
/**
 * @file  c_linear.c
 * @brief Implements the linear constraints for the suite of 
 *        constrained problems.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#line 14 "code-experiments/src/c_linear.c"
#line 15 "code-experiments/src/c_linear.c"
/**
 * @brief Data type for the linear constraints.
 */
typedef struct {
  double *gradient;
  double *x;
} linear_constraint_data_t;	

static void c_sum_variables_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y);
                                     
static void c_linear_single_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y);
                                        
static coco_problem_t *c_guarantee_feasible_point(coco_problem_t *problem,
                                                  const double *feasible_point);
                                               
static void c_linear_gradient_free(void *thing);

static coco_problem_t *c_sum_variables_allocate(const size_t number_of_variables);

static coco_problem_t *c_linear_transform(coco_problem_t *inner_problem, 
                                          const double *gradient);
         
static coco_problem_t *c_linear_single_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t constraint_number,
                                                      const double factor1,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      double *gradient,
                                                      const double *feasible_direction);
                                                      
static coco_problem_t *c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      const double *feasible_direction);

/**
 * @brief Evaluates the linear constraint with all-ones gradient at
 *        the point 'x' and stores the result into 'y'.
 */
static void c_sum_variables_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y) {
	
  size_t i;

  assert(self->number_of_constraints == 1);
  
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i)
    y[0] += x[i];
}	

/**
 * @brief Evaluates the linear constraint at the point 'x' and stores
 *        the result in 'y'.
 */
static void c_linear_single_evaluate(coco_problem_t *self, 
                                     const double *x, 
                                     double *y) {
	
  size_t i;
  
  linear_constraint_data_t *data;
  coco_problem_t *inner_problem;
  
  data = (linear_constraint_data_t *) coco_problem_transformed_get_data(self);
  inner_problem = coco_problem_transformed_get_inner_problem(self);
  
  assert(self->number_of_constraints == 1);
			
  for (i = 0; i < self->number_of_variables; ++i)
    data->x[i] = (data->gradient[i])*x[i];
  
  coco_evaluate_constraint(inner_problem, data->x, y);
  
  inner_problem = NULL;
  data = NULL;
}

/**
 * @brief Guarantees that "feasible_direction" is feasible w.r.t. 
 *        the constraint in "problem" and records it as the 
 *        initial feasible solution to this coco_problem.
 */
static coco_problem_t *c_guarantee_feasible_point(coco_problem_t *problem,
                                                const double *feasible_direction) {
  
  size_t i;
  linear_constraint_data_t *data;
  double constraint_value = 0.0;
  
  data = (linear_constraint_data_t *) coco_problem_transformed_get_data(problem);
  
  assert(problem->number_of_constraints == 1);
  
  /* Let p be the gradient of the constraint in "problem".
   * Check whether p' * (feasible_direction) <= 0.
   */
  coco_evaluate_constraint(problem, feasible_direction, &constraint_value);
  
  /* Flip the constraint in "problem" if feasible_direction
   * is not feasible w.r.t. the constraint in "problem".
   */
  if (constraint_value > 0)
    for (i = 0; i < problem->number_of_variables; ++i)
      data->gradient[i] *= -1.0;
          
  problem->initial_solution = coco_duplicate_vector(feasible_direction, 
      problem->number_of_variables);
 
  data = NULL;  
  return problem;
}

/**
 * @brief Frees the data object.
 */
static void c_linear_gradient_free(void *thing) {
	
  linear_constraint_data_t *data = (linear_constraint_data_t *) thing;
  coco_free_memory(data->gradient);
  coco_free_memory(data->x);
}

/**
 * @brief Allocates a linear constraint coco_problem_t with all-ones gradient.
 */
static coco_problem_t *c_sum_variables_allocate(const size_t number_of_variables) {

  size_t i;
  coco_problem_t *problem = coco_problem_allocate(number_of_variables, 0, 1);

  problem->evaluate_constraint = c_sum_variables_evaluate;
  
  coco_problem_set_id(problem, "%s_d%02lu", "linearconstraint", number_of_variables);

  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
  }
  return problem;
}

/**
 * @brief Transforms a linear constraint with all-ones gradient
 *        into a linear constraint whose gradient is passed 
 *        as argument.
 */
static coco_problem_t *c_linear_transform(coco_problem_t *inner_problem, 
                                          const double *gradient) {
  
  linear_constraint_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->gradient = coco_duplicate_vector(gradient, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  self = coco_problem_transformed_allocate(inner_problem, data, 
      c_linear_gradient_free, "gradient_linear_constraint");
  self->evaluate_constraint = c_linear_single_evaluate;

  return self;
}

/**
 * @brief Builds a coco_problem_t containing one single linear constraint.
 * 
 * This function is called by c_linear_cons_bbob_problem_allocate(),
 * the central function that stacks all the constraints built by
 * c_linear_single_cons_bbob_problem_allocate() into one single
 * coco_problem_t object.
 */
static coco_problem_t *c_linear_single_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t constraint_number,
                                                      const double factor1,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      double *gradient,
                                                      const double *feasible_direction) {
																			
  size_t i;
  
  double *gradient_linear_constraint = NULL;
  coco_problem_t *problem = NULL;
  coco_random_state_t *random_generator;
  long seed_cons_i;
  double factor2;
  
  problem = c_sum_variables_allocate(dimension);
  
  seed_cons_i = (long)(function + 10000 * instance 
                                + 50000 * constraint_number);
  random_generator = coco_random_new((uint32_t) seed_cons_i);
  
  /* The constraints gradients are scaled with random numbers
   * 10**U[0,1] and 10**U_i[0,2], where U[a, b] is uniform in [a,b] 
   * and only U_i is drawn for each constraint individually. 
   * The random number 10**U[0,1] is given by the variable 'factor1' 
   * while the random number 10**U_i[0,2] is calculated below and 
   * stored as 'factor2'. (The exception is when the number of
   * constraints is n+1, in which case 'factor2' defines a random
   * number 10**U_i[0,1])
   */
     
  factor2 = pow(100.0, coco_random_uniform(random_generator));
    
  
  /* Set the gradient of the linear constraint if it is given.
   * This should be the case of the construction of the first 
   * linear constraint only.
   */
  if(gradient) {
	  
    coco_vector_scale(gradient, dimension,
                      factor1 * factor2,
                      coco_vector_norm(gradient, dimension));
    problem = c_linear_transform(problem, gradient);

  }
  else{ /* Randomly generate the gradient of the linear constraint */
	  
    gradient_linear_constraint = coco_allocate_vector(dimension);
     
    /* Generate a pseudorandom vector with distribution N_i(0, I)
     * and scale it with 'factor1' and 'factor2' (see comments above)
     */
    for (i = 0; i < dimension; ++i)
      gradient_linear_constraint[i] = factor1 *
                coco_random_normal(random_generator) * factor2 / sqrt((double)dimension);

    problem = c_linear_transform(problem, gradient_linear_constraint);
    coco_free_memory(gradient_linear_constraint);
  }
  
  /* Guarantee that the vector feasible_point is feasible w.r.t. to
   * this constraint and set it as the initial solution.
   * The initial solution will be copied later to the constrained function
   * coco_problem_t object once the objective function and the constraint(s) 
   * are stacked together in coco_problem_stacked_allocate().
   */
  if(feasible_direction)
    problem = c_guarantee_feasible_point(problem, feasible_direction);
  
  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);
  coco_problem_set_type(problem, "linear");
  coco_random_free(random_generator);
  return problem;
  
}

/**
 * @brief Builds a coco_problem_t containing all the linear constraints
 *        by stacking them all.
 * 
 * The constraints' gradients are randomly generated with distribution
 * 10**U[0,1] * N_i(0, I) * 10**U_i[0,2], where U[a, b] is uniform 
 * in [a,b] and only U_i is drawn for each constraint individually. 
 * The exception is the first constraint, whose gradient is given by
 * 10**U[0,1] * (-feasible_direction) * 10**U_i[0,2].
 * 
 * Each constraint is built by calling the function
 * c_linear_single_cons_bbob_problem_allocate(), which returns a
 * coco_problem_t object that defines the constraint. The resulting
 * coco_problem_t objects are then stacked together into one single
 * coco_problem_t object that is returned by the function.
 */
static coco_problem_t *c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template,
                                                      const double *feasible_direction) {
																																			
  const double global_scaling_factor = 100.;
  size_t i;
  
  coco_problem_t *problem_c = NULL;
  coco_problem_t *problem_c2 = NULL;
  coco_random_state_t *random_generator;
  double *gradient_c1 = NULL;
  double *gradient;
  long seed_cons;
  double exp1, factor1;
  
  gradient_c1 = coco_allocate_vector(dimension);
  																	
  for (i = 0; i < dimension; ++i)
    gradient_c1[i] = -feasible_direction[i];

  /* Build a coco_problem_t object for each constraint. 
   * The constraints' gradients are generated randomly with
   * distriution 10**U[0,1] * N_i(0, I) * 10**U_i[0,2]
   * where U[a, b] is uniform in [a,b] and only U_i is drawn 
   * for each constraint individually.
   */
  
  /* Calculate the first random factor 10**U[0,1]. */
  seed_cons = (long)(function + 10000 * instance);
  random_generator = coco_random_new((uint32_t) seed_cons);
  exp1 = coco_random_uniform(random_generator);
  factor1 = global_scaling_factor * pow(10.0, exp1);

  /* Build the first linear constraint using 'gradient_c1' to build
   * its gradient.
   */ 
  /* set gradient depending on instance number */
  gradient = instance % number_of_linear_constraints ? NULL : gradient_c1;
  problem_c = c_linear_single_cons_bbob_problem_allocate(function,
      dimension, instance, 1, factor1,
      problem_id_template, problem_name_template, gradient,
      feasible_direction);
	 
  /* Instantiate the other linear constraints (if any) and stack them 
   * all into problem_c
   */     
  for (i = 2; i <= number_of_linear_constraints; ++i) {
	 
    /* Instantiate a new problem containing one linear constraint only */
    /* set gradient depending on instance number */
    gradient = (i - 1 + instance) % number_of_linear_constraints ? NULL : gradient_c1;
    problem_c2 = c_linear_single_cons_bbob_problem_allocate(function,
        dimension, instance, i, factor1,
        problem_id_template, problem_name_template, gradient,
        feasible_direction);
		
    problem_c = coco_problem_stacked_allocate(problem_c, problem_c2,
        problem_c2->smallest_values_of_interest, problem_c2->largest_values_of_interest);
	 
    /* Use the standard stacked problem_id as problem_name and 
     * construct a new suite-specific problem_id 
     */
    coco_problem_set_name(problem_c, problem_c->problem_id);
    coco_problem_set_id(problem_c, "bbob-constrained_f%02lu_i%02lu_d%02lu", 
        (unsigned long)function, (unsigned long)instance, (unsigned long)dimension);

    /* Construct problem type */
    coco_problem_set_type(problem_c, "%s_%s", problem_c2->problem_type, 
        problem_c2->problem_type);
  }
  
  coco_free_memory(gradient_c1);
  coco_random_free(random_generator);
  
  return problem_c;
 
}

#line 88 "code-experiments/src/suite_cons_bbob_problems.c"
#line 89 "code-experiments/src/suite_cons_bbob_problems.c"
#line 90 "code-experiments/src/suite_cons_bbob_problems.c"
#line 91 "code-experiments/src/suite_cons_bbob_problems.c"
#line 92 "code-experiments/src/suite_cons_bbob_problems.c"
#line 93 "code-experiments/src/suite_cons_bbob_problems.c"
#line 94 "code-experiments/src/suite_cons_bbob_problems.c"

/**
 * @brief Calculates the objective function type based on the value
 *        of "function"
 */
static size_t obj_function_type(const size_t function) {
  
  size_t problems_per_obj_function_type = 6;
  return (size_t)ceil((double)function/(double)problems_per_obj_function_type);
  
}

/**
 * @brief Returns the number of linear constraints associated to the
 *        value of "function"
 */
static size_t nb_of_linear_constraints(const size_t function,
                                       const size_t dimension) {
  
  int problems_per_obj_function_type = 6;
  int p;
  
  /* Map "function" value into {1, ..., problems_per_obj_function_type} */
  p = (((int)function - 1) % problems_per_obj_function_type) + 1;
  
  if (p == 1) return 1;
  else if (p == 2) return 2;
  else if (p == 3) return 6;
  else if (p == 4) return 6 + dimension/2;
  else if (p == 5) return 6 + dimension;
  else return 6 + 3*dimension;
  
}

/**
 * @brief Scale feasible direction depending on xopt such that
 *        xopt + feasible_direction remains in [-5, 5].
 *
 *
 */
static void feasible_direction_set_length(double * feasible_direction,
                                          const double *xopt,
                                          size_t dimension,
                                          long rseed) {
  const long seed_offset = 412;  /* was sampled uniform in 0-999 */
  const double feas_shrink = 0.75;  /* scale randomly between 0.75 and 1.0 */
  const double feas_bound = 5.0;

  int i;
  double r[1], maxabs, maxrel;

  for (maxabs = maxrel = i = 0; i < dimension; ++i) {
    maxabs = coco_double_max(maxabs, fabs(xopt[i]));
    maxrel = coco_double_max(maxrel, feasible_direction[i] / (feas_bound - xopt[i]));
    maxrel = coco_double_max(maxrel, feasible_direction[i] / (-feas_bound - xopt[i]));
  }
  if (maxabs > 4.01)
    coco_warning("feasible_direction_set_length: a component of fabs(xopt) was greater than 4.01");
  if (maxabs > 5.0)
    coco_error("feasible_direction_set_length: a component of fabs(xopt) was greater than 5.0");
  bbob2009_unif(r, 1, rseed + seed_offset);
  coco_vector_scale(feasible_direction, dimension,
                    feas_shrink + r[0] * (1 - feas_shrink),  /* nominator */
                    maxrel);  /* denominator */
}

/**
 * @brief Objective function: sphere
 *        Constraint(s): linear
 */
static coco_problem_t *f_sphere_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
                                                         		
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;  
  
  all_zeros = coco_allocate_vector(dimension);             
  
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;    
  
  /* Create the objective function */
  problem = f_sphere_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
	 
  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);	 
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);
	 
  /* Create the constraints. Use the gradient of the objective
   * function at the origin to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
	    
  problem_type_temp = coco_strdup(problem->problem_type);
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
	    
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value.
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin.
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
	
  return problem;
 
}

/**
 * @brief Objective function: ellipsoid
 *        Constraint(s): linear
 */
static coco_problem_t *f_ellipsoid_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;

  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
     
  /* Create the objective function */
  problem = f_ellipsoid_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);

  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);

  /* Create the constraints. Use the gradient of the objective
   * function at the origin to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  
  /* Build the final constrained function by stacking the objective
   * function and the constraints into one coco_problem_t type.
   */
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value.
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;

  problem = transform_vars_oscillate(problem);
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin.
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: rotated ellipsoid
 *        Constraint(s): linear
 */
static coco_problem_t *f_ellipsoid_rotated_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  /* Create the objective function */
  problem = f_ellipsoid_rotated_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);
  
  /* Create the constraints. Use the gradient of the objective
   * function at the origin to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  
  /* Build the final constrained function by stacking the objective
   * function and the constraints into one coco_problem_t type.
   */
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
     
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value.
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
  
  problem = transform_vars_oscillate(problem);
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin .
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: linear slope
 *        Constraint(s): linear
 */
static coco_problem_t *f_linear_slope_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  /* Create the objective function */
  problem = f_linear_slope_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);
  
  /* Create the constraints. Use the gradient of the objective
   * function at the origin to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  
  /* Build the final constrained function by stacking the objective
   * function and the constraints into one coco_problem_t type.
   */
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value.
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin.
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: discus
 *        Constraint(s): linear
 */
static coco_problem_t *f_discus_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;

  /* Create the objective function */
  problem = f_discus_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);

  /* Create the constraints. Use the gradient of the objective
   * function at the origin to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  
  /* Build the final constrained function by stacking the objective
   * function and the constraints into one coco_problem_t type.
   */
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
     
  problem = transform_vars_oscillate(problem);
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin. 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: bent cigar
 *        Constraint(s): linear
 */
static coco_problem_t *f_bent_cigar_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  /* Create the objective function */
  problem = f_bent_cigar_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
      
  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);

  /* Create the constraints. Use the gradient of the objective
   * function at the origin to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  
  /* Build the final constrained function by stacking the objective
   * function and the constraints into one coco_problem_t type.
   */
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value.
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
     
  problem = transform_vars_asymmetric(problem, 0.2);
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin. 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: different powers
 *        Constraint(s): linear
 */
static coco_problem_t *f_different_powers_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;
  
  char *problem_type_temp = NULL;
  double *all_zeros = NULL;
  
  all_zeros = coco_allocate_vector(dimension);
 
  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;
	 
  /* Create the objective function */
  problem = f_different_powers_bbob_constrained_problem_allocate(function, dimension,
      instance, rseed, problem_id_template, problem_name_template);
      
  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);
  
  /* Create the constraints. Use the gradient of the objective
   * function at the origin to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function, 
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  
  /* Build the final constrained function by stacking the objective
   * function and the constraints into one coco_problem_t type.
   */
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value.
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
     
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin.
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}

/**
 * @brief Objective function: rastrigin
 *        Constraint(s): linear
 */
static coco_problem_t *f_rastrigin_c_linear_cons_bbob_problem_allocate(const size_t function,
                                                      const size_t dimension,
                                                      const size_t instance,
                                                      const size_t number_of_linear_constraints,
                                                      const long rseed,
                                                      double *feasible_direction,
                                                      const double *xopt,
                                                      const char *problem_id_template,
                                                      const char *problem_name_template) {
																			
  size_t i;
  coco_problem_t *problem = NULL;
  coco_problem_t *problem_c = NULL;

  double *all_zeros = NULL;
  char *problem_type_temp = NULL;

  all_zeros = coco_allocate_vector(dimension);

  for (i = 0; i < dimension; ++i)
    all_zeros[i] = 0.0;

  /* Create the objective function */
  problem = f_rastrigin_cons_bbob_problem_allocate(function, dimension, 
      instance, rseed, problem_id_template, problem_name_template);
  
  bbob_evaluate_gradient(problem, all_zeros, feasible_direction);
  feasible_direction_set_length(feasible_direction, xopt, dimension, rseed);
     
  /* Create the constraints. Use the feasible direction above
   * to build the first constraint. 
   */
  problem_c = c_linear_cons_bbob_problem_allocate(function,
      dimension, instance, number_of_linear_constraints,
      problem_id_template, problem_name_template, feasible_direction);
      
  problem_type_temp = coco_strdup(problem->problem_type);
  
  /* Build the final constrained function by stacking the objective
   * function and the constraints into one coco_problem_t type.
   */
  problem = coco_problem_stacked_allocate(problem, problem_c,
      problem_c->smallest_values_of_interest, 
      problem_c->largest_values_of_interest);
  
  /* Define problem->best_parameter as the origin and store its
   * objective function value into problem->best_value.
   */
  for (i = 0; i < dimension; ++i)
    problem->best_parameter[i] = 0.0;
  assert(problem->evaluate_function != NULL);
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  
  /* To be sure that everything is correct, reset the f-evaluations
   * and g-evaluations counters to zero.
   */
  problem->evaluations = 0;
  problem->evaluations_constraints = 0;
  
  problem = transform_vars_asymmetric(problem, 0.2);
  problem = transform_vars_oscillate(problem);
  
  /* Apply a translation to the whole problem so that the constrained 
   * minimum is no longer at the origin. 
   */
  problem = transform_vars_shift(problem, xopt, 0);
 
  /* Construct problem type */
  coco_problem_set_type(problem, "%s_%s", problem_type_temp, 
  problem_c->problem_type);
 
  coco_free_memory(problem_type_temp);
  coco_free_memory(all_zeros);
  
  return problem;
 
}
#line 10 "code-experiments/src/suite_cons_bbob.c"
#line 1 "code-experiments/src/transform_obj_scale.c"
/**
 * @file transform_obj_scale.c
 * @brief Implementation of scaling the objective value by the given factor.
 */

#include <assert.h>

#line 9 "code-experiments/src/transform_obj_scale.c"
#line 10 "code-experiments/src/transform_obj_scale.c"

/**
 * @brief Data type for transform_obj_scale.
 */
typedef struct {
  double factor;
} transform_obj_scale_data_t;

/**
 * @brief Evaluates the transformed function.
 */
static void transform_obj_scale_evaluate_function(coco_problem_t *problem, const double *x, double *y) {
  transform_obj_scale_data_t *data;
  double *cons_values;
  int is_feasible;
  size_t i;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  data = (transform_obj_scale_data_t *) coco_problem_transformed_get_data(problem);
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);

  for (i = 0; i < problem->number_of_objectives; i++)
    y[i] *= data->factor;

  if (problem->number_of_constraints > 0) {
    cons_values = coco_allocate_vector(problem->number_of_constraints);
    is_feasible = coco_is_feasible(problem, x, cons_values);
    coco_free_memory(cons_values);
    if (is_feasible)
      assert(y[0] + 1e-13 >= problem->best_value[0]);
  }
  else assert(y[0] + 1e-13 >= problem->best_value[0]);
}

/**
 * @brief Evaluates the gradient of the transformed function at x
 */
static void transform_obj_scale_evaluate_gradient(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  transform_obj_scale_data_t *data;

  if (coco_vector_contains_nan(x, coco_problem_get_dimension(problem))) {
    coco_vector_set_to_nan(y, coco_problem_get_number_of_objectives(problem));
    return;
  }

  bbob_evaluate_gradient(coco_problem_transformed_get_inner_problem(problem), x, y);

  data = (transform_obj_scale_data_t *) coco_problem_transformed_get_data(problem);
  for (i = 0; i < problem->number_of_variables; ++i) {
    y[i] *= data->factor;
  }
}

/**
 * @brief Creates the transformation.
 */
static coco_problem_t *transform_obj_scale(coco_problem_t *inner_problem, const double factor) {
  coco_problem_t *problem;
  transform_obj_scale_data_t *data;
  size_t i;
  data = (transform_obj_scale_data_t *) coco_allocate_memory(sizeof(*data));
  data->factor = factor;

  problem = coco_problem_transformed_allocate(inner_problem, data,
    NULL, "transform_obj_scale");

  if (inner_problem->number_of_objectives > 0)
    problem->evaluate_function = transform_obj_scale_evaluate_function;

  problem->evaluate_gradient = transform_obj_scale_evaluate_gradient;

  for (i = 0; i < problem->number_of_objectives; ++i)
    problem->best_value[i] *= factor;

  return problem;
}
#line 11 "code-experiments/src/suite_cons_bbob.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the bbob suite.
 */
static coco_suite_t *suite_cons_bbob_initialize(void) {

  coco_suite_t *suite;
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };

  /* IMPORTANT: Make sure to change the default instance for every new workshop! */
  suite = coco_suite_allocate("bbob-constrained", 48, 6, dimensions, "year: 2016");

  return suite;
}

/**
 * @brief Sets the instances associated with years for the constrained
 *        bbob suite.
 */
static const char *suite_cons_bbob_get_instances_by_year(const int year) {

  if ((year == 2016) || (year == 0)) {
    return "1-15";
  }
  else {
    coco_error("suite_cons_bbob_get_instances_by_year(): year %d not defined for suite_cons_bbob", year);
    return NULL;
  }
}

/**
 * @brief Creates and returns a constrained BBOB problem.
 */
static coco_problem_t *coco_get_cons_bbob_problem(const size_t function,
                                                  const size_t dimension,
                                                  const size_t instance) {
  
  size_t number_of_linear_constraints; 
  coco_problem_t *problem = NULL;
  
  double *feasible_direction = coco_allocate_vector(dimension);  
  double *xopt = coco_allocate_vector(dimension);  
  double f_0, exponent;

  const char *problem_id_template = "bbob-constrained_f%03lu_i%02lu_d%02lu";
  const char *problem_name_template = "bbob-constrained suite problem f%lu instance %lu in %luD";
  
  /* Seed value used for shifting the whole constrained problem */
  long rseed = (long) (function + 10000 * instance);
  bbob2009_compute_xopt(xopt, rseed, dimension);
  
  /* Choose a different seed value for building the objective function */
  rseed = (long) (function + 20000 * instance);
  
  number_of_linear_constraints = nb_of_linear_constraints(function, dimension);
  
  if (obj_function_type(function) == 1) {
	  
    problem = f_sphere_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	 
  } else if (obj_function_type(function) == 2) {
	  
    problem = f_ellipsoid_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 3) {
	  
    problem = f_linear_slope_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 4) {
	  
    problem = f_ellipsoid_rotated_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 5) {
	  
    problem = f_discus_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 6) {
	  
    problem = f_bent_cigar_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 7) {
	  
    problem = f_different_powers_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else if (obj_function_type(function) == 8) {
	  
    problem = f_rastrigin_c_linear_cons_bbob_problem_allocate(function, 
        dimension, instance, number_of_linear_constraints, rseed,
        feasible_direction, xopt, problem_id_template, 
        problem_name_template);
	  
  } else {
    coco_error("get_cons_bbob_problem(): cannot retrieve problem f%lu instance %lu in %luD", 
        function, instance, dimension);
    coco_free_memory(xopt);
    coco_free_memory(feasible_direction);
    return NULL; /* Never reached */
  }

  /* Scale down the objective function value */
  exponent = -2./3;
  f_0 = coco_problem_get_best_value(problem);
  if (f_0 > 1e3) {
    problem = transform_obj_scale(problem, pow(f_0, exponent));
  }

  coco_free_memory(xopt);
  coco_free_memory(feasible_direction);
  
  return problem;
}

/**
 * @brief Returns the problem from the constrained bbob suite that 
 *        corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_cons_bbob_get_problem(coco_suite_t *suite,
                                                   const size_t function_idx,
                                                   const size_t dimension_idx,
                                                   const size_t instance_idx) {
  
  coco_problem_t *problem = NULL;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  problem = coco_get_cons_bbob_problem(function, dimension, instance);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);
  
  /* Use the standard stacked problem_id as problem_name and 
   * construct a new suite-specific problem_id 
   */
  coco_problem_set_name(problem, problem->problem_id);
  coco_problem_set_id(problem, "bbob-constrained_f%02lu_i%02lu_d%02lu", 
  (unsigned long)function, (unsigned long)instance, (unsigned long)dimension);
  
  return problem;
}
#line 25 "code-experiments/src/coco_suite.c"

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
  } else if ((strcmp(suite_name, "bbob-biobj") == 0) ||
      (strcmp(suite_name, "bbob-biobj-ext") == 0)) {
    suite = suite_biobj_initialize(suite_name);
  } else if (strcmp(suite_name, "bbob-largescale") == 0) {
    suite = suite_largescale_initialize();
  } else if (strcmp(suite_name, "bbob-constrained") == 0) {
    suite = suite_cons_bbob_initialize();
  } else if (strcmp(suite_name, "bbob-mixint-1") == 0) {
    suite = suite_bbob_mixint_initialize(suite_name);
  } else if (strcmp(suite_name, "bbob-mixint-2") == 0) {
    suite = suite_bbob_mixint_initialize(suite_name);
  } else if (strcmp(suite_name, "bbob-biobj-mixint") == 0) {
    suite = suite_biobj_mixint_initialize();
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
  } else if (strcmp(suite->suite_name, "bbob-constrained") == 0) {
    year_string = suite_cons_bbob_get_instances_by_year(year);
  } else if ((strcmp(suite->suite_name, "bbob-biobj") == 0) ||
      (strcmp(suite->suite_name, "bbob-biobj-ext") == 0)) {
    year_string = suite_biobj_get_instances_by_year(year);
  } else if (strcmp(suite->suite_name, "bbob-mixint-1") == 0) {
    year_string = suite_bbob_get_instances_by_year(year);
  } else if (strcmp(suite->suite_name, "bbob-mixint-2") == 0) {
    year_string = suite_bbob_get_instances_by_year(year);
  } else if (strcmp(suite->suite_name, "bbob-biobj-mixint") == 0) {
    year_string = suite_biobj_mixint_get_instances_by_year(year);
  } else {
    coco_error("coco_suite_get_instances_by_year(): suite '%s' has no years defined", suite->suite_name);
    return NULL;
  }

  return year_string;
}

/**
 * @brief Calls the function that returns the problem corresponding to the given suite, function index,
 * dimension index and instance index. If the indices don't correspond to a problem because of suite
 * filtering, it returns NULL.
 *
 * @note This function needs to be updated when a new suite is added to COCO.
 */
static coco_problem_t *coco_suite_get_problem_from_indices(coco_suite_t *suite,
                                                           const size_t function_idx,
                                                           const size_t dimension_idx,
                                                           const size_t instance_idx) {

  coco_problem_t *problem;
  
  if ((suite->functions[function_idx] == 0) ||
      (suite->dimensions[dimension_idx] == 0) ||
    (suite->instances[instance_idx] == 0)) {
    return NULL;
  }

  if (strcmp(suite->suite_name, "toy") == 0) {
    problem = suite_toy_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob") == 0) {
    problem = suite_bbob_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if ((strcmp(suite->suite_name, "bbob-biobj") == 0) ||
      (strcmp(suite->suite_name, "bbob-biobj-ext") == 0)) {
    problem = suite_biobj_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob-largescale") == 0) {
    problem = suite_largescale_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob-constrained") == 0) {
    problem = suite_cons_bbob_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob-mixint-1") == 0) {
    problem = suite_bbob_mixint_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob-mixint-2") == 0) {
    problem = suite_bbob_mixint_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else if (strcmp(suite->suite_name, "bbob-biobj-mixint") == 0) {
    problem = suite_biobj_mixint_get_problem(suite, function_idx, dimension_idx, instance_idx);
  } else {
    coco_error("coco_suite_get_problem_from_indices(): unknown problem suite");
    return NULL;
  }

  coco_problem_set_suite(problem, suite);

  return problem;
}

/**
 * @brief Returns the best indicator value for the given problem.
 *
 * @note This function needs to be updated when a new biobjective suite is added to COCO.
 */
static double coco_suite_get_best_indicator_value(const coco_suite_t *suite,
                                                  const coco_problem_t *problem,
                                                  const char *indicator_name) {
  double result = 0;

  if (strcmp(suite->suite_name, "bbob-biobj") == 0) {
    result = suite_biobj_get_best_value(indicator_name, problem->problem_id);
  } else {
    coco_error("coco_suite_get_best_indicator_value_for_problem(): unknown problem suite");
    return 0; /* Never reached */
  }
  return result;
}

/**
 * @note: While a suite can contain multiple problems with equal function, dimension and instance, this
 * function always returns the first problem in the suite with the given function, dimension and instance
 * values. If the given values don't correspond to a problem, the function returns NULL.
 */
coco_problem_t *coco_suite_get_problem_by_function_dimension_instance(coco_suite_t *suite,
                                                                      const size_t function,
                                                                      const size_t dimension,
                                                                      const size_t instance) {

  size_t i;
  int function_idx, dimension_idx, instance_idx;
  int found;

  found = 0;
  for (i = 0; i < suite->number_of_functions; i++) {
    if (suite->functions[i] == function) {
      function_idx = (int) i;
      found = 1;
      break;
    }
  }
  if (!found)
    return NULL;

  found = 0;
  for (i = 0; i < suite->number_of_dimensions; i++) {
    if (suite->dimensions[i] == dimension) {
      dimension_idx = (int) i;
      found = 1;
      break;
    }
  }
  if (!found)
    return NULL;

  found = 0;
  for (i = 0; i < suite->number_of_instances; i++) {
    if (suite->instances[i] == instance) {
      instance_idx = (int) i;
      found = 1;
      break;
    }
  }
  if (!found)
    return NULL;

  return coco_suite_get_problem_from_indices(suite, (size_t) function_idx, (size_t) dimension_idx, (size_t) instance_idx);
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
 * @return The problem of the suite defined by problem_index (NULL if this problem has been filtered out
 * from the suite).
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
  int parse_year = 1, parse_instances = 1;
  size_t *result = NULL;

  if (suite_instance == NULL)
    return NULL;

  year_found = coco_strfind(suite_instance, "year");
  instances_found = coco_strfind(suite_instance, "instances");

  if ((year_found < 0) && (instances_found < 0))
    return NULL;

  if ((year_found > 0) && (instances_found > 0)) {
    if (year_found < instances_found) {
      parse_instances = 0;
      coco_warning("coco_suite_get_instance_indices(): 'instances' suite option ignored because it follows 'year'");
    }
    else {
      parse_year = 0;
      coco_warning("coco_suite_get_instance_indices(): 'year' suite option ignored because it follows 'instances'");
    }
  }

  if ((year_found >= 0) && (parse_year == 1)) {
    if (coco_options_read_int(suite_instance, "year", &(year)) != 0) {
      year_string = coco_suite_get_instances_by_year(suite, year);
      result = coco_string_parse_ranges(year_string, 1, 0, "instances", COCO_MAX_INSTANCES);
    } else {
      coco_warning("coco_suite_get_instance_indices(): problems parsing the 'year' suite_instance option, ignored");
    }
  }

  instances = coco_allocate_string(COCO_MAX_INSTANCES);
  if ((instances_found >= 0) && (parse_instances == 1)) {
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
 * Currently, six suites are supported.
 * Seven suites with artificial test functions:
 * - "bbob" contains 24 <a href="http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf">
 * single-objective functions</a> in 6 dimensions (2, 3, 5, 10, 20, 40)
 * - "bbob-biobj" contains 55 <a href="http://numbbo.github.io/coco-doc/bbob-biobj/functions">bi-objective
 * functions</a> in 6 dimensions (2, 3, 5, 10, 20, 40)
 * - "bbob-biobj-ext" as an extension of "bbob-biobj" contains 92
 * <a href="http://numbbo.github.io/coco-doc/bbob-biobj/functions">bi-objective functions</a> in 6 dimensions 
 * (2, 3, 5, 10, 20, 40)
 * - "bbob-largescale" contains 24 <a href="http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf">
 * single-objective functions</a> in 6 large dimensions (40, 80, 160, 320, 640, 1280)
 * - "bbob-constrained" contains 48 linearly-constrained problems, which are combinations of 8 single 
 * objective functions with 6 different numbers of linear constraints (1, 2, 10, dimension/2, dimension-1, 
 * dimension+1), in 6 dimensions (2, 3, 5, 10, 20, 40).
 * - "bbob-mixint" contains mixed-integer single-objective functions in 6 dimensions (2, 3, 5, 10, 20, 40)
 * - "toy" contains 6 <a href="http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf">
 * single-objective functions</a> in 5 dimensions (2, 3, 5, 10, 20)
 *
 * Only the suite_name parameter needs to be non-empty. The suite_instance and suite_options can be "" or
 * NULL. In this case, default values are taken (default instances of a suite are those used in the last year
 * and the suite is not filtered by default).
 *
 * @param suite_name A string containing the name of the suite. Currently supported suite names are "bbob",
 * "bbob-biobj", "bbob-biobj-ext", "bbob-largescale", "bbob-constrained", and "toy".
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
    option_string = coco_allocate_string(COCO_PATH_MAX + 1);
    if (coco_options_read_values(suite_options, "function_indices", option_string) > 0) {
      indices = coco_string_parse_ranges(option_string, 1, suite->number_of_functions, "function_indices", COCO_MAX_INSTANCES);
      if (indices != NULL) {
        coco_suite_filter_indices(suite->functions, suite->number_of_functions, indices, "function_indices");
        coco_free_memory(indices);
      }
    }
    coco_free_memory(option_string);

    option_string = coco_allocate_string(COCO_PATH_MAX + 1);
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

    option_string = coco_allocate_string(COCO_PATH_MAX + 1);
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

    option_string = coco_allocate_string(COCO_PATH_MAX + 1);
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
      coco_info_partial("COCO INFO: %s, d=%lu, running: f%02lu", time_string,
          (unsigned long) suite->dimensions[dimension_idx], (unsigned long) suite->functions[function_idx]);
      coco_free_memory(time_string);
    }
    else if ((long) function_idx != previous_function_idx){
      /* A new function started */
      coco_info_partial("f%02lu", (unsigned long) suite->functions[function_idx]);
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
 * @param instance_idx Index of the instance (starting with 0).
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

    if (current_exponent < targets->exponent) {
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

    /* Compute the adjusted_exponent in such a way, that it is always diminishing in value. The adjusted
     * exponent can only be used to verify if a new target has been hit. To compute the actual target
     * value, the current_exponent needs to be used. */
    adjusted_exponent = 2 * (int) (ceil(log10(targets->precision / 10.0) * number_of_targets_double))
        - current_exponent - 1;

    if (adjusted_exponent < targets->exponent) {
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

  if (evaluation_number >= evaluations->value1) {
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

  if (evaluation_number >= evaluations->value2) {
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
                                               const int precision_f,
                                               const int precision_g,
                                               const int log_discrete_as_int) {

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
  observer->precision_g = precision_g;
  observer->log_discrete_as_int = log_discrete_as_int;
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

static const double fvalue_logged_for_infinite = 3e21;   /* value used for logging try */
static const double fvalue_logged_for_nan = 2e21;
/* static const double fvalue_logged_for_infeasible = 1e21;  only in first evaluation */
static const double weight_constraints = 1e0;  /* factor used in logged indicator (f-f*)^+ + sum_i g_i^+ in front of the sum */

/*static const size_t bbob_nbpts_nbevals = 20; Wassim: tentative, are now observer options with these default values*/
/*static const size_t bbob_nbpts_fval = 5;*/
static size_t bbob_current_dim = 0;
static size_t bbob_current_funId = 0;
static size_t bbob_infoFile_firstInstance = 0;
char *bbob_infoFile_firstInstance_char;
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
  size_t number_of_evaluations_constraints;
  double best_fvalue;
  double last_fvalue;
  short written_last_eval; /* allows writing the data of the final fun eval in the .tdat file if not already written by the t_trigger*/
  double *best_solution;
  /* The following are to only pass data as a parameter in the free function. The
   * interface should probably be the same for all free functions so passing the
   * problem as a second parameter is not an option even though we need info
   * form it.*/
  size_t function_id; /*TODO: consider changing name*/
  size_t instance_id;
  size_t number_of_variables;
  size_t number_of_integer_variables;
  int log_discrete_as_int;            /**< @brief Whether to output discrete variables in int or double format. */
  double optimal_fvalue;
  char *suite_name;

  coco_observer_targets_t *targets;          /**< @brief Triggers based on target values. */
  coco_observer_evaluations_t *evaluations;  /**< @brief Triggers based on the number of evaluations. */

} logger_bbob_data_t;

/**
 * @brief Discretized constraint value, ~8 + log10(c), in a single digit.
 *
 * -\infty..0 -> 0
 *    0..1e-7 -> 1
 * 1e-7..1e-6 -> 2
 *    ...
 * 1e-1..1    -> 8
 *   >1       -> 9
 */
static int single_digit_constraint_value(const double c) {
  const double limits[9] = {0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1};
  int i;

  for (i = 0; i < 9; ++i)
    if (c <= limits[i])
      return i;
  return 9;
}

/* was (old):
 * function evaluation |
 * noise-free fitness - Fopt (7.948000000000e+01) |
 * best noise-free fitness - Fopt |
 * measured fitness |
 * best measured fitness |
 * x1 | x2...
 was (bbob-new):
    "f evaluations | "
    "g evaluations | "
    "best noise-free fitness - Fopt | "
    "noise-free fitness - Fopt (%13.12e) | "
    "measured fitness | "
    "best measured fitness | "
    "x1 | "
    "x2...\n";
 */
static const char *bbob_file_header_str = "%% "
    "f evaluations | "
    "g evaluations | "
    "best noise-free fitness - Fopt (%13.12e) + sum g_i+ | "
    "measured fitness | "
    "best measured fitness or single-digit g-values | "
    "x1 | "
    "x2...\n";

static const char *logger_name = "bbob";
static const char *data_format = "bbob-new2"; /* back to 5 columns, 5-th column writes single digit constraint values */

/**
 * adds a formated line to a data file
 */
static void logger_bbob_write_data(FILE *target_file,
                                   size_t number_of_f_evaluations,
                                   size_t number_of_cons_evaluations,
                                   double fvalue,
                                   double best_fvalue,
                                   double best_value,
                                   const double *x,
                                   size_t number_of_variables,
                                   size_t number_of_integer_variables,
                                   const double *constraints,
                                   size_t number_of_constraints,
                                   const int log_discrete_as_int) {
  size_t i;
  /* for some reason, it's %.0f in the old code instead of the 10.9e
   * in the documentation
   */
  fprintf(target_file, "%lu %lu %+10.9e %+10.9e ",
          (unsigned long) number_of_f_evaluations,
    	  (unsigned long) number_of_cons_evaluations,
          best_fvalue - best_value,
    	  fvalue);

  if (number_of_constraints > 0)
    for (i = 0; i < number_of_constraints; ++i)
      fprintf(target_file, "%d",
              constraints ? single_digit_constraint_value(constraints[i])
                          : (int) (i % 10)); /* print 01234567890123..., may happen in last line of .tdat */
  else
    fprintf(target_file, "%+10.9e", best_fvalue);

  if ((number_of_variables - number_of_integer_variables) < 22) {
    for (i = 0; i < number_of_variables; i++) {
      if ((i < number_of_integer_variables) && (log_discrete_as_int))
        fprintf(target_file, " %d", coco_double_to_int(x[i]));
      else
        fprintf(target_file, " %+5.4e", x[i]);
    }
  }
  fprintf(target_file, "\n");

  /* Flush output so that impatient users can see progress.
   * Otherwise it can take a long time until the output appears.
   */
  fflush(target_file);
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
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  char relative_filePath[COCO_PATH_MAX + 2] = { 0 };
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
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  char relative_filePath[COCO_PATH_MAX + 2] = { 0 };
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
 * folder_path
 */
static void logger_bbob_openIndexFile(logger_bbob_data_t *logger,
                                      const char *folder_path,
                                      const char *indexFile_prefix,
                                      const char *function_id,
                                      const char *dataFile_path,
                                      const char *suite_name) {
  /* to add the instance number TODO: this should be done outside to avoid redoing this for the .*dat files */
  char used_dataFile_path[COCO_PATH_MAX + 2] = { 0 };
  int errnum, newLine; /* newLine is at 1 if we need a new line in the info file */
  char *function_id_char; /* TODO: consider adding them to logger */
  char file_name[COCO_PATH_MAX + 2] = { 0 };
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  FILE **target_file;
  FILE *tmp_file;
  strncpy(used_dataFile_path, dataFile_path, COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
  if (bbob_infoFile_firstInstance == 0) {
    bbob_infoFile_firstInstance = logger->instance_id;
  }
  function_id_char = coco_strdupf("%lu", (unsigned long) logger->function_id);
  bbob_infoFile_firstInstance_char = coco_strdupf("%lu", (unsigned long) bbob_infoFile_firstInstance);
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
            coco_free_memory(bbob_infoFile_firstInstance_char);
            bbob_infoFile_firstInstance_char = coco_strdupf("%lu", (unsigned long) bbob_infoFile_firstInstance);
            strncat(file_path, "_i", COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, bbob_infoFile_firstInstance_char, COCO_PATH_MAX - strlen(file_name) - 1);
            strncat(file_path, ".info", COCO_PATH_MAX - strlen(file_name) - 1);
          } else { /* we have all dimensions */
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
      /* data_format = coco_strdup("bbob-constrained"); */
      fprintf(*target_file,
              "suite = '%s', funcId = %d, DIM = %lu, Precision = %.3e, algId = '%s', coco_version = '%s', logger = '%s', data_format = '%s'\n",
              suite_name,
              (int) strtol(function_id, NULL, 10),
              (unsigned long) logger->number_of_variables,
              pow(10, -8),
              logger->observer->algorithm_name,
              coco_version,
              logger_name,
              data_format);

      fprintf(*target_file, "%%\n");
      strncat(used_dataFile_path, "_i", COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      strncat(used_dataFile_path, bbob_infoFile_firstInstance_char,
      COCO_PATH_MAX - strlen(used_dataFile_path) - 1);
      fprintf(*target_file, "%s.dat", used_dataFile_path); /* dataFile_path does not have the extension */
      bbob_current_dim = logger->number_of_variables;
      bbob_current_funId = logger->function_id;
    }
  }
  coco_free_memory(function_id_char);
}

/**
 * Generates the different files and folder needed by the logger to store the
 * data if these don't already exist
 */
static void logger_bbob_initialize(logger_bbob_data_t *logger, coco_problem_t *inner_problem) {
  /*
   Creates/opens the data and index files
   */
  char dataFile_path[COCO_PATH_MAX + 2] = { 0 }; /* relative path to the .dat file from where the .info file is */
  char folder_path[COCO_PATH_MAX + 2] = { 0 };
  char *tmpc_funId; /* serves to extract the function id as a char *. There should be a better way of doing this! */
  char *tmpc_dim; /* serves to extract the dimension as a char *. There should be a better way of doing this! */
  char indexFile_prefix[10] = "bbobexp"; /* TODO (minor): make the prefix bbobexp a parameter that the user can modify */

  assert(logger != NULL);
  assert(inner_problem != NULL);
  assert(inner_problem->problem_id != NULL);

  tmpc_funId = coco_strdupf("%lu", (unsigned long) coco_problem_get_suite_dep_function(inner_problem));
  tmpc_dim = coco_strdupf("%lu", (unsigned long) inner_problem->number_of_variables);

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
  assert(coco_problem_get_suite(inner_problem));
  logger_bbob_openIndexFile(logger, logger->observer->result_folder, indexFile_prefix, tmpc_funId,
      dataFile_path, coco_problem_get_suite(inner_problem)->suite_name);
  fprintf(logger->index_file, ", %lu", (unsigned long) coco_problem_get_suite_dep_instance(inner_problem));
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
  coco_free_memory(bbob_infoFile_firstInstance_char);
}

/**
 * Layer added to the transformed-problem evaluate_function by the logger
 */
static void logger_bbob_evaluate(coco_problem_t *problem, const double *x, double *y) {
  size_t i;
  double y_logged, max_fvalue, sum_cons;
  double *cons;
  logger_bbob_data_t *logger = (logger_bbob_data_t *) coco_problem_transformed_get_data(problem);
  coco_problem_t *inner_problem = coco_problem_transformed_get_inner_problem(problem);
  const int is_feasible = problem->number_of_constraints <= 0
                            || coco_is_feasible(inner_problem, x, NULL);

  if (!logger->is_initialized) {
    logger_bbob_initialize(logger, inner_problem);
  }
  if ((coco_log_level >= COCO_DEBUG) && logger->number_of_evaluations == 0) {
    coco_debug("%4lu: ", (unsigned long) inner_problem->suite_dep_index);
    coco_debug("on problem %s ... ", coco_problem_get_id(inner_problem));
  }

  coco_evaluate_function(inner_problem, x, y); /* fulfill contract as "being" a coco evaluate function */

  logger->number_of_evaluations_constraints = coco_problem_get_evaluations_constraints(problem);
  logger->number_of_evaluations++; /* could be != coco_problem_get_evaluations(problem) for non-anytime logging? */
  logger->written_last_eval = 0; /* flag whether the current evaluation was logged? */
  logger->last_fvalue = y[0]; /* asma: should be: max(y[0], logger->optimal_fvalue) */

  y_logged = y[0];
  if (coco_is_nan(y_logged))
    y_logged = fvalue_logged_for_nan;
  else if (coco_is_inf(y_logged))
    y_logged = fvalue_logged_for_infinite;
  /* do sanity check */
  if (is_feasible)  /* infeasible solutions can have much better y0 values */
    assert(y_logged + 1e-13 >= logger->optimal_fvalue);

  /* Evaluate the constraints */
  if (problem->number_of_constraints > 0) {
    cons = coco_allocate_vector(problem->number_of_constraints);
    inner_problem->evaluate_constraint(inner_problem, x, cons);
  }

  /* Compute the sum of positive constraint values */
  sum_cons = 0;
  for (i = 0; i < problem->number_of_constraints; ++i) {
    if (cons[i] > 0)
        sum_cons += cons[i];
  }
  sum_cons *= weight_constraints;  /* do this before the checks */
  if (coco_is_nan(sum_cons))
    sum_cons = fvalue_logged_for_nan;
  else if (coco_is_inf(sum_cons))
    sum_cons = fvalue_logged_for_infinite;

  max_fvalue =  y_logged > logger->optimal_fvalue ? y_logged : logger->optimal_fvalue;

  /* Update logger state.
   *   at logger->number_of_evaluations == 1 the logger->best_fvalue is not initialized,
   *   also compare to y_logged to not potentially be thrown off by weird values in y[0]
   */
  if (logger->number_of_evaluations == 1 || (max_fvalue + sum_cons < logger->best_fvalue)) {
    logger->best_fvalue = max_fvalue + sum_cons;
    for (i = 0; i < problem->number_of_variables; i++)
      logger->best_solution[i] = x[i]; /* may well be infeasible */

    /* Add a line in the .dat file for each logging target reached
     * by a feasible solution and always at evaluation one
     */
    if (logger->number_of_evaluations == 1 || coco_observer_targets_trigger(logger->targets,
                                        logger->best_fvalue - logger->optimal_fvalue)) {
      logger_bbob_write_data(
          logger->fdata_file,
          logger->number_of_evaluations,
          logger->number_of_evaluations_constraints,
          y_logged,
          logger->best_fvalue,
          logger->optimal_fvalue,
          x,
          problem->number_of_variables,
          problem->number_of_integer_variables,
          cons,
          problem->number_of_constraints,
          logger->log_discrete_as_int);
    }
  }

  /* Add a line in the .tdat file each time an fevals trigger is reached.*/
  if (coco_observer_evaluations_trigger(logger->evaluations,
        logger->number_of_evaluations + logger->number_of_evaluations_constraints)) {
    logger_bbob_write_data(
        logger->tdata_file,
        logger->number_of_evaluations,
        logger->number_of_evaluations_constraints,
        y_logged,
        logger->best_fvalue,
        logger->optimal_fvalue,
        x,
        problem->number_of_variables,
        problem->number_of_integer_variables,
        cons,
        problem->number_of_constraints,
        logger->log_discrete_as_int);
    logger->written_last_eval = 1;
  }

  /* Free allocated memory */
  if (problem->number_of_constraints > 0)
    coco_free_memory(cons);

}  /* end logger_bbob_evaluate */

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
    coco_debug("best f=%e after %lu fevals (done observing)\n", logger->best_fvalue,
    		(unsigned long) logger->number_of_evaluations);
  }
  if (logger->index_file != NULL) {
    fprintf(logger->index_file, ":%lu|%.1e",
            (unsigned long) logger->number_of_evaluations,
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
      logger_bbob_write_data(logger->tdata_file, 
          logger->number_of_evaluations,
          logger->number_of_evaluations_constraints,
          logger->best_fvalue,
          logger->best_fvalue,
          logger->optimal_fvalue,
          logger->best_solution,
          logger->number_of_variables,
          logger->number_of_integer_variables,
          NULL,
          0,
          logger->log_discrete_as_int);
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
  logger_bbob_data_t *logger_data;
  coco_problem_t *problem;

  logger_data = (logger_bbob_data_t *) coco_allocate_memory(sizeof(*logger_data));
  logger_data->observer = observer;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_bbob(): The bbob logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }

  if (bbob_logger_is_open)
    coco_error("The current bbob_logger (observer) must be closed before a new one is opened");
  /* This is the name of the folder which happens to be the algName */
  /*logger->path = coco_strdup(observer->output_folder);*/
  logger_data->index_file = NULL;
  logger_data->fdata_file = NULL;
  logger_data->tdata_file = NULL;
  logger_data->rdata_file = NULL;
  logger_data->number_of_variables = inner_problem->number_of_variables;
  logger_data->number_of_integer_variables = inner_problem->number_of_integer_variables;
  if (inner_problem->best_value == NULL) {
    /* coco_error("Optimal f value must be defined for each problem in order for the logger to work properly"); */
    /* Setting the value to 0 results in the assertion y>=optimal_fvalue being susceptible to failure */
    coco_warning("undefined optimal f value. Set to 0");
    logger_data->optimal_fvalue = 0;
  } else {
    logger_data->optimal_fvalue = *(inner_problem->best_value);
  }

  logger_data->number_of_evaluations = 0;
  logger_data->number_of_evaluations_constraints = 0;
  logger_data->best_solution = coco_allocate_vector(inner_problem->number_of_variables);
  /* TODO: the following inits are just to be in the safe side and
   * should eventually be removed. Some fields of the bbob_logger struct
   * might be useless
   */
  logger_data->function_id = coco_problem_get_suite_dep_function(inner_problem);
  logger_data->instance_id = coco_problem_get_suite_dep_instance(inner_problem);
  logger_data->written_last_eval = 0;
  logger_data->last_fvalue = DBL_MAX;
  logger_data->is_initialized = 0;
  logger_data->log_discrete_as_int = observer->log_discrete_as_int;
    
  /* Initialize triggers based on target values and number of evaluations */
  logger_data->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  logger_data->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, inner_problem->number_of_variables);

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_bbob_free, observer->observer_name);

  problem->evaluate_function = logger_bbob_evaluate;
  bbob_logger_is_open = 1;
  return problem;
}
#line 357 "code-experiments/src/coco_observer.c"
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
 * observer_biobj() for more information. One .adat file is created for each problem function, dimension
 * and instance.
 *
 * @note Whenever in this file a ROI is mentioned, it means the (normalized) region of interest in the
 * objective space. The non-normalized ROI is a rectangle with the ideal and nadir points as its two
 * opposite vertices, while the normalized ROI is the square [0, 1]^2. If not specifically mentioned, the
 * normalized ROI is assumed.
 *
 * @note This logger can handle both the original bbob-biobj test suite with 55 and the extended
 * bbob-biobj-ext test suite with 96 functions.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#line 38 "code-experiments/src/logger_biobj.c"
#line 39 "code-experiments/src/logger_biobj.c"

#line 41 "code-experiments/src/logger_biobj.c"
#line 42 "code-experiments/src/logger_biobj.c"
#line 43 "code-experiments/src/logger_biobj.c"
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
static avl_tree_t *avl_tree_init(avl_tree_t *avltree, avl_compare_t cmp, avl_free_t free_item) {
  if (avltree) {
    avltree->head = NULL;
    avltree->tail = NULL;
    avltree->top = NULL;
    avltree->cmpitem = cmp;
    avltree->freeitem = free_item;
    avltree->userdata = NULL;
    avltree->allocator = NULL;
  }
  return avltree;
}

/* Allocates and initializes a new tree for elements that will be
 * ordered using the supplied strcmp()-like function.
 * Returns NULL if memory could not be allocated.
 * O(1) */
static avl_tree_t *avl_tree_construct(avl_compare_t cmp, avl_free_t free_item) {
  return avl_tree_init((avl_tree_t *) malloc(sizeof(avl_tree_t)), cmp, free_item);
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
#line 44 "code-experiments/src/logger_biobj.c"
#line 1 "code-experiments/src/observer_biobj.c"
/**
 * @file observer_biobj.c
 * @brief Implementation of the bbob-biobj observer.
 */

#line 7 "code-experiments/src/observer_biobj.c"
#line 8 "code-experiments/src/observer_biobj.c"

#line 10 "code-experiments/src/observer_biobj.c"
#line 11 "code-experiments/src/observer_biobj.c"

/** @brief Enum for denoting the way in which the nondominated solutions are treated. */
typedef enum {
  LOG_NONDOM_NONE, LOG_NONDOM_FINAL, LOG_NONDOM_ALL, LOG_NONDOM_READ
} observer_biobj_log_nondom_e;

/** @brief Enum for denoting when the decision variables are logged. */
typedef enum {
  LOG_VARS_NEVER, LOG_VARS_LOW_DIM, LOG_VARS_ALWAYS
} observer_biobj_log_vars_e;

/**
 * @brief The bbob-biobj observer data type.
 */
typedef struct {
  observer_biobj_log_nondom_e log_nondom_mode; /**< @brief Handling of the nondominated solutions. */
  observer_biobj_log_vars_e log_vars_mode;     /**< @brief When the decision variables are logged. */

  int compute_indicators;                      /**< @brief Whether to compute indicators. */
  int produce_all_data;                        /**< @brief Whether to produce all data. */

  long previous_function;                      /**< @brief Function of the previous logged problem. */
  long previous_dimension;                     /**< @brief Dimension of the previous logged problem */

} observer_biobj_data_t;

static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *problem);
static void logger_biobj_free(void *logger);

/**
 * @brief Initializes the bi-objective observer.
 *
 * Possible options:
 *
 * - "log_nondominated: STRING" determines how the nondominated solutions are handled. STRING can take on the
 * values "none" (don't log nondominated solutions), "final" (log only the final nondominated solutions),
 * "all" (log every solution that is nondominated at creation time) and "read" (the nondominated solutions
 * are not logged, but are passed to the logger as input - this is a functionality needed in pre-processing
 * of the data). The default value is "all".
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

  observer_biobj_data_t *observer_data;
  char string_value[COCO_PATH_MAX + 1];

  /* Sets the valid keys for bbob-biobj observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "log_nondominated", "log_decision_variables", "compute_indicators",
      "produce_all_data" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_biobj_data_t *) coco_allocate_memory(sizeof(*observer_data));

  observer_data->log_nondom_mode = LOG_NONDOM_ALL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_NONE;
    else if (strcmp(string_value, "final") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_FINAL;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_ALL;
    else if (strcmp(string_value, "read") == 0)
      observer_data->log_nondom_mode = LOG_NONDOM_READ;
  }

  observer_data->log_vars_mode = LOG_VARS_LOW_DIM;
  if (coco_options_read_string(options, "log_decision_variables", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_vars_mode = LOG_VARS_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_vars_mode = LOG_VARS_ALWAYS;
    else if (strcmp(string_value, "low_dim") == 0)
      observer_data->log_vars_mode = LOG_VARS_LOW_DIM;
  }

  if (coco_options_read_int(options, "compute_indicators", &(observer_data->compute_indicators)) == 0)
    observer_data->compute_indicators = 1;

  if (coco_options_read_int(options, "produce_all_data", &(observer_data->produce_all_data)) == 0)
    observer_data->produce_all_data = 0;

  if (observer_data->produce_all_data) {
    observer_data->compute_indicators = 1;
    observer_data->log_nondom_mode = LOG_NONDOM_ALL;
    observer_data->log_vars_mode = LOG_VARS_LOW_DIM;
  }

  if (observer_data->compute_indicators) {
    observer_data->previous_function = -1;
    observer_data->previous_dimension = -1;
  }

  observer->logger_allocate_function = logger_biobj;
  observer->logger_free_function = logger_biobj_free;
  observer->data_free_function = NULL;
  observer->data = observer_data;

  if ((observer_data->log_nondom_mode == LOG_NONDOM_NONE) && (!observer_data->compute_indicators)) {
    /* No logging required */
    observer->is_active = 0;
  }
}
#line 45 "code-experiments/src/logger_biobj.c"

#line 47 "code-experiments/src/logger_biobj.c"

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
  double previous_value;     /**< @brief The previous overall value of the indicator. */

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
  FILE *adat_file;                    /**< @brief File for archiving nondominated solutions (all or final). */

  int log_vars;                       /**< @brief Whether to log the decision values. */

  int precision_x;                    /**< @brief Precision for outputting decision values. */
  int precision_f;                    /**< @brief Precision for outputting objective values. */
  int log_discrete_as_int;            /**< @brief Whether to output discrete variables in int or double format. */

  size_t number_of_evaluations;       /**< @brief The number of evaluations performed so far. */
  size_t number_of_variables;         /**< @brief Dimension of the problem. */
  size_t number_of_integer_variables; /**< @brief Number of integer variables. */
  size_t number_of_objectives;        /**< @brief Number of objectives (clearly equal to 2). */
  size_t suite_dep_instance;          /**< @brief Suite-dependent instance number of the observed problem. */

  size_t previous_evaluations;        /**< @brief The number of evaluations from the previous call to the logger. */

  avl_tree_t *archive_tree;           /**< @brief The tree keeping currently non-dominated solutions. */
  avl_tree_t *buffer_tree;            /**< @brief The tree with pointers to nondominated solutions that haven't
                                           been logged yet. */

  /* Indicators (TODO: Implement others!) */
  int compute_indicators;             /**< @brief Whether to compute the indicators. */
  logger_biobj_indicator_t *indicators[LOGGER_BIOBJ_NUMBER_OF_INDICATORS];
                                      /**< @brief The implemented indicators. */
} logger_biobj_data_t;

/**
 * @brief The type for the node's item in the AVL tree as used by the bi-objective logger.
 *
 * Contains information on the exact objective values (y) and their rounded normalized values (normalized_y).
 * The exact values are used for output, while archive update and indicator computation use the normalized
 * values.
 */
typedef struct {
  double *x;                 /**< @brief The decision values of this solution. */
  double *y;                 /**< @brief The values of objectives of this solution. */
  double *normalized_y;      /**< @brief The values of normalized objectives of this solution. */
  size_t evaluation_number;  /**< @brief The evaluation number of when the solution was created. */

  double indicator_contribution[LOGGER_BIOBJ_NUMBER_OF_INDICATORS];
                      /**< @brief The contribution of this solution to the overall indicator values. */
  int within_ROI;     /**< @brief Whether the solution is within the region of interest (ROI). */

} logger_biobj_avl_item_t;

/**
 * @brief Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static logger_biobj_avl_item_t* logger_biobj_node_create(const coco_problem_t *problem,
                                                         const double *x,
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

  /* Compute the normalized y */
  item->normalized_y = mo_normalize(item->y, problem->best_value, problem->nadir_value, num_obj);
  item->within_ROI = mo_is_within_ROI(item->normalized_y, num_obj);

  item->evaluation_number = evaluation_number;
  for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++)
    item->indicator_contribution[i] = 0;

  return item;
}

/**
 * @brief Frees the data of the given logger_biobj_avl_item_t.
 */
static void logger_biobj_node_free(logger_biobj_avl_item_t *item, void *userdata) {

  coco_free_memory(item->x);
  coco_free_memory(item->y);
  coco_free_memory(item->normalized_y);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the value of the last objective.
 *
 * @note This ordering is used by the archive_tree.
 */
static int avl_tree_compare_by_last_objective(const logger_biobj_avl_item_t *item1,
                                              const logger_biobj_avl_item_t *item2,
                                              void *userdata) {
  if (coco_double_almost_equal(item1->normalized_y[1], item2->normalized_y[1], mo_precision))
    return 0;
  else if (item1->normalized_y[1] < item2->normalized_y[1])
    return -1;
  else
    return 1;

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
                                       const size_t num_int_vars,
                                       const size_t num_obj,
                                       const int log_vars,
                                       const int precision_x,
                                       const int precision_f,
                                       const int log_discrete_as_int) {

  avl_node_t *solution;
  size_t i;
  size_t j;
  size_t number_of_nodes = 0;

  if (tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = tree->head;
    while (solution != NULL) {
      fprintf(file, "%lu\t", (unsigned long) ((logger_biobj_avl_item_t*) solution->item)->evaluation_number);
      for (j = 0; j < num_obj; j++)
        fprintf(file, "%.*e\t", precision_f, ((logger_biobj_avl_item_t*) solution->item)->y[j]);
      if (log_vars) {
        for (i = 0; i < dim; i++)
          if ((i < num_int_vars) && (log_discrete_as_int))
            fprintf(file, "%d\t", coco_double_to_int(((logger_biobj_avl_item_t*) solution->item)->x[i]));
          else
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
                                    logger_biobj_avl_item_t *node_item) {

  avl_node_t *node, *next_node, *new_node;
  int trigger_update = 0;
  int dominance;
  size_t i;
  int previous_unavailable = 0;

  /* Find the first point that is not worse than the new point (NULL if such point does not exist) */
  node = avl_item_search_right(logger->archive_tree, node_item, NULL);

  if (node == NULL) {
    /* The new point is an extreme point */
    trigger_update = 1;
    next_node = logger->archive_tree->head;
  } else {
    dominance = mo_get_dominance(node_item->normalized_y,
        ((logger_biobj_avl_item_t*) node->item)->normalized_y, logger->number_of_objectives);
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
      /* The new point is dominated or equal to an existing one, nothing more to do */
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
      dominance = mo_get_dominance(node_item->normalized_y,
          ((logger_biobj_avl_item_t*) node->item)->normalized_y, logger->number_of_objectives);
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
    assert(new_node != NULL);
    avl_item_insert(logger->buffer_tree, node_item);

    if (logger->compute_indicators) {
      if (node_item->within_ROI) {
        /* Compute indicator value for new node and update the indicator value of the affected nodes */
        logger_biobj_avl_item_t *next_item, *previous_item;

        if (new_node->next != NULL) {
          next_item = (logger_biobj_avl_item_t*) new_node->next->item;
          if (next_item->within_ROI) {
            for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              logger->indicators[i]->current_value -= next_item->indicator_contribution[i];
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                next_item->indicator_contribution[i] = (node_item->normalized_y[0] - next_item->normalized_y[0])
                    * (1 - next_item->normalized_y[1]);
                assert(next_item->indicator_contribution[i] >= 0);
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
                node_item->indicator_contribution[i] = (previous_item->normalized_y[0] - node_item->normalized_y[0])
                    * (1 - node_item->normalized_y[1]);
                assert(node_item->indicator_contribution[i] >= 0);
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
              node_item->indicator_contribution[i] = (1 - node_item->normalized_y[0])
                  * (1 - node_item->normalized_y[1]);
              assert(node_item->indicator_contribution[i] >= 0);
            } else {
              coco_error(
                  "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                  logger->indicators[i]->name);
            }
          }
        }

        for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
          if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
            assert(node_item->indicator_contribution[i] >= 0);
            logger->indicators[i]->current_value += node_item->indicator_contribution[i];
          }
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

  observer_biobj_data_t *observer_data;
  logger_biobj_indicator_t *indicator;
  char *prefix, *file_name, *path_name;
  int info_file_exists = 0;

  indicator = (logger_biobj_indicator_t *) coco_allocate_memory(sizeof(*indicator));
  assert(observer);
  assert(observer->data);
  observer_data = (observer_biobj_data_t *) observer->data;

  indicator->name = coco_strdup(indicator_name);

  assert(problem->suite);
  indicator->best_value = coco_suite_get_best_indicator_value(problem->suite, problem, indicator->name);
  indicator->target_hit = 0;
  indicator->evaluation_logged = 0;
  indicator->current_value = 0;
  indicator->additional_penalty = DBL_MAX;
  indicator->overall_value = 0;
  indicator->previous_value = 0;

  indicator->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  indicator->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, problem->number_of_variables);

  /* Prepare the info file */
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
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
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
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
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
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
    assert(problem->suite);
    fprintf(indicator->info_file,
        "suite = '%s', algorithm = '%s', indicator = '%s', folder = '%s', coco_version = '%s'\n%% %s",
        problem->suite->suite_name, observer->algorithm_name, indicator_name, problem->problem_type,
        coco_version, observer->algorithm_info);
    if (logger->log_nondom_mode == LOG_NONDOM_READ)
      fprintf(indicator->info_file, " (reconstructed)");
  }
  if ((observer_data->previous_function != problem->suite_dep_function)
    || (observer_data->previous_dimension != problem->number_of_variables)) {
    fprintf(indicator->info_file, "\nfunction = %2lu, ", (unsigned long) problem->suite_dep_function);
    fprintf(indicator->info_file, "dim = %2lu, ", (unsigned long) problem->number_of_variables);
    fprintf(indicator->info_file, "%s", file_name);
  }

  coco_free_memory(prefix);
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Output header information to the dat file */
  fprintf(indicator->dat_file, "%%\n%% index = %lu, name = %s\n", (unsigned long) problem->suite_dep_index,
      problem->problem_name);
  fprintf(indicator->dat_file, "%% instance = %lu, reference value = %.*e\n",
      (unsigned long) problem->suite_dep_instance, logger->precision_f, indicator->best_value);
  fprintf(indicator->dat_file, "%% function evaluation | indicator value | target hit\n");

  /* Output header information to the tdat file */
  fprintf(indicator->tdat_file, "%%\n%% index = %lu, name = %s\n", (unsigned long) problem->suite_dep_index,
      problem->problem_name);
  fprintf(indicator->tdat_file, "%% instance = %lu, reference value = %.*e\n",
      (unsigned long) problem->suite_dep_instance, logger->precision_f, indicator->best_value);
  fprintf(indicator->tdat_file, "%% function evaluation | indicator value\n");

  return indicator;
}

/**
 * @brief Outputs the final information about this indicator.
 */
static void logger_biobj_indicator_finalize(logger_biobj_indicator_t *indicator, const logger_biobj_data_t *logger) {

  /* Log the last eval_number in the dat file if wasn't already logged */
  if (!indicator->target_hit) {
    fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", (unsigned long) logger->number_of_evaluations,
        logger->precision_f, indicator->overall_value, logger->precision_f,
        ((coco_observer_targets_t *) indicator->targets)->value);
  }

  /* Log the last eval_number in the tdat file if wasn't already logged */
  if (!indicator->evaluation_logged) {
    fprintf(indicator->tdat_file, "%lu\t%.*e\n", (unsigned long) logger->number_of_evaluations,
        logger->precision_f, indicator->overall_value);
  }

  /* Log the information in the info file */
  fprintf(indicator->info_file, ", %lu:%lu|%.1e", (unsigned long) logger->suite_dep_instance,
      (unsigned long) logger->number_of_evaluations, indicator->overall_value);
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

/*
 * @brief Outputs the information according to the observer options.
 *
 * Outputs to the:
 * - dat file, if the archive was updated and a new target was reached for an indicator;
 * - tdat file, if the number of evaluations matches one of the predefined numbers.
 *
 * Note that a target is reached when
 * best_value - current_value + additional_penalty <= relative_target_value
 *
 * The relative_target_value is a target for indicator difference, not the actual indicator value!
 */
static void logger_biobj_output(logger_biobj_data_t *logger,
                                const int update_performed,
                                const logger_biobj_avl_item_t *node_item) {

  size_t i, j;
  logger_biobj_indicator_t *indicator;

  if (logger->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {

      indicator = logger->indicators[i];
      indicator->target_hit = 0;
      indicator->previous_value = indicator->overall_value;

      /* If the update was performed, update the overall indicator value */
      if (update_performed) {
        /* Compute the overall_value of an indicator */
        if (strcmp(indicator->name, "hyp") == 0) {
          if (coco_double_almost_equal(indicator->current_value, 0, mo_precision)) {
            /* Update the additional penalty for hypervolume (the minimal distance from the nondominated set
             * to the ROI) */
            double new_distance = mo_get_distance_to_ROI(node_item->normalized_y, logger->number_of_objectives);
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
        fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", (unsigned long) logger->number_of_evaluations,
            logger->precision_f, indicator->overall_value, logger->precision_f,
            ((coco_observer_targets_t *) indicator->targets)->value);
      }

      if (logger->log_nondom_mode == LOG_NONDOM_READ) {
        /* Log to the tdat file the previous indicator value if any evaluation number between the previous and
         * this one matches one of the predefined evaluation numbers. */
        for (j = logger->previous_evaluations + 1; j < logger->number_of_evaluations; j++) {
          indicator->evaluation_logged = coco_observer_evaluations_trigger(indicator->evaluations, j);
          if (indicator->evaluation_logged) {
            fprintf(indicator->tdat_file, "%lu\t%.*e\n", (unsigned long) j, logger->precision_f,
                indicator->previous_value);
          }
        }
      }

      /* Log to the tdat file if the number of evaluations matches one of the predefined numbers */
      indicator->evaluation_logged = coco_observer_evaluations_trigger(indicator->evaluations,
          logger->number_of_evaluations);
      if (indicator->evaluation_logged) {
        fprintf(indicator->tdat_file, "%lu\t%.*e\n", (unsigned long) logger->number_of_evaluations,
            logger->precision_f, indicator->overall_value);
      }

    }
  }
}

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information according to
 * observer options.
 */
static void logger_biobj_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_biobj_data_t *logger;
  logger_biobj_avl_item_t *node_item;
  int update_performed;
  coco_problem_t *inner_problem;

  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* Evaluate function */
  coco_evaluate_function(inner_problem, x, y);
  logger->number_of_evaluations++;

  node_item = logger_biobj_node_create(inner_problem, x, y, logger->number_of_evaluations, logger->number_of_variables,
      logger->number_of_objectives);

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in
   * the archive */
  update_performed = logger_biobj_tree_update(logger, node_item);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to
   * nondom_file */
  if (update_performed && (logger->log_nondom_mode == LOG_NONDOM_ALL)) {
    logger_biobj_tree_output(logger->adat_file, logger->buffer_tree, logger->number_of_variables,
        logger->number_of_integer_variables, logger->number_of_objectives, logger->log_vars,
        logger->precision_x, logger->precision_f, logger->log_discrete_as_int);
    avl_tree_purge(logger->buffer_tree);

    /* Flush output so that impatient users can see progress. */
    fflush(logger->adat_file);
  }

  /* Output according to observer options */
  logger_biobj_output(logger, update_performed, node_item);
}

/**
 * Sets the number of evaluations, adds the objective vector to the archive and outputs information according
 * to observer options (but does not output the archive).
 *
 * @note Vector y must point to a correctly sized allocated memory region and the given evaluation number must
 * be larger than the existing one.
 *
 * @param problem The given COCO problem.
 * @param evaluation The number of evaluations.
 * @param y The objective vector.
 * @return 1 if archive was updated was done and 0 otherwise.
 */
int coco_logger_biobj_feed_solution(coco_problem_t *problem, const size_t evaluation, const double *y) {

  logger_biobj_data_t *logger;
  logger_biobj_avl_item_t *node_item;
  int update_performed;
  coco_problem_t *inner_problem;
  double *x;
  size_t i;

  assert(problem != NULL);
  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  assert(logger->log_nondom_mode == LOG_NONDOM_READ);

  /* Set the number of evaluations */
  logger->previous_evaluations = logger->number_of_evaluations;
  if (logger->previous_evaluations >= evaluation)
    coco_error("coco_logger_biobj_reconstruct(): Evaluation %lu came before evaluation %lu. Note that "
        "the evaluations need to be always increasing.", logger->previous_evaluations, evaluation);
  logger->number_of_evaluations = evaluation;

  /* Update the archive with the new solution */
  x = coco_allocate_vector(problem->number_of_variables);
  for (i = 0; i < problem->number_of_variables; i++)
    x[i] = 0;
  node_item = logger_biobj_node_create(inner_problem, x, y, logger->number_of_evaluations,
      logger->number_of_variables, logger->number_of_objectives);
  coco_free_memory(x);

  /* Update the archive */
  update_performed = logger_biobj_tree_update(logger, node_item);

  /* Output according to observer options */
  logger_biobj_output(logger, update_performed, node_item);

  return update_performed;
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
      logger->number_of_integer_variables, logger->number_of_objectives, logger->log_vars,
      logger->precision_x, logger->precision_f, logger->log_discrete_as_int);

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

  if (((logger->log_nondom_mode == LOG_NONDOM_ALL) || (logger->log_nondom_mode == LOG_NONDOM_FINAL)) &&
      (logger->adat_file != NULL)) {
    fprintf(logger->adat_file, "%% evaluations = %lu\n", (unsigned long) logger->number_of_evaluations);
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
  logger_biobj_data_t *logger_data;
  observer_biobj_data_t *observer_data;
  const char nondom_folder_name[] = "archive";
  char *path_name, *file_name = NULL;
  size_t i;

  if (inner_problem->number_of_objectives != 2) {
    coco_error("logger_biobj(): The bi-objective logger cannot log a problem with %d objective(s)",
        inner_problem->number_of_objectives);
    return NULL; /* Never reached. */
  }

  logger_data = (logger_biobj_data_t *) coco_allocate_memory(sizeof(*logger_data));

  logger_data->number_of_evaluations = 0;
  logger_data->previous_evaluations = 0;
  logger_data->number_of_variables = inner_problem->number_of_variables;
  logger_data->number_of_integer_variables = inner_problem->number_of_integer_variables;
  logger_data->number_of_objectives = inner_problem->number_of_objectives;
  logger_data->suite_dep_instance = inner_problem->suite_dep_instance;

  observer_data = (observer_biobj_data_t *) observer->data;
  /* Copy values from the observes that you might need even if they do not exist any more */
  logger_data->log_nondom_mode = observer_data->log_nondom_mode;
  logger_data->compute_indicators = observer_data->compute_indicators;
  logger_data->precision_x = observer->precision_x;
  logger_data->precision_f = observer->precision_f;
  logger_data->log_discrete_as_int = observer->log_discrete_as_int;

  if (((observer_data->log_vars_mode == LOG_VARS_LOW_DIM) && (inner_problem->number_of_variables > 5))
      || (observer_data->log_vars_mode == LOG_VARS_NEVER))
    logger_data->log_vars = 0;
  else
    logger_data->log_vars = 1;

  /* Initialize logging of nondominated solutions into the archive file */
  if ((logger_data->log_nondom_mode == LOG_NONDOM_ALL) ||
      (logger_data->log_nondom_mode == LOG_NONDOM_FINAL)) {

    /* Create the path to the file */
    path_name = coco_allocate_string(COCO_PATH_MAX + 1);
    memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
    coco_join_path(path_name, COCO_PATH_MAX, nondom_folder_name, NULL);
    coco_create_directory(path_name);

    /* Construct file name */
    if (logger_data->log_nondom_mode == LOG_NONDOM_ALL)
      file_name = coco_strdupf("%s_nondom_all.adat", inner_problem->problem_id);
    else if (logger_data->log_nondom_mode == LOG_NONDOM_FINAL)
      file_name = coco_strdupf("%s_nondom_final.adat", inner_problem->problem_id);
    coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
    coco_free_memory(file_name);

    /* Open and initialize the archive file */
    logger_data->adat_file = fopen(path_name, "a");
    if (logger_data->adat_file == NULL) {
      coco_error("logger_biobj() failed to open file '%s'.", path_name);
      return NULL; /* Never reached */
    }
    coco_free_memory(path_name);

    /* Output header information */
    fprintf(logger_data->adat_file, "%% instance = %lu, name = %s\n",
        (unsigned long) inner_problem->suite_dep_instance, inner_problem->problem_name);
    if (logger_data->log_vars) {
      fprintf(logger_data->adat_file, "%% function evaluation | %lu objectives | %lu variables\n",
          (unsigned long) inner_problem->number_of_objectives,
          (unsigned long) inner_problem->number_of_variables);
    } else {
      fprintf(logger_data->adat_file, "%% function evaluation | %lu objectives \n",
          (unsigned long) inner_problem->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  logger_data->archive_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_last_objective,
      (avl_free_t) logger_biobj_node_free);
  logger_data->buffer_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_eval_number, NULL);

  /* Initialize the indicators */
  if (logger_data->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++)
      logger_data->indicators[i] = logger_biobj_indicator(logger_data, observer, inner_problem, logger_biobj_indicators[i]);

    observer_data->previous_function = (long) inner_problem->suite_dep_function;
    observer_data->previous_dimension = (long) inner_problem->number_of_variables;
  }

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_biobj_free, observer->observer_name);
  problem->evaluate_function = logger_biobj_evaluate;

  return problem;
}
#line 358 "code-experiments/src/coco_observer.c"
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

  observer_toy_data_t *observer_data;
  char *string_value;
  char *file_name;

  /* Sets the valid keys for toy observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "file_name" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_toy_data_t *) coco_allocate_memory(sizeof(*observer_data));

  /* Read file_name and number_of_targets from the options and use them to initialize the observer */
  string_value = coco_allocate_string(COCO_PATH_MAX + 1);
  if (coco_options_read_string(options, "file_name", string_value) == 0) {
    strcpy(string_value, "first_hitting_times.dat");
  }

  /* Open log_file */
  file_name = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(file_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_create_directory(file_name);
  coco_join_path(file_name, COCO_PATH_MAX, string_value, NULL);

  observer_data->log_file = fopen(file_name, "a");
  if (observer_data->log_file == NULL) {
    coco_error("observer_toy(): failed to open file %s.", file_name);
    return; /* Never reached */
  }

  coco_free_memory(string_value);
  coco_free_memory(file_name);

  observer->logger_allocate_function = logger_toy;
  observer->logger_free_function = logger_toy_free;
  observer->data_free_function = observer_toy_free;
  observer->data = observer_data;
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
    fprintf(logger->log_file, "%lu\t%.*e\t%.*e", (unsigned long) logger->number_of_evaluations,
    		logger->precision_f, y[0], logger->precision_f, logger->targets->value);
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

  logger_toy_data_t *logger_data;
  coco_problem_t *problem;

  if (inner_problem->number_of_objectives != 1) {
    coco_warning("logger_toy(): The toy logger shouldn't be used to log a problem with %d objectives",
        inner_problem->number_of_objectives);
  }

  /* Initialize the logger_toy_data_t object instance */
  logger_data = (logger_toy_data_t *) coco_allocate_memory(sizeof(*logger_data));
  logger_data->number_of_evaluations = 0;
  logger_data->targets = coco_observer_targets(observer->number_target_triggers, observer->target_precision);
  logger_data->log_file = ((observer_toy_data_t *) observer->data)->log_file;
  logger_data->precision_x = observer->precision_x;
  logger_data->precision_f = observer->precision_f;

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_toy_free, observer->observer_name);
  problem->evaluate_function = logger_toy_evaluate;

  /* Output initial information */
  assert(coco_problem_get_suite(inner_problem));
  fprintf(logger_data->log_file, "\n");
  fprintf(logger_data->log_file, "suite = '%s', problem_id = '%s', problem_name = '%s', coco_version = '%s'\n",
          coco_problem_get_suite(inner_problem)->suite_name, coco_problem_get_id(inner_problem),
          coco_problem_get_name(inner_problem), coco_version);
  fprintf(logger_data->log_file, "%% evaluation number | function value | target hit | %lu variables \n",
  		(unsigned long) inner_problem->number_of_variables);

  return problem;
}
#line 359 "code-experiments/src/coco_observer.c"
#line 1 "code-experiments/src/logger_rw.c"
/**
 * @file logger_rw.c
 * @brief Implementation of the real-world logger.
 *
 * Can be used to log all (or just those that are better than the preceding) solutions with information
 * about objectives, decision variables (optional) and constraints (optional). See observer_rw() for
 * more information on the options. Produces one "txt" file for each problem function, dimension and
 * instance.
 *
 * @note This logger can be used with single- and multi-objective problems, but in the multi-objective
 * case, all solutions are always logged.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#line 19 "code-experiments/src/logger_rw.c"
#line 20 "code-experiments/src/logger_rw.c"

#line 22 "code-experiments/src/logger_rw.c"
#line 23 "code-experiments/src/logger_rw.c"
#line 24 "code-experiments/src/logger_rw.c"
#line 1 "code-experiments/src/observer_rw.c"
/**
 * @file observer_rw.c
 * @brief Implementation of an observer for real-world problems.
 */

#line 7 "code-experiments/src/observer_rw.c"
#line 8 "code-experiments/src/observer_rw.c"

#line 10 "code-experiments/src/observer_rw.c"

/** @brief Enum for denoting when the decision variables and constraints are logged. */
typedef enum {
  LOG_NEVER, LOG_LOW_DIM, LOG_ALWAYS
} observer_rw_log_e;

/**
 * @brief The real-world observer data type.
 */
typedef struct {
  observer_rw_log_e log_vars_mode;   /**< @brief When the decision variables are logged. */
  observer_rw_log_e log_cons_mode;   /**< @brief When the constraints are logged. */
  size_t low_dim_vars;               /**< @brief "Low dimension" for decision variables. */
  size_t low_dim_cons;               /**< @brief "Low dimension" for constraints. */
  int log_only_better;               /**< @brief Whether to log only solutions that are better than previous
                                                 ones (only for the single-objective problems). */
  int log_time;                      /**< @brief Whether to log time. */
} observer_rw_data_t;

static coco_problem_t *logger_rw(coco_observer_t *observer, coco_problem_t *problem);
static void logger_rw_free(void *logger);

/**
 * @brief Initializes the observer for real-world problems.
 *
 * Possible options:
 *
 * - "log_variables: STRING" determines whether the decision variables are to be logged. STRING can take on
 * the values "none" (don't output decision variables), "low_dim"(output decision variables only for
 * dimensions lower or equal to low_dim_vars) and "all" (output all decision variables). The default value
 * is "all".
 *
 * - "log_constraints: STRING" determines whether the constraints are to be logged. STRING can take on the
 * values "none" (don't output constraints), "low_dim"(output constraints only for dimensions lower or equal
 * to low_dim_cons) and "all" (output all constraints). The default value is "all".
 *
 * - "low_dim_vars: VALUE" determines the value used to define "low_dim" for decision variables. The default
 * value is 10.
 *
 * - "low_dim_cons: VALUE" determines the value used to define "low_dim" for constraints. The default value
 * is 10.
 *
 * - "log_only_better: 0/1" determines whether all solutions are logged (0) or only the ones that are better
 * than previous ones (1). This is applicable only for the single-objective problems, where the default value
 * is 1, while for multi-objective problems all solutions are always logged.
 *
 * - "log_time: 0/1" determines whether the time needed to evaluate each solution is logged (0) or not (1).
 * The default value is 0.
 */
static void observer_rw(coco_observer_t *observer, const char *options, coco_option_keys_t **option_keys) {

  observer_rw_data_t *observer_data;
  char string_value[COCO_PATH_MAX + 1];

  /* Sets the valid keys for rw observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "log_variables", "log_constraints", "low_dim_vars", "low_dim_cons",
      "log_only_better", "log_time" };
  *option_keys = coco_option_keys_allocate(sizeof(known_keys) / sizeof(char *), known_keys);

  observer_data = (observer_rw_data_t *) coco_allocate_memory(sizeof(*observer_data));

  observer_data->log_vars_mode = LOG_ALWAYS;
  if (coco_options_read_string(options, "log_variables", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_vars_mode = LOG_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_vars_mode = LOG_ALWAYS;
    else if (strcmp(string_value, "low_dim") == 0)
      observer_data->log_vars_mode = LOG_LOW_DIM;
  }

  observer_data->log_cons_mode = LOG_ALWAYS;
  if (coco_options_read_string(options, "log_constraints", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
      observer_data->log_cons_mode = LOG_NEVER;
    else if (strcmp(string_value, "all") == 0)
      observer_data->log_cons_mode = LOG_ALWAYS;
    else if (strcmp(string_value, "low_dim") == 0)
      observer_data->log_cons_mode = LOG_LOW_DIM;
  }

  if (coco_options_read_size_t(options, "low_dim_vars", &(observer_data->low_dim_vars)) == 0)
    observer_data->low_dim_vars = 10;

  if (coco_options_read_size_t(options, "low_dim_cons", &(observer_data->low_dim_cons)) == 0)
    observer_data->low_dim_cons = 10;

  if (coco_options_read_int(options, "log_only_better", &(observer_data->log_only_better)) == 0)
    observer_data->log_only_better = 1;

  if (coco_options_read_int(options, "log_time", &(observer_data->log_time)) == 0)
    observer_data->log_time = 0;

  observer->logger_allocate_function = logger_rw;
  observer->logger_free_function = logger_rw_free;
  observer->data_free_function = NULL;
  observer->data = observer_data;
}
#line 25 "code-experiments/src/logger_rw.c"

/**
 * @brief The rw logger data type.
 *
 * @note Some fields from the observers (coco_observer as well as observer_rw) need to be copied here
 * because the observers can be deleted before the logger is finalized and we need these fields for
 * finalization.
 */
typedef struct {
  FILE *out_file;                /**< @brief File for logging. */
  size_t number_of_evaluations;  /**< @brief The number of evaluations performed so far. */

  double best_value;             /**< @brief The best-so-far value. */
  double current_value;          /**< @brief The current value. */

  int log_vars;                  /**< @brief Whether to log the decision values. */
  int log_cons;                  /**< @brief Whether to log the constraints. */
  int log_only_better;           /**< @brief Whether to log only solutions that are better than previous ones. */
  int log_time;                  /**< @brief Whether to log evaluation time. */

  int precision_x;               /**< @brief Precision for outputting decision values. */
  int precision_f;               /**< @brief Precision for outputting objective values. */
  int precision_g;               /**< @brief Precision for outputting constraint values. */
} logger_rw_data_t;

/**
 * @brief Evaluates the function and constraints and outputs the information according to the
 * observer options.
 */
static void logger_rw_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_rw_data_t *logger;
  coco_problem_t *inner_problem;
  double *constraints;
  size_t i;
  int log_this_time = 1;
  time_t start, end;

  logger = (logger_rw_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* Time the evaluations */
  if (logger->log_time)
    time(&start);

  /* Evaluate the objective(s) */
  coco_evaluate_function(inner_problem, x, y);
  logger->number_of_evaluations++;
  if (problem->number_of_objectives == 1)
    logger->current_value = y[0];

  /* Evaluate the constraints */
  if (problem->number_of_constraints > 0) {
    constraints = coco_allocate_vector(problem->number_of_constraints);
    inner_problem->evaluate_constraint(inner_problem, x, constraints);
  }

  /* Time the evaluations */
  if (logger->log_time)
    time(&end);

  /* Log to the output file */
  if ((problem->number_of_objectives == 1) && (logger->current_value < logger->best_value))
    logger->best_value = logger->current_value;
  else
    log_this_time = !logger->log_only_better;
  if (log_this_time) {
    fprintf(logger->out_file, "%lu\t", (unsigned long) logger->number_of_evaluations);
    for (i = 0; i < problem->number_of_objectives; i++)
      fprintf(logger->out_file, "%.*e\t", logger->precision_f, y[i]);
    if (logger->log_vars) {
      for (i = 0; i < problem->number_of_variables; i++)
        fprintf(logger->out_file, "%.*e\t", logger->precision_x, x[i]);
    }
    if (logger->log_cons) {
      for (i = 0; i < problem->number_of_constraints; i++)
        fprintf(logger->out_file, "%.*e\t", logger->precision_g, constraints[i]);
    }
    /* Log time in seconds */
    if (logger->log_time)
      fprintf(logger->out_file, "%.0f\t", difftime(end, start));
    fprintf(logger->out_file, "\n");
  }
  fflush(logger->out_file);

  if (problem->number_of_constraints > 0)
    coco_free_memory(constraints);
}

/**
 * @brief Frees the memory of the given rw logger.
 */
static void logger_rw_free(void *stuff) {

  logger_rw_data_t *logger;

  assert(stuff != NULL);
  logger = (logger_rw_data_t *) stuff;

  if (logger->out_file != NULL) {
    fclose(logger->out_file);
    logger->out_file = NULL;
  }
}

/**
 * @brief Initializes the rw logger.
 *
 * Copies all observer field values that are needed after initialization into logger field values for two
 * reasons:
 * - If the observer is deleted before the suite, the observer is not available anymore when the logger
 * is finalized.
 * - This reduces function calls.
 */
static coco_problem_t *logger_rw(coco_observer_t *observer, coco_problem_t *inner_problem) {

  coco_problem_t *problem;
  logger_rw_data_t *logger_data;
  observer_rw_data_t *observer_data;
  char *path_name, *file_name = NULL;

  logger_data = (logger_rw_data_t *) coco_allocate_memory(sizeof(*logger_data));
  logger_data->number_of_evaluations = 0;

  observer_data = (observer_rw_data_t *) observer->data;
  /* Copy values from the observes that you might need even if they do not exist any more */
  logger_data->precision_x = observer->precision_x;
  logger_data->precision_f = observer->precision_f;
  logger_data->precision_g = observer->precision_g;

  if (((observer_data->log_vars_mode == LOG_LOW_DIM) &&
      (inner_problem->number_of_variables > observer_data->low_dim_vars))
      || (observer_data->log_vars_mode == LOG_NEVER))
    logger_data->log_vars = 0;
  else
    logger_data->log_vars = 1;

  if (((observer_data->log_cons_mode == LOG_LOW_DIM) &&
      (inner_problem->number_of_constraints > observer_data->low_dim_cons))
      || (observer_data->log_cons_mode == LOG_NEVER)
      || (inner_problem->number_of_constraints == 0))
    logger_data->log_cons = 0;
  else
    logger_data->log_cons = 1;

  logger_data->log_only_better = (observer_data->log_only_better) &&
      (inner_problem->number_of_objectives == 1);
  logger_data->log_time = observer_data->log_time;

  logger_data->best_value = DBL_MAX;
  logger_data->current_value = DBL_MAX;

  /* Construct file name */
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_create_directory(path_name);
  file_name = coco_strdupf("%s_rw.txt", coco_problem_get_id(inner_problem));
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);

  /* Open and initialize the output file */
  logger_data->out_file = fopen(path_name, "a");
  if (logger_data->out_file == NULL) {
    coco_error("logger_rw() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(path_name);
  coco_free_memory(file_name);

  /* Output header information */
  fprintf(logger_data->out_file, "\n%% suite = '%s', problem_id = '%s', problem_name = '%s', coco_version = '%s'\n",
          coco_problem_get_suite(inner_problem)->suite_name, coco_problem_get_id(inner_problem),
          coco_problem_get_name(inner_problem), coco_version);
  fprintf(logger_data->out_file, "%% evaluation | %lu objective",
      (unsigned long) inner_problem->number_of_objectives);
  if (inner_problem->number_of_objectives > 1)
    fprintf(logger_data->out_file, "s");
  if (logger_data->log_vars)
    fprintf(logger_data->out_file, " | %lu variable",
        (unsigned long) inner_problem->number_of_variables);
  if (inner_problem->number_of_variables > 1)
    fprintf(logger_data->out_file, "s");
  if (logger_data->log_cons)
    fprintf(logger_data->out_file, " | %lu constraint",
        (unsigned long) inner_problem->number_of_constraints);
  if (inner_problem->number_of_constraints > 1)
    fprintf(logger_data->out_file, "s");
  if (logger_data->log_time)
    fprintf(logger_data->out_file, " | evaluation time (s)");
  fprintf(logger_data->out_file, "\n");

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_rw_free, observer->observer_name);
  problem->evaluate_function = logger_rw_evaluate;

  return problem;
}
#line 360 "code-experiments/src/coco_observer.c"

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
 * - "precision_g: VALUE" defines the precision used when outputting constraints and corresponds to the number
 * of digits to be printed after the decimal point. The default value is 3.
 * - "log_discrete_as_int: VALUE" determines whether the values of integer variables (in mixed-integer problems)
 * are logged as integers (1) or not (0 - in this case they are logged as doubles). The default value is 1.
 *
 * @return The constructed observer object or NULL if observer_name equals NULL, "" or "no_observer".
 */
coco_observer_t *coco_observer(const char *observer_name, const char *observer_options) {

  coco_observer_t *observer;
  char *path, *result_folder, *algorithm_name, *algorithm_info;
  const char *outer_folder_name = "exdata";
  int precision_x, precision_f, precision_g, log_discrete_as_int;

  size_t number_target_triggers;
  size_t number_evaluation_triggers;
  double target_precision;
  char *base_evaluation_triggers;

  coco_option_keys_t *known_option_keys, *given_option_keys, *additional_option_keys, *redundant_option_keys;

  /* Sets the valid keys for observer options
   * IMPORTANT: This list should be up-to-date with the code and the documentation */
  const char *known_keys[] = { "result_folder", "algorithm_name", "algorithm_info",
      "number_target_triggers", "target_precision", "number_evaluation_triggers", "base_evaluation_triggers",
      "precision_x", "precision_f", "precision_g", "log_discrete_as_int" };
  additional_option_keys = NULL; /* To be set by the chosen observer */

  if (0 == strcmp(observer_name, "no_observer")) {
    return NULL;
  } else if (strlen(observer_name) == 0) {
    coco_warning("Empty observer_name has no effect. To prevent this warning use 'no_observer' instead");
    return NULL;
  }

  result_folder = coco_allocate_string(COCO_PATH_MAX + 1);
  algorithm_name = coco_allocate_string(COCO_PATH_MAX + 1);
  algorithm_info = coco_allocate_string(5 * COCO_PATH_MAX);
  /* Read result_folder, algorithm_name and algorithm_info from the observer_options and use
   * them to initialize the observer */
  if (coco_options_read_string(observer_options, "result_folder", result_folder) == 0) {
    strcpy(result_folder, "default");
  }
  /* Create the result_folder inside the "exdata" folder */
  path = coco_allocate_string(COCO_PATH_MAX + 1);
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

  precision_g = 3;
  if (coco_options_read_int(observer_options, "precision_g", &precision_g) != 0) {
    if ((precision_g < 1) || (precision_g > 32))
      precision_g = 3;
  }

  log_discrete_as_int = 1;
  if (coco_options_read_int(observer_options, "log_discrete_as_int", &log_discrete_as_int) != 0) {
    if ((log_discrete_as_int < 0) || (log_discrete_as_int > 1))
      log_discrete_as_int = 1;
  }

  observer = coco_observer_allocate(path, observer_name, algorithm_name, algorithm_info,
      number_target_triggers, target_precision, number_evaluation_triggers, base_evaluation_triggers,
      precision_x, precision_f, precision_g, log_discrete_as_int);

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
  } else if (0 == strcmp(observer_name, "bbob-biobj-ext")) {
    observer_biobj(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "bbob-largescale")) {
    observer_bbob(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "bbob-constrained")) {
    observer_bbob(observer, observer_options, &additional_option_keys);
  } else if (0 == strcmp(observer_name, "rw")) {
    observer_rw(observer, observer_options, &additional_option_keys);
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
 * @return The observed problem in the form of a new COCO problem instance or the same problem if the
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
 * @return The unobserved problem as a pointer to the inner problem or the same problem if the problem
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

/**
 * Get the result folder name, which is a unique folder name constructed
 * from the result_folder option.
 *
 * @param observer The COCO observer, whose logger may be wrapping a problem.
 *
 * @return The result folder name, where the logger writes its output.
 */
const char *coco_observer_get_result_folder(const coco_observer_t *observer) {
  if (observer == NULL) {
    coco_warning("coco_observer_get_result_folder: no observer to get result_folder from");
    return "";
  }
  else if (observer->is_active == 0) {
    coco_warning("coco_observer_get_result_folder: observer is not active, returning empty string");
    return "";
  }
  return observer->result_folder;
}

#line 1 "code-experiments/src/coco_archive.c"
/**
 * @file coco_archive.c
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
 *
 * Contains information on the rounded normalized objective values (normalized_y), which are used for
 * computing the indicators and the text, which is used for output.
 */
typedef struct {
  double *normalized_y;      /**< @brief The values of normalized objectives of this solution. */
  char *text;                /**< @brief The text describing the solution (the whole line of the archive). */
} coco_archive_avl_item_t;

/**
 * @brief Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static coco_archive_avl_item_t* coco_archive_node_item_create(const double *y,
                                                              const double *ideal,
                                                              const double *nadir,
                                                              const size_t num_obj,
                                                              const char *text) {

  /* Allocate memory to hold the data structure coco_archive_avl_item_t */
  coco_archive_avl_item_t *item = (coco_archive_avl_item_t*) coco_allocate_memory(sizeof(*item));

  /* Compute the normalized y */
  item->normalized_y = mo_normalize(y, ideal, nadir, num_obj);

  item->text = coco_strdup(text);
  return item;
}

/**
 * @brief Frees the data of the given coco_archive_avl_item_t.
 */
static void coco_archive_node_item_free(coco_archive_avl_item_t *item, void *userdata) {
  coco_free_memory(item->normalized_y);
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
  if (coco_double_almost_equal(item1->normalized_y[1], item2->normalized_y[1], mo_precision))
    return 0;
  else if (item1->normalized_y[1] < item2->normalized_y[1])
    return -1;
  else
    return 1;

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
  char *suite_instance = coco_strdupf("instances: %lu", (unsigned long) instance);
  char *suite_options = coco_strdupf("dimensions: %lu function_indices: %lu",
  		(unsigned long) dimension, (unsigned long) function);
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
    coco_error("coco_archive(): cannot create problem f%02lu_i%02lu_d%02lu in suite '%s'",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension, suite_name);
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

int coco_archive_add_solution(coco_archive_t *archive, const double y1, const double y2, const char *text) {

  coco_archive_avl_item_t* insert_item;
  avl_node_t *node, *next_node;
  int update = 0;
  int dominance;

  double *y = coco_allocate_vector(2);
  y[0] = y1;
  y[1] = y2;
  insert_item = coco_archive_node_item_create(y, archive->ideal, archive->nadir,
      archive->number_of_objectives, text);
  coco_free_memory(y);

  /* Find the first point that is not worse than the new point (NULL if such point does not exist) */
  node = avl_item_search_right(archive->tree, insert_item, NULL);

  if (node == NULL) {
    /* The new point is an extreme point */
    update = 1;
    next_node = archive->tree->head;
  } else {
    dominance = mo_get_dominance(insert_item->normalized_y, ((coco_archive_avl_item_t*) node->item)->normalized_y,
        archive->number_of_objectives);
    if (dominance > -1) {
      update = 1;
      next_node = node->next;
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
      	assert((node != archive->extreme1) && (node != archive->extreme2));
      	avl_node_delete(archive->tree, node);
      }
    } else {
      /* The new point is dominated or equal to an existing one, ignore */
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
      dominance = mo_get_dominance(insert_item->normalized_y, ((coco_archive_avl_item_t*) node->item)->normalized_y,
          archive->number_of_objectives);
      if (dominance == 1) {
        next_node = node->next;
        /* The new point dominates the next point, remove the next point */
        assert((node != archive->extreme1) && (node != archive->extreme2));
      	avl_node_delete(archive->tree, node);
      } else {
        break;
      }
    }

    if(avl_item_insert(archive->tree, insert_item) == NULL) {
      coco_warning("Solution %s did not update the archive", text);
      update = 0;
    }

    archive->is_up_to_date = 0;
  }

  return update;
}

/**
 * @brief Updates the archive fields returned by the getters.
 */
static void coco_archive_update(coco_archive_t *archive) {

  double hyp;

  if (!archive->is_up_to_date) {

    avl_node_t *node, *left_node;
    coco_archive_avl_item_t *node_item, *left_node_item;

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
      if (mo_is_within_ROI(left_node_item->normalized_y, archive->number_of_objectives)) {
        hyp = 0;
        if (mo_is_within_ROI(node_item->normalized_y, archive->number_of_objectives))
          hyp = (node_item->normalized_y[0] - left_node_item->normalized_y[0]) * (1 - left_node_item->normalized_y[1]);
        else
          hyp = (1 - left_node_item->normalized_y[0]) * (1 - left_node_item->normalized_y[1]);
        assert(hyp >= 0);
         archive->hypervolume += hyp;
      }
      node = left_node;
    }

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
