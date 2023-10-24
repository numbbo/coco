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

/* Macro to tag function declarations:
 *
 * COCO_NORETURN:
 *   Function never returns. Mainly useful for linters and static analysis 
 *   tools.
 *
 * COCO_UNUSED:
 *   Function is not used. Useful to suppress false warnings.
 *
 * The next two macros are useful to give hints to the optimizer. This can
 * lead to much bette code generation and ultimately faster runtimes.
 *
 * COCO_LIKELY(e):
 *   Mark expression `e` as likely true. Usually used in if() or while() 
 *   statements to give the compiler a hint.
 *
 * COCO_UNLIKELY(e):
 *   Mark expression `e` as likely false.
 *
 */
#ifdef __GNUC__
#define COCO_NORETURN __attribute__((noreturn))
#define COCO_UNUSED __attribute__((unused))
#define COCO_LIKELY(x) __builtin_expect((x),1)
#define COCO_UNLIKELY(x) __builtin_expect((x),0)
#elif __clang__
#define COCO_NORETURN __attribute__((noreturn))
#define COCO_UNUSED __attribute__((unused))
#define COCO_LIKELY(x) __builtin_expect((x),1)
#define COCO_UNLIKELY(x) __builtin_expect((x),0)
#elif _MSC_VER
#define COCO_NORETURN __declspec(noreturn)
#define COCO_UNUSED
#define COCO_LIKELY(x) (x)
#define COCO_UNLIKELY(x) (x)
#else
#define COCO_NORETURN 
#define COCO_UNUSED
#define COCO_LIKELY(x) (x)
#define COCO_UNLIKELY(x) (x)
#endif

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
static const char coco_version[32] = "$COCO_VERSION";
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
/**
 * @name Structures representing the additional arguments to be passed to the functions
 */
/**{@*/

/**
 * @brief The extra argument to be passed to the step ellipsoid function
 * only penalty scale, because in the legacy code is different 
 * between the noisy and the noise free implementations 
 */
typedef struct{
  double penalty_scale;
} f_step_ellipsoid_args_t;

/**
 * @brief The extra argument to be passed to the step ellipsoid function
 * only conditioning, because in the legacy code is different 
 * between the noisy and the noise free implementations 
 */
typedef struct{
  double conditioning;
} f_ellipsoid_args_t;

/**
 * @brief The extra argument to be passed to the step ellipsoid function
 * conditioning and penalty scale, because in the legacy code were different 
 * between the noisy and the noise free implementations 
 */
typedef struct{
  double conditioning;
  double penalty_scale;
} f_schaffers_args_t;

/**
 * @brief The extra argument to be passed to the step ellipsoid function
 * facftrue, because in the legacy code is different 
 * between the noisy and the noise free implementations 
 */
typedef struct{
  double facftrue;
}f_griewank_rosenbrock_args_t;

/**
 * @brief The extra argument to be passed to the step ellipsoid function
 * facftrue, because in the legacy code is different 
 * between the noisy and the noise free implementations 
 */
typedef struct{
  size_t number_of_peaks;
  double penalty_scale;
}f_gallagher_args_t;
/**@}*/

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
 * See coco_random_state_s for more information on its fields. 
 */
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
 * @name Methods regarding noisy problems
 */
/**@{*/

/**
 * @brief Resets seeds
 */
void reset_seeds(void);

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
void COCO_NORETURN coco_error(const char *message, ...);

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

void bbob_problem_best_parameter_print(const coco_problem_t *problem);
void bbob_biobj_problem_best_parameter_print(const coco_problem_t *problem);


#ifdef __cplusplus
}
#endif
#endif
