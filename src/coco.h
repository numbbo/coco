/*
 * Public NumBBO interface
 *
 * All public functions, constants and variables are defined in this
 * file. It is the authorative reference, if any function deviates
 * from the documented behaviour it is considered a bug.
 */
#ifndef __NUMBBO_H__
#define __NUMBBO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> /* For size_t */
#include <stdint.h>

/**
 * Our very own pi constant. Simplifies the case, when the value of pi changes.
 */
static const double coco_pi = 3.14159265358979323846;
static const double coco_two_pi = 2.0 * 3.14159265358979323846;

struct coco_problem;
typedef struct coco_problem coco_problem_t;
typedef void (*coco_optimizer_t)(coco_problem_t *problem);

/**
 * Evaluate the NUMBB problem represented by ${self} with the
 * parameter settings ${x} and save the result in ${y}.
 *
 * @note Both x and y must point to correctly sized allocated memory
 * regions.
 */
void coco_evaluate_function(coco_problem_t *self, double *x, double *y);

/**
 * Evaluate the constraints of the NUMBB problem represented by
 * ${self} with the parameter settings ${x} and save the result in
 * ${y}.
 *
 * Note: ${x} and ${y} are expected to be of the correct sizes.
 */
void coco_evaluate_constraint(coco_problem_t *self, double *x, double *y);

/**
 * Recommend ${number_of_solutions} parameter settings (stored in
 * ${x}) as the current best guess solutions to the problem ${self}.
 */
void coco_recommend_solutions(coco_problem_t *self,
                                double *x, size_t number_of_solutions);

/**
 * Free the NUMBBO problem represented by ${self}.
 */
void coco_free_problem(coco_problem_t *self);

/**
 * Return the name of a NUMBBO problem.
 *
 * @note Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 *
 * @see coco_strdup()
 */
const char *coco_get_problem_name(coco_problem_t *self);

/**
 * Return the ID of the NUMBBO problem ${self}. The ID is guaranteed to
 * contain only characters in the set [a-z0-9_-]. It should therefore
 * be safe to use the ID to construct filenames or other identifiers.
 *
 * @note Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 *
 * @see coco_strdup
 */
const char *coco_get_problem_id(coco_problem_t *self);

/**
 * Return the number of variables of a COCO problem.
 */
size_t coco_get_number_of_variables(const coco_problem_t *self);

/**
 * Return the number of objectives of a COCO problem.
 */
size_t coco_get_number_of_objectives(const coco_problem_t *self);

/**
 * Get the ${function_index}-th problem of the ${problem_suit} test
 * suit.
 */
coco_problem_t *coco_get_problem(const char *problem_suit,
                                     const int function_index);

/**
 * tentative getters for region of interest
 */
const double * coco_get_smallest_values_of_interest(const coco_problem_t *self);
const double * coco_get_largest_values_of_interest(const coco_problem_t *self);

/**
 * Return an initial solution, i.e. a feasible variable setting, to the
 * problem.
 *
 * By default, the center of the problems region of interest
 * is the initial solution.
 *
 * @see coco_get_smallest_values_of_interest() and coco_get_largest_values_of_interest()
 */
void coco_get_initial_solution(const coco_problem_t *self,
                                 double *initial_solution);

/**
 * Add the observer named ${observer_name} to ${problem}. An
 * observer is a wrapper around a coco_problem_t. This allows the
 * observer to see all interactions between the algorithm and the
 * optimization problem.
 *
 * ${options} is a string that can be used to pass options to an
 * observer. The format is observer dependent.
 *
 * @note There is a special observer names "no_observer" which simply
 * returns the original problem. This is largely to simplify the
 * interface design for interpreted languages. A short hand for this
 * observer is the empty string ("").
 */
coco_problem_t *coco_observe_problem(const char *observer_name,
                                         coco_problem_t *problem,
                                         const char *options);

void coco_benchmark(const char *problem_suit,
                      const char *observer,
                      const char *options,
                      coco_optimizer_t optimizer);

/**************************************************************************
 * Random number generator
 */

struct coco_random_state;
typedef struct coco_random_state coco_random_state_t;

/**
 * Create a new random number stream using ${seed} and return its state.
 */
coco_random_state_t *coco_new_random(uint32_t seed);

/**
 * Free all memory associated with the RNG state.
 */
void coco_free_random(coco_random_state_t *state);

/**
 * Return one uniform [0, 1) random value from the random number
 * generator associated with ${state}.
 */
double coco_uniform_random(coco_random_state_t *state);

/**
 * Generate an approximately normal random number.
 *
 * Instead of using the (expensive) polar method, we may cheat and
 * abuse the central limit theorem. The sum of 12 uniform RVs has mean
 * 6, variance 1 and is approximately normal. Subtract 6 and you get
 * an approximately N(0, 1) random number.
 */
double coco_normal_random(coco_random_state_t *state);

/**
 * Function to signal a fatal error conditions.
 */
void coco_error(const char *message);

/**
 * Function to warn about eror conditions.
 */
void coco_warning(const char *message);

/* Memory managment routines.
 *
 * Their implementation may never fail. They either return a valid
 * pointer or terminate the program.
 */
void *coco_allocate_memory(const size_t size);
double *coco_allocate_vector(const size_t size);
void coco_free_memory(void *data);


/**
 * Create a duplicate of a string and return a pointer to
 * it. The caller is responsible for releasing the allocated memory
 * using coco_free_memory().
 *
 * @see coco_free_memory()
 */
char *coco_strdup(const char *string);

/* FIXME: Find portable 64 bit integer type. */
typedef uint64_t coco_nanotime_t;

/**
 * Return a monotonic nanosecond value. Useful to measure execution times.
 */
coco_nanotime_t coco_get_nanotime(void);

#ifdef __cplusplus
}
#endif
#endif
