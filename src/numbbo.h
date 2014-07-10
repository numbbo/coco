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
static const double numbbo_pi = 3.14159265358979323846;

struct numbbo_problem;
typedef struct numbbo_problem numbbo_problem_t;

typedef void (*numbbo_optimizer_t)(numbbo_problem_t *problem);

/**
 * Evaluate the NUMBB problem represented by ${self} with the
 * parameter settings ${x} and save the result in ${y}.
 *
 * @note Both x and y must point to correctly sized allocated memory
 * regions.
 */
void numbbo_evaluate_function(numbbo_problem_t *self, double *x, double *y);

/**
 * Evaluate the constraints of the NUMBB problem represented by
 * ${self} with the parameter settings ${x} and save the result in
 * ${y}.
 *
 * Note: ${x} and ${y} are expected to be of the correct sizes.
 */
void numbbo_evaluate_constraint(numbbo_problem_t *self, double *x, double *y);

/**
 * Recommend ${number_of_solutions} parameter settings (stored in
 * ${x}) as the current best guess solutions to the problem ${self}.
 */
void numbbo_recommend_solutions(numbbo_problem_t *self, 
                                double *x, size_t number_of_solutions);

/**
 * Free the NUMBBO problem represented by ${self}.
 */
void numbbo_free_problem(numbbo_problem_t *self);

/**
 * Return the name of a NUMBBO problem.
 *
 * @note Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 *
 * @ref numbbo_strdup
 */
const char *numbbo_get_problem_name(numbbo_problem_t *self);

/**
 * Return the ID of the NUMBBO problem ${self}. The ID is guaranteed to
 * contain only characters in the set [a-z0-9_-]. It should therefore
 * be safe to use the ID to construct filenames or other identifiers.
 *
 * @note Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 *
 * @ref numbbo_strdup
 */
const char *numbbo_get_problem_id(numbbo_problem_t *self);

/**
 * Return the number of variables of a NUMBBO problem.
 */
const size_t numbbo_get_number_of_variables(const numbbo_problem_t *self);

/**
 * Get the ${function_index}-th problem of the ${problem_suit} test
 * suit. 
 */
numbbo_problem_t *numbbo_get_problem(const char *problem_suit,
                                     const int function_index);

/**
 * tentative getters for region of interest 
 */
const double * numbbo_get_smallest_values_of_interest(const numbbo_problem_t *self);
const double * numbbo_get_largest_values_of_interest(const numbbo_problem_t *self);

/**
 * Write initial variable values for the problem ${self} into
 * ${initial_solution}.
 *
 * @ref numbbo_get_smallest_values_of_interest
 * @ref numbbo_get_largest_values_of_interest
 */
void numbbo_get_initial_solution(const numbbo_problem_t *self, 
                                 double *initial_solution);

/**
 * Add the observer named ${observer_name} to ${problem}. An
 * observer is a wrapper around a numbbo_problem_t. This allows the
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
numbbo_problem_t *numbbo_observe_problem(const char *observer_name,
                                         numbbo_problem_t *problem,
                                         const char *options);
    
void numbbo_benchmark(const char *problem_suit,
                      const char *observer,
                      const char *options,
                      numbbo_optimizer_t optimizer);

/**************************************************************************
 * Random number generator
 */

struct numbbo_random_state;
typedef struct numbbo_random_state numbbo_random_state_t;

/**
 * Create a new random number stream using ${seed} and return its state.
 */
numbbo_random_state_t *numbbo_new_random(uint32_t seed);

/**
 * Free all memory associated with the RNG state.
 */
void numbbo_free_random(numbbo_random_state_t *state);

/**
 * Return one uniform [0, 1) random value from the random number
 * generator associated with ${state}.
 */
double numbbo_uniform_random(numbbo_random_state_t *state);

/**
 * Generate an approximately normal random number.
 *
 * Instead of using the (expensive) polar method, we may cheat and
 * abuse the central limit theorem. The sum of 12 uniform RVs has mean
 * 6, variance 1 and is approximately normal. Subtract 6 and you get
 * an approximately N(0, 1) random number.
 */
double numbbo_normal_random(numbbo_random_state_t *state);

/**
 * Function to signal a fatal error conditions.
 */
void numbbo_error(const char *message);

/**
 * Function to warn about eror conditions.
 */
void numbbo_warning(const char *message);

/* Memory managment routines. 
 *
 * Their implementation may never fail. They either return a valid
 * pointer or terminate the program.
 */
void *numbbo_allocate_memory(size_t size);
double *numbbo_allocate_vector(size_t size);
void numbbo_free_memory(void *data);


/**
 * Create a duplicate of a string and return a pointer to
 * it. The caller is responsible for releasing the allocated memory
 * using numbbo_free_memory().
 *
 * @ref numbbo_free_memory
 */
char *numbbo_strdup(const char *string);

#ifdef __cplusplus
}
#endif
#endif
