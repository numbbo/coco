#ifndef __NUMBBO_H__
#define __NUMBBO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h> /* For size_t */
#include <stdint.h>

/**
 * Our very own pi constant.
 */
static const double numbbo_pi = 3.14159265358979323846;

struct numbbo_problem;

typedef void (*numbbo_evaluate_function_t) (struct numbbo_problem *self, 
                                            double *x, 
                                            double *y);
typedef void (*numbbo_recommendation_function_t) (struct numbbo_problem *self,
                                                  double *x,
                                                  size_t number_of_solutions);

typedef void (*numbbo_free_function_t) (struct numbbo_problem *self);

/**
 * Description of a NUMBBO problem (instance)
 *
 * evaluate and free are opaque pointers which should not be called
 * directly. Instead they are used by the numbbo_* functions in
 * numbbo_generics.c. This indirection gives us the flexibility to add
 * generic checks or more sophisticated dispatch methods later on.
 *
 * Fields:
 *
 * number_of_parameters - Number of parameters expected by the
 *   function and contraints.
 *
 * number_of_objectives - Number of objectives.
 *
 * number_of_constraints - Number of constraints.
 *
 * lower_bounds, upper_bounds - Vector of length
 *   'number_of_parameters'. Lower/Upper bounds of parameter space.
 *
 * problem_name - Descriptive name for the test problem. May be NULL
 *   to indicate that no name is known.
 *
 * problem_id - Short name consisting of _only_ [a-z0-9], '-' and '_'
 *   that, when not NULL, must be unique. It can for example be used
 *   to generate valid directory names under which to store results.
 */
typedef struct numbbo_problem {
    numbbo_evaluate_function_t evaluate_function;
    numbbo_evaluate_function_t evaluate_constraint;
    numbbo_recommendation_function_t recommend_solutions;
    numbbo_free_function_t free_problem;
    size_t number_of_parameters;
    size_t number_of_objectives;
    size_t number_of_constraints;
    double *lower_bounds;
    double *upper_bounds;
    double *best_value;
    double *best_parameter;
    char *problem_name;
    char *problem_id;
} numbbo_problem_t;

typedef void (*numbbo_optimizer_t)(numbbo_problem_t *problem);

/**
 * numbbo_evaluate_function(self, x, y):
 *
 * Evaluate the NUMBB problem represented by ${self} with the
 * parameter settings ${x} and save the result in ${y}.
 *
 * Note: ${x} and ${y} are expected to be of the correct sizes.
 */
void numbbo_evaluate_function(numbbo_problem_t *self, double *x, double *y);

/**
 * numbbo_evaluate_constraint(self, x, y):
 *
 * Evaluate the constraints of the NUMBB problem represented by
 * ${self} with the parameter settings ${x} and save the result in
 * ${y}.
 *
 * Note: ${x} and ${y} are expected to be of the correct sizes.
 */
void numbbo_evaluate_constraint(numbbo_problem_t *self, double *x, double *y);

/**
 * numbbo_recommend_solutions(self, x, number_of_solutions):
 *
 * Recommend ${number_of_solutions} parameter settings (stored in
 * ${x}) as the current best guess solutions to the problem ${self}.
 */
void numbbo_recommend_solutions(numbbo_problem_t *self, 
                                double *x, size_t number_of_solutions);

/**
 * numbbo_free(self):
 *
 * Free the NUMBBO problem represented by ${self}.
 */
void numbbo_free_problem(numbbo_problem_t *self);

/**
 * numbbo_get_problem_name(self):
 *
 * Return the name of the NUMBBO problem ${self}.
 *
 * Note: Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 */
const char *numbbo_get_problem_name(numbbo_problem_t *self);

/**
 * numbbo_get_problem_id(self):
 *
 * Return the ID of the NUMBBO problem ${self}. The ID is guaranteed to
 * contain only characters in the set [a-z0-9_-]. It should therefore
 * be safe to use the ID to construct filenames or other identifiers.
 *
 * Note: Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 */
const char *numbbo_get_problem_id(numbbo_problem_t *self);

/**
 * numbbo_get_number_of_variables(self):
 *
 * Return the number of variables of the NUMBBO problem ${self}.
 */
const size_t numbbo_get_number_of_variables(const numbbo_problem_t *self);

/**
 * numbbo_get_problem(problem_suit, function_index):
 *
 * Get the ${function_index}-th problem of the ${problem_suit} test
 * suit. 
 */
numbbo_problem_t *numbbo_get_problem(const char *problem_suit,
                                     const int function_index);

/** tentative getters for region of interest 
*/
const double * numbbo_get_lowest_values_of_interest(const numbbo_problem_t *self);
const double * numbbo_get_highest_values_of_interest(const numbbo_problem_t *self);

/** tentative getter for initial variable vector 
*/
const double * numbbo_get_initial_solution(const numbbo_problem_t *self);


/**
 * numbbo_observe_problem(observer_name, problem, options):;
 *
 * Add the observer named ${observer_name} to ${problem}. An
 * observer is a wrapper around a numbbo_problem_t. This allows the
 * observer to see all interactions between the algorithm and the
 * optimization problem.
 *
 * ${options} is a string that can be used to pass options to an
 * observer. The format is observer dependent.
 *
 * NOTE: There is a special observer names "no_observer" which simply
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
 * numbbo_new_random(seed):
 *
 * Create a new random number stream using ${seed} and return its state.
 */
numbbo_random_state_t *numbbo_new_random(uint32_t seed);

/**
 * numbbo_free_random(state):
 *
 * Free all memory associated with the RNG ${state}.
 */
void numbbo_free_random(numbbo_random_state_t *state);

/**
 * numbbo_uniform_random(state):
 *
 * Return one uniform [0, 1) random value from the random number
 * generator associated with ${state}.
 */
double numbbo_uniform_random(numbbo_random_state_t *state);

/**
 * numbbo_normal_random(state):
 *
 * Generate an approximately normal random number.
 *
 * Instead of using the (expensive) polar method, we may cheat and
 * abuse the central limit theorem. The sum of 12 uniform RVs has mean
 * 6, variance 1 and is approximately normal. Subtract 6 and you get
 * an approximately N(0, 1) random number.
 */
double numbbo_normal_random(numbbo_random_state_t *state);

/**
 * numbbo_error(message):
 * numbbo_warning(message):
 *
 * Functions to signal error conditions.
 */
void numbbo_error(const char *message);
void numbbo_warning(const char *message);


/* Memory managment routines. 
 *
 * Their implementation may never fail. They either return a valid
 * pointer or terminate the program.
 */
void *numbbo_allocate_memory(size_t size);
void numbbo_free_memory(void *data);

#ifdef __cplusplus
}
#endif
#endif

