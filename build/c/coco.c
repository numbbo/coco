
/************************************************************************
 * WARNING
 *
 * This file is an auto-generated amalgamation. Any changes made to this
 * file will be lost when it is regenerated!
 ************************************************************************/

#line 1 "src/coco_benchmark.c"
#line 1 "src/coco_platform.h"
/*
 * Automatic platform dependent "configuration" of COCO framework
 *
 * Some platforms and standard conforming compilers require extra defines or
 * includes to provide some functionality. 
 *
 * Because most feature defines need to be set before the first system header
 * is included and we do not know when a system header is included for the
 * first time in the amalgamation, all internal files should include this file
 * before any system headers.
 *
 */
#ifndef __COCO_PLATFORM__ 
#define __COCO_PLATFORM__

/* Because C89 does not have a round() function, dance around and try to force
 * a definition.
 */
#if defined(unix) || defined(__unix__) || defined(__unix)
/* On Unix like platforms, force POSIX 2008 behaviour which gives us fmin(),
 * fmax(), round() and snprintf() even if we do not have a C99 compiler.
 */
#define _POSIX_C_SOURCE 200809L
#endif

#endif
#line 2 "src/coco_benchmark.c"

#include <string.h>
#include <stdio.h>

#line 1 "src/coco.h"
/*
 * Public CoCO/NumBBO experiments interface
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
#include <math.h> /* For NAN among other things */

#ifndef NAN
#define NAN 0.0 / 0.0
#endif

/**
 * Our very own pi constant. Simplifies the case, when the value of pi changes.
 */
static const double coco_pi = 3.14159265358979323846;
static const double coco_two_pi = 2.0 * 3.14159265358979323846;

struct coco_problem;
typedef struct coco_problem coco_problem_t;
typedef void (*coco_optimizer_t)(coco_problem_t *problem);

/**
 * Evaluate the NUMBBO problem represented by ${self} with the
 * parameter settings ${x} and save the result in ${y}.
 *
 * @note Both x and y must point to correctly sized allocated memory
 * regions.
 */
void coco_evaluate_function(coco_problem_t *self, const double *x, double *y);

/**
 * Evaluate the constraints of the NUMBB problem represented by
 * ${self} with the parameter settings ${x} and save the result in
 * ${y}.
 *
 * @note ${x} and ${y} are expected to be of the correct sizes.
 */
void coco_evaluate_constraint(coco_problem_t *self, const double *x, double *y);

/**
 * Recommend ${number_of_solutions} parameter settings (stored in
 * ${x}) as the current best guess solutions to the problem ${self}.
 *
 * @note ${number_of_solutions} is expected to be larger than 1 only
 * if coco_get_number_of_objectives(self) is larger than 1. 
 */
void coco_recommend_solutions(coco_problem_t *self, const double *x,
                              size_t number_of_solutions);

/**
 * Free the NUMBBO problem represented by ${self}.
 */
void coco_free_problem(coco_problem_t *self);

/**
 * Return the name of a COCO problem.
 *
 * @note Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 *
 * @see coco_strdup()
 */
const char *coco_get_problem_name(const coco_problem_t *self);

/**
 * Return the ID of the COCO problem ${self}. The ID is guaranteed to
 * contain only characters in the set [a-z0-9_-]. It should therefore
 * be safe to use the ID to construct filenames or other identifiers.
 *
 * Each problem ID should be unique within each benchmark suite. 
 *
 * @note Do not modify the returned string! If you free the problem,
 * the returned pointer becomes invalid. When in doubt, strdup() the
 * returned value.
 *
 * @see coco_strdup
 */
const char *coco_get_problem_id(const coco_problem_t *self);

/**
 * Return the number of variables of a COCO problem.
 */
size_t coco_get_number_of_variables(const coco_problem_t *self);

/**
 * Return the number of objectives of a COCO problem.
 */
size_t coco_get_number_of_objectives(const coco_problem_t *self);

/**
 * Return the number of constraints of a COCO problem.
 */
size_t coco_get_number_of_constraints(const coco_problem_t *self);

/**
 * Get the ${problem_index}-th problem of the ${problem_suit} test
 * suit.
 */
coco_problem_t *coco_get_problem(const char *problem_suite,
                                 const long problem_index);

/**
 * Return the successor index of ${problem_index} in ${problem_suit},
 * or the first index if ${problem_index} < 0,
 * or -1 otherwise (no successor problem is available).
 *
 * int index = -1;
 * while (-1 < (index = coco_next_problem_index(suite, index, ""))) {
 *   coco_problem_t *problem = coco_get_problem(suite, index); 
 *   ...
 *   coco_free_problem(problem);
 * }
 * 
 * loops over all indices and problems consequently. 
 */
long coco_next_problem_index(const char *problem_suite,
                            const long problem_index,
                            const char *select_options);

/**
 * Number of evaluations done on problem ${self}. 
 * Tentative and yet versatile. 
 */
long coco_get_evaluations(coco_problem_t *self);
double coco_get_best_observed_fvalue1(const coco_problem_t *self);

/**
 * Return target value for first objective. Values below are not
 * relevant in the performance assessment. 
 *
 * This function breaks the black-box property: the returned 
 * value is not meant to be used by the optimization algorithm 
 * other than for final termination. 


 */
double coco_get_final_target_fvalue1(const coco_problem_t *self);

/**
 * tentative getters for region of interest
 */
const double *coco_get_smallest_values_of_interest(const coco_problem_t *self);
const double *coco_get_largest_values_of_interest(const coco_problem_t *self);

/**
 * Return an initial solution, i.e. a feasible variable setting, to the
 * problem.
 *
 * By default, the center of the problems region of interest
 * is the initial solution.
 *
 * @see coco_get_smallest_values_of_interest() and
 *coco_get_largest_values_of_interest()
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

void coco_benchmark(const char *problem_suite, const char *observer,
                    const char *observer_options, coco_optimizer_t optimizer);

/* shall replace the above? */
void new_coco_benchmark(const char *problem_suite, const char *problem_suit_options,
                     const char *observer, const char *observer_options,
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
void coco_error(const char *message, ...);

/**
 * Function to warn about eror conditions.
 */
void coco_warning(const char *message, ...);

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

/* TODO: These bbob2009... functions should probably not be in
 * this header.
 */
/* but they are necessary for Builder fbsd9-amd64-test-gcc at
   http://numbbo.p-value.net/buildbot/builders/fbsd9-amd64-test-gcc
   (not for the others) */
/**
 * Return the function ID of a BBOB 2009 problem or -1.
 */
/* int bbob2009_get_function_id(const coco_problem_t *problem);
*/
/**
 * Return the function ID of a BBOB 2009 problem or -1.
 */
/* int bbob2009_get_instance_id(const coco_problem_t *problem);
*/

#ifdef __cplusplus
}
#endif
#endif
#line 7 "src/coco_benchmark.c"

#line 1 "src/toy_suit.c"
#line 1 "src/coco_generics.c"
#include <assert.h>

#line 4 "src/coco_generics.c"
#line 1 "src/coco_internal.h"
/*
 * Internal NumBBO structures and typedefs.
 *
 * These are used throughout the NumBBO code base but should not be
 * used by any external code.
 */

#ifndef __NUMBBO_INTERNAL__
#define __NUMBBO_INTERNAL__

typedef void (*coco_initial_solution_function_t)(
    const struct coco_problem *self, double *y);
typedef void (*coco_evaluate_function_t)(struct coco_problem *self,
                                         const double *x,
                                         double *y);
typedef void (*coco_recommendation_function_t)(struct coco_problem *self,
                                               const double *x,
                                               size_t number_of_solutions);

typedef void (*coco_free_function_t)(struct coco_problem *self);

/**
 * Description of a COCO problem (instance)
 *
 * evaluate and free are opaque pointers which should not be called
 * directly. Instead they are used by the coco_* functions in
 * coco_generics.c. This indirection gives us the flexibility to add
 * generic checks or more sophisticated dispatch methods later on.
 *
 * Fields:
 *
 * number_of_variables - Number of parameters expected by the
 *   function and constraints.
 *
 * number_of_objectives - Number of objectives.
 *
 * number_of_constraints - Number of constraints.
 *
 * smallest_values_of_interest, largest_values_of_interest - Vectors
 *   of length 'number_of_variables'. Lower/Upper bounds of the
 *   region of interest.
 *
 * problem_name - Descriptive name for the test problem. May be NULL
 *   to indicate that no name is known.
 *
 * problem_id - Short name consisting of _only_ [a-z0-9], '-' and '_'
 *   that, when not NULL, must be unique. It can for example be used
 *   to generate valid directory names under which to store results.
 *
 * data - Void pointer that can be used to store problem specific data
 *   needed by any of the methods.
 */
struct coco_problem {
  coco_initial_solution_function_t initial_solution;
  coco_evaluate_function_t evaluate_function;
  coco_evaluate_function_t evaluate_constraint;
  coco_recommendation_function_t recommend_solutions;
  coco_free_function_t free_problem; /* AKA free_self */
  size_t number_of_variables;
  size_t number_of_objectives;
  size_t number_of_constraints;
  double *smallest_values_of_interest;
  double *largest_values_of_interest;
  double *best_value; /* means: f-value */
  double *best_parameter;
  char *problem_name; /* problem is redundant but useful when searching */
  char *problem_id; /* problem is redundant but useful when searching */
  long index; /* unique index within the current/parent benchmark suite */
  long evaluations;
  double final_target_delta[1];
  double best_observed_fvalue[1];
  long best_observed_evaluation[1];
  void *data;
  /* The prominent usecase for data is coco_transform_data_t*, making an
   * "onion of problems", initialized in coco_allocate_transformed_problem(...).
   * This makes the current ("outer" or "transformed") problem a "derived
   * problem class", which inherits from the "inner" problem, the "base class".
   *   - data holds the meta-information to administer the inheritance
   *   - data->data holds the additional fields of the derived class (the outer problem)
   * Specifically:  
   * data = coco_transform_data_t *  / * mnemonic: inheritance data or onion data or link data
   *          - coco_problem_t *inner_problem;  / * now we have a linked list
   *          - void *data;  / * defines the additional attributes/fields etc. to be used by the "outer" problem (derived class)
   *          - coco_transform_free_data_t free_data;  / * deleter for allocated memory in (not of) data->data
   */
};

#endif
#line 5 "src/coco_generics.c"

void coco_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  /* implements a safer version of self->evaluate(self, x, y) */
  assert(self != NULL);
  assert(self->evaluate_function != NULL);
  self->evaluate_function(self, x, y);
  self->evaluations++; /* each derived class has its own counter, only the most outer will be visible */
#if 1
 /* A little bit of bookkeeping */
  if (y[0] < self->best_observed_fvalue[0]) {
    self->best_observed_fvalue[0] = y[0];
    self->best_observed_evaluation[0] = self->evaluations;
  }
#endif
}

long coco_get_evaluations(coco_problem_t *self) {
  assert(self != NULL);
  return self->evaluations;  
}

#if 1  /* tentative */
double coco_get_best_observed_fvalue1(const coco_problem_t *self) {
  assert(self != NULL);
  return self->best_observed_fvalue[0];
}
double coco_get_final_target_fvalue1(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->best_value != NULL);
  assert(self->final_target_delta != NULL);
  return self->best_value[0] + self->final_target_delta[0];
}
#endif

void coco_evaluate_constraint(coco_problem_t *self, const double *x, double *y) {
  /* implements a safer version of self->evaluate(self, x, y) */
  assert(self != NULL);
  assert(self->evaluate_constraint != NULL);
  self->evaluate_constraint(self, x, y);
}

void coco_recommend_solutions(coco_problem_t *self, const double *x,
                              size_t number_of_solutions) {
  assert(self != NULL);
  assert(self->recommend_solutions != NULL);
  self->recommend_solutions(self, x, number_of_solutions);
}

void coco_free_problem(coco_problem_t *self) {
  assert(self != NULL);
  if (self->free_problem != NULL) {
    self->free_problem(self);
  } else {
    /* Best guess at freeing all relevant structures */
    if (self->smallest_values_of_interest != NULL)
      coco_free_memory(self->smallest_values_of_interest);
    if (self->largest_values_of_interest != NULL)
      coco_free_memory(self->largest_values_of_interest);
    if (self->best_parameter != NULL)
      coco_free_memory(self->best_parameter);
    if (self->best_value != NULL)
      coco_free_memory(self->best_value);
    if (self->problem_name != NULL)
      coco_free_memory(self->problem_name);
    if (self->problem_id != NULL)
      coco_free_memory(self->problem_id);
    if (self->data != NULL)
      coco_free_memory(self->data);
    self->smallest_values_of_interest = NULL;
    self->largest_values_of_interest = NULL;
    self->best_parameter = NULL;
    self->best_value = NULL;
    self->data = NULL;
    coco_free_memory(self);
  }
}

const char *coco_get_problem_name(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_name != NULL);
  return self->problem_name;
}

const char *coco_get_problem_id(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->problem_id;
}

size_t coco_get_number_of_variables(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->number_of_variables;
}

size_t coco_get_number_of_objectives(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->number_of_objectives;
}

size_t coco_get_number_of_constraints(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->number_of_constraints;
}

const double *coco_get_smallest_values_of_interest(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->smallest_values_of_interest;
}

const double *coco_get_largest_values_of_interest(const coco_problem_t *self) {
  assert(self != NULL);
  assert(self->problem_id != NULL);
  return self->largest_values_of_interest;
}

void coco_get_initial_solution(const coco_problem_t *self,
                               double *initial_solution) {
  assert(self != NULL);
  if (self->initial_solution != NULL) {
    self->initial_solution(self, initial_solution);
  } else {
    size_t i;
    assert(self->smallest_values_of_interest != NULL);
    assert(self->largest_values_of_interest != NULL);
    for (i = 0; i < self->number_of_variables; ++i)
      initial_solution[i] = 0.5 * (self->smallest_values_of_interest[i] +
                                   self->largest_values_of_interest[i]);
  }
}
#line 2 "src/toy_suit.c"

#line 1 "src/log_hitting_times.c"
#include <stdio.h>
#include <assert.h>

#line 5 "src/log_hitting_times.c"

#line 1 "src/coco_utilities.c"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#line 7 "src/coco_utilities.c"
#line 8 "src/coco_utilities.c"
#line 1 "src/coco_strdup.c"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#line 6 "src/coco_strdup.c"

char *coco_strdup(const char *string);
char * coco_strdupf(const char *str, ...);
char * coco_vstrdupf(const char *str, va_list args);
char *coco_strconcat(const char *s1, const char *s2); 
long coco_strfind(const char *base, const char *seq); 

/**
 * coco_strdup(string):
 *
 * Create a duplicate copy of ${string} and return a pointer to
 * it. The caller is responsible for free()ing the memory allocated
 * using coco_free_memory().
 */
char *coco_strdup(const char *string) {
  size_t len;
  char *duplicate;
  if (string == NULL)
    return NULL;
  len = strlen(string);
  duplicate = (char *)coco_allocate_memory(len + 1);
  memcpy(duplicate, string, len + 1);
  return duplicate;
}
/**
 * Formatted string duplication. Optional arguments are
 * used like in sprintf. 
 */ 
char * coco_strdupf(const char *str, ...) {
  va_list args;
  char *s;

  va_start(args, str);
  s = coco_vstrdupf(str, args);
  va_end(args);
  return s;
}

#define coco_vstrdupf_buflen 444
/**
 * va_list version of formatted string duplication coco_strdupf()
 */
char * coco_vstrdupf(const char *str, va_list args) {
  static char buf[coco_vstrdupf_buflen];
  long written; 
  /* apparently args can only be used once, therefore
   * len = vsnprintf(NULL, 0, str, args) to find out the
   * length does not work. Therefore we use a buffer
   * which limits the max length. Longer strings should
   * never appear anyway, so this is rather a non-issue. */
  
#if 1 
  written = vsnprintf(buf, coco_vstrdupf_buflen - 2, str, args); 
  if (written < 0)
    coco_error("coco_vstrdupf(): vsnprintf failed on '%s'", str);
#else /* less save alternative, if vsnprintf is not available */
  assert(strlen(str) < coco_vstrdupf_buflen / 2 - 2);
  written = vsprintf(buf, str, args); 
  if (written < 0)
    coco_error("coco_vstrdupf(): vsprintf failed on '%s'", str);
#endif
  if (written > coco_vstrdupf_buflen - 3)
    coco_error("coco_vstrdupf(): A suspiciously long string is tried to being duplicated '%s'", buf);
  return coco_strdup(buf);
}
#undef coco_vstrdupf_buflen

/**
 * coco_strconcat(string1, string2):
 *
 * Return a concatenate copy of ${string1} + ${string2}. 
 * The caller is responsible for free()ing the memory allocated
 * using coco_free_memory().
 */
char *coco_strconcat(const char *s1, const char *s2) {
  size_t len1 = strlen(s1);
  size_t len2 = strlen(s2);
  char *s = (char *)coco_allocate_memory(len1 + len2 + 1);
  
  memcpy(s, s1, len1);
  memcpy(&s[len1], s2, len2 + 1);
  return s;
}

/**
 * return first index where ${seq} occurs in ${base}, -1 if it doesn't.
 *
 * If there is an equivalent standard C function, this can/should be removed. 
 */
long coco_strfind(const char *base, const char *seq) {
  const size_t strlen_seq = strlen(seq);
  const size_t last_first_idx = strlen(base) - strlen(seq);
  size_t i, j;
  
  for (i = 0; i <= last_first_idx; ++i) {
    if (base[i] == seq[0]) {
      for (j = 0; j < strlen_seq; ++j) {
        if (base[i+j] != seq[j])
          break;
      }
      if (j == strlen_seq) {
        if (i > 1e9)
          coco_error("coco_strfind(): strange values observed i=%lu, j=%lu, strlen(base)=%lu",
                     i, j, strlen(base));
        return (long)i;
      }
    }
  }
  return -1;
}
#line 9 "src/coco_utilities.c"

/* Figure out if we are on a sane platform or on the dominant platform. */
#if defined(_WIN32) || defined(_WIN64) || defined(__MINGW64__) || defined(__CYGWIN__)
#include <windows.h>
static const char *coco_path_separator = "\\";
#define NUMBBO_PATH_MAX MAX_PATH
#define HAVE_GFA 1
#elif defined(__gnu_linux__)
#include <sys/stat.h>
#include <sys/types.h>
#include <linux/limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define NUMBBO_PATH_MAX PATH_MAX
#elif defined(__APPLE__)
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syslimits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define NUMBBO_PATH_MAX PATH_MAX
#elif defined(__FreeBSD__)
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define NUMBBO_PATH_MAX PATH_MAX
#elif defined(__sun) || defined(sun)
#if defined(__SVR4) || defined(__svr4__)
/* Solaris */
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define NUMBBO_PATH_MAX PATH_MAX
#endif
#else
#error Unknown platform
#endif

#if !defined(NUMBBO_PATH_MAX)
#error NUMBBO_PATH_MAX undefined
#endif

/***********************************
 * Global definitions in this file
 * which are not in coco.h 
 ***********************************/
void coco_join_path(char *path, size_t path_max_length, ...);
int coco_path_exists(const char *path);
void coco_create_path(const char *path);
void coco_create_new_path(const char *path, size_t maxlen, char *new_path);
double *coco_duplicate_vector(const double *src, const size_t number_of_elements);
/***********************************/

void coco_join_path(char *path, size_t path_max_length, ...) {
  const size_t path_separator_length = strlen(coco_path_separator);
  va_list args;
  char *path_component;
  size_t path_length = strlen(path);

  va_start(args, path_max_length);
  while (NULL != (path_component = va_arg(args, char *))) {
    size_t component_length = strlen(path_component);
    if (path_length + path_separator_length + component_length >=
        path_max_length) {
      coco_error("coco_file_path() failed because the ${path} is too short.");
      return; /* never reached */
    }
    /* Both should be safe because of the above check. */
    if (strlen(path) > 0)
      strncat(path, coco_path_separator, path_max_length - strlen(path) - 1);
    strncat(path, path_component, path_max_length - strlen(path) - 1);
  }
  va_end(args);
}

int coco_path_exists(const char *path) {
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributes(path);
  return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
          (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#elif defined(HAVE_STAT)
  struct stat buf;
  int res = stat(path, &buf);
  return res == 0;
#else
#error Ooops
#endif
}

void coco_create_path(const char *path) {
#if defined(HAVE_GFA)
  /* FIXME: Unimplemented for now. */
  /* Nothing to do if the path exists. */
  if (coco_path_exists(path))
    return;
  mkdir(path);

#elif defined(HAVE_STAT)
  char *tmp = NULL;
  char buf[4096];
  char *p;
  size_t len = strlen(path);
  assert(strcmp(coco_path_separator, "/") == 0);

  /* Nothing to do if the path exists. */
  if (coco_path_exists(path))
    return;

  tmp = coco_strdup(path);
  /* Remove possible trailing slash */
  if (tmp[len - 1] == '/')
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      if (!coco_path_exists(tmp)) {
        if (0 != mkdir(tmp, S_IRWXU))
          goto error;
      }
      *p = '/';
    }
  }
  if (0 != mkdir(tmp, S_IRWXU))
    goto error;
  coco_free_memory(tmp);
  return;
error:
  snprintf(buf, sizeof(buf), "mkdir(\"%s\") failed.", tmp);
  coco_error(buf);
  return; /* never reached */
#else
#error Ooops
#endif
}

#if 0
/** path and new_path can be the same argument. 
 */
void coco_create_new_path(const char *path, size_t maxlen, char *new_path) {
  char sep = '_';
  size_t oldlen, len;
  time_t now;
  const char *snow;
  int i, tries;
  
  if (!coco_path_exists(path)) {
    coco_create_path(path);
    return;
  }

  maxlen -= 1; /* prevent failure from misinterpretation of what maxlen is */
  new_path[maxlen] = '\0';
  oldlen = strlen(path);
  assert(maxlen > oldlen);
  if (new_path != path)
    strncpy(new_path, path, maxlen);
  
  /* modify new_path name until path does not exist */
  for (tries = 0; tries <= (int)('z' - 'a'); ++tries) {
    /* create new name */
    now = time(NULL);
    snow = ctime(&now);
    /*                 012345678901234567890123
     * snow =         "Www Mmm dd hh:mm:ss yyyy"
     * new_path = "oldname_Mmm_dd_hh_mm_ss_yyyy[a-z]"
     *                    ^ oldlen
     */
    new_path[oldlen] = sep;
    strncpy(&new_path[oldlen+1], &snow[4], maxlen - oldlen - 1);
    for (i = oldlen; i < maxlen; ++i) {
      if (new_path[i] == ' ' || new_path[i] == ':') 
        new_path[i] = sep;
      if (new_path[i] == '\n')
        new_path[i] = '\0';
      if (new_path[i] == '\0')
        break;
    }
    len = strlen(new_path);
    if (len > 6) {
      new_path[len - 5] = (char)(tries + 'a');
      new_path[len - 4] = '\0';
    }
      
    /* try new name */
    if (!coco_path_exists(new_path)) {
      /* not thread safe until path is created */
      coco_create_path(new_path);
      tries = -1;
      break;
    }
  }
  if (tries > 0) {
    char *message = "coco_create_new_path: could not create a new path from '%s' (%d attempts)";
    coco_warning(message, path, tries);
    coco_error(message);
  } 
}
#endif

double *coco_allocate_vector(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(double);
  return (double *)coco_allocate_memory(block_size);
}

double *coco_duplicate_vector(const double *src,
                              const size_t number_of_elements) {
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
#line 7 "src/log_hitting_times.c"
#line 1 "src/coco_problem.c"
#include <float.h>
#line 3 "src/coco_problem.c"

#line 5 "src/coco_problem.c"

/***********************************
 * Global definitions in this file
 *
 * TODO: are these really needed? 
 * Only if they would need to be used from
 * outside. Benchmarks that are included in
 * coco_benchmark.c can include coco_problem.c
 * directly due to the amalgamate magic.
 * 
 ***********************************/

coco_problem_t *
coco_allocate_problem(const size_t number_of_variables,
                      const size_t number_of_objectives,
                      const size_t number_of_constraints); 
coco_problem_t *
coco_duplicate_problem(coco_problem_t *other);
typedef void (*coco_transform_free_data_t)(void *data);

/* typedef coco_transform_data_t; */
coco_problem_t *
coco_allocate_transformed_problem(coco_problem_t *inner_problem, void *userdata,
                                  coco_transform_free_data_t free_data);
void *coco_get_transform_data(coco_problem_t *self);
coco_problem_t *
coco_get_transform_inner_problem(coco_problem_t *self);

/* typedef coco_stacked_problem_data_t; */
coco_problem_t *
coco_stacked_problem_allocate(coco_problem_t *problem1_to_be_stacked,
                              coco_problem_t *problem2_to_be_stacked);

/***********************************/

/**
 * coco_allocate_problem(number_of_variables):
 *
 * Allocate and pre-populate a new coco_problem_t for a problem with
 * ${number_of_variables}.
 */
coco_problem_t *
coco_allocate_problem(const size_t number_of_variables,
                      const size_t number_of_objectives,
                      const size_t number_of_constraints) {
  coco_problem_t *problem;
  problem = (coco_problem_t *)coco_allocate_memory(sizeof(coco_problem_t));
  /* Initialize fields to sane/safe defaults */
  problem->initial_solution = NULL;
  problem->evaluate_function = NULL;
  problem->evaluate_constraint = NULL;
  problem->recommend_solutions = NULL;
  problem->free_problem = NULL;
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = number_of_objectives;
  problem->number_of_constraints = number_of_constraints;
  problem->smallest_values_of_interest =
      coco_allocate_vector(number_of_variables);
  problem->largest_values_of_interest =
      coco_allocate_vector(number_of_variables);
  problem->best_parameter = coco_allocate_vector(number_of_variables);
  problem->best_value = coco_allocate_vector(number_of_objectives);
  problem->problem_name = NULL;
  problem->problem_id = NULL;
  problem->evaluations = 0;
  problem->final_target_delta[0] = 1e-8; /* in case to be modified by the benchmark */
  problem->best_observed_fvalue[0] = DBL_MAX; 
  problem->best_observed_evaluation[0] = 0;
  problem->data = NULL;
  return problem;
}

coco_problem_t *
coco_duplicate_problem(coco_problem_t *other) {
  size_t i;
  coco_problem_t *problem;
  problem = coco_allocate_problem(other->number_of_variables,
                                  other->number_of_objectives,
                                  other->number_of_constraints);

  problem->evaluate_function = other->evaluate_function;
  problem->evaluate_constraint = other->evaluate_constraint;
  problem->recommend_solutions = other->recommend_solutions;
  problem->free_problem = NULL;

  for (i = 0; i < problem->number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] =
        other->smallest_values_of_interest[i];
    problem->largest_values_of_interest[i] =
        other->largest_values_of_interest[i];
    if (other->best_parameter)
      problem->best_parameter[i] = other->best_parameter[i];
  }

  if (other->best_value)
    for (i = 0; i < problem->number_of_objectives; ++i) {
        problem->best_value[i] = other->best_value[i];
    }

  problem->problem_name = coco_strdup(other->problem_name);
  problem->problem_id = coco_strdup(other->problem_id);
  return problem;
}

/**
 * Generic data member of a transformed (or "outer") coco_problem_t.
 */
typedef struct {
  coco_problem_t *inner_problem;
  void *data;
  coco_transform_free_data_t free_data;
} coco_transform_data_t;

static void _tfp_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  coco_transform_data_t *data;
  assert(self != NULL);
  assert(self->data != NULL);
  data = self->data;
  assert(data->inner_problem != NULL);

  coco_evaluate_function(data->inner_problem, x, y);
}

static void _tfp_evaluate_constraint(coco_problem_t *self, const double *x,
                                     double *y) {
  coco_transform_data_t *data;
  assert(self != NULL);
  assert(self->data != NULL);
  data = self->data;
  assert(data->inner_problem != NULL);

  coco_evaluate_constraint(data->inner_problem, x, y);
}

static void _tfp_recommend_solutions(coco_problem_t *self, const double *x,
                                     size_t number_of_solutions) {
  coco_transform_data_t *data;
  assert(self != NULL);
  assert(self->data != NULL);
  data = self->data;
  assert(data->inner_problem != NULL);

  coco_recommend_solutions(data->inner_problem, x, number_of_solutions);
}

static void _tfp_free_problem(coco_problem_t *self) {
  coco_transform_data_t *data;
  assert(self != NULL);
  assert(self->data != NULL);
  data = self->data;
  assert(data->inner_problem != NULL);

  if (data->inner_problem != NULL) {
    coco_free_problem(data->inner_problem);
    data->inner_problem = NULL;
  }
  if (data->data != NULL) {
    if (data->free_data != NULL) {
      data->free_data(data->data);
      data->free_data = NULL;
    }
    coco_free_memory(data->data);
    data->data = NULL;
  }
  /* Let the generic free problem code deal with the rest of the
   * fields. For this we clear the free_problem function pointer and
   * recall the generic function.
   */
  self->free_problem = NULL;
  coco_free_problem(self);
}

/**
 * coco_allocate_transformed_problem(inner_problem):
 *
 * Allocate a transformed problem that wraps ${inner_problem}. By
 * default all methods will dispatch to the ${inner_problem} method.
 *
 */
coco_problem_t *
coco_allocate_transformed_problem(coco_problem_t *inner_problem, void *userdata,
                                  coco_transform_free_data_t free_data) {
  coco_transform_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->inner_problem = inner_problem;
  data->data = userdata;
  data->free_data = free_data;

  self = coco_duplicate_problem(inner_problem);
  self->evaluate_function = _tfp_evaluate_function;
  self->evaluate_constraint = _tfp_evaluate_constraint;
  self->recommend_solutions = _tfp_recommend_solutions;
  self->free_problem = _tfp_free_problem;
  self->data = data;
  return self;
}

void *coco_get_transform_data(coco_problem_t *self) {
  assert(self != NULL);
  assert(self->data != NULL);
  assert(((coco_transform_data_t *)self->data)->data != NULL);

  return ((coco_transform_data_t *)self->data)->data;
}

coco_problem_t *
coco_get_transform_inner_problem(coco_problem_t *self) {
  assert(self != NULL);
  assert(self->data != NULL);
  assert(((coco_transform_data_t *)self->data)->inner_problem != NULL);

  return ((coco_transform_data_t *)self->data)->inner_problem;
}

/** type provided coco problem data for a stacked coco problem
 */
typedef struct {
  coco_problem_t *problem1;
  coco_problem_t *problem2;
} coco_stacked_problem_data_t;

static void stacked_problem_evaluate(coco_problem_t *self, const double *x, double *y) {
  coco_stacked_problem_data_t* data = (coco_stacked_problem_data_t*)self->data; 

  assert(coco_get_number_of_objectives(self) ==
           coco_get_number_of_objectives(data->problem1) + coco_get_number_of_objectives(data->problem2));

  coco_evaluate_function(data->problem1, x, &y[0]);
  coco_evaluate_function(data->problem2, x, &y[coco_get_number_of_objectives(data->problem1)]);
}

static void stacked_problem_evaluate_constraint(coco_problem_t *self, const double *x, double *y) {
  coco_stacked_problem_data_t* data = (coco_stacked_problem_data_t*)self->data;

  assert(coco_get_number_of_constraints(self) ==
           coco_get_number_of_constraints(data->problem1) + coco_get_number_of_constraints(data->problem2));

  if (coco_get_number_of_constraints(data->problem1) > 0)
    coco_evaluate_constraint(data->problem1, x, y);
  if (coco_get_number_of_constraints(data->problem2) > 0)
    coco_evaluate_constraint(data->problem2, x, &y[coco_get_number_of_constraints(data->problem1)]);  
}

static void stacked_problem_free(coco_problem_t *self) {
  coco_stacked_problem_data_t *data;
  assert(self != NULL);
  assert(self->data != NULL);
  data = self->data;

  if (data->problem1 != NULL) {
    coco_free_problem(data->problem1);
    data->problem1 = NULL;
  }
  if (data->problem2 != NULL) {
    coco_free_problem(data->problem2);
    data->problem2 = NULL;
  }
  /* Let the generic free problem code deal with the rest of the
   * fields. For this we clear the free_problem function pointer and
   * recall the generic function.
   */
  self->free_problem = NULL;
  coco_free_problem(self);
}

/**
 * Return a problem that stacks the output of two problems, namely
 * of coco_evaluate_function and coco_evaluate_constraint. The accepted
 * input remains the same and must be identical for the stacked
 * problems. 
 * 
 * This is particularly useful to generate multiobjective problems,
 * e.g. a biobjective problem from two single objective problems.
 *
 * Details: regions of interest must either agree or at least one
 * of them must be NULL. Best parameter becomes somewhat meaningless. 
 */
coco_problem_t *coco_stacked_problem_allocate(coco_problem_t *problem1, coco_problem_t *problem2) {
  const size_t number_of_variables = coco_get_number_of_variables(problem1); 
  const size_t number_of_objectives =
      coco_get_number_of_objectives(problem1) + coco_get_number_of_objectives(problem2);
  const size_t number_of_constraints =
      coco_get_number_of_constraints(problem1) + coco_get_number_of_constraints(problem2);
  size_t i;
  char *s;
  const double *smallest, *largest;
  coco_stacked_problem_data_t *data;
  coco_problem_t *problem; /* the new coco problem */

  assert(coco_get_number_of_variables(problem1) == coco_get_number_of_variables(problem2));

  problem = coco_allocate_problem(number_of_variables, number_of_objectives, number_of_constraints);

  s = coco_strconcat(coco_get_problem_id(problem1), "_-_");
  problem->problem_id = coco_strconcat(s, coco_get_problem_id(problem2));
  coco_free_memory(s);
  s = coco_strconcat(coco_get_problem_name(problem1), " + ");
  problem->problem_name = coco_strconcat(s, coco_get_problem_name(problem2));
  coco_free_memory(s);
             
  problem->evaluate_function = stacked_problem_evaluate;
  if (number_of_constraints > 0)
    problem->evaluate_constraint = stacked_problem_evaluate_constraint;
  
  /* set/copy "boundaries" and best_parameter */
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
      
    if (problem->best_parameter) /* bbob2009 logger doesn't work then anymore */
      coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
    if (problem->best_value)
      coco_free_memory(problem->best_value);
    problem->best_value = NULL;  /* bbob2009 logger doesn't work */
  }
  
  /* setup data holder */
  data = coco_allocate_memory(sizeof(*data));
  data->problem1 = problem1;
  data->problem2 = problem2;

  problem->data = data;
  problem->free_problem = stacked_problem_free; /* free self->data and coco_free_problem(self) */

  return problem;
}
#line 8 "src/log_hitting_times.c"
#line 9 "src/log_hitting_times.c"

typedef struct {
  char *path;
  FILE *logfile;
  double *target_values;
  size_t number_of_target_values;
  size_t next_target_value;
  long number_of_evaluations;
} _log_hitting_time_t;

static void _lht_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  _log_hitting_time_t *data;
  data = coco_get_transform_data(self);

  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  data->number_of_evaluations++;

  /* Open logfile if it is not alread open */
  if (data->logfile == NULL) {
    data->logfile = fopen(data->path, "w");
    if (data->logfile == NULL) {
      char *buf;
      const char *error_format =
          "lht_evaluate_function() failed to open log file '%s'.";
      size_t buffer_size = snprintf(NULL, 0, error_format, data->path);
      buf = (char *)coco_allocate_memory(buffer_size);
      snprintf(buf, buffer_size, error_format, data->path);
      coco_error(buf);
      coco_free_memory(buf); /* Never reached */
    }
    fputs("target_value function_value number_of_evaluations\n", data->logfile);
  }

  /* Add a line for each hitting level we have reached. */
  while (y[0] <= data->target_values[data->next_target_value] &&
         data->next_target_value < data->number_of_target_values) {
    fprintf(data->logfile, "%e %e %li\n",
            data->target_values[data->next_target_value], y[0],
            data->number_of_evaluations);
    data->next_target_value++;
  }
  /* Flush output so that impatient users can see progress. */
  fflush(data->logfile);
}

static void _lht_free_data(void *stuff) {
  _log_hitting_time_t *data;
  assert(stuff != NULL);
  data = stuff;

  if (data->path != NULL) {
    coco_free_memory(data->path);
    data->path = NULL;
  }
  if (data->target_values != NULL) {
    coco_free_memory(data->target_values);
    data->target_values = NULL;
  }
  if (data->logfile != NULL) {
    fclose(data->logfile);
    data->logfile = NULL;
  }
}

static coco_problem_t *log_hitting_times(coco_problem_t *inner_problem,
                                  const double *target_values,
                                  const size_t number_of_target_values,
                                  const char *path) {
  _log_hitting_time_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->number_of_evaluations = 0;
  data->path = coco_strdup(path);
  data->logfile = NULL; /* Open lazily in lht_evaluate_function(). */
  data->target_values =
      coco_duplicate_vector(target_values, number_of_target_values);
  data->number_of_target_values = number_of_target_values;
  data->next_target_value = 0;

  self = coco_allocate_transformed_problem(inner_problem, data, _lht_free_data);
  self->evaluate_function = _lht_evaluate_function;
  return self;
}
#line 4 "src/toy_suit.c"

#line 1 "src/f_sphere.c"
#include <stdio.h>
#include <assert.h>

#line 5 "src/f_sphere.c"

#line 7 "src/f_sphere.c"

static void f_sphere_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  assert(self->number_of_objectives == 1);
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    y[0] += x[i] * x[i];
  }
}

static coco_problem_t *sphere_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("sphere function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "sphere", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "sphere",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = f_sphere_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0; /* FIXME: this is not sphere-specific */
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  f_sphere_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 6 "src/toy_suit.c"
#line 1 "src/f_ellipsoid.c"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#line 6 "src/f_ellipsoid.c"

#line 8 "src/f_ellipsoid.c"

static void _ellipsoid_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i = 0;
  static const double condition = 1.0e6;
  assert(self->number_of_objectives == 1);
  assert(self->number_of_variables > 0);
  y[0] = x[i] * x[i];
  for (i = 1; i < self->number_of_variables; ++i) {
    const double exponent = i / (self->number_of_variables - 1.0);
    y[0] += pow(condition, exponent) * x[i] * x[i];
  }
}

static coco_problem_t *ellipsoid_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem;

  problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("ellipsoid function");
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "ellipsoid", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02i", "ellipsoid",
           (int)number_of_variables);
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _ellipsoid_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _ellipsoid_evaluate(problem, problem->best_parameter, problem->best_value);

  return problem;
}
#line 7 "src/toy_suit.c"
#line 1 "src/f_rastrigin.c"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#line 6 "src/f_rastrigin.c"
#line 7 "src/f_rastrigin.c"

static void _rastrigin_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double sum1 = 0.0, sum2 = 0.0;
  assert(self->number_of_objectives == 1);

  for (i = 0; i < self->number_of_variables; ++i) {
    sum1 += cos(coco_two_pi * x[i]);
    sum2 += x[i] * x[i];
  }
  y[0] = 10.0 * (self->number_of_variables - sum1) + sum2;
}

static coco_problem_t *rastrigin_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("rastrigin function");
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "rastrigin", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02i", "rastrigin",
           (int)number_of_variables);
  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _rastrigin_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _rastrigin_evaluate(problem, problem->best_parameter, problem->best_value);

  return problem;
}
#line 8 "src/toy_suit.c"
#line 1 "src/f_rosenbrock.c"
#include <assert.h>

#line 4 "src/f_rosenbrock.c"
#line 5 "src/f_rosenbrock.c"

static void _rosenbrock_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double s1 = 0.0, s2 = 0.0, tmp;
  assert(self->number_of_objectives == 1);
  assert(self->number_of_variables > 1);
  for (i = 0; i < self->number_of_variables - 1; ++i) {
    tmp = (x[i] * x[i] - x[i + 1]);
    s1 += tmp * tmp;
    tmp = (x[i] - 1.0);
    s2 += tmp * tmp;
  }
  y[0] = 100.0 * s1 + s2;
}

static coco_problem_t *rosenbrock_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("rosenbrock function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "rosenbrock", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "rosenbrock",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _rosenbrock_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 1.0;
  }
  /* Calculate best parameter value */
  _rosenbrock_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 9 "src/toy_suit.c"
#line 1 "src/f_bueche-rastrigin.c"
#include <math.h>
#include <assert.h>

#line 5 "src/f_bueche-rastrigin.c"

#line 7 "src/f_bueche-rastrigin.c"

static void _bueche_rastrigin_evaluate(coco_problem_t *self, const double *x,
                                       double *y) {
  size_t i;
  double tmp = 0., tmp2 = 0.;
  assert(self->number_of_objectives == 1);
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    tmp += cos(2 * coco_pi * x[i]);
    tmp2 += x[i] * x[i];
  }
  y[0] = 10 * (self->number_of_variables - tmp) + tmp2 + 0;
}

static coco_problem_t *
bueche_rastrigin_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("Bueche-Rastrigin function");
  /* Construct a meaningful problem id */
  problem_id_length = snprintf(NULL, 0, "%s_%02i", "bueche-rastrigin",
                               (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "skewRastriginBueche", (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _bueche_rastrigin_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _bueche_rastrigin_evaluate(problem, problem->best_parameter,
                             problem->best_value);
  return problem;
}
#line 10 "src/toy_suit.c"
#line 1 "src/f_linear_slope.c"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#line 6 "src/f_linear_slope.c"
#line 7 "src/f_linear_slope.c"

static void _linear_slope_evaluate(coco_problem_t *self, const double *x, double *y) {
  static const double alpha = 100.0;
  size_t i;
  assert(self->number_of_objectives == 1);
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    double base, exponent, si;

    base = sqrt(alpha);
    exponent = i * 1.0 / (self->number_of_variables - 1);
    if (self->best_parameter[i] > 0.0) {
      si = pow(base, exponent);
    } else {
      si = -pow(base, exponent);
    }
    y[0] += 5.0 * fabs(si) - si * x[i];
  }
}

static coco_problem_t *linear_slope_problem(const size_t number_of_variables,
                                            const double *best_parameter) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("linear slope function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "linear_slope", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "linear_slope", (int)number_of_variables);

  problem->evaluate_function = _linear_slope_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    if (best_parameter[i] < 0.0) {
      problem->best_parameter[i] = problem->smallest_values_of_interest[i];
    } else {
      problem->best_parameter[i] = problem->largest_values_of_interest[i];
    }
  }
  /* Calculate best parameter value */
  _linear_slope_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 11 "src/toy_suit.c"

/**
 * toy_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem in the toy
 * benchmark suit. If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *toy_suit(const long function_index) {
  static const size_t dims[] = {2, 3, 5, 10, 20};
  const long fid = function_index % 6;
  const long did = function_index / 6;
  coco_problem_t *problem;
  if (did >= 1)
    return NULL;

  if (fid == 0) {
    problem = sphere_problem(dims[did]);
  } else if (fid == 1) {
    problem = ellipsoid_problem(dims[did]);
  } else if (fid == 2) {
    problem = rastrigin_problem(dims[did]);
  } else if (fid == 3) {
    problem = bueche_rastrigin_problem(dims[did]);
  } else if (fid == 4) {
    double xopt[20] = {5.0};
    problem = linear_slope_problem(dims[did], xopt);
  } else if (fid == 5) {
    problem = rosenbrock_problem(dims[did]);
  } else {
    return NULL;
  }
  return problem;
}
#line 9 "src/coco_benchmark.c"
#line 1 "src/toy_observer.c"
#line 2 "src/toy_observer.c"
#line 3 "src/toy_observer.c"
#line 4 "src/toy_observer.c"

static coco_problem_t *toy_observer(coco_problem_t *problem, const char *options) {
  size_t i;
  static const size_t number_of_targets = 20;
  double targets[20];
  char base_path[NUMBBO_PATH_MAX] = {0};
  char filename[NUMBBO_PATH_MAX] = {0};

  /* Calculate target levels: */
  for (i = number_of_targets; i > 0; --i) {
    targets[i - 1] = pow(10.0, (number_of_targets - i) - 9.0);
  }

  coco_join_path(base_path, sizeof(base_path), options, "toy_so",
                 coco_get_problem_id(problem), NULL);
  if (coco_path_exists(base_path)) {
    coco_error("Result directory exists.");
    return NULL; /* never reached */
  }
  coco_create_path(base_path);
  coco_join_path(filename, sizeof(filename), base_path,
                 "first_hitting_times.txt", NULL);
  problem = log_hitting_times(problem, targets, number_of_targets, filename);
  return problem;
}
#line 10 "src/coco_benchmark.c"

#line 1 "src/bbob2009_suite.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#line 7 "src/bbob2009_suite.c"

#line 1 "src/bbob2009_legacy_code.c"
/*
 * Legacy code from BBOB2009 required to replicate the 2009 functions.
 *
 * All of this code should only be used by the bbob2009_suite functions
 * to provide compatibility to the legacy code. New test beds should
 * strive to use the new numbbo facilities for random number
 * generation etc.
 */

#include <math.h>
#include <stdio.h>
#line 13 "src/bbob2009_legacy_code.c"
#define BBOB2009_MAX_DIM 40

static double bbob2009_fmin(double a, double b) { return (a < b) ? a : b; }

static double bbob2009_fmax(double a, double b) { return (a > b) ? a : b; }

static double bbob2009_round(double x) { return floor(x + 0.5); }

/**
 * bbob2009_allocate_matrix(n, m):
 *
 * Allocate a ${n} by ${m} matrix structured as an array of pointers
 * to double arrays.
 */
static double **bbob2009_allocate_matrix(const size_t n, const size_t m) {
  double **matrix = NULL;
  size_t i;
  matrix = (double **)coco_allocate_memory(sizeof(double *) * n);
  for (i = 0; i < n; ++i) {
    matrix[i] = coco_allocate_vector(m);
  }
  return matrix;
}

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
* bbob2009_unif(r, N, inseed):
 *
 * Generate N uniform random numbers using ${inseed} as the seed and
 * store them in ${r}.
 */
static void bbob2009_unif(double *r, long N, long inseed) {
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
    tmp = (int)floor((double)aktseed / (double)127773);
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    if (aktseed < 0)
      aktseed = aktseed + 2147483647;
    if (i < 32)
      rgrand[i] = aktseed;
  }
  aktrand = rgrand[0];
  for (i = 0; i < N; i++) {
    tmp = (int)floor((double)aktseed / (double)127773);
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    if (aktseed < 0)
      aktseed = aktseed + 2147483647;
    tmp = (int)floor((double)aktrand / (double)67108865);
    aktrand = rgrand[tmp];
    rgrand[tmp] = aktseed;
    r[i] = (double)aktrand / 2.147483647e9;
    if (r[i] == 0.) {
      r[i] = 1e-99;
    }
  }
  return;
}

/**
 * bbob2009_reshape(B, vector, m, n):
 *
 * Convert from packed matrix storage to an array of array of double
 * representation.
 */
static double **bbob2009_reshape(double **B, double *vector, int m, int n) {
  int i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      B[i][j] = vector[j * m + i];
    }
  }
  return B;
}

/**
 * bbob2009_gauss(g, N, seed)
 *
 * Generate ${N} Gaussian random numbers using the seed ${seed} and
 * store them in ${g}.
 */
static void bbob2009_gauss(double *g, long N, long seed) {
  int i;
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
 * bbob2009_compute_rotation(B, seed, DIM):
 *
 * Compute a ${DIM}x${DIM} rotation matrix based on ${seed} and store
 * it in ${B}.
 */
static void bbob2009_compute_rotation(double **B, long seed, long DIM) {
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

/**
 * bbob2009_compute_xopt(xopt, seed, DIM):
 *
 * Randomly compute the location of the global optimum.
 */
static void bbob2009_compute_xopt(double *xopt, long seed, long DIM) {
  long i;
  bbob2009_unif(xopt, DIM, seed);
  for (i = 0; i < DIM; i++) {
    xopt[i] = 8 * floor(1e4 * xopt[i]) / 1e4 - 4;
    if (xopt[i] == 0.0)
      xopt[i] = -1e-5;
  }
}

/**
 * bbob2009_compute_fopt(function_id, instance_id):
 *
 * Randomly choose the objective offset for function ${function_id}
 * and instance ${instance_id}.
 */
static double bbob2009_compute_fopt(int function_id, long instance_id) {
  long rseed, rrseed;
  double gval, gval2;

  if (function_id == 4)
    rseed = 3;
  else if (function_id == 18)
    rseed = 17;
  else if (function_id == 101 || function_id == 102 || function_id == 103 ||
           function_id == 107 || function_id == 108 || function_id == 109)
    rseed = 1;
  else if (function_id == 104 || function_id == 105 || function_id == 106 ||
           function_id == 110 || function_id == 111 || function_id == 112)
    rseed = 8;
  else if (function_id == 113 || function_id == 114 || function_id == 115)
    rseed = 7;
  else if (function_id == 116 || function_id == 117 || function_id == 118)
    rseed = 10;
  else if (function_id == 119 || function_id == 120 || function_id == 121)
    rseed = 14;
  else if (function_id == 122 || function_id == 123 || function_id == 124)
    rseed = 17;
  else if (function_id == 125 || function_id == 126 || function_id == 127)
    rseed = 19;
  else if (function_id == 128 || function_id == 129 || function_id == 130)
    rseed = 21;
  else
    rseed = function_id;

  rrseed = rseed + 10000 * instance_id;
  bbob2009_gauss(&gval, 1, rrseed);
  bbob2009_gauss(&gval2, 1, rrseed + 1);
  return bbob2009_fmin(
      1000.,
      bbob2009_fmax(-1000., bbob2009_round(100. * 100. * gval / gval2) / 100.));
}
#line 9 "src/bbob2009_suite.c"

#line 1 "src/f_bbob_step_ellipsoid.c"
/*
 * f_bbob_step_ellipsoid.c
 *
 * The BBOB step ellipsoid function intertwins the variable and
 * objective transformations in such a way that it is hard to devise a
 * composition of generic transformations to implement it. In the end
 * one would have to implement several custom transformations which
 * would be used soley by this problem. Therefore we opt to implement
 * it as a monolithic function instead.
 *
 * TODO: It would be nice to have a generic step ellipsoid function to
 * complement this one.
 */
#line 15 "src/f_bbob_step_ellipsoid.c"

#include <assert.h>
#include <math.h>

#line 20 "src/f_bbob_step_ellipsoid.c"
#line 21 "src/f_bbob_step_ellipsoid.c"
#line 22 "src/f_bbob_step_ellipsoid.c"

typedef struct {
  double *x, *xx;
  double *xopt, fopt;
  double **rot1, **rot2;
} _bbob_step_ellipsoid_t;

static void _bbob_step_ellipsoid_evaluate(coco_problem_t *self, const double *x,
                                          double *y) {
  static const double condition = 100;
  static const double alpha = 10.0;
  size_t i, j;
  double penalty = 0.0, x1;
  _bbob_step_ellipsoid_t *data;

  assert(self->number_of_variables > 1);
  assert(self->number_of_objectives == 1);

  data = self->data;
  for (i = 0; i < self->number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  for (i = 0; i < self->number_of_variables; ++i) {
    double c1;
    data->x[i] = 0.0;
    c1 = sqrt(pow(condition / 10., (double) i / (double) (self->number_of_variables - 1)));
    for (j = 0; j < self->number_of_variables; ++j) {
      data->x[i] += c1 * data->rot2[i][j] * (x[j] - data->xopt[j]);
    }
  }
  x1 = data->x[0];

  for (i = 0; i < self->number_of_variables; ++i) {
    if (fabs(data->x[i]) > 0.5)
      data->x[i] = round(data->x[i]);
    else
      data->x[i] = round(alpha * data->x[i]) / alpha;
  }

  for (i = 0; i < self->number_of_variables; ++i) {
    data->xx[i] = 0.0;
    for (j = 0; j < self->number_of_variables; ++j) {
      data->xx[i] += data->rot1[i][j] * data->x[j];
    }
  }

  /* Computation core */
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    double exponent;
    exponent = i / (self->number_of_variables - 1.0);
    y[0] += pow(condition, exponent) * data->xx[i] * data->xx[i];
    ;
  }
  y[0] = 0.1 * fmax(fabs(x1) * 1.0e-4, y[0]) + penalty + data->fopt;
}

static void _bbob_step_ellipsoid_free(coco_problem_t *self) {
  _bbob_step_ellipsoid_t *data;
  data = self->data;
  coco_free_memory(data->x);
  coco_free_memory(data->xx);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, self->number_of_variables);
  bbob2009_free_matrix(data->rot2, self->number_of_variables);
  /* Let the generic free problem code deal with all of the
   * coco_problem_t fields.
   */
  self->free_problem = NULL;
  coco_free_problem(self);
}

static coco_problem_t *
bbob_step_ellipsoid_problem(const size_t number_of_variables,
                            const long instance_id) {
  size_t i, problem_id_length;
  long rseed;
  coco_problem_t *problem;
  _bbob_step_ellipsoid_t *data;

  rseed = 7 + 10000 * instance_id;

  data = coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x = coco_allocate_vector(number_of_variables);
  data->xx = coco_allocate_vector(number_of_variables);
  data->xopt = coco_allocate_vector(number_of_variables);
  data->rot1 =
      bbob2009_allocate_matrix(number_of_variables, number_of_variables);
  data->rot2 =
      bbob2009_allocate_matrix(number_of_variables, number_of_variables);

  data->fopt = bbob2009_compute_fopt(7, instance_id);
  bbob2009_compute_xopt(data->xopt, rseed, (long)number_of_variables);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, (long)number_of_variables);
  bbob2009_compute_rotation(data->rot2, rseed, (long)number_of_variables);

  problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("BBOB f7");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "bbob2009_f7", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "bbob2009_f7",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->data = data;
  problem->evaluate_function = _bbob_step_ellipsoid_evaluate;
  problem->free_problem = _bbob_step_ellipsoid_free;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = NAN;
  }
  /* "Calculate" best parameter value.
   *
   * OME: Dirty hack for now because I did not want to invert the
   * transformations to find the best_parameter :/
   */
  problem->best_value[0] = data->fopt;
  return problem;
}
#line 11 "src/bbob2009_suite.c"
#line 1 "src/f_attractive_sector.c"
#include <assert.h>
#include <math.h>

#line 5 "src/f_attractive_sector.c"

#line 7 "src/f_attractive_sector.c"

typedef struct { double *xopt; } coco_bbob_attractive_sector_problem_data_t;

static void _attractive_sector_evaluate(coco_problem_t *self, const double *x,
                                        double *y) {
  size_t i;
  coco_bbob_attractive_sector_problem_data_t *data;

  assert(self->number_of_objectives == 1);
  data = self->data;
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    if (data->xopt[i] * x[i] > 0.0) {
      y[0] += 100.0 * 100.0 * x[i] * x[i];
    } else {
      y[0] += x[i] * x[i];
    }
  }
}

static void _attractive_sector_free(coco_problem_t *self) {
  coco_bbob_attractive_sector_problem_data_t *data;
  data = self->data;
  coco_free_memory(data->xopt);
  self->free_problem = NULL;
  coco_free_problem(self);
}

static coco_problem_t *
attractive_sector_problem(const size_t number_of_variables,
                          const double *xopt) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  coco_bbob_attractive_sector_problem_data_t *data;
  data = coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, number_of_variables);

  problem->problem_name = coco_strdup("attractive sector function");
  /* Construct a meaningful problem id */
  problem_id_length = snprintf(NULL, 0, "%s_%02i", "attractive_sector",
                               (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "attractive_sector", (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->data = data;
  problem->evaluate_function = _attractive_sector_evaluate;
  problem->free_problem = _attractive_sector_free;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _attractive_sector_evaluate(problem, problem->best_parameter,
                              problem->best_value);
  return problem;
}
#line 12 "src/bbob2009_suite.c"
#line 1 "src/f_bent_cigar.c"
#include <stdio.h>
#include <assert.h>

#line 5 "src/f_bent_cigar.c"

#line 7 "src/f_bent_cigar.c"

static void _bent_cigar_evaluate(coco_problem_t *self, const double *x, double *y) {
  static const double condition = 1.0e6;
  size_t i;
  assert(self->number_of_objectives == 1);

  y[0] = x[0] * x[0];
  for (i = 1; i < self->number_of_variables; ++i) {
    y[0] += condition * x[i] * x[i];
  }
}

static coco_problem_t *bent_cigar_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("bent cigar function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "bent_cigar", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "bent_cigar",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _bent_cigar_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _bent_cigar_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 13 "src/bbob2009_suite.c"
#line 14 "src/bbob2009_suite.c"
#line 1 "src/f_different_powers.c"
#include <assert.h>
#include <math.h>

#line 5 "src/f_different_powers.c"
#line 6 "src/f_different_powers.c"

static void _different_powers_evaluate(coco_problem_t *self, const double *x,
                                       double *y) {
  size_t i;
  double sum = 0.0;

  assert(self->number_of_objectives == 1);
  for (i = 0; i < self->number_of_variables; ++i) {
    double exponent = 2.0 + (4.0 * i) / (self->number_of_variables - 1.0);
    sum += pow(fabs(x[i]), exponent);
  }
  y[0] = sqrt(sum);
}

static coco_problem_t *
different_powers_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("different powers function");
  /* Construct a meaningful problem id */
  problem_id_length = snprintf(NULL, 0, "%s_%02i", "different powers",
                               (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "different powers", (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _different_powers_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _different_powers_evaluate(problem, problem->best_parameter,
                             problem->best_value);
  return problem;
}
#line 15 "src/bbob2009_suite.c"
#line 1 "src/f_discus.c"
#include <assert.h>

#line 4 "src/f_discus.c"
#line 5 "src/f_discus.c"

static void _discus_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  static const double condition = 1.0e6;
  assert(self->number_of_objectives == 1);

  y[0] = condition * x[0] * x[0];
  for (i = 1; i < self->number_of_variables; ++i) {
    y[0] += x[i] * x[i];
  }
}

static coco_problem_t *discus_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("discus function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "discus", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "discus",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _discus_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _discus_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 16 "src/bbob2009_suite.c"
#line 17 "src/bbob2009_suite.c"
#line 1 "src/f_griewankRosenbrock.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 6 "src/f_griewankRosenbrock.c"

#line 8 "src/f_griewankRosenbrock.c"

static void f_griewankRosenbrock_evaluate(coco_problem_t *self, const double *x,
                                          double *y) {
  size_t i;
  double tmp = 0;
  assert(self->number_of_objectives == 1);

  /* Computation core */
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables - 1; ++i) {
    const double c1 = x[i] * x[i] - x[i + 1];
    const double c2 = 1.0 - x[i];
    tmp = 100.0 * c1 * c1 + c2 * c2;
    y[0] += tmp / 4000. - cos(tmp);
  }
  y[0] = 10. + 10. * y[0] / (double)(self->number_of_variables - 1);
}

static coco_problem_t *
griewankRosenbrock_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("griewank rosenbrock function");
  /* Construct a meaningful problem id */
  problem_id_length = snprintf(NULL, 0, "%s_%02i", "griewank rosenbrock",
                               (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "griewank rosenbrock", (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = f_griewankRosenbrock_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 1.0; /* z^opt = 1*/
  }
  /* Calculate best parameter value */
  f_griewankRosenbrock_evaluate(problem, problem->best_parameter,
                                problem->best_value);
  return problem;
}
#line 18 "src/bbob2009_suite.c"
#line 19 "src/bbob2009_suite.c"
#line 20 "src/bbob2009_suite.c"
#line 21 "src/bbob2009_suite.c"
#line 1 "src/f_schaffers.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 6 "src/f_schaffers.c"

#line 8 "src/f_schaffers.c"

/* Schaffers F7 function, transformations not implemented for the moment  */

static void _schaffers_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  assert(self->number_of_variables > 1);
  assert(self->number_of_objectives == 1);

  /* Computation core */
  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables - 1; ++i) {
    const double tmp = x[i] * x[i] + x[i + 1] * x[i + 1];
    y[0] += pow(tmp, 0.25) * (1.0 + pow(sin(50.0 * pow(tmp, 0.1)), 2.0));
  }
  y[0] = pow(y[0] / (self->number_of_variables - 1.0), 2.0);
}

static coco_problem_t *schaffers_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("schaffers function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "schaffers", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "schaffers",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _schaffers_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _schaffers_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 22 "src/bbob2009_suite.c"
#line 1 "src/f_sharp_ridge.c"
#include <assert.h>
#include <math.h>

#line 5 "src/f_sharp_ridge.c"
#line 6 "src/f_sharp_ridge.c"

static void _sharp_ridge_evaluate(coco_problem_t *self, const double *x, double *y) {
  static const double alpha = 100.0;
  size_t i;
  assert(self->number_of_variables > 1);
  assert(self->number_of_objectives == 1);

  y[0] = 0.0;
  for (i = 1; i < self->number_of_variables; ++i) {
    y[0] += x[i] * x[i];
  }
  y[0] = alpha * sqrt(y[0]) + x[0] * x[0];
}

static coco_problem_t *sharp_ridge_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("sharp ridge function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "sharp_ridge", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "sharp_ridge",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _sharp_ridge_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }
  /* Calculate best parameter value */
  _sharp_ridge_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 23 "src/bbob2009_suite.c"
#line 24 "src/bbob2009_suite.c"
#line 1 "src/f_weierstrass.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 6 "src/f_weierstrass.c"

#line 8 "src/f_weierstrass.c"

/* Number of summands in the Weierstrass problem. */
#define WEIERSTRASS_SUMMANDS 12
typedef struct {
  double f0;
  double ak[WEIERSTRASS_SUMMANDS];
  double bk[WEIERSTRASS_SUMMANDS];
} _wss_problem_t;

static void _weierstrass_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i, j;
  _wss_problem_t *data = self->data;
  assert(self->number_of_objectives == 1);

  y[0] = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    for (j = 0; j < WEIERSTRASS_SUMMANDS; ++j) {
      y[0] += cos(2 * coco_pi * (x[i] + 0.5) * data->bk[j]) * data->ak[j];
    }
  }
  y[0] = 10.0 * pow(y[0] / self->number_of_variables - data->f0, 3.0);
}

static coco_problem_t *weierstrass_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  _wss_problem_t *data;
  data = coco_allocate_memory(sizeof(*data));

  data->f0 = 0.0;
  for (i = 0; i < WEIERSTRASS_SUMMANDS; ++i) {
    data->ak[i] = pow(0.5, (double)i);
    data->bk[i] = pow(3., (double)i);
    data->f0 += data->ak[i] * cos(2 * coco_pi * data->bk[i] * 0.5);
  }

  problem->problem_name = coco_strdup("weierstrass function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "weierstrass", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "weierstrass",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = _weierstrass_evaluate;
  problem->data = data;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 0.0;
  }

  /* Calculate best parameter value */
  _weierstrass_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}

#undef WEIERSTRASS_SUMMANDS
#line 25 "src/bbob2009_suite.c"
#line 26 "src/bbob2009_suite.c"
#line 1 "src/f_katsuura.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 6 "src/f_katsuura.c"

#line 8 "src/f_katsuura.c"

static void f_katsuura_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i, j;
  double tmp, tmp2;
  assert(self->number_of_objectives == 1);

  /* Computation core */
  y[0] = 1.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    tmp = 0;
    for (j = 1; j < 33; ++j) {
      tmp2 = pow(2., (double)j);
      tmp += fabs(tmp2 * x[i] - round(tmp2 * x[i])) / tmp2;
    }
    tmp = 1. + (i + 1) * tmp;
    y[0] *= tmp;
  }
  y[0] = 10. / ((double)self->number_of_variables) /
         ((double)self->number_of_variables) *
         (-1. + pow(y[0], 10. / pow((double)self->number_of_variables, 1.2)));
}

static coco_problem_t *katsuura_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("katsuura function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "katsuura", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "katsuura",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = f_katsuura_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = 1.0; /* z^opt = 1*/
  }
  /* Calculate best parameter value */
  f_katsuura_evaluate(problem, problem->best_parameter, problem->best_value);
  return problem;
}
#line 27 "src/bbob2009_suite.c"
#line 1 "src/f_schwefel.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#line 6 "src/f_schwefel.c"
#line 7 "src/f_schwefel.c"
#line 8 "src/f_schwefel.c"
#line 9 "src/f_schwefel.c"

static void f_schwefel_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double penalty, sum;
  assert(self->number_of_objectives == 1);

  /* Boundary handling*/
  penalty = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    const double tmp = fabs(x[i]) - 500.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* Computation core */
  sum = 0.0;
  for (i = 0; i < self->number_of_variables; ++i) {
    sum += x[i] * sin(sqrt(fabs(x[i])));
  }
  y[0] = 0.01 * (penalty + 418.9828872724339 -
                 sum / (double)self->number_of_variables);
  assert(y[0] >= self->best_value[0]);
}

static coco_problem_t *schwefel_problem(const size_t number_of_variables) {
  size_t i, problem_id_length;
  coco_problem_t *problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("schwefel function");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "schwefel", (int)number_of_variables);
  problem->problem_id = (char *)coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d", "schwefel",
           (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->evaluate_function = f_schwefel_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
    problem->best_parameter[i] = NAN;
  }
  /* "Calculate" best parameter value
   *
   * OME: Hard code optimal value for now...
   */
  problem->best_value[0] = 0.0;

  return problem;
}
#line 28 "src/bbob2009_suite.c"
#line 1 "src/f_lunacek_bi_rastrigin.c"
#line 2 "src/f_lunacek_bi_rastrigin.c"
#include <assert.h>
#include <math.h>

#line 6 "src/f_lunacek_bi_rastrigin.c"
#line 7 "src/f_lunacek_bi_rastrigin.c"
#line 8 "src/f_lunacek_bi_rastrigin.c"

typedef struct {
  double *x_hat, *z;
  double *xopt, fopt;
  double **rot1, **rot2;
  long rseed;
  coco_free_function_t old_free_problem;
} _bbob_lunacek_bi_rastrigin_t;

static void _bbob_lunacek_bi_rastrigin_evaluate(coco_problem_t *self, const double *x,
                                                double *y) {
  static const double condition = 100.;
  size_t i, j;
  double penalty = 0.0;
  static const double mu0 = 2.5;
  static const double d = 1.;
  const double s = 1. - 0.5 / (sqrt((double)(self->number_of_variables + 20)) - 4.1);
  const double mu1 = -sqrt((mu0 * mu0 - d) / s);
  _bbob_lunacek_bi_rastrigin_t *data;
  double *tmpvect, sum1 = 0., sum2 = 0., sum3 = 0.;

  assert(self->number_of_variables > 1);
  assert(self->number_of_objectives == 1);
  data = self->data;
  for (i = 0; i < self->number_of_variables; ++i) {
    double tmp;
    tmp = fabs(x[i]) - 5.0;
    if (tmp > 0.0)
      penalty += tmp * tmp;
  }

  /* x_hat */
  for (i = 0; i < self->number_of_variables; ++i) {
    data->x_hat[i] = 2. * x[i];
    if (data->xopt[i] < 0.) {
      data->x_hat[i] *= -1.;
    }
  }

  tmpvect = coco_allocate_vector(self->number_of_variables);
  /* affine transformation */
  for (i = 0; i < self->number_of_variables; ++i) {
    double c1;
    tmpvect[i] = 0.0;
    c1 = pow(sqrt(condition),
             ((double)i) / (double)(self->number_of_variables - 1));
    for (j = 0; j < self->number_of_variables; ++j) {
      tmpvect[i] += c1 * data->rot2[i][j] * (data->x_hat[j] - mu0);
    }
  }
  for (i = 0; i < self->number_of_variables; ++i) {
    data->z[i] = 0;
    for (j = 0; j < self->number_of_variables; ++j) {
      data->z[i] += data->rot1[i][j] * tmpvect[j];
    }
  }
  /* Computation core */
  for (i = 0; i < self->number_of_variables; ++i) {
    sum1 += (data->x_hat[i] - mu0) * (data->x_hat[i] - mu0);
    sum2 += (data->x_hat[i] - mu1) * (data->x_hat[i] - mu1);
    sum3 += cos(2 * coco_pi * data->z[i]);
  }
  y[0] = fmin(sum1, d * (double)self->number_of_variables + s * sum2) +
         10. * ((double)self->number_of_variables - sum3) + 1e4 * penalty;
  coco_free_memory(tmpvect);
}

static void _bbob_lunacek_bi_rastrigin_free(coco_problem_t *self) {
  _bbob_lunacek_bi_rastrigin_t *data;
  data = self->data;
  coco_free_memory(data->x_hat);
  coco_free_memory(data->z);
  coco_free_memory(data->xopt);
  bbob2009_free_matrix(data->rot1, self->number_of_variables);
  bbob2009_free_matrix(data->rot2, self->number_of_variables);
  /* Let the generic free problem code deal with all of the
   * coco_problem_t fields.
   */
  self->free_problem = NULL;
  coco_free_problem(self);
}

static coco_problem_t *
bbob_lunacek_bi_rastrigin_problem(const size_t number_of_variables,
                                  const long instance_id) {
  double *tmpvect;
  size_t i, problem_id_length;
  long rseed;
  coco_problem_t *problem;
  _bbob_lunacek_bi_rastrigin_t *data;
  static const double mu0 = 2.5;

  rseed = 24 + 10000 * instance_id;

  data = coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->x_hat = coco_allocate_vector(number_of_variables);
  data->z = coco_allocate_vector(number_of_variables);
  data->xopt = coco_allocate_vector(number_of_variables);
  data->rot1 =
      bbob2009_allocate_matrix(number_of_variables, number_of_variables);
  data->rot2 =
      bbob2009_allocate_matrix(number_of_variables, number_of_variables);
  data->rseed = rseed;

  data->fopt = bbob2009_compute_fopt(24, instance_id);
  bbob2009_compute_xopt(data->xopt, rseed, (long)number_of_variables);
  bbob2009_compute_rotation(data->rot1, rseed + 1000000, (long)number_of_variables);
  bbob2009_compute_rotation(data->rot2, rseed, (long)number_of_variables);

  problem = coco_allocate_problem(number_of_variables, 1, 0);
  problem->problem_name = coco_strdup("BBOB f24");
  /* Construct a meaningful problem id */
  problem_id_length =
      snprintf(NULL, 0, "%s_%02i", "bbob2009_f24", (int)number_of_variables);
  problem->problem_id = coco_allocate_memory(problem_id_length + 1);
  snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
           "bbob2009_f24", (int)number_of_variables);

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->data = data;
  problem->evaluate_function = _bbob_lunacek_bi_rastrigin_evaluate;
  problem->free_problem = _bbob_lunacek_bi_rastrigin_free;

  /* Computing xopt  */
  tmpvect = coco_allocate_vector(number_of_variables);
  bbob2009_gauss(tmpvect, (long)number_of_variables, rseed);
  for (i = 0; i < number_of_variables; ++i) {
    data->xopt[i] = 0.5 * mu0;
    if (tmpvect[i] < 0.0) {
      data->xopt[i] *= -1.0;
    }
    problem->best_parameter[i] = data->xopt[i];
  }
  coco_free_memory(tmpvect);

  for (i = 0; i < number_of_variables; ++i) {

    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
  }
  /* Calculate best parameter value */
  problem->evaluate_function(problem, problem->best_parameter,
                             problem->best_value);
  return problem;
}
#line 29 "src/bbob2009_suite.c"
#line 1 "src/f_gallagher.c"
#include <assert.h>
#include <math.h>

#line 5 "src/f_gallagher.c"
#line 6 "src/f_gallagher.c"
#line 7 "src/f_gallagher.c"

#define NB_PEAKS_21 101
#define NB_PEAKS_22 21
#define MAX_DIM BBOB2009_MAX_DIM

static double *bbob_gallagher_peaks;
/* To make dimension free of restrictions (and save memory for large MAX_DIM),
   these should be allocated in bbob_gallagher_problem */
static double bbob_gallagher_peaks21[NB_PEAKS_21 * MAX_DIM];
static double bbob_gallagher_peaks22[NB_PEAKS_22 * MAX_DIM];
static int compare_doubles(const void *, const void *);

typedef struct {
  long rseed;
  unsigned int number_of_peaks;
  double *xopt;
  double **rotation, **Xlocal, **arrScales;
  double *peakvalues;
  coco_free_function_t old_free_problem;
} _bbob_gallagher_t;

static void _bbob_gallagher_evaluate(coco_problem_t *self, const double *x,
                                     double *y) {
  size_t i, j; /*Loop over dim*/
  double *tmx;
  _bbob_gallagher_t *data = self->data;
  double a = 0.1;
  double tmp2, f = 0., Fadd, tmp, Fpen = 0., Ftrue = 0.;
  double fac = -0.5 / (double)self->number_of_variables;

  assert(self->number_of_objectives == 1);
  assert(self->number_of_variables > 0);
  /* Boundary handling */
  for (i = 0; i < self->number_of_variables; ++i) {
    tmp = fabs(x[i]) - 5.;
    if (tmp > 0.) {
      Fpen += tmp * tmp;
    }
  }
  Fadd = Fpen;
  /* Transformation in search space */
  /* FIXME: this should rather be done in bbob_gallagher_problem */
  tmx = (double *)calloc(self->number_of_variables, sizeof(double));
  for (i = 0; i < self->number_of_variables; i++) {
    for (j = 0; j < self->number_of_variables; ++j) {
      tmx[i] += data->rotation[i][j] * x[j];
    }
  }
  /* Computation core*/
  for (i = 0; i < data->number_of_peaks; ++i) {
    tmp2 = 0.;
    for (j = 0; j < self->number_of_variables; ++j) {
      tmp = (tmx[j] - data->Xlocal[j][i]);
      tmp2 += data->arrScales[i][j] * tmp * tmp;
    }
    tmp2 = data->peakvalues[i] * exp(fac * tmp2);
    f = fmax(f, tmp2);
  }

  f = 10. - f;
  if (f > 0) {
    Ftrue = log(f) / a;
    Ftrue = pow(exp(Ftrue + 0.49 * (sin(Ftrue) + sin(0.79 * Ftrue))), a);
  } else if (f < 0) {
    Ftrue = log(-f) / a;
    Ftrue =
        -pow(exp(Ftrue + 0.49 * (sin(0.55 * Ftrue) + sin(0.31 * Ftrue))), a);
  } else
    Ftrue = f;

  Ftrue *= Ftrue;
  Ftrue += Fadd;
  y[0] = Ftrue;
  assert(y[0] >= self->best_value[0]);
  /* FIXME: tmx hasn't been allocated with coco_allocate... */
  coco_free_memory(tmx);
}

static void _bbob_gallagher_free(coco_problem_t *self) {
  _bbob_gallagher_t *data;
  data = self->data;
  coco_free_memory(data->xopt);
  coco_free_memory(data->peakvalues);
  bbob2009_free_matrix(data->rotation, self->number_of_variables);
  bbob2009_free_matrix(data->Xlocal, self->number_of_variables);
  bbob2009_free_matrix(data->arrScales, data->number_of_peaks);
  self->free_problem = NULL;
  coco_free_problem(self);
}

static coco_problem_t *bbob_gallagher_problem(const size_t number_of_variables,
                                              const long instance_id,
                                              const unsigned int number_of_peaks) {
  size_t i, j, k, problem_id_length, *rperm;
  long rseed;
  coco_problem_t *problem;
  _bbob_gallagher_t *data;
  double maxcondition = 1000., maxcondition1 = 1000., *arrCondition,
         fitvalues[2] = {1.1, 9.1}; /*maxcondition1 satisfies the old code and
                                       the doc but seems wrong in that it is,
                                       with very high probabiliy, not the
                                       largest condition level!!!*/
  double b,
      c; /* Parameters for generating local optima. In the old code, they are
            different in f21 and f22 */
      
  assert(number_of_variables <= MAX_DIM);
  if (number_of_peaks == 101) {
    rseed = 21 + 10000 * instance_id;
    /* FIXME: rather use coco_allocate_vector here */
    bbob_gallagher_peaks = bbob_gallagher_peaks21;
    maxcondition1 = sqrt(maxcondition1);
  } else {
    rseed = 22 + 10000 * instance_id;
    bbob_gallagher_peaks = bbob_gallagher_peaks22;
  }

  data = coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->rseed = rseed;
  data->number_of_peaks = number_of_peaks;
  data->xopt = coco_allocate_vector(number_of_variables);
  data->rotation =
      bbob2009_allocate_matrix(number_of_variables, number_of_variables);
  data->Xlocal = bbob2009_allocate_matrix(number_of_variables, number_of_peaks);
  data->arrScales =
      bbob2009_allocate_matrix(number_of_peaks, number_of_variables);
  bbob2009_compute_rotation(data->rotation, rseed, (long)number_of_variables);
  problem = coco_allocate_problem(number_of_variables, 1, 0);
  /* Construct a meaningful problem id */
  if (number_of_peaks == NB_PEAKS_21) {
    problem->problem_name = coco_strdup("BBOB f21");
    problem_id_length =
        snprintf(NULL, 0, "%s_%02i", "bbob2009_f21", (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
             "bbob2009_f21", (int)number_of_variables);
    b = 10.;
    c = 5.;
  } else if (number_of_peaks == NB_PEAKS_22) {
    problem->problem_name = coco_strdup("BBOB f22");
    problem_id_length =
        snprintf(NULL, 0, "%s_%02i", "bbob2009_f22", (int)number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, "%s_%02d",
             "bbob2009_f22", (int)number_of_variables);
    b = 9.8;
    c = 4.9;
  } else {
    b = 0.0;
    c = 0.0;
    coco_error("Bad number of peaks in f_gallagher");
  }

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->data = data;
  problem->free_problem = _bbob_gallagher_free;
  problem->evaluate_function = _bbob_gallagher_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
  }

  /* Initialize all the data of the inner problem */
  bbob2009_unif(bbob_gallagher_peaks, number_of_peaks - 1, data->rseed);
  rperm = (size_t *)malloc((number_of_peaks - 1) * sizeof(size_t));
  for (i = 0; i < number_of_peaks - 1; ++i)
    rperm[i] = i;
  qsort(rperm, number_of_peaks - 1, sizeof(size_t), compare_doubles);

  /* Random permutation */
  arrCondition = coco_allocate_vector(number_of_peaks);
  arrCondition[0] = maxcondition1;
  data->peakvalues = coco_allocate_vector(number_of_peaks);
  data->peakvalues[0] = 10;
  for (i = 1; i < number_of_peaks; ++i) {
    arrCondition[i] = pow(maxcondition, (double)(rperm[i - 1]) /
                                            ((double)(number_of_peaks - 2)));
    data->peakvalues[i] = (double)(i - 1) / (double)(number_of_peaks - 2) *
                              (fitvalues[1] - fitvalues[0]) +
                          fitvalues[0];
  }
  coco_free_memory(rperm);

  rperm = (size_t *)malloc(number_of_variables * sizeof(size_t));
  for (i = 0; i < number_of_peaks; ++i) {
    bbob2009_unif(bbob_gallagher_peaks, (long)number_of_variables, data->rseed + 1000 * (long)i);
    for (j = 0; j < number_of_variables; ++j)
      rperm[j] = j;
    qsort(rperm, number_of_variables, sizeof(size_t), compare_doubles);
    for (j = 0; j < number_of_variables; ++j) {
      data->arrScales[i][j] =
          pow(arrCondition[i],
              ((double)rperm[j]) / ((double)(number_of_variables - 1)) - 0.5);
    }
  }
  coco_free_memory(rperm);

  bbob2009_unif(bbob_gallagher_peaks, (long)(number_of_variables * number_of_peaks), data->rseed);
  for (i = 0; i < number_of_variables; ++i) {
    data->xopt[i] = 0.8 * (b * bbob_gallagher_peaks[i] - c);
    problem->best_parameter[i] = 0.8 * (b * bbob_gallagher_peaks[i] - c);
    for (j = 0; j < number_of_peaks; ++j) {
      data->Xlocal[i][j] = 0.;
      for (k = 0; k < number_of_variables; ++k) {
        data->Xlocal[i][j] +=
            data->rotation[i][k] * (b * bbob_gallagher_peaks[j * number_of_variables + k] - c);
      }
      if (j == 0) {
        data->Xlocal[i][j] *= 0.8;
      }
    }
  }

  coco_free_memory(arrCondition);

  /* Calculate best parameter value */
  problem->evaluate_function(problem, problem->best_parameter,
                             problem->best_value);
  return problem;
}

static int compare_doubles(const void *a, const void *b) {
  double temp = bbob_gallagher_peaks[*(const int *)a] -
                bbob_gallagher_peaks[*(const int *)b]; /* TODO: replace int by size_t? */
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}

/* Be nice and remove defines from amalgamation */
#undef NB_PEAKS_21
#undef NB_PEAKS_22
#undef MAX_DIM
#line 30 "src/bbob2009_suite.c"

#line 1 "src/shift_objective.c"
#include <assert.h>

#line 4 "src/shift_objective.c"
#line 5 "src/shift_objective.c"

typedef struct { double offset; } _shift_objective_t;

static void _so_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  _shift_objective_t *data;
  data = coco_get_transform_data(self);
  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  y[0] += data->offset; /* FIXME: shifts only the first objective */
}

/**
 * Shift the objective value of the inner problem by offset.
 */
static coco_problem_t *shift_objective(coco_problem_t *inner_problem,
                                const double offset) {
  coco_problem_t *self;
  _shift_objective_t *data;
  data = coco_allocate_memory(sizeof(*data));
  data->offset = offset;

  self = coco_allocate_transformed_problem(inner_problem, data, NULL);
  self->evaluate_function = _so_evaluate_function;
  self->best_value[0] += offset; /* FIXME: shifts only the first objective */
  return self;
}
#line 32 "src/bbob2009_suite.c"
#line 1 "src/oscillate_objective.c"
#include <assert.h>
#include <math.h>

#line 5 "src/oscillate_objective.c"
#line 6 "src/oscillate_objective.c"

static void _oo_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  static const double factor = 0.1;
  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  if (y[0] != 0) {
    double log_y;
    log_y = log(fabs(y[0])) / factor;
    if (y[0] > 0) {
      y[0] = pow(exp(log_y + 0.49 * (sin(log_y) + sin(0.79 * log_y))), factor);
    } else {
      y[0] = -pow(exp(log_y + 0.49 * (sin(0.55 * log_y) + sin(0.31 * log_y))),
                  factor);
    }
  }
}

/**
 * Oscillate the objective value of the inner problem.
 *
 * Caveat: this can change best_parameter and best_value. 
 */
static coco_problem_t *oscillate_objective(coco_problem_t *inner_problem) {
  coco_problem_t *self;
  self = coco_allocate_transformed_problem(inner_problem, NULL, NULL);
  self->evaluate_function = _oo_evaluate_function;
  return self;
}
#line 33 "src/bbob2009_suite.c"
#line 1 "src/power_objective.c"
#include <assert.h>
#include <math.h>

#line 5 "src/power_objective.c"
#line 6 "src/power_objective.c"

typedef struct { double exponent; } _powo_data_t;

static void _powo_evaluate_function(coco_problem_t *self, const double *x,
                                    double *y) {
  _powo_data_t *data;
  data = coco_get_transform_data(self);
  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  y[0] = pow(y[0], data->exponent);
}

/**
 * Raise the objective value to the power of a given exponent.
 */
static coco_problem_t *power_objective(coco_problem_t *inner_problem,
                                const double exponent) {
  _powo_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->exponent = exponent;

  self = coco_allocate_transformed_problem(inner_problem, data, NULL);
  self->evaluate_function = _powo_evaluate_function;
  return self;
}
#line 34 "src/bbob2009_suite.c"

#line 1 "src/affine_transform_variables.c"
#include <stdbool.h>
#include <assert.h>

#line 5 "src/affine_transform_variables.c"
#line 6 "src/affine_transform_variables.c"

/*
 * Perform an affine transformation of the variable vector:
 *
 *   x |-> Mx + b
 *
 * The matrix M is stored in row-major format.
 */

typedef struct { double *M, *b, *x; } _atv_data_t;

static void _atv_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  size_t i, j;
  _atv_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);

  for (i = 0; i < inner_problem->number_of_variables; ++i) {
    /* data->M has self->number_of_variables columns and
     * problem->inner_problem->number_of_variables rows.
     */
    const double *current_row = data->M + i * self->number_of_variables;
    data->x[i] = data->b[i];
    for (j = 0; j < self->number_of_variables; ++j) {
      data->x[i] += x[j] * current_row[j];
    }
  }
  coco_evaluate_function(inner_problem, data->x, y);
}

static void _atv_free_data(void *thing) {
  _atv_data_t *data = thing;
  coco_free_memory(data->M);
  coco_free_memory(data->b);
  coco_free_memory(data->x);
}

/*
 * FIXMEs:
 * - Calculate new smallest/largest values of interest?
 * - Resize bounds vectors if input and output dimensions do not match
 * - problem_id and problem_name need to be adjusted
 */
static coco_problem_t *affine_transform_variables(coco_problem_t *inner_problem,
                                           const double *M, const double *b,
                                           const size_t number_of_variables) {
  coco_problem_t *self;
  _atv_data_t *data;
  size_t entries_in_M;

  entries_in_M = inner_problem->number_of_variables * number_of_variables;
  data = coco_allocate_memory(sizeof(*data));
  data->M = coco_duplicate_vector(M, entries_in_M);
  data->b = coco_duplicate_vector(b, inner_problem->number_of_variables);
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  self = coco_allocate_transformed_problem(inner_problem, data, _atv_free_data);
  self->evaluate_function = _atv_evaluate_function;
  return self;
}
#line 36 "src/bbob2009_suite.c"
#line 1 "src/asymmetric_variable_transform.c"
/*
 * Implementation of the BBOB T_asy transformation for variables.
 */
#include <math.h>
#include <assert.h>

#line 8 "src/asymmetric_variable_transform.c"
#line 9 "src/asymmetric_variable_transform.c"

typedef struct {
  double *x;
  double beta;
} _avt_data_t;

static void _avt_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double exponent;
  _avt_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
    if (x[i] > 0.0) {
      exponent =
          1.0 +
          (data->beta * i) / (self->number_of_variables - 1.0) * sqrt(x[i]);
      data->x[i] = pow(x[i], exponent);
    } else {
      data->x[i] = x[i];
    }
  }
  coco_evaluate_function(inner_problem, data->x, y);
}

static void _avt_free_data(void *thing) {
  _avt_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
static coco_problem_t *asymmetric_variable_transform(coco_problem_t *inner_problem,
                                              const double beta) {
  _avt_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->beta = beta;
  self = coco_allocate_transformed_problem(inner_problem, data, _avt_free_data);
  self->evaluate_function = _avt_evaluate_function;
  return self;
}
#line 37 "src/bbob2009_suite.c"
#line 1 "src/brs_transform.c"
/*
 * Implementation of the ominuous 's_i scaling' of the BBOB Bueche-Rastrigin
 * function.
 */
#include <math.h>
#include <assert.h>

#line 9 "src/brs_transform.c"
#line 10 "src/brs_transform.c"

typedef struct { double *x; } _brs_data_t;

static void _brs_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  double factor;
  _brs_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
    /* Function documentation says we should compute 10^(0.5 *
     * (i-1)/(D-1)). Instead we compute the equivalent
     * sqrt(10)^((i-1)/(D-1)) just like the legacy code.
     */
    factor = pow(sqrt(10.0), i / (self->number_of_variables - 1.0));
    /* Documentation specifies odd indexes and starts indexing
     * from 1, we use all even indexes since C starts indexing
     * with 0.
     */
    if (x[i] > 0.0 && i % 2 == 0) {
      factor *= 10.0;
    }
    data->x[i] = factor * x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
}

static void _brs_free_data(void *thing) {
  _brs_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
static coco_problem_t *brs_transform(coco_problem_t *inner_problem) {
  _brs_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  self = coco_allocate_transformed_problem(inner_problem, data, _brs_free_data);
  self->evaluate_function = _brs_evaluate_function;
  return self;
}
#line 38 "src/bbob2009_suite.c"
#line 1 "src/condition_variables.c"
/*
 * Implementation of the BBOB Gamma transformation for variables.
 */
#include <math.h>
#include <assert.h>

#line 8 "src/condition_variables.c"
#line 9 "src/condition_variables.c"

typedef struct {
  double *x;
  double alpha;
} _cv_data_t;

static void _cv_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  _cv_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
    /* OME: We could precalculate the scaling coefficients if we
     * really wanted to.
     */
    data->x[i] =
        pow(data->alpha, 0.5 * i / (self->number_of_variables - 1.0)) * x[i];
  }
  coco_evaluate_function(inner_problem, data->x, y);
}

static void _cv_free_data(void *thing) {
  _cv_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
static coco_problem_t *condition_variables(coco_problem_t *inner_problem,
                                    const double alpha) {
  _cv_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->x = coco_allocate_vector(inner_problem->number_of_variables);
  data->alpha = alpha;
  self = coco_allocate_transformed_problem(inner_problem, data, _cv_free_data);
  self->evaluate_function = _cv_evaluate_function;
  return self;
}
#line 39 "src/bbob2009_suite.c"
#line 1 "src/oscillate_variables.c"
/*
 * Implementation of the BBOB T_osz transformation for variables.
 */

#include <math.h>
#include <assert.h>

#line 9 "src/oscillate_variables.c"
#line 10 "src/oscillate_variables.c"

typedef struct { double *oscillated_x; } _ov_data_t;

static void _ov_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  static const double alpha = 0.1;
  double tmp, base, *oscillated_x;
  size_t i;
  _ov_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_get_transform_data(self);
  oscillated_x = data->oscillated_x; /* short cut to make code more readable */
  inner_problem = coco_get_transform_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
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
}

static void _ov_free_data(void *thing) {
  _ov_data_t *data = thing;
  coco_free_memory(data->oscillated_x);
}

/**
 * Perform monotone oscillation transformation on input variables.
 */
static coco_problem_t *oscillate_variables(coco_problem_t *inner_problem) {
  _ov_data_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->oscillated_x = coco_allocate_vector(inner_problem->number_of_variables);

  self = coco_allocate_transformed_problem(inner_problem, data, _ov_free_data);
  self->evaluate_function = _ov_evaluate_function;
  return self;
}
#line 40 "src/bbob2009_suite.c"
#line 1 "src/scale_variables.c"
/*
 * Scale variables by a given factor.
 */
#include <assert.h>

#line 7 "src/scale_variables.c"
#line 8 "src/scale_variables.c"

typedef struct {
  double factor;
  double *x;
} _scv_data_t;

static void _scv_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  _scv_data_t *data;
  coco_problem_t *inner_problem;
  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);
  do {
    const double factor = data->factor;

    for (i = 0; i < self->number_of_variables; ++i) {
      data->x[i] = factor * x[i];
    }
    coco_evaluate_function(inner_problem, data->x, y);
    assert(y[0] >= self->best_value[0]);
  } while (0);
}

static void _scv_free_data(void *thing) {
  _scv_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Scale all variables by factor before evaluation.
 */
static coco_problem_t *scale_variables(coco_problem_t *inner_problem,
                                const double factor) {
  _scv_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->factor = factor;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  self = coco_allocate_transformed_problem(inner_problem, data, _scv_free_data);
  self->evaluate_function = _scv_evaluate_function;
  return self;
}
#line 41 "src/bbob2009_suite.c"
#line 1 "src/shift_variables.c"
#include <assert.h>

#line 4 "src/shift_variables.c"
#line 5 "src/shift_variables.c"

typedef struct {
  double *offset;
  double *shifted_x;
  coco_free_function_t old_free_problem;
} _sv_data_t;

static void _sv_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  size_t i;
  _sv_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);

  for (i = 0; i < self->number_of_variables; ++i) {
    data->shifted_x[i] = x[i] - data->offset[i];
  }
  coco_evaluate_function(inner_problem, data->shifted_x, y);
  assert(y[0] >= self->best_value[0]);
}

static void _sv_free_data(void *thing) {
  _sv_data_t *data = thing;
  coco_free_memory(data->shifted_x);
  coco_free_memory(data->offset);
}

/* Shift all variables of ${inner_problem} by ${amount}.
 */
static coco_problem_t *shift_variables(coco_problem_t *inner_problem,
                                const double *offset, const int shift_bounds) {
  _sv_data_t *data;
  coco_problem_t *self;
  if (shift_bounds)
    coco_error("shift_bounds not implemented.");

  data = coco_allocate_memory(sizeof(*data));
  data->offset =
      coco_duplicate_vector(offset, inner_problem->number_of_variables);
  data->shifted_x = coco_allocate_vector(inner_problem->number_of_variables);

  self = coco_allocate_transformed_problem(inner_problem, data, _sv_free_data);
  self->evaluate_function = _sv_evaluate_function;
  return self;
}
#line 42 "src/bbob2009_suite.c"
#line 1 "src/x_hat_schwefel.c"
#include <assert.h>

#line 4 "src/x_hat_schwefel.c"
#line 5 "src/x_hat_schwefel.c"
#line 6 "src/x_hat_schwefel.c"

typedef struct {
  long seed;
  double *x;
  coco_free_function_t old_free_problem;
} _x_hat_data_t;

static void _x_hat_evaluate_function(coco_problem_t *self, const double *x,
                                     double *y) {
  size_t i;
  _x_hat_data_t *data;
  coco_problem_t *inner_problem;
  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);
  do {
    bbob2009_unif(data->x, (long)self->number_of_variables, data->seed);

    for (i = 0; i < self->number_of_variables; ++i) {
      if (data->x[i] - 0.5 < 0.0) {
        data->x[i] = -x[i];
      } else {
        data->x[i] = x[i];
      }
    }
    coco_evaluate_function(inner_problem, data->x, y);
  } while (0);
}

static void _x_hat_free_data(void *thing) {
  _x_hat_data_t *data = thing;
  coco_free_memory(data->x);
}

/**
 * Multiply the x-vector by the vector 2 * 1+-
 */
static coco_problem_t *x_hat(coco_problem_t *inner_problem, long seed) {
  _x_hat_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->seed = seed;
  data->x = coco_allocate_vector(inner_problem->number_of_variables);

  self =
      coco_allocate_transformed_problem(inner_problem, data, _x_hat_free_data);
  self->evaluate_function = _x_hat_evaluate_function;
  return self;
}
#line 43 "src/bbob2009_suite.c"
#line 1 "src/z_hat_schwefel.c"
#include <assert.h>

#line 4 "src/z_hat_schwefel.c"
#line 5 "src/z_hat_schwefel.c"

typedef struct {
  double *xopt;
  double *z;
  coco_free_function_t old_free_problem;
} _z_hat_data_t;

static void _z_hat_evaluate_function(coco_problem_t *self, const double *x,
                                     double *y) {
  size_t i;
  _z_hat_data_t *data;
  coco_problem_t *inner_problem;

  data = coco_get_transform_data(self);
  inner_problem = coco_get_transform_inner_problem(self);

  data->z[0] = x[0];

  for (i = 1; i < self->number_of_variables; ++i) {
    data->z[i] = x[i] + 0.25 * (x[i - 1] - 2.0 * fabs(data->xopt[i - 1]));
  }
  coco_evaluate_function(inner_problem, data->z, y);
  assert(y[0] >= self->best_value[0]);
}

static void _z_hat_free_data(void *thing) {
  _z_hat_data_t *data = thing;
  coco_free_memory(data->xopt);
  coco_free_memory(data->z);
}

/* Compute the vector {z^hat} for f_schwefel
 */
static coco_problem_t *z_hat(coco_problem_t *inner_problem, const double *xopt) {
  _z_hat_data_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->xopt = coco_duplicate_vector(xopt, inner_problem->number_of_variables);
  data->z = coco_allocate_vector(inner_problem->number_of_variables);

  self =
      coco_allocate_transformed_problem(inner_problem, data, _z_hat_free_data);
  self->evaluate_function = _z_hat_evaluate_function;
  return self;
}
#line 44 "src/bbob2009_suite.c"
#line 1 "src/penalize_uninteresting_values.c"
#include <assert.h>

#line 4 "src/penalize_uninteresting_values.c"
#line 5 "src/penalize_uninteresting_values.c"

typedef struct {
  double factor;
} _puv_data_t;

static void _puv_evaluate_function(coco_problem_t *self, const double *x, double *y) {
	_puv_data_t *data = coco_get_transform_data(self);
	const double *lower_bounds = self->smallest_values_of_interest;
	const double *upper_bounds = self->largest_values_of_interest;
	double penalty = 0.0;
	size_t i;
	for (i = 0; i < self->number_of_variables; ++i) {
		const double c1 = x[i] - upper_bounds[i];
		const double c2 = lower_bounds[i] - x[i];
		assert(lower_bounds[i] < upper_bounds[i]);
		if (c1 > 0.0) {
			penalty += c1 * c1;
		} else if (c2 > 0.0) {
			penalty += c2 * c2;
		}
	}
	assert(coco_get_transform_inner_problem(self) != NULL);
	/*assert(problem->state != NULL);*/
	coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
	for (i = 0; i < self->number_of_objectives; ++i) {
		y[i] += data->factor * penalty;
	}
}

/**
 * penalize_uninteresting_values(inner_problem):
 *
 * Add a penalty to all evaluations outside of the region of interest
 * of ${inner_problem}.
 */
static coco_problem_t *penalize_uninteresting_values(coco_problem_t *inner_problem, const double factor) {
	coco_problem_t *self;
	_puv_data_t *data;
	assert(inner_problem != NULL);
	/* assert(offset != NULL); */
	
	data = coco_allocate_memory(sizeof(*data));
	data->factor = factor;
	self = coco_allocate_transformed_problem(inner_problem, data, NULL);
	self->evaluate_function = _puv_evaluate_function;
	return self;
}
#line 45 "src/bbob2009_suite.c"

#define MAX_DIM BBOB2009_MAX_DIM
#define BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES 5
#define BBOB2009_NUMBER_OF_FUNCTIONS 24
#define BBOB2009_NUMBER_OF_DIMENSIONS 6
static const unsigned BBOB2009_DIMS[] = {2, 3, 5, 10, 20, 40};/*might end up useful outside of bbob2009_decode_problem_index*/

/**
 * bbob2009_decode_problem_index(problem_index, function_id, instance_id,
 *dimension):
 *
 * Decode the new problem_index into the old convention of function,
 * instance and dimension. We have 24 functions in 6 different
 * dimensions so a total of 144 functions and any number of
 * instances. A natural thing would be to order them so that the
 * function varies faster than the dimension which is still faster
 * than the instance. For analysis reasons we want something
 * different. Our goal is to quickly produce 5 repetitions of a single
 * function in one dimension, then vary the function, then the
 * dimension.
 *
 * TODO: this is the default prescription for 2009. This is typically
 *       not what we want _now_, as the instances change in each
 *       workshop. We should have provide-problem-instance-indices
 *       methods to be able to run useful subsets of instances.
 * 
 * This gives us:
 *
 * problem_index | function_id | instance_id | dimension
 * ---------------+-------------+-------------+-----------
 *              0 |           1 |           1 |         2
 *              1 |           1 |           2 |         2
 *              2 |           1 |           3 |         2
 *              3 |           1 |           4 |         2
 *              4 |           1 |           5 |         2
 *              5 |           2 |           1 |         2
 *              6 |           2 |           2 |         2
 *             ...           ...           ...        ...
 *            119 |          24 |           5 |         2
 *            120 |           1 |           1 |         3
 *            121 |           1 |           2 |         3
 *             ...           ...           ...        ...
 *           2157 |          24 |           13|        40
 *           2158 |          24 |           14|        40
 *           2159 |          24 |           15|        40
 *
 * The quickest way to decode this is using integer division and
 * remainders.
 */

static void bbob2009_decode_problem_index(const long problem_index, int *function_id,
                                    long *instance_id, long *dimension) {
  const long high_instance_id =
      problem_index / (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCTIONS *
                        BBOB2009_NUMBER_OF_DIMENSIONS);
  long low_instance_id;
  long rest = problem_index % (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES *
                               BBOB2009_NUMBER_OF_FUNCTIONS * BBOB2009_NUMBER_OF_DIMENSIONS);
  *dimension =
      BBOB2009_DIMS[rest / (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCTIONS)];
  rest = rest % (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCTIONS);
  *function_id = (int)(rest / BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES + 1);
  rest = rest % BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
  low_instance_id = rest + 1;
  *instance_id = low_instance_id + 5 * high_instance_id;
}

/* Encodes a triplet of (function_id, instance_id, dimension_idx) into a problem_index
 * The problem index can, then, be used to directly generate a problem
 * It helps allow easier control on instances, functions and dimensions one wants to run
 * all indices start from 0 TODO: start at 1 instead?
 */
static long bbob2009_encode_problem_index(int function_id, long instance_id, int dimension_idx){
    long cycleLength = BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCTIONS * BBOB2009_NUMBER_OF_DIMENSIONS;
    long tmp1 = instance_id % BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
    long tmp2 = function_id * BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES;
    long tmp3 = dimension_idx * (BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES * BBOB2009_NUMBER_OF_FUNCTIONS);
    long tmp4 = ((long)(instance_id / BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES) ) * cycleLength; /* just for safety */
    
    return tmp1 + tmp2 + tmp3 + tmp4;
}

static void bbob2009_copy_rotation_matrix(double **rot, double *M, double *b,
                                          const size_t dimension) {
  size_t row, column;
  double *current_row;

  for (row = 0; row < dimension; ++row) {
    current_row = M + row * dimension;
    for (column = 0; column < dimension; ++column) {
      current_row[column] = rot[row][column];
    }
    b[row] = 0.0;
  }
}

static coco_problem_t *bbob2009_problem(int function_id, long dimension_, long instance_id) {
  size_t len;
  long rseed;
  coco_problem_t *problem = NULL;
  const size_t dimension = (unsigned long) dimension_;
  
  /* This assert is a hint for the static analyzer. */
  assert(dimension > 1);
  if (dimension > MAX_DIM)
    coco_error("bbob2009_suite currently supports dimension up to %ld (%ld given)", 
        MAX_DIM, dimension);

#if 0
  {  /* to be removed */
    int dimension_idx;
    switch (dimension) {/*TODO: make this more dynamic*//* This*/
            case 2:
            dimension_idx = 0;
            break;
            case 3:
            dimension_idx = 1;
            break;
            case 5:
            dimension_idx = 2;
            break;
            case 10:
            dimension_idx = 3;
            break;
            case 20:
            dimension_idx = 4;
            break;
            case 40:
            dimension_idx = 5;
            break;
        default:
            dimension_idx = -1;
            break;
    }
    assert(problem_index == bbob2009_encode_problem_index(function_id - 1, instance_id - 1 , dimension_idx));
  }
#endif 
  rseed = function_id + 10000 * instance_id;

  /* Break if we are past our 15 instances. */
  if (instance_id > 15)
    return NULL;

  if (function_id == 1) {
    double xopt[MAX_DIM], fopt;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    fopt = bbob2009_compute_fopt(function_id, instance_id);

    problem = sphere_problem(dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 2) {
    double xopt[MAX_DIM], fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    problem = ellipsoid_problem(dimension);
    problem = oscillate_variables(problem);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 3) {
    double xopt[MAX_DIM], fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    problem = rastrigin_problem(dimension);
    problem = condition_variables(problem, 10.0);
    problem = asymmetric_variable_transform(problem, 0.2);
    problem = oscillate_variables(problem);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 4) {
    unsigned i; /*to prevent warnings, changed for all i,j and k variables used to iterate over coordinates*/
    double xopt[MAX_DIM], fopt, penalty_factor = 100.0;
    rseed = 3 + 10000 * instance_id;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    /*
     * OME: This step is in the legacy C code but _not_ in the
     * function description.
     */
    for (i = 0; i < dimension; i += 2) {
      xopt[i] = fabs(xopt[i]);
    }

    problem = bueche_rastrigin_problem(dimension);
    problem = brs_transform(problem);
    problem = oscillate_variables(problem);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
    problem = penalize_uninteresting_values(problem, penalty_factor);
  } else if (function_id == 5) {
    double xopt[MAX_DIM], fopt;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = linear_slope_problem(dimension, xopt);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 6) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = k * 1.0 / (dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(10.0), exponent) * rot2[k][j];
        }
      }
    }
    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);

    problem = attractive_sector_problem(dimension, xopt);
    problem = oscillate_objective(problem);
    problem = power_objective(problem, 0.9);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
  } else if (function_id == 7) {
    problem = bbob_step_ellipsoid_problem(dimension, instance_id);
  } else if (function_id == 8) {
    unsigned i;
    double xopt[MAX_DIM], minus_one[MAX_DIM], fopt, factor;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      minus_one[i] = -1.0;
      xopt[i] *= 0.75;
    }
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    /* C89 version of
     *   fmax(1.0, sqrt(dimension) / 8.0);
     * follows
     */
    factor = sqrt(dimension) / 8.0;
    if (factor < 1.0)
      factor = 1.0;

    problem = rosenbrock_problem(dimension);
    problem = shift_variables(problem, minus_one, 0);
    problem = scale_variables(problem, factor);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 9) {
    unsigned row, column;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], fopt, factor, *current_row;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed, dimension_);
    /* C89 version of
     *   fmax(1.0, sqrt(dimension) / 8.0);
     * follows
     */
    factor = sqrt(dimension) / 8.0;
    if (factor < 1.0)
      factor = 1.0;
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

    problem = rosenbrock_problem(dimension);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 10) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    bbob2009_compute_xopt(xopt, rseed, dimension_);
    fopt = bbob2009_compute_fopt(function_id, instance_id);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = ellipsoid_problem(dimension);
    problem = oscillate_variables(problem);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 11) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = discus_problem(dimension);
    problem = oscillate_variables(problem);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 12) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed + 1000000, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = bent_cigar_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = asymmetric_variable_transform(problem, 0.5);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
  } else if (function_id == 13) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = k * 1.0 / (dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
        }
      }
    }
    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
    problem = sharp_ridge_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
  } else if (function_id == 14) {
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt;
    double **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    bbob2009_free_matrix(rot1, dimension);

    problem = different_powers_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
  } else if (function_id == 15) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = k * 1.0 / (dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(10), exponent) * rot2[k][j];
        }
      }
    }

    problem = rastrigin_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = asymmetric_variable_transform(problem, 0.2);
    problem = oscillate_variables(problem);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 16) {
    unsigned i, j, k;
    static double condition = 100.;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0/(double)dimension;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          const double base = 1.0 / sqrt(condition);
          const double exponent = k * 1.0 / (dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(base, exponent) * rot2[k][j];
        }
      }
    }

    problem = weierstrass_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = oscillate_variables(problem);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 17) {
    unsigned i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        double exponent = i / (dimension - 1.0);
        current_row[j] = rot2[i][j] * pow(sqrt(10), exponent);
      }
    }

    problem = schaffers_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = asymmetric_variable_transform(problem, 0.5);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 18) {
    unsigned i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row, penalty_factor = 10.0;
    double **rot1, **rot2;
    /* Reuse rseed from f17. */
    rseed = 17 + 10000 * instance_id;

    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        double exponent = i * 1.0 / (dimension - 1.0);
        current_row[j] = rot2[i][j] * pow(sqrt(1000), exponent);
      }
    }

    problem = schaffers_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = asymmetric_variable_transform(problem, 0.5);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 19) {
    unsigned i, j;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], shift[MAX_DIM], fopt;
    double scales, **rot1;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    for (i = 0; i < dimension; ++i) {
      shift[i] = -0.5;
    }

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed, dimension_);
    scales = fmax(1., sqrt((double)dimension) / 8.);
    for (i = 0; i < dimension; ++i) {
      for (j = 0; j < dimension; ++j) {
        rot1[i][j] *= scales;
      }
    }

    problem = griewankRosenbrock_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = shift_variables(problem, shift, 0);
    bbob2009_copy_rotation_matrix(rot1, M, b, dimension);
    problem = affine_transform_variables(problem, M, b, dimension);

    bbob2009_free_matrix(rot1, dimension);

  } else if (function_id == 20) {
    unsigned i, j;
    static double condition = 10.;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], fopt, *current_row,
        *tmp1 = coco_allocate_vector(dimension),
        *tmp2 = coco_allocate_vector(dimension);
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_unif(tmp1, dimension_, rseed);
    for (i = 0; i < dimension; ++i) {
      xopt[i] = 0.5 * 4.2096874633;
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
          double exponent = (double)i / (dimension - 1);
          current_row[j] = pow(sqrt(condition), exponent);
        }
      }
    }
    for (i = 0; i < dimension; ++i) {
      tmp1[i] = -2 * fabs(xopt[i]);
      tmp2[i] = 2 * fabs(xopt[i]);
    }
    problem = schwefel_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = scale_variables(problem, 100);
    problem = shift_variables(problem, tmp1, 0);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, tmp2, 0);
    problem = z_hat(problem, xopt);
    problem = scale_variables(problem, 2);
    problem = x_hat(problem, rseed);
    coco_free_memory(tmp1);
    coco_free_memory(tmp2);
  } else if (function_id == 21) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = bbob_gallagher_problem(dimension, instance_id, 101);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 22) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = bbob_gallagher_problem(dimension, instance_id, 21);
    problem = shift_objective(problem, fopt);
  } else if (function_id == 23) {
    unsigned i, j, k;
    double M[MAX_DIM * MAX_DIM], b[MAX_DIM], xopt[MAX_DIM], *current_row, fopt, penalty_factor = 1.0;
    double **rot1, **rot2;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    bbob2009_compute_xopt(xopt, rseed, dimension_);

    rot1 = bbob2009_allocate_matrix(dimension, dimension);
    rot2 = bbob2009_allocate_matrix(dimension, dimension);
    bbob2009_compute_rotation(rot1, rseed + 1000000, dimension_);
    bbob2009_compute_rotation(rot2, rseed, dimension_);
    for (i = 0; i < dimension; ++i) {
      b[i] = 0.0;
      current_row = M + i * dimension;
      for (j = 0; j < dimension; ++j) {
        current_row[j] = 0.0;
        for (k = 0; k < dimension; ++k) {
          double exponent = k * 1.0 / (dimension - 1.0);
          current_row[j] += rot1[i][k] * pow(sqrt(100), exponent) * rot2[k][j];
        }
      }
    }
    problem = katsuura_problem(dimension);
    problem = shift_objective(problem, fopt);
    problem = affine_transform_variables(problem, M, b, dimension);
    problem = shift_variables(problem, xopt, 0);
    problem = penalize_uninteresting_values(problem, penalty_factor);

    bbob2009_free_matrix(rot1, dimension);
    bbob2009_free_matrix(rot2, dimension);
  } else if (function_id == 24) {
    double fopt;
    fopt = bbob2009_compute_fopt(function_id, instance_id);
    problem = bbob_lunacek_bi_rastrigin_problem(dimension, instance_id);
    problem = shift_objective(problem, fopt);
  } else {
    return NULL;
  }

  /* Now set the problem name and problem id of the final problem */
  coco_free_memory(problem->problem_name);
  coco_free_memory(problem->problem_id);

  /* Construct a meaningful problem id */
  len = snprintf(NULL, 0, "bbob2009_f%02i_i%02li_d%02lu", function_id,
                 instance_id, dimension);
  problem->problem_id = coco_allocate_memory(len + 1);
  snprintf(problem->problem_id, len + 1, "bbob2009_f%02i_i%02li_d%02lu",
           function_id, instance_id, dimension);

  len = snprintf(NULL, 0, "BBOB2009 f%02i instance %li in %luD", function_id,
                 instance_id, dimension);
  problem->problem_name = coco_allocate_memory(len + 1);
  snprintf(problem->problem_name, len + 1, "BBOB2009 f%02i instance %li in %luD",
           function_id, instance_id, dimension);
  return problem;
}

/* Return the bbob2009 function id of the problem or -1 if it is not a bbob2009
 * problem. */
static int bbob2009_get_function_id(const coco_problem_t *problem) {
  static const char *bbob_prefix = "bbob2009_";
  const char *problem_id = coco_get_problem_id(problem);
  assert(strlen(problem_id) >= 20);

  if (strncmp(bbob_prefix, problem_id, strlen(bbob_prefix)) != 0) {
    return -1;
  }

  /* OME: Ugly hardcoded extraction. In a perfect world, we would
   * parse the problem id by splitting on _ and then finding the 'f'
   * field. Instead, we cound out the position of the function id in
   * the string
   *
   *   01234567890123456789
   *   bbob2009_fXX_iYY_dZZ
   */
  return (problem_id[10] - '0') * 10 + (problem_id[11] - '0');
}

/* Return the bbob2009 instance id of the problem or -1 if it is not a bbob2009
 * problem. */
static int bbob2009_get_instance_id(const coco_problem_t *problem) {
  static const char *bbob_prefix = "bbob2009_";
  const char *problem_id = coco_get_problem_id(problem);
  assert(strlen(problem_id) >= 20);

  if (strncmp(bbob_prefix, problem_id, strlen(bbob_prefix)) != 0) {
    return -1;
  }

  /* OME: Ugly hardcoded extraction. In a perfect world, we would
   * parse the problem id by splitting on _ and then finding the 'i'
   * field. Instead, we cound out the position of the instance id in
   * the string
   *
   *   01234567890123456789
   *   bbob2009_fXX_iYY_dZZ
   */
  return (problem_id[14] - '0') * 10 + (problem_id[15] - '0');
}

/* TODO: specify selection_descriptor and implement
 *
 * Possible example for a descriptor: "instance:1-5, dimension:-20",
 * where instances are relative numbers (w.r.t. to the instances in
 * test bed), dimensions are absolute.
 *
 * Return successor of problem_index or first index if problem_index < 0 or -1 otherwise.
 *
 * Details: this function is not necessary unless selection is implemented. 
*/
static long bbob2009_next_problem_index(long problem_index, const char *selection_descriptor) {
  const long first_index = 0;
  const long last_index = 2159;
  
  if (problem_index < 0)
    problem_index = first_index - 1;
    
  if (strlen(selection_descriptor) == 0) {
    if (problem_index < last_index)
      return problem_index + 1;
    return -1;
  }
  
  /* TODO:
     o parse the selection_descriptor -> value bounds on funID, dimension, instance
     o inrement problem_index until funID, dimension, instance match the restrictions
       or max problem_index is succeeded. 
    */
  
  coco_error("next_problem_index is yet to be implemented for specific selections");
  return -1;
}

/**
 * bbob2009_suite(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from the BBOB2009
 * benchmark suit. If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *bbob2009_suite(long problem_index) {
  coco_problem_t *problem;
  int function_id;
  long dimension, instance_id;
  
  if (problem_index < 0)
    return NULL; 
  bbob2009_decode_problem_index(problem_index, &function_id, &instance_id,
                                 &dimension);
  problem = bbob2009_problem(function_id, dimension, instance_id);
  problem->index = problem_index;
  return problem;
}

/* Undefine constants */
#undef MAX_DIM
#undef BBOB2009_NUMBER_OF_CONSECUTIVE_INSTANCES 
#undef BBOB2009_NUMBER_OF_FUNCTIONS 
#undef BBOB2009_NUMBER_OF_DIMENSIONS 
#line 12 "src/coco_benchmark.c"
#line 1 "src/bbob2009_observer.c"
#line 2 "src/bbob2009_observer.c"

#line 1 "src/bbob2009_logger.c"
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <errno.h>

#line 9 "src/bbob2009_logger.c"

#line 11 "src/bbob2009_logger.c"
#line 12 "src/bbob2009_logger.c"
#line 13 "src/bbob2009_logger.c"

static int bbob2009_logger_verbosity = 3;  /* TODO: make this an option the user can modify */
static int raisedOptValWarning;
static int bbob2009_get_function_id(const coco_problem_t *problem);
static int bbob2009_get_instance_id(const coco_problem_t *problem);

/* FIXME: these names could easily created conflicts with other coco.c-global names. Use bbob2009 as prefix to prevent conflicts. */
static const size_t bbob2009_nbpts_nbevals = 20;
static const size_t bbob2009_nbpts_fval = 5;
static size_t current_dim = 0;
static long current_funId = 0;
static int infoFile_firstInstance = 0;
char infoFile_firstInstance_char[3];
/*a possible solution: have a list of dims that are already in the file, if the ones we're about to log is != current_dim and the funId is currend_funId, create a new .info file with as suffix the number of the first instance */
static const int bbob2009_number_of_dimensions = 6;
static size_t dimensions_in_current_infoFile[6] = {0,0,0,0,0,0}; /*TODO should use BBOB2009_NUMBER_OF_DIMENSIONS*/


/* The current_... mechanism fails if several problems are open. 
 * For the time being this should lead to an error.
 *
 * A possible solution: bbob2009_logger_is_open becomes a reference
 * counter and as long as another logger is open, always a new info
 * file is generated. 
 */
static int bbob2009_logger_is_open = 0;  /* this could become lock-list of .info files */

/*TODO: add possibility of adding a prefix to the index files*/

typedef struct {
  int is_initialized;
  char *path; /*relative path to the data folder. Simply the Algname*/
  const char *
      alg_name;      /*the alg name, for now, temporarly the same as the path*/
  FILE *index_file;  /*index file*/
  FILE *fdata_file;  /*function value aligned data file*/
  FILE *tdata_file;  /*number of function evaluations aligned data file*/
  FILE *rdata_file;  /*restart info data file*/
  double f_trigger;  /* next upper bound on the fvalue to trigger a log in the
                        .dat file*/
  long t_trigger;    /* next lower bound on nb fun evals to trigger a log in the
                        .tdat file*/
  int idx_f_trigger; /* allows to track the index i in logging target =
                        {10**(i/bbob2009_nbpts_fval), i \in Z} */
  int idx_t_trigger; /* allows to track the index i in logging nbevals  =
                        {int(10**(i/bbob2009_nbpts_nbevals)), i \in Z} */
  int idx_tdim_trigger; /* allows to track the index i in logging nbevals  =
                           {dim * 10**i, i \in Z} */
  long number_of_evaluations;
  double best_fvalue;
  double last_fvalue;
  short written_last_eval; /*allows writing the the data of the final fun eval
                              in the .tdat file if not already written by the
                              t_trigger*/
  double *best_solution;
  /*the following are to only pass data as a parameter in the free function. The
   * interface should probably be the same for all free functions so passing the
   * problem as a second parameter is not an option even though we need info
   * form it.*/
  int function_id; /*TODO: consider changing name*/
  int instance_id;
  size_t number_of_variables;
  double optimal_fvalue;
} bbob2009_logger_t; 

static const char *_file_header_str = "%% function evaluation | "
                                      "noise-free fitness - Fopt (%13.12e) | "
                                      "best noise-free fitness - Fopt | "
                                      "measured fitness | "
                                      "best measured fitness | "
                                      "x1 | "
                                      "x2...\n";

static void _bbob2009_logger_update_f_trigger(bbob2009_logger_t *data,
                                              double fvalue) {
  /* "jump" directly to the next closest (but larger) target to the
   * current fvalue from the initial target
   */

  if (fvalue - data->optimal_fvalue <= 0.) {
    data->f_trigger = -DBL_MAX;
  } else {
    if (data->idx_f_trigger == INT_MAX) { /* first time*/
      data->idx_f_trigger =
          (int)(ceil(log10(fvalue - data->optimal_fvalue)) * bbob2009_nbpts_fval);
    } else { /* We only call this function when we reach the current f_trigger*/
      data->idx_f_trigger--;
    }
    data->f_trigger = pow(10, data->idx_f_trigger * 1.0 / bbob2009_nbpts_fval);
    while (fvalue - data->optimal_fvalue <= data->f_trigger) {
      data->idx_f_trigger--;
      data->f_trigger = pow(10, data->idx_f_trigger * 1.0 / bbob2009_nbpts_fval);
    }
  }
}

static void _bbob2009_logger_update_t_trigger(bbob2009_logger_t *data,
                                              size_t number_of_variables) {
  while (data->number_of_evaluations >=
         floor(pow(10, (double)data->idx_t_trigger / (double)bbob2009_nbpts_nbevals)))
    data->idx_t_trigger++;

  while (data->number_of_evaluations >=
         number_of_variables * pow(10, (double)data->idx_tdim_trigger))
    data->idx_tdim_trigger++;

  data->t_trigger =
      (long)fmin(floor(pow(10, (double)data->idx_t_trigger / (double)bbob2009_nbpts_nbevals)),
           number_of_variables * pow(10, (double)data->idx_tdim_trigger));
}

/**
 * adds a formated line to a data file
 */
static void _bbob2009_logger_write_data(FILE *target_file,
                                        long number_of_evaluations,
                                        double fvalue, double best_fvalue,
                                        double best_value, const double *x,
                                        size_t number_of_variables) {
  /* for some reason, it's %.0f in the old code instead of the 10.9e
   * in the documentation
   */
  fprintf(target_file, "%ld %+10.9e %+10.9e %+10.9e %+10.9e",
          number_of_evaluations, fvalue - best_value, best_fvalue - best_value,
          fvalue, best_fvalue);
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
static void _bbob2009_logger_error_io(FILE *path, int errnum) {
  char *buf;
  const char *error_format = "Error opening file: %s\n ";
                             /*"bbob2009_logger_prepare() failed to open log "
                             "file '%s'.";*/
  size_t buffer_size = (size_t)(snprintf(NULL, 0, error_format, path));/*to silence warning*/
  buf = (char *)coco_allocate_memory(buffer_size);
  snprintf(buf, buffer_size, error_format, strerror(errnum), path);
  coco_error(buf);
  coco_free_memory(buf);
}

/**
 * Creates the data files or simply opens it
 */

/*
 calling sequence:
 _bbob2009_logger_open_dataFile(&(data->fdata_file), data->path, dataFile_path,
 ".dat");
 */

static void _bbob2009_logger_open_dataFile(FILE **target_file, const char *path,
                                           const char *dataFile_path,
                                           const char *file_extension) {
    char file_path[NUMBBO_PATH_MAX] = {0};
    char relative_filePath[NUMBBO_PATH_MAX] = {0};
    int errnum;
    strncpy(relative_filePath, dataFile_path,
            NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
    strncat(relative_filePath, file_extension,
            NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
    coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
    if (*target_file == NULL) {
        *target_file = fopen(file_path, "a+");
        errnum = errno;
        if (*target_file == NULL) {
            _bbob2009_logger_error_io(*target_file, errnum);
        }
    }
}

/*
static void _bbob2009_logger_open_dataFile(FILE **target_file, const char *path,
                                           const char *dataFile_path,
                                           const char *file_extension) {
  char file_path[NUMBBO_PATH_MAX] = {0};
  char relative_filePath[NUMBBO_PATH_MAX] = {0};
  int errnum;
  strncpy(relative_filePath, dataFile_path,
          NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
  strncat(relative_filePath, file_extension,
          NUMBBO_PATH_MAX - strlen(relative_filePath) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_filePath, NULL);
  if (*target_file == NULL) {
    *target_file = fopen(file_path, "a+");
    errnum = errno;
    if (*target_file == NULL) {
      _bbob2009_logger_error_io(*target_file, errnum);
    }
  }
}*/

/**
 * Creates the index file fileName_prefix+problem_id+file_extension in
 * folde_path
 */
static void _bbob2009_logger_openIndexFile(bbob2009_logger_t *data,
                                           const char *folder_path,
                                           const char *indexFile_prefix,
                                           const char *function_id,
                                           const char *dataFile_path) {
    /*to add the instance number TODO: this should be done outside to avoid redoing this for the .*dat files */
    char used_dataFile_path[NUMBBO_PATH_MAX] = {0};
    int errnum, newLine;/*newLine is at 1 if we need a new line in the info file*/
    char function_id_char[3];/*TODO: consider adding them to data*/
    char file_name[NUMBBO_PATH_MAX] = {0};
    char file_path[NUMBBO_PATH_MAX] = {0};
    FILE **target_file;
    FILE *tmp_file;
    strncpy(used_dataFile_path, dataFile_path, NUMBBO_PATH_MAX - strlen(used_dataFile_path) - 1);
    if (infoFile_firstInstance == 0) {
        infoFile_firstInstance = data->instance_id;
    }
    sprintf(function_id_char, "%d", data->function_id);
    sprintf(infoFile_firstInstance_char, "%d", infoFile_firstInstance);
    target_file = &(data->index_file);
    tmp_file = NULL; /*to check whether the file already exists. Don't want to use
           target_file*/
    strncpy(file_name, indexFile_prefix, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, "_f", NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, function_id_char, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, "_i", NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, infoFile_firstInstance_char, NUMBBO_PATH_MAX - strlen(file_name) - 1);
    strncat(file_name, ".info", NUMBBO_PATH_MAX - strlen(file_name) - 1);
    coco_join_path(file_path, sizeof(file_path), folder_path, file_name, NULL);
    if (*target_file == NULL) {
        tmp_file = fopen(file_path, "r");/*to check for existance*/
        if ((tmp_file ) &&
            (current_dim == data->number_of_variables) &&
            (current_funId == data->function_id)) {/*new instance of current funId and current dim*/
            newLine = 0;
            *target_file = fopen(file_path, "a+");
            if (*target_file == NULL) {
                errnum = errno;
                _bbob2009_logger_error_io(*target_file, errnum);
            }
            fclose(tmp_file);
        }
        else { /* either file doesn't exist (new funId) or new Dim*/
            /*check that the dim was not already present earlier in the file, if so, create a new info file*/
            if (current_dim != data->number_of_variables) {
                int i, j;
                for (i=0; i<bbob2009_number_of_dimensions && dimensions_in_current_infoFile[i]!=0 &&
                     dimensions_in_current_infoFile[i]!=data->number_of_variables;i++) {
                    ;/*checks whether dimension already present in the current infoFile*/
                }
                if (i<bbob2009_number_of_dimensions && dimensions_in_current_infoFile[i]==0) {
                    /*new dimension seen for the first time*/
                    dimensions_in_current_infoFile[i]=data->number_of_variables;
                    newLine = 1;
                }
                else{
                        if (i<bbob2009_number_of_dimensions) {/*dimension already present, need to create a new file*/
                            newLine = 0;
                            file_path[strlen(file_path)-strlen(infoFile_firstInstance_char) - 7] = 0;/*truncate the instance part*/
                            infoFile_firstInstance = data->instance_id;
                            sprintf(infoFile_firstInstance_char, "%d", infoFile_firstInstance);
                            strncat(file_path, "_i", NUMBBO_PATH_MAX - strlen(file_name) - 1);
                            strncat(file_path, infoFile_firstInstance_char, NUMBBO_PATH_MAX - strlen(file_name) - 1);
                            strncat(file_path, ".info", NUMBBO_PATH_MAX - strlen(file_name) - 1);
                        }
                        else{/*we have all dimensions*/
                            newLine = 1;
                        }
                        for (j=0; j<bbob2009_number_of_dimensions;j++){/*new info file, reinitilize list of dims*/
                            dimensions_in_current_infoFile[j]= 0;
                        }
                    dimensions_in_current_infoFile[i]=data->number_of_variables;
                }
            }
            *target_file = fopen(file_path, "a+");/*in any case, we append*/
            if (*target_file == NULL) {
                errnum = errno;
                _bbob2009_logger_error_io(*target_file, errnum);
            }
            if (tmp_file) { /*File already exists, new dim so just a new line. ALso, close the tmp_file*/
                if (newLine) {
                    fprintf(*target_file, "\n");
                }
                
                fclose(tmp_file);
            }
            
            fprintf(*target_file,
                    /* TODO: z-modifier is bound to fail as being incompatible to standard C */
                    "funcId = %d, DIM = %ld, Precision = %.3e, algId = '%s'\n",
                    (int)strtol(function_id, NULL, 10), (long)data->number_of_variables,
                    pow(10, -8), data->alg_name);
            fprintf(*target_file, "%%\n");
            strncat(used_dataFile_path, "_i", NUMBBO_PATH_MAX - strlen(used_dataFile_path) - 1);
            strncat(used_dataFile_path, infoFile_firstInstance_char,
                    NUMBBO_PATH_MAX - strlen(used_dataFile_path) - 1);
            fprintf(*target_file, "%s.dat",
                    used_dataFile_path); /*dataFile_path does not have the extension*/
            current_dim = data->number_of_variables;
            current_funId = data->function_id;
        }
    }
}


/**
 * Generates the different files and folder needed by the logger to store the
 * data if theses don't already exist
 */
static void _bbob2009_logger_initialize(bbob2009_logger_t *data,
                                        coco_problem_t *inner_problem) {
  /*
    Creates/opens the data and index files
  */
  char dataFile_path[NUMBBO_PATH_MAX] = {
      0}; /*relative path to the .dat file from where the .info file is*/
  char folder_path[NUMBBO_PATH_MAX] = {0};
  char tmpc_funId[3]; /*servs to extract the function id as a char *. There
                         should be a better way of doing this! */
  char tmpc_dim[3];   /*servs to extract the dimension as a char *. There should
                         be a better way of doing this! */
  char indexFile_prefix[10] = "bbobexp"; /* TODO (minor): make the prefix bbobexp a
                                            parameter that the user can modify */  
  assert(data != NULL);
  assert(inner_problem != NULL);
  assert(inner_problem->problem_id != NULL);

  sprintf(tmpc_funId, "%d", bbob2009_get_function_id(inner_problem));
  sprintf(tmpc_dim, "%lu", (unsigned long) inner_problem->number_of_variables);
  
  /* prepare paths and names*/
  strncpy(dataFile_path, "data_f", NUMBBO_PATH_MAX);
  strncat(dataFile_path, tmpc_funId,
          NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  coco_join_path(folder_path, sizeof(folder_path), data->path, dataFile_path,
                 NULL);
  coco_create_path(folder_path);
  strncat(dataFile_path, "/bbobexp_f",
          NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_funId,
          NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, "_DIM", NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, tmpc_dim, NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);

  /* index/info file*/
  _bbob2009_logger_openIndexFile(data, data->path, indexFile_prefix, tmpc_funId,
                                 dataFile_path);
  fprintf(data->index_file, ", %d", bbob2009_get_instance_id(inner_problem));
  /* data files*/
  /*TODO: definitely improvable but works for now*/
  strncat(dataFile_path, "_i", NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  strncat(dataFile_path, infoFile_firstInstance_char,
            NUMBBO_PATH_MAX - strlen(dataFile_path) - 1);
  _bbob2009_logger_open_dataFile(&(data->fdata_file), data->path, dataFile_path,
                                 ".dat");
  fprintf(data->fdata_file, _file_header_str, data->optimal_fvalue);

  _bbob2009_logger_open_dataFile(&(data->tdata_file), data->path, dataFile_path,
                                 ".tdat");
  fprintf(data->tdata_file, _file_header_str, data->optimal_fvalue);

  _bbob2009_logger_open_dataFile(&(data->rdata_file), data->path, dataFile_path,
                                 ".rdat");
  fprintf(data->rdata_file, _file_header_str, data->optimal_fvalue);
  /* TODO: manage duplicate filenames by either using numbers or raising an
   * error */
  data->is_initialized = 1;
}

/**
 * Layer added to the transformed-problem evaluate_function by the logger
 */
static void _bbob2009_logger_evaluate_function(coco_problem_t *self, const double *x,
                                               double *y) {
  bbob2009_logger_t *data = coco_get_transform_data(self);
  coco_problem_t * inner_problem = coco_get_transform_inner_problem(self);
  
  if (!data->is_initialized) {
    _bbob2009_logger_initialize(data, inner_problem);
  }
  if (bbob2009_logger_verbosity > 2 && data->number_of_evaluations == 0) {
    if (inner_problem->index >= 0) {
      printf("%4ld: ", inner_problem->index);
    }
    printf("on problem %s ... ", coco_get_problem_id(inner_problem));
  }
  coco_evaluate_function(inner_problem, x, y);
  data->last_fvalue = y[0];
  data->written_last_eval = 0;
  if (data->number_of_evaluations == 0 || y[0] < data->best_fvalue) {
    size_t i;
    data->best_fvalue = y[0];
    for (i = 0; i < self->number_of_variables; i++)
      data->best_solution[i] = x[i];
  }
  data->number_of_evaluations++;

  /* Add sanity check for optimal f value */
  /*assert(y[0] >= data->optimal_fvalue);*/
  if (!raisedOptValWarning && y[0] < data->optimal_fvalue) {
      coco_warning("Observed fitness is smaller than supposed optimal fitness.");
      raisedOptValWarning = 1;
  }

  /* Add a line in the .dat file for each logging target reached. */
  if (y[0] - data->optimal_fvalue <= data->f_trigger) {

    _bbob2009_logger_write_data(data->fdata_file, data->number_of_evaluations,
                                y[0], data->best_fvalue, data->optimal_fvalue,
                                x, self->number_of_variables);
    _bbob2009_logger_update_f_trigger(data, y[0]);
  }

  /* Add a line in the .tdat file each time an fevals trigger is reached. */
  if (data->number_of_evaluations >= data->t_trigger) {
    data->written_last_eval = 1;
    _bbob2009_logger_write_data(data->tdata_file, data->number_of_evaluations,
                                y[0], data->best_fvalue, data->optimal_fvalue,
                                x, self->number_of_variables);
    _bbob2009_logger_update_t_trigger(data, self->number_of_variables);
  }

  /* Flush output so that impatient users can see progress. */
  fflush(data->fdata_file);
}

/**
 * Also serves as a finalize run method so. Must be called at the end
 * of Each run to correctly fill the index file
 *
 * TODO: make sure it is called at the end of each run or move the
 * writing into files to another function
 */
static void _bbob2009_logger_free_data(void *stuff) {
  /*TODO: do all the "non simply freeing" stuff in another function
   * that can have problem as input
   */
  bbob2009_logger_t *data = stuff;

  if (bbob2009_logger_verbosity > 2 && data && data->number_of_evaluations > 0) {
    printf("best f=%e after %ld fevals (done observing)\n",
           data->best_fvalue, (long)data->number_of_evaluations);
    }
  if (data->alg_name != NULL) {
    coco_free_memory((void*)data->alg_name);
    data->alg_name = NULL;
  }
    
  if (data->path != NULL) {
    coco_free_memory(data->path);
    data->path = NULL;
  }
  if (data->index_file != NULL) {
    fprintf(data->index_file, ":%ld|%.1e", (long)data->number_of_evaluations,
            data->best_fvalue - data->optimal_fvalue);
    fclose(data->index_file);
    data->index_file = NULL;
  }
  if (data->fdata_file != NULL) {
    fclose(data->fdata_file);
    data->fdata_file = NULL;
  }
  if (data->tdata_file != NULL) {
    /* TODO: make sure it handles restarts well. i.e., it writes
     * at the end of a single run, not all the runs on a given
     * instance. Maybe start with forcing it to generate a new
     * "instance" of problem for each restart in the beginning
     */
    if (!data->written_last_eval) {
      _bbob2009_logger_write_data(data->tdata_file, data->number_of_evaluations,
                                  data->last_fvalue, data->best_fvalue,
                                  data->optimal_fvalue, data->best_solution,
                                  data->number_of_variables);
    }
    fclose(data->tdata_file);
    data->tdata_file = NULL;
  }

  if (data->rdata_file != NULL) {
    fclose(data->rdata_file);
    data->rdata_file = NULL;
  }

  if (data->best_solution != NULL) {
    coco_free_memory(data->best_solution);
    data->best_solution = NULL;
  }
  bbob2009_logger_is_open = 0;
}

static coco_problem_t *bbob2009_logger(coco_problem_t *inner_problem,
                                const char *alg_name) {
  bbob2009_logger_t *data;
  coco_problem_t *self;
  data = coco_allocate_memory(sizeof(*data));
  data->alg_name = coco_strdup(alg_name);
  if (bbob2009_logger_is_open)
    coco_error("The current bbob2009_logger (observer) must be closed before a new one is opened");
  /* This is the name of the folder which happens to be the algName */
  data->path = coco_strdup(alg_name);
  data->index_file = NULL;
  data->fdata_file = NULL;
  data->tdata_file = NULL;
  data->rdata_file = NULL;
  data->number_of_variables = inner_problem->number_of_variables;
  if (inner_problem->best_value == NULL) {
      /*coco_error("Optimal f value must be defined for each problem in order for the logger to work propertly");*/
      /*Setting the value to 0 results in the assertion y>=optimal_fvalue being susceptible to failure*/
      coco_warning("undefined optimal f value. Set to 0");
      data->optimal_fvalue = 0;
  }
  else
  {
      data->optimal_fvalue = *(inner_problem->best_value);
  }
  raisedOptValWarning = 0;

  data->idx_f_trigger = INT_MAX;
  data->idx_t_trigger = 0;
  data->idx_tdim_trigger = 0;
  data->f_trigger = DBL_MAX;
  data->t_trigger = 0;
  data->number_of_evaluations = 0;
  data->best_solution =
      coco_allocate_vector(inner_problem->number_of_variables);
  /* TODO: the following inits are just to be in the safe side and
   * should eventually be removed. Some fileds of the bbob2009_logger struct
   * might be useless
   */
  data->function_id = bbob2009_get_function_id(inner_problem);
  data->instance_id = bbob2009_get_instance_id(inner_problem);
  data->written_last_eval = 1;
  data->last_fvalue = DBL_MAX;
  data->is_initialized = 0;

  self = coco_allocate_transformed_problem(inner_problem, data,
                                           _bbob2009_logger_free_data);
  self->evaluate_function = _bbob2009_logger_evaluate_function;
  bbob2009_logger_is_open = 1;
  return self;
}
#line 4 "src/bbob2009_observer.c"

/* TODO:
 *
 * o here needs to go the docstring for this function
 * 
 * o parse options that look like "folder:foo; verbose:bar" (use coco_strfind and/or sscanf and/or ??)
 *   Ideally, valid options should be
 *      "my_folder_name verbose : 3",
 *      "folder: my_folder_name",
 *      "verbose : 4 folder:another_folder"
 *      "folder:yet_another verbose: -2 "
 *   This could be done with a coco_get_option(options, name, format, pointer)
 *   function with code snippets like (approximately)

        logger->folder = coco_allocate_memory(sizeof(char) * (strlen(options) + 1));
        if (!coco_options_read(options, "folder", " %s", logger->folder))
            sscanf(options, " %s", logger->folder);
        coco_options_read(options, "verbose", " %i", &(logger->verbose));

    with 
        
        # caveat: "folder: name; " might fail, use spaces for separations
        int coco_options_read(const char *options, const char *name, const char *format, void *pointer) {
            long i1 = coco_strfind(options, name);
            long i2;
            
            if (i1 < 0)
                return 0;
            i2 = i1 + coco_strfind(&options[i1], ":") + 1;
            if (i2 <= i1)
                return 0;
            return sscanf(&options[i2], format, pointer);
        }
 * 
 */   
static coco_problem_t *bbob2009_observer(coco_problem_t *problem,
                                  const char *options) {
  if (problem == NULL)
    return problem;
  /* TODO: " */
  coco_create_path(options); 
  problem = bbob2009_logger(problem, options);
  return problem;
}
#line 13 "src/coco_benchmark.c"

#line 1 "src/mo_suite_first_attempt.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#line 7 "src/mo_suite_first_attempt.c"
#line 8 "src/mo_suite_first_attempt.c"
#line 9 "src/mo_suite_first_attempt.c"
#line 10 "src/mo_suite_first_attempt.c"
/* #include "biobjective_problem.c"
*/

int coco_problem_id_is_fine(const char *id, ...) {
  va_list args;
  const int reject = 0;
  const int OK = 1;
  const char *cp;
  char *s;
  int result = OK;
  
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
 * Formatted printing of a problem ID, mimicking
 * sprintf(coco_get_problem_id(problem), id, ...) while taking care
 * of memory (de-)allocations. 
 *
 */
void coco_problem_setf_id(coco_problem_t *problem, const char *id, ...) {
  va_list args;

  va_start(args, id);
  coco_free_memory(problem->problem_id);
  problem->problem_id = coco_vstrdupf(id, args);
  va_end(args);
  if (!coco_problem_id_is_fine(problem->problem_id)) {
    coco_error("Problem id should only contain standard chars, not like '%s'",
               coco_get_problem_id(problem));
  }
}
/**
 * Formatted printing of a problem name, mimicking
 * sprintf(coco_get_problem_name(problem), name, ...) while taking care
 * of memory (de-)allocation, tentative, needs at the minimum some (more) testing. 
 *
 */
void coco_problem_setf_name(coco_problem_t *problem, const char *name, ...) {
  va_list args;
  
  va_start(args, name);
  coco_free_memory(problem->problem_name);
  problem->problem_name = coco_vstrdupf(name, args);
  va_end(args);
}

/**
 * mo_suit...(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from...
 * If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *mo_suite_first_attempt(const long problem_index) {
  coco_problem_t *problem, *problem2;
  long dimension, instance, instance2;
  int f, f2;
  
  if (problem_index < 0) 
    return NULL;

  if (problem_index < 24) { 
  
    /* here we compute the mapping from problem index to the following five values */
    
    dimension = 10;
    f = 1;
    f2 = 1 + (int)(problem_index % 24);
    instance = 0;
    instance2 = 1;
    
    problem = bbob2009_problem(f, dimension, instance);

    problem2 = bbob2009_problem(f2, dimension, instance2);
    problem = coco_stacked_problem_allocate(problem, problem2);
    /* repeat the last two lines to add more objectives */
#if 0
    coco_problem_setf_id(problem, "ID-F%03d-F%03d-d03%ld-%06ld", f, f2, dimension, problem_index);
    coco_problem_setf_name(problem, "%s + %s",
                          coco_get_problem_name(problem), coco_get_problem_name(problem2));
#endif
    problem->index = problem_index;
    
    return problem; 
  } /* else if ... */
  return NULL;
}

        
        
#line 15 "src/coco_benchmark.c"

#line 1 "src/biobjective_suite_300.c"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#line 7 "src/biobjective_suite_300.c"
#line 8 "src/biobjective_suite_300.c"
#line 9 "src/biobjective_suite_300.c"
#line 10 "src/biobjective_suite_300.c"

#define BIOBJECTIVE_NUMBER_OF_COMBINATIONS 300
#define BIOBJECTIVE_NUMBER_OF_INSTANCES 5
#define BIOBJECTIVE_NUMBER_OF_DIMENSIONS 5


/**
 * The biobjective suite of 300 function combinations generated
 * using the bbob2009_suite. For each function combination, there are
 * 11 preset instances in each of the five dimensions {2, 3, 5, 10, 20}.
 * Ref: Benchmarking Numerical Multiobjective Optimizers Revisited @ GECCO'15
 */

// const size_t DIMENSIONS[6] = {2, 3, 5, 10, 20, 40};
static const size_t instance_list[5][2] = { {2, 4},
                                            {3, 5},
                                            {7, 8},
                                            {9, 10},
                                            {11, 12} };
//--> we must map this number to the two corresponding BBOB functions
static int biobjective_list[300][2]; // 300 is the total number of 2-obj combinations (< 24*24)
static int defined = 0;

/**
 * How: instance varies faster than combination which is still faster than dimension
 * 
 *  problem_index | instance | combination | dimension
 * ---------------+----------+-------------+-----------
 *              0 |        1 |           1 |         2
 *              1 |        2 |           1 |         2
 *              2 |        3 |           1 |         2
 *              3 |        4 |           1 |         2
 *              4 |        5 |           1 |         2
 *              5 |        1 |           2 |         2
 *              6 |        2 |           2 |         2
 *             ...        ...           ...        ...
 *           1499 |        5 |         300 |         2
 *           1500 |        1 |           1 |         3
 *           1501 |        2 |           1 |         3
 *             ...        ...           ...        ...
 *           7497 |        3 |         300 |        20
 *           7498 |        4 |         300 |        20
 *           7499 |        5 |         300 |        20
 */
static long biobjective_encode_problem_index(int combination_idx, long instance_idx, int dimension_idx) {
    long problem_index;
    problem_index = instance_idx + 
                    combination_idx * BIOBJECTIVE_NUMBER_OF_INSTANCES + 
                    dimension_idx * (BIOBJECTIVE_NUMBER_OF_INSTANCES * BIOBJECTIVE_NUMBER_OF_COMBINATIONS);
    return problem_index;
}

static void biobjective_decode_problem_index(const long problem_index, int *combination_idx,
                                             long *instance_idx, long *dimension_idx) {
    *dimension_idx = problem_index / (BIOBJECTIVE_NUMBER_OF_INSTANCES * BIOBJECTIVE_NUMBER_OF_COMBINATIONS);
    long rest = problem_index % (BIOBJECTIVE_NUMBER_OF_INSTANCES * BIOBJECTIVE_NUMBER_OF_COMBINATIONS);
    *combination_idx = (int)(rest / BIOBJECTIVE_NUMBER_OF_INSTANCES);
    *instance_idx = rest % BIOBJECTIVE_NUMBER_OF_INSTANCES;
}

static coco_problem_t *biobjective_suite_300(const long problem_index) {  
    if (problem_index < 0) 
        return NULL;
    
    if (defined == 0) {
        int k = 0;
        size_t i, j;
        for (i = 1; i <= 24; ++i) {
            for (j = i; j <= 24; ++j) {
                biobjective_list[k][0] = i;
                biobjective_list[k][1] = j;
                k++;
            }
        }
        defined = 1;
    }
    
    int combination_idx;
    long instance_idx, dimension_idx;
    int problem1_index, problem2_index;
    coco_problem_t *problem1, *problem2, *problem;
    biobjective_decode_problem_index(problem_index, &combination_idx, &instance_idx, &dimension_idx);
    
    problem1 = bbob2009_problem(biobjective_list[combination_idx][0],
                                BBOB2009_DIMS[dimension_idx],
                                instance_list[instance_idx][0]);
    problem2 = bbob2009_problem(biobjective_list[combination_idx][1],
                                BBOB2009_DIMS[dimension_idx],
                                instance_list[instance_idx][1]);
    problem = coco_stacked_problem_allocate(problem1, problem2);
    problem->index = problem_index;
    
    return problem; 
}


/* Undefine constants */
#undef BIOBJECTIVE_NUMBER_OF_COMBINATIONS
#undef BIOBJECTIVE_NUMBER_OF_INSTANCES
#undef BIOBJECTIVE_NUMBER_OF_DIMENSIONS
#line 17 "src/coco_benchmark.c"
#line 1 "src/biobjective_observer.c"
#line 2 "src/biobjective_observer.c"
#line 3 "src/biobjective_observer.c"
#line 1 "src/log_nondominating.c"
#include <stdio.h>
#include <assert.h>

#line 5 "src/log_nondominating.c"

#line 7 "src/log_nondominating.c"
#line 8 "src/log_nondominating.c"
#line 9 "src/log_nondominating.c"

/* For making my multiobjective recorder work */
#line 1 "src/mo_recorder.h"
#ifndef MO_RECORDER_H
#define	MO_RECORDER_H


#ifdef	__cplusplus
extern "C" {
#endif


struct mococo_solution_entry {
    int status;    // 0: inactive | 1: active
    size_t birth;  // timestamp to know which are newly created
    double *var;
    double *obj;
};

struct mococo_solutions_archive {
    size_t maxsize;
    size_t maxupdatesize;
    size_t size;
    size_t updatesize;
    size_t numvar;
    size_t numobj;
    struct mococo_solution_entry *entry;
    struct mococo_solution_entry **active;
    struct mococo_solution_entry **update;
};

void mococo_allocate_archive(struct mococo_solutions_archive *archive, size_t maxsize, size_t sizeVar, size_t sizeObj, size_t maxUpdate);
void mococo_free_archive(struct mococo_solutions_archive *archive);
void mococo_reset_archive(struct mococo_solutions_archive *archive);
void mococo_push_to_archive(double **pop, double **obj, struct mococo_solutions_archive *archive, size_t nPop, size_t timestamp);
void mococo_mark_updates(struct mococo_solutions_archive *archive, size_t timestamp);
// void mococo_recorder(const char *mode);


#ifdef	__cplusplus
}
#endif

#endif	// MO_RECORDER_H#line 12 "src/log_nondominating.c"
#line 1 "src/mo_recorder.c"
#include <sys/types.h> // for creating folder
#include <sys/stat.h>  // for creating folder
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#line 8 "src/mo_recorder.c"


void mococo_allocate_archive(struct mococo_solutions_archive *archive, size_t maxsize, size_t sizeVar, size_t sizeObj, size_t maxUpdate) {
    archive->maxsize = maxsize;
    archive->maxupdatesize = maxUpdate;
    archive->size = 0;
    archive->updatesize = 0;
    archive->numvar = sizeVar;
    archive->numobj = sizeObj;
    archive->entry  = (struct mococo_solution_entry *) malloc(maxsize * sizeof(struct mococo_solution_entry));
    archive->active = (struct mococo_solution_entry **) malloc(maxsize * sizeof(struct mococo_solution_entry*));
    archive->update = (struct mococo_solution_entry **) malloc(maxUpdate * sizeof(struct mococo_solution_entry*));
    if (archive->entry == NULL || archive->active == NULL || archive->update == NULL) {
        fprintf(stderr, "ERROR in allocating memory for the archive.\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate memory for each solution entry
    for (size_t i=0; i < maxsize; i++) {
        archive->entry[i].status = 0;  // 0: inactive | 1: active
        archive->entry[i].birth = 0;
        archive->entry[i].var = (double*) malloc(sizeVar * sizeof(double));
        archive->entry[i].obj = (double*) malloc(sizeObj * sizeof(double));
        if (archive->entry[i].var == NULL || archive->entry[i].obj == NULL) {
            fprintf(stderr, "ERROR in allocating memory for some entry of the archive.\n");
            exit(EXIT_FAILURE);
        }
    }
}

void mococo_reset_archive(struct mococo_solutions_archive *archive) {
    archive->size = 0;
    archive->updatesize = 0;
    for (size_t i=0; i < archive->maxsize; i++) {
        archive->entry[i].status = 0;
        archive->entry[i].birth = 0;
    }
}

void mococo_free_archive(struct mococo_solutions_archive *archive) {
    for (size_t i=0; i < archive->maxsize; i++) {
        free(archive->entry[i].var);
        free(archive->entry[i].obj);
    }
    free(archive->update);
    free(archive->active);
    free(archive->entry);
}


void mococo_push_to_archive(double **pop, double **obj, struct mococo_solutions_archive *archive, size_t nPop, size_t timestamp) {
    struct mococo_solution_entry *entry;
    size_t s = archive->size;
    size_t tnext = 0;
    for (size_t i=0; i < nPop; i++) {
        // Find a non-active slot for the new i-th solution
        for (size_t t = tnext; t < archive->maxsize; t++) {
            if (archive->entry[t].status == 0) {
                archive->active[s] = &(archive->entry[t]);
                tnext = t + 1;
                break;
            }
        }
        // Keep the i-th solution in the slot found
        entry = archive->active[s];
        entry->status = 1;
        entry->birth = timestamp;
        for (size_t j=0; j < archive->numvar; j++)   // all decision variables of a solution
            entry->var[j] = pop[i][j];
        for (size_t k=0; k < archive->numobj; k++)   // all objective values of a solution
            entry->obj[k] = obj[i][k];
        s++;
    }
    archive->size = s;
}


void mococo_mark_updates(struct mococo_solutions_archive *archive, size_t timestamp) {
    size_t u = 0;
    for (size_t i=0; i < archive->size; i++) {
        if (archive->active[i]->birth == timestamp) {
            archive->update[u] = archive->active[i];
            u++;
        }
    }
    archive->updatesize = u;
}

#line 13 "src/log_nondominating.c"
#line 1 "src/mo_paretofiltering.c"
// paretofront returns the logical Pareto membership of a set of points
// synopsis:  frontFlag = paretofront(objMat)
// Created by Yi Cao: y.cao@cranfield.ac.uk
// for compiling type:
//   mex paretofront.c

#include <stdio.h>
#include <stdlib.h>  // memory, e.g. malloc
#include <stdbool.h> // to use the bool datatype, required C99
#include <math.h>
#line 12 "src/mo_paretofiltering.c"
#include <stdbool.h> // to use the bool datatype, required C99

#ifdef __cplusplus
extern "C" {
#endif


void mococo_pareto_front(bool *frontFlag, double *obj, unsigned nrow, unsigned ncol);


void mococo_pareto_filtering(struct mococo_solutions_archive *archive) {
    // Create the objective vectors and frontFlag of appropriate format for paretofront()
    size_t len = archive->size;
    size_t nObjs = archive->numobj;
    bool *frontFlag = (bool*) malloc(len * sizeof(bool));
    double *obj = (double*) malloc(len * nObjs * sizeof(double));
    for (size_t i=0; i < len; i++) {
        for (size_t k=0; k < nObjs; k++) {
            obj[i + k*len] = archive->active[i]->obj[k];
        }
        frontFlag[i] = false;
    }
    
    // Call the non-dominated sorting engine
    mococo_pareto_front(frontFlag, obj, len, nObjs);
    
    // Mark non-dominated solutions and filter out dominated ones
    size_t s = 0;
    for (size_t i=0; i < len; i++) {
        if (frontFlag[i] == true) {
            archive->active[i]->status = 1;
            if (i != s)
                archive->active[s] = archive->active[i];
            s++;
        } else {
            archive->active[i]->status = 0; // filter out dominated solutions
        }
    }
    archive->size = s;
    
    free(obj);
    free(frontFlag);
}


void mococo_pareto_front(bool *frontFlag, double *obj, unsigned nrow, unsigned ncol) {
    unsigned t, s, i, j, j1, j2;
    bool *checklist, colDominatedFlag;
    
    checklist = (bool*)malloc(nrow*sizeof(bool));
    
    for(t=0; t<nrow; t++)
        checklist[t] = true;
    for(s=0; s<nrow; s++) {
        t = s;
        if (!checklist[t])
            continue;
        checklist[t] = false;
        colDominatedFlag = true;
        for(i=t+1; i<nrow; i++) {
            if (!checklist[i])
                continue;
            checklist[i] = false;
            for (j=0,j1=i,j2=t; j<ncol; j++,j1+=nrow,j2+=nrow) {
                if (obj[j1] < obj[j2]) {
                    checklist[i] = true;
                    break;
                }
            }
            if (!checklist[i])
                continue;
            colDominatedFlag = false;
            for (j=0,j1=i,j2=t; j<ncol; j++,j1+=nrow,j2+=nrow) {
                if (obj[j1] > obj[j2]) {
                    colDominatedFlag = true;
                    break;
                }
            }
            if (!colDominatedFlag) { //swap active index continue checking
                frontFlag[t] = false;
                checklist[i] = false;
                colDominatedFlag = true;
                t = i;
            }
        }
        frontFlag[t] = colDominatedFlag;
        if (t>s) {
            for (i=s+1; i<t; i++) {
                if (!checklist[i])
                    continue;
                checklist[i] = false;
                for (j=0,j1=i,j2=t; j<ncol; j++,j1+=nrow,j2+=nrow) {
                    if (obj[j1] < obj[j2]) {
                        checklist[i] = true;
                        break;
                    }
                }
            }
        }
    }
    free(checklist); 
}

#ifdef __cplusplus
}
#endif
#line 14 "src/log_nondominating.c"


typedef struct {
    char *path;
    FILE *logfile;
    size_t max_size_of_archive;
    long number_of_evaluations;
} _log_nondominating_t;

static struct mococo_solutions_archive *mo_archive;
static struct mococo_solution_entry *entry;


static void lnd_evaluate_function(coco_problem_t *self, const double *x, double *y) {
  _log_nondominating_t *data;
  data = coco_get_transform_data(self);

  coco_evaluate_function(coco_get_transform_inner_problem(self), x, y);
  data->number_of_evaluations++;

  /* Open logfile if it is not alread open */
  if (data->logfile == NULL) {
    data->logfile = fopen(data->path, "w");
    if (data->logfile == NULL) {
      char *buf;
      const char *error_format =
          "lnd_evaluate_function() failed to open log file '%s'.";
      size_t buffer_size = snprintf(NULL, 0, error_format, data->path);
      buf = (char *)coco_allocate_memory(buffer_size);
      snprintf(buf, buffer_size, error_format, data->path);
      coco_error(buf);
      coco_free_memory(buf); /* Never reached */
    }
    fprintf(data->logfile, "# %zu variables  |  %zu objectives  |  func eval number\n",
            coco_get_number_of_variables(coco_get_transform_inner_problem(self)),
            coco_get_number_of_objectives(coco_get_transform_inner_problem(self)));
    
    /*********************************************************************/
    /* TODO: Temporary put it here, to check later */
    /* Allocate memory for the archive */
    mo_archive = (struct mococo_solutions_archive *) malloc(1 * sizeof(struct mococo_solutions_archive));
    mococo_allocate_archive(mo_archive, data->max_size_of_archive,
                          coco_get_number_of_variables(coco_get_transform_inner_problem(self)),
                          coco_get_number_of_objectives(coco_get_transform_inner_problem(self)), 1);
    /*********************************************************************/
  }
  
  /********************************************************************************/
  /* Finish evaluations of 1 single solution of the pop, with nObj objectives,
   * now update the archive with this newly evaluated solution and check its nondomination. */
  mococo_push_to_archive(&x, &y, mo_archive, 1, data->number_of_evaluations);
  mococo_pareto_filtering(mo_archive);  /***** TODO: IMPROVE THIS ROUTINE *****/
  mococo_mark_updates(mo_archive, data->number_of_evaluations);
  
  /* Write out a line for this newly evaluated solution if it is nondominated */
  // write main info to the log file for pfront
  for (size_t i=0; i < mo_archive->updatesize; i++) {
      entry = mo_archive->update[i];
      for (size_t j=0; j < coco_get_number_of_variables(coco_get_transform_inner_problem(self)); j++) // all decision variables of a solution
          fprintf(data->logfile, "%13.10e\t", entry->var[j]);
      for (size_t k=0; k < coco_get_number_of_objectives(coco_get_transform_inner_problem(self)); k++) // all objective values of a solution
          fprintf(data->logfile, "%13.10e\t", entry->obj[k]);
      fprintf(data->logfile, "%zu", entry->birth);  // its timestamp (FEval)
      fprintf(data->logfile, "\n");  // go to the next line for another solution
  }
  /********************************************************************************/
  
  /* Flush output so that impatient users can see progress. */
  fflush(data->logfile);
}

static void _lnd_free_data(void *stuff) {
  _log_nondominating_t *data;
  assert(stuff != NULL);
  data = stuff;

  if (data->path != NULL) {
    coco_free_memory(data->path);
    data->path = NULL;
  }
  // if (data->target_values != NULL) {
  //   coco_free_memory(data->target_values);
  //   data->target_values = NULL;
  // }
  if (data->logfile != NULL) {
    fclose(data->logfile);
    data->logfile = NULL;
    
    /***************************************************************/
    /* TODO: Temporary put it here, to check later */
    mococo_free_archive(mo_archive); // free the archive
    free(mo_archive);
    /***************************************************************/
  }
}

static coco_problem_t *log_nondominating(coco_problem_t *inner_problem,
                                  const size_t max_size_of_archive,
                                  const char *path) {
  _log_nondominating_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->number_of_evaluations = 0;
  data->path = coco_strdup(path);
  data->logfile = NULL; /* Open lazily in lht_evaluate_function(). */
  data->max_size_of_archive = max_size_of_archive;
  self = coco_allocate_transformed_problem(inner_problem, data, _lnd_free_data);
  self->evaluate_function = lnd_evaluate_function;
  return self;
}
#line 4 "src/biobjective_observer.c"

/**
 * Multiobjective observer for logging all nondominated solutions found when
 * a new solution is generated and evaluated.
 */
static coco_problem_t *mo_toy_observer(coco_problem_t *problem, const char *options) {
    /* Calculate target levels for first hitting times */
    static const size_t max_size_of_archive = 100000;
    char base_path[NUMBBO_PATH_MAX] = {0};
    char filename[NUMBBO_PATH_MAX] = {0};
    coco_join_path(base_path, sizeof(base_path), options, "log_nondominated_solutions",
                   coco_get_problem_id(problem), NULL);
    if (coco_path_exists(base_path)) {
        coco_error("Result directory exists.");
        return NULL; /* never reached */
    }
    coco_create_path(base_path);
    coco_join_path(filename, sizeof(filename), 
                   base_path, "nondominated_at_birth.txt", NULL);
    problem = log_nondominating(problem, max_size_of_archive, filename);
    /* To control which information to be logged at each func eval, modify
     * the function 'lht_evaluate_function' in the file 'log_hitting_times.c' */
    return problem;
}

#line 18 "src/coco_benchmark.c"

/**
 * A new benchmark suite must providing a function that returns a
 * coco_problem or NULL when given an problem_index as input.
 *
 * The file containing this (static) function must be included above
 * and the function call must be added to coco_get_problem below.
 * 
 * If the benchmark does not have continuous problem indices starting with,
 * zero, additional functionality must also be added to coco_next_problem_index
 * (it should done in any case for efficiency).
 *
 * To construct a benchmark suite, useful tools are coco_transformed...
 * coco_stacked..., bbob2009_problem() and the various existing base
 * functions and transformations like shift_variables...
 */

/** return next problem_index or -1
 */
long coco_next_problem_index(const char *problem_suite,
                            long problem_index,
                            const char *select_options) {
  coco_problem_t *problem; /* to check validity */
  long last_index = -1;
  
  /* code specific to known benchmark suites */
  /* for efficiency reasons, each test suit should define
   * at least its last_index here */
  if (0 == strcmp(problem_suite, "bbob2009")) {
    /* without selection_options: last_index = 2159; */
    return bbob2009_next_problem_index(problem_index, select_options);
  }

  /** generic implementation:
   *   first index == 0,
   *   ++index until index > max_index or problem(index) == NULL
   **/
  
  if (problem_index < 0)
    problem_index = -1;
    
  ++problem_index;
  if (last_index >= 0) {
    if (problem_index <= last_index)
      return problem_index;
    else
      return -1;
  }
  
  /* last resort: last_index is not known */
  problem = coco_get_problem(problem_suite, problem_index);
  if (problem == NULL) {
    return -1;
  }
  coco_free_problem(problem);
  return problem_index;
}

coco_problem_t *coco_get_problem(const char *problem_suite,
                                 const long problem_index) {
  if (0 == strcmp(problem_suite, "toy_suit")) {
    return toy_suit(problem_index);
  } else if (0 == strcmp(problem_suite, "bbob2009")) {
    return bbob2009_suite(problem_index);
  } else if (0 == strcmp(problem_suite, "mo_suite_first_attempt")) {
    return mo_suite_first_attempt(problem_index);
  } else if (0 == strcmp(problem_suite, "biobjective_combinations")) {
    return biobjective_suite_300(problem_index);
  } else {
    coco_warning("Unknown problem suite.");
    return NULL;
  }
}

coco_problem_t *coco_observe_problem(const char *observer,
                                     coco_problem_t *problem,
                                     const char *options) {
  if (problem == NULL) {
    coco_warning("Trying to observe a NULL problem has no effect.");
    return problem;
  }
  if (0 == strcmp(observer, "toy_observer")) {
    return toy_observer(problem, options);
  } else if (0 == strcmp(observer, "bbob2009_observer")) {
    return bbob2009_observer(problem, options);
  } else if (0 == strcmp(observer, "mo_toy_observer")) {
    return mo_toy_observer(problem, options);
  }
  
  /* here each observer must have another entry */
  
  if (0 == strcmp(observer, "no_observer")) {
    return problem;
  } else if (strlen(observer) == 0) {
    coco_warning("Empty observer '' has no effect. To prevent this warning use 'no_observer' instead");
    return problem;
  } else {
    /* not so clear whether an error is better, depends on the usecase */
    coco_warning(observer);
    coco_warning("is an unkown observer which has no effect (the reason might just be a typo)");
    return problem;
  }
  coco_error("Unknown observer.");
  return NULL; /* Never reached */
}

#if 1
void coco_benchmark(const char *problem_suite, const char *observer,
                    const char *options, coco_optimizer_t optimizer) {
  int problem_index;
  coco_problem_t *problem;
  for (problem_index = 0;; ++problem_index) {
    problem = coco_get_problem(problem_suite, problem_index);
    if (NULL == problem)
      break;
    problem = coco_observe_problem(observer, problem, options); /* should remain invisible to the user*/
    optimizer(problem);
    /* Free problem after optimization. */
    coco_free_problem(problem);
  }
}

#else
/** "improved" interface for coco_benchmark: is it worth-while to have suite-options on the C-level? 
 */
void coco_benchmark(const char *problem_suite, const char *problem_suite_options,
                     const char *observer, const char *observer_options,
                     coco_optimizer_t optimizer) {
  int problem_index;
  int is_instance;
  coco_problem_t *problem;
  char buf[222]; /* TODO: this is ugly, how to improve? */
  for (problem_index = -1; ; ) {
    problem_index = coco_next_problem_index(problem_suite, problem_suite_options, problem_index); 
    if (problem_index < 0)
      break;
    problem = coco_get_problem(problem_suite, problem_index);
    if (problem == NULL)
      snprintf(buf, 221, "problem index %d not found in problem suit %s (this is probably a bug)",
               problem_index, problem_suite); 
      coco_warning(buf);
      break;
    problem = coco_observe_problem(observer, problem, observer_options); /* should remain invisible to the user*/
    optimizer(problem);
    coco_free_problem(problem);
  }
}
#endif
#line 1 "src/coco_random.c"
#include <math.h>

#line 4 "src/coco_random.c"

#define NUMBBO_NORMAL_POLAR /* Use polar transformation method */

#define SHORT_LAG 273
#define LONG_LAG 607

struct coco_random_state {
  double x[LONG_LAG];
  size_t index;
};

/**
 * coco_random_generate(state):
 *
 * This is a lagged Fibonacci generator that is nice because it is
 * reasonably small and directly generates double values. The chosen
 * lags (607 and 273) lead to a generator with a period in excess of
 * 2^607-1.
 */
static void coco_random_generate(coco_random_state_t *state) {
  size_t i;
  for (i = 0; i < SHORT_LAG; ++i) {
    double t = state->x[i] + state->x[i + (LONG_LAG - SHORT_LAG)];
    if (t >= 1.0)
      t -= 1.0;
    state->x[i] = t;
  }
  for (i = SHORT_LAG; i < LONG_LAG; ++i) {
    double t = state->x[i] + state->x[i - SHORT_LAG];
    if (t >= 1.0)
      t -= 1.0;
    state->x[i] = t;
  }
  state->index = 0;
}

coco_random_state_t *coco_new_random(uint32_t seed) {
  coco_random_state_t *state =
      (coco_random_state_t *)coco_allocate_memory(sizeof(coco_random_state_t));
  size_t i;
  /* Expand seed to fill initial state array. */
  for (i = 0; i < LONG_LAG; ++i) {
    state->x[i] = ((double)seed) / (double)((1ULL << 32) - 1);
    /* Advance seed based on simple RNG from TAOCP */
    seed = 1812433253UL * (seed ^ (seed >> 30)) + (i + 1);
  }
  state->index = 0;
  return state;
}

void coco_free_random(coco_random_state_t *state) { coco_free_memory(state); }

double coco_uniform_random(coco_random_state_t *state) {
  /* If we have consumed all random numbers in our archive, it is
   * time to run the actual generator for one iteration to refill
   * the state with 'LONG_LAG' new values.
   */
  if (state->index >= LONG_LAG)
    coco_random_generate(state);
  return state->x[state->index++];
}

double coco_normal_random(coco_random_state_t *state) {
  double normal;
#ifdef NUMBBO_NORMAL_POLAR
  const double u1 = coco_uniform_random(state);
  const double u2 = coco_uniform_random(state);
  normal = sqrt(-2 * log(u1)) * cos(2 * coco_pi * u2);
#else
  int i;
  normal = 0.0;
  for (i = 0; i < 12; ++i) {
    normal += coco_uniform_random(state);
  }
  normal -= 6.0;
#endif
  return normal;
}

/* Be hygenic (for amalgamation) and undef lags. */
#undef SHORT_LAG
#undef LONG_LAG
#line 1 "src/coco_c_runtime.c"
/*
 * Generic NUMBBO runtime implementation.
 *
 * Other language interfaces might want to replace this so that memory
 * allocation and error handling go through the respective language
 * runtime.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#line 13 "src/coco_c_runtime.c"

void coco_error(const char *message, ...) {
  va_list args;

  fprintf(stderr, "FATAL ERROR: ");
  va_start(args, message);
  vfprintf(stderr, message, args);
  va_end(args);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

void coco_warning(const char *message, ...) {
  va_list args;
  
  fprintf(stderr, "WARNING: ");
  va_start(args, message);
  vfprintf(stderr, message, args);
  va_end(args);
  fprintf(stderr, "\n");
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

void coco_free_memory(void *data) { free(data); }
