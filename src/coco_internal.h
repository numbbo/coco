/*
 * Internal COCO structures and typedefs.
 *
 * These are used throughout the COCO code base but should not be
 * used by any external code.
 */

#ifndef __COCO_INTERNAL__
#define __COCO_INTERNAL__

typedef void (*coco_initial_solution_function_t)(const coco_problem_t *self, double *y);
typedef void (*coco_evaluate_function_t)(coco_problem_t *self, const double *x, double *y);
typedef void (*coco_recommendation_function_t)(coco_problem_t *self,
                                               const double *x,
                                               size_t number_of_solutions);

typedef void (*coco_free_function_t)(coco_problem_t *self);

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
  /* The prominent usecase for data is coco_transformed_data_t*, making an
   * "onion of problems", initialized in coco_transformed_allocate(...).
   * This makes the current ("outer" or "transformed") problem a "derived
   * problem class", which inherits from the "inner" problem, the "base class".
   *   - data holds the meta-information to administer the inheritance
   *   - data->data holds the additional fields of the derived class (the outer problem)
   * Specifically:  
   * data = coco_transformed_data_t *  / * mnemonic: inheritance data or onion data or link data
   *          - coco_problem_t *inner_problem;  / * now we have a linked list
   *          - void *data;  / * defines the additional attributes/fields etc. to be used by the "outer" problem (derived class)
   *          - coco_transformed_free_data_t free_data;  / * deleter for allocated memory in (not of) data->data
   */
};

#endif
