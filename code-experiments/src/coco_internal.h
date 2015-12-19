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
 * Evaluate and free are opaque pointers which should not be called
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
 * problem_type - Type of the problem. May be NULL to indicate that no type is known.
 *
 * suite_dep_index - Index of the problem in the current/parent benchmark suite
 *
 * suite_dep_function - Problem function in the current/parent benchmark suite
 *
 * suite_dep_instanced - Problem instance the current/parent benchmark suite
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
  double *best_value; /* means: smallest possible f-value */
  double *best_parameter;
  char *problem_name;
  char *problem_id;
  char *problem_type;
  long evaluations;
  /* Convenience fields for output generation */
  double final_target_delta[1];
  double best_observed_fvalue[1];
  long best_observed_evaluation[1];
  /* Fields depending on the current/parent benchmark suite */
  size_t suite_dep_index;
  size_t suite_dep_function;
  size_t suite_dep_instance;
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

typedef void (*coco_observer_data_free_function_t)(void *data);
typedef coco_problem_t *(*coco_logger_initialize_function_t)(coco_observer_t *self, coco_problem_t *problem);

/**
 * Description of a COCO observer (instance)
 *
 * Fields:
 *
 * output_folder - Name of the output folder
 *
 * algorithm_name - Name of the algorithm to be used in logger output and plots
 *
 * algorithm_info - Additional information on the algorithm to be used in logger output
 *
 * data - Void pointer that can be used to store data specific to any observer
 *
 */
struct coco_observer {

  int is_active;
  char *output_folder;
  char *algorithm_name;
  char *algorithm_info;
  int precision_x;
  int precision_f;
  void *data;

  coco_observer_data_free_function_t data_free_function;
  coco_logger_initialize_function_t logger_initialize_function;
};

struct coco_suite {

  char *suite_name;

  size_t number_of_dimensions;
  size_t *dimensions;
  long current_dimension_idx;

  size_t number_of_functions;
  size_t *functions;
  long current_function_idx;

  size_t number_of_instances;
  size_t *instances;
  long current_instance_idx;
  char *default_instances;

  coco_problem_t *current_problem;

};

#endif
