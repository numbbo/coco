/**
 * @file toy_socket.c
 *
 * @brief Implementation of a toy problem to demonstrate communication with an external evaluator
 * through sockets.
 */
#include "coco.h"
#include "coco_platform.h"
#include "socket_communication.c"

/**
 * @brief Calls the external evaluator to evaluate x.
 */
static void toy_socket_evaluate(coco_problem_t *problem, const double *x, double *y) {

  char *message;
  socket_communication_data_t *data = (socket_communication_data_t *) problem->suite->data;

  message = socket_communication_get_message(
      problem->suite->suite_name,
      problem->number_of_objectives,
      problem->suite_dep_function,
      problem->suite_dep_instance,
      problem->number_of_variables,
      x,
      data->precision_x
  );
  socket_communication_evaluate(
      data->host_name,
      data->port,
      message,
      problem->number_of_objectives,
      y
  );
  coco_free_memory(message);
}


/**
 * @brief Creates the toy-socket problem.
 */
static coco_problem_t *toy_socket_problem_allocate(const size_t number_of_objectives,
                                                   const size_t function,
                                                   const size_t dimension,
                                                   const size_t instance,
                                                   const char *problem_id_template,
                                                   const char *problem_name_template) {

  coco_problem_t *problem = NULL;
  size_t i;

  if ((number_of_objectives != 1) && (number_of_objectives != 2))
    coco_error("toy_socket_problem_allocate(): %lu objectives are not supported (only 1 or 2)",
        (unsigned long)number_of_objectives);

  /* Provide the region of interest */
  problem = coco_problem_allocate(dimension, number_of_objectives, 0);
  for (i = 0; i < dimension; ++i) {
    problem->smallest_values_of_interest[i] = -1;
    problem->largest_values_of_interest[i] = 1;
  }
  problem->number_of_integer_variables = 0;
  problem->evaluate_function = toy_socket_evaluate;

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);

  if (number_of_objectives == 1) {
    problem->best_value[0] = -1000;
    for (i = 0; i < dimension; ++i)
      problem->best_parameter[i] = 0; /* TODO Support unknown optimum */
  }
  /* Need to provide estimation for the ideal and nadir points in the bi-objective case */
  else if (number_of_objectives == 2) {
    problem->best_value[0] = -1000;
    problem->best_value[1] = -1000;
    problem->nadir_value[0] = 1000;
    problem->nadir_value[1] = 1000;
  }
  return problem;
}
