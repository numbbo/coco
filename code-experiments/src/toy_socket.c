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
 * @brief Data type used by the toy socket problem.
 */
typedef struct {
  unsigned short port;
  char *host_name;
} toy_socket_data_t;

/**
 * @brief Frees the memory of the toy-socket data.
 */
static void toy_socket_free(coco_problem_t *problem) {

  toy_socket_data_t *data = (toy_socket_data_t *) problem->data;
  coco_free_memory(data->host_name);
  problem->problem_free_function = NULL;
  coco_problem_free(problem);
}

/**
 * @brief Calls the external evaluator to evaluate x.
 */
static void toy_socket_evaluate(coco_problem_t *problem, const double *x, double *y) {

  char *message;
  toy_socket_data_t *data = (toy_socket_data_t *) problem->data;

  /* TODO Prepare the message for evaluation */
  message = coco_strdupf("toy_socket_f01_i01_d02 2 0.5 0.2");
  socket_communication_evaluate(data->host_name, data->port, message,
      problem->number_of_objectives, y);
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
                                                   const char *problem_name_template,
                                                   const char *host_name,
                                                   const unsigned short port) {

  coco_problem_t *problem = NULL;
  toy_socket_data_t *data;
  size_t i;

  if ((number_of_objectives != 1) && (number_of_objectives != 2))
    coco_error("toy_socket_problem_allocate(): %lu objectives are not supported (only 1 or 2)",
        (unsigned long)number_of_objectives);

  /* Provide the region of interest */
  problem = coco_problem_allocate(dimension, number_of_objectives, 0);
  for (i = 0; i < dimension; ++i) {
    problem->smallest_values_of_interest[i] = -1;
    problem->largest_values_of_interest[i] = 1;
    problem->best_parameter[i] = 0; /* TODO */
  }
  problem->number_of_integer_variables = 0;
  problem->evaluate_function = toy_socket_evaluate;

  coco_problem_set_id(problem, problem_id_template, function, instance, dimension);
  coco_problem_set_name(problem, problem_name_template, function, instance, dimension);

  /* TODO: Support unknown optimum */
  /*if (problem->best_parameter != NULL) {
    coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
  }
  if (number_of_objectives == 1) {
    if (problem->best_value != NULL) {
      coco_free_memory(problem->best_value);
      problem->best_value = NULL;
    }
  }*/

  if (number_of_objectives == 1) {
    problem->best_value[0] = -1000;
  }
  /* Need to provide estimation for the ideal and nadir points in the bi-objective case */
  else if (number_of_objectives == 2) {
    problem->best_value[0] = -1000;
    problem->best_value[1] = -1000;
    problem->nadir_value[0] = 1000;
    problem->nadir_value[1] = 1000;
  }

  data = (toy_socket_data_t *) coco_allocate_memory(sizeof(*data));
  data->port = port;
  data->host_name = coco_strdup(host_name);
  problem->data = data;
  problem->problem_free_function = toy_socket_free;
  return problem;
}
