/**
 * @file suite_toy_socket_biobj.c
 *
 * @brief Implementation of a bi-objective suite containing one problem to show the functioning
 * of socket communication for function evaluation. A single-objective version can be found in the
 * file suite_toy_socket.c
 *
 * The suite contains 1 problem with dimension 2 and 1 instance.
 */

#include "coco.h"
#include "toy_socket.c"
#include "suite_toy_socket.c"

static coco_suite_t *coco_suite_allocate(const char *suite_name,
                                         const size_t number_of_functions,
                                         const size_t number_of_dimensions,
                                         const size_t *dimensions,
                                         const char *default_instances);

/**
 * @brief Sets the dimensions and default instances for the toy-socket-biobj suite.
 */
static coco_suite_t *suite_toy_socket_biobj_initialize(void) {

  coco_suite_t *suite;
  suite_toy_socket_data_t *data;
  const size_t dimensions[] = { 2 };

  suite = coco_suite_allocate("toy-socket-biobj", 1, 1, dimensions, "instances: 1");

  data = (suite_toy_socket_data_t *) coco_allocate_memory(sizeof(*data));
  data->host_name = coco_strdup("localhost");     /* TODO: Read this from the suite options! */
  data->port = 7251;                              /* TODO: Read this from the suite options! */
  suite->data = data;
  suite->data_free_function = suite_toy_socket_data_free;
  return suite;
}

/**
 * @brief Returns the problem from the toy-socket-biobj suite that corresponds to the given parameters.
 *
 * @param suite The COCO suite.
 * @param function_idx Index of the function (starting from 0).
 * @param dimension_idx Index of the dimension (starting from 0).
 * @param instance_idx Index of the instance (starting from 0).
 * @return The problem that corresponds to the given parameters.
 */
static coco_problem_t *suite_toy_socket_biobj_get_problem(coco_suite_t *suite,
                                                          const size_t function_idx,
                                                          const size_t dimension_idx,
                                                          const size_t instance_idx) {

  coco_problem_t *problem = NULL;
  suite_toy_socket_data_t *data;
  data = (suite_toy_socket_data_t *) suite->data;

  const size_t function = suite->functions[function_idx];
  const size_t dimension = suite->dimensions[dimension_idx];
  const size_t instance = suite->instances[instance_idx];

  const char *problem_id_template = "toy_socket_biobj_f%02lu_i%02lu_d%02lu";
  const char *problem_name_template = "bi-objective toy socket suite problem f%lu instance %lu in %luD";

  problem = toy_socket_problem_allocate(2, function, dimension, instance,
      problem_id_template, problem_name_template, data->host_name, data->port);
  assert(problem != NULL);

  problem->suite_dep_function = function;
  problem->suite_dep_instance = instance;
  problem->suite_dep_index = coco_suite_encode_problem_index(suite, function_idx, dimension_idx, instance_idx);

  return problem;
}
