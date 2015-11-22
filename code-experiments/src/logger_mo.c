#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

#include "observer_mo.c"

#include "mo_avl_tree.c"
#include "mo_generics.c"
#include "mo_targets.c"

/* Static variables needed to know whether the logger was called for the first time */
static size_t DEPRECATED__LOGGER_MO_CALLS = 0;
static char DEPRECATED__LOGGER_MO_RESULT_FOLDER[COCO_PATH_MAX] = "";

/* Mode of logging nondominated solutions */
typedef enum {DEPRECATED__NONE, DEPRECATED__FINAL, DEPRECATED__ALL} deprecated__logger_mo_nondom_e;

/* Data for each indicator */
typedef struct {
  /* Name of the indicator to be used in the output files */
  char *name;

  /* File for logging indicator values at target hits */
  FILE *log_file;
  /* File for logging summary information on algorithm performance */
  FILE *info_file;

  double *target_values;
  size_t number_of_targets;
  size_t next_target_id;
} logger_mo_indicator_t;

/* Data for the multiobjective logger */
typedef struct {
  coco_observer_t *observer;

  /* File for logging nondominated solutions (either all or final) */
  FILE *nondom_file;

  size_t number_of_evaluations;
  size_t number_of_variables;
  size_t number_of_objectives;

  /* The tree keeping currently non-dominated solutions */
  avl_tree_t *archive_tree;
  /* The tree with pointers to nondominated solutions that haven't been logged yet */
  avl_tree_t *buffer_tree;

  /* The indicators */
  logger_mo_indicator_t *hypervolume;
} logger_mo_t;

/* Data for the multiobjective logger */
typedef struct {
  char *result_folder;

  /* File for logging nondominated solutions (either all or final) */
  FILE *nondom_file;
  deprecated__logger_mo_nondom_e log_mode;

  int include_decision_variables;
  int compute_indicators;
  int produce_all_data;

  size_t number_of_evaluations;
  size_t number_of_variables;
  size_t number_of_objectives;

  /* The tree keeping currently non-dominated solutions */
  avl_tree_t *archive_tree;
  /* The tree with pointers to nondominated solutions that haven't been logged yet */
  avl_tree_t *buffer_tree;

  logger_mo_indicator_t *hypervolume;

  int is_initialized;
} deprecated__logger_mo_t;

/* Data contained in the node's item in the AVL tree */
typedef struct {
  double *x;
  double *y;
  size_t time_stamp;
} logger_mo_avl_item_t;

/**
 * Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static logger_mo_avl_item_t* logger_mo_node_create(const double *x, const double *y, const size_t time_stamp,
    const size_t dim, const size_t num_obj) {

  size_t i;

  /* Allocate memory to hold the data structure logger_mo_node_t */
  logger_mo_avl_item_t *item = (logger_mo_avl_item_t*) coco_allocate_memory(sizeof(logger_mo_avl_item_t));

  /* Allocate memory to store the (copied) data of the new node */
  item->x = coco_allocate_vector(dim);
  item->y = coco_allocate_vector(num_obj);

  /* Copy the data */
  for (i = 0; i < dim; i++)
    item->x[i] = x[i];
  for (i = 0; i < num_obj; i++)
    item->y[i] = y[i];
  item->time_stamp = time_stamp;
  return item;
}

/**
 * Frees the data of the given logger_mo_avl_item_t.
 */
static void logger_mo_node_free(logger_mo_avl_item_t *item, void *userdata) {

  coco_free_memory(item->x);
  coco_free_memory(item->y);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * Defines the ordering of AVL tree nodes based on the value of the last objective.
 */
static int avl_tree_compare_by_last_objective(const logger_mo_avl_item_t *item1, const logger_mo_avl_item_t *item2, void *userdata) {
  /* This ordering is used by the archive_tree. */

  if (item1->y[1] < item2->y[1])
    return -1;
  else if (item1->y[1] > item2->y[1])
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * Defines the ordering of AVL tree nodes based on the time stamp.
 */
static int avl_tree_compare_by_time_stamp(const logger_mo_avl_item_t *item1, const logger_mo_avl_item_t *item2, void *userdata) {
  /* This ordering is used by the buffer_tree. */

  if (item1->time_stamp < item2->time_stamp)
    return -1;
  else if (item1->time_stamp > item2->time_stamp)
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * Outputs the AVL tree to the given file. Returns the number of nodes in the tree.
 */
static size_t logger_mo_tree_output(FILE *file, avl_tree_t *tree, const size_t dim,
    const size_t num_obj, const int output_x) {

  avl_node_t *solution;
  size_t i;
  size_t j;
  size_t number_of_nodes = 0;

  if (tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = tree->head;
    while (solution != NULL) {
      fprintf(file, "%lu\t", ((logger_mo_avl_item_t*) solution->item)->time_stamp);
      for (j = 0; j < num_obj; j++)
        fprintf(file, "%13.10e\t", ((logger_mo_avl_item_t*) solution->item)->y[j]);
      if (output_x) {
        for (i = 0; i < dim; i++)
          fprintf(file, "%13.10e\t", ((logger_mo_avl_item_t*) solution->item)->x[i]);
      }
      fprintf(file, "\n");
      solution = solution->next;
      number_of_nodes++;
    }
  }

  return number_of_nodes;
}

/**
 * Checks for domination and updates the archive tree if the given node is not dominated by or equal to existing
 * nodes in the archive tree.
 * Returns 1 if the update was performed and 0 otherwise.
 */
static int deprecated__logger_mo_tree_update(deprecated__logger_mo_t *data, logger_mo_avl_item_t *node_item) {

  avl_node_t *node, *next_node;
  int trigger_update = 0;
  int dominance;

  node = avl_item_search_right(data->archive_tree, node_item, NULL);

  if (node == NULL) { /* The given node is an extremal point */
    trigger_update = 1;
    next_node = data->archive_tree->head;
  } else {
    dominance = mo_get_dominance(node_item->y, ((logger_mo_avl_item_t*) node->item)->y, data->number_of_objectives);
    if (dominance > -1) {
      trigger_update = 1;
      next_node = node->next;
      if (dominance == 1) {
        avl_item_delete(data->buffer_tree, node->item);
        avl_node_delete(data->archive_tree, node);
      }
    } else {
      trigger_update = 0;
    }
  }

  if (!trigger_update) {
    logger_mo_node_free(node_item, NULL);
  } else {
    /* Perform tree update */
    while (next_node != NULL) {
      /* Check the dominance relation between the given node and the next node. There are only two possibilities:
       * dominance = 0: the given node and the next node are nondominated
       * dominance = 1: the given node dominates the next node */
      node = next_node;
      dominance = mo_get_dominance(node_item->y, ((logger_mo_avl_item_t*) node->item)->y,
          data->number_of_objectives);
      if (dominance == 1) {
        next_node = node->next;
        avl_item_delete(data->buffer_tree, node->item);
        avl_node_delete(data->archive_tree, node);
      } else {
        break;
      }
    }

    avl_item_insert(data->archive_tree, node_item);
    avl_item_insert(data->buffer_tree, node_item);
  }

  return trigger_update;
}

static void deprecated__logger_mo_initialize(coco_problem_t *problem) {
  deprecated__logger_mo_t *data;

  const char nondom_folder_name[] = "archive";
  char *path_name;
  char *file_name;

  data = coco_transformed_get_data(problem);

  if (data->log_mode != DEPRECATED__NONE) {

    /* Create the path to the file */
    path_name = (char *) coco_allocate_memory(COCO_PATH_MAX);
    memcpy(path_name, data->result_folder, strlen(data->result_folder) + 1);
    coco_join_path(path_name, COCO_PATH_MAX, nondom_folder_name, NULL);
    coco_create_path(path_name);

    /* Construct file name (use coco_create_unique_filename to make it unique) */
    if (data->log_mode == DEPRECATED__ALL)
      file_name = coco_strdupf("%s_nondom_all.dat", problem->problem_id);
    else if (data->log_mode == DEPRECATED__FINAL)
      file_name = coco_strdupf("%s_nondom_final.dat", problem->problem_id);
    coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);

    /* Open and initialize the file */
    data->nondom_file = fopen(path_name, "a");
    if (data->nondom_file == NULL) {
      coco_error("private_logger_mo_initialize() failed to open file '%s'.", path_name);
      coco_free_memory(file_name);
      coco_free_memory(path_name);
      return; /* Never reached */
    }

    coco_free_memory(file_name);
    coco_free_memory(path_name);

    fprintf(data->nondom_file, "%% instance = %lu\n", 224L); /* TODO: Output the instance_id!*/
    if (data->include_decision_variables) {
      fprintf(data->nondom_file, "%% function evaluation | %lu objectives | %lu variables\n",
          data->number_of_objectives, data->number_of_variables);
    } else {
      fprintf(data->nondom_file, "%% function evaluation | %lu objectives \n", data->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  data->archive_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_last_objective,
      (avl_free_t) logger_mo_node_free);
  data->buffer_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

  data->is_initialized = 1;
}

static void deprecated__logger_mo_evaluate(coco_problem_t *problem, const double *x, double *y) {
  deprecated__logger_mo_t *data;
  logger_mo_avl_item_t *node_item;
  int update_performed;

  data = coco_transformed_get_data(problem);

  /* If not yet initialized, initialize it */
  if (!data->is_initialized)
    deprecated__logger_mo_initialize(problem);

  /* Evaluate function */
  coco_evaluate_function(coco_transformed_get_inner_problem(problem), x, y);
  data->number_of_evaluations++;

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in the archive. */
  node_item = logger_mo_node_create(x, y, data->number_of_evaluations, data->number_of_variables,
      data->number_of_objectives);
  update_performed = deprecated__logger_mo_tree_update(data, node_item);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to nondom_file */
  if (update_performed && (data->log_mode == DEPRECATED__ALL)) {
    logger_mo_tree_output(data->nondom_file, data->buffer_tree, data->number_of_variables,
        data->number_of_objectives, data->include_decision_variables);
    avl_tree_purge(data->buffer_tree);
  }

  /* TODO If the archive was updated, compute and output a bunch of indicators */

  /* Flush output so that impatient users can see progress. */
  fflush(data->nondom_file);
}

static void deprecated__logger_mo_finalize(deprecated__logger_mo_t *data) {

  if (data->log_mode == DEPRECATED__FINAL) {
    /* Resort archive_tree according to time stamp and then output it */

    avl_tree_t *resorted_tree;
    avl_node_t *solution;
    resorted_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

    if (data->archive_tree->tail) {
      /* There is at least a solution in the tree to output */
      solution = data->archive_tree->head;
      while (solution != NULL) {
        avl_item_insert(resorted_tree, solution->item);
        solution = solution->next;
      }
    }

    logger_mo_tree_output(data->nondom_file, resorted_tree, data->number_of_variables,
        data->number_of_objectives, data->include_decision_variables);
  }

}

static void deprecated__logger_mo_free(void *stuff) {

  deprecated__logger_mo_t *data;

  assert(stuff != NULL);
  data = stuff;

  deprecated__logger_mo_finalize(data);

  if (data->result_folder != NULL) {
    coco_free_memory(data->result_folder);
    data->result_folder = NULL;
  }

  if (data->nondom_file != NULL) {
    fclose(data->nondom_file);
    data->nondom_file = NULL;
  }

  avl_tree_destruct(data->archive_tree);
  avl_tree_destruct(data->buffer_tree);
}

/**
 * Sets up the multiobjective logger. Possible options:
 * - result_folder : folder_name (folder for result output; if not given, "results" is used instead)
 * - log_nondominated : none (don't log nondominated solutions)
 * - log_nondominated : final (log only the final nondominated solutions; default value)
 * - log_nondominated : all (log every solution that is nondominated at creation time)
 * - include_decision_variables : 0 / 1 (whether to include decision variables when logging nondominated solutions;
 * default value is 0)
 * - compute_indicators : 0 / 1 (whether to compute performance indicators; default value is 1)
 * - produce_all_data: 0 / 1 (whether to produce all data; if set to 1, overwrites other options and is equivalent to
 * setting log_nondominated to all, include_decision_variables to 1 and compute_indicators to 1; if set to 0, it
 * does not change the values of other options; default value is 0)
 */
coco_problem_t *deprecated__logger_mo(coco_problem_t *problem, const char *options) {

  deprecated__logger_mo_t *data;
  coco_problem_t *self;
  char string_value[COCO_PATH_MAX];

  data = coco_allocate_memory(sizeof(*data));

  data->number_of_evaluations = 0;
  data->number_of_variables = coco_problem_get_dimension(problem);
  data->number_of_objectives = coco_problem_get_number_of_objectives(problem);

  /* Creates a unique output folder if this is a first call to this logger */
  if (DEPRECATED__LOGGER_MO_CALLS == 0) {
    /* First call to the logger, create a unique output folder */
    if (coco_options_read_string(options, "result_folder", string_value) > 0) {
      data->result_folder = coco_strdup(string_value);
    } else {
      coco_warning("logger_mo(): using results as the name of the result folder");
      data->result_folder = coco_strdup("results");
    }
    coco_create_unique_path(&(data->result_folder));
    strcpy(DEPRECATED__LOGGER_MO_RESULT_FOLDER, data->result_folder);
  }
  else {
    data->result_folder = coco_strdup(DEPRECATED__LOGGER_MO_RESULT_FOLDER);
  }

  data->log_mode = DEPRECATED__FINAL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
        data->log_mode = DEPRECATED__NONE;
    else if (strcmp(string_value, "all") == 0)
      data->log_mode = DEPRECATED__ALL;
  }

  if (coco_options_read_int(options, "include_decision_variables", &(data->include_decision_variables)) == 0)
    data->include_decision_variables = 0;

  if (coco_options_read_int(options, "compute_indicators", &(data->compute_indicators)) == 0)
    data->compute_indicators = 1;

  if (coco_options_read_int(options, "produce_all_data", &(data->produce_all_data)) == 0)
    data->produce_all_data = 0;

  if (data->produce_all_data) {
    data->include_decision_variables = 1;
    data->compute_indicators = 1;
    data->log_mode = DEPRECATED__ALL;
  }

  if ((data->log_mode == DEPRECATED__NONE) && (!data->compute_indicators)) {
    /* No logging required, return NULL */
    return NULL;
  }

  data->nondom_file = NULL; /* Open lazily in private_logger_mo_evaluate(). */
  data->is_initialized = 0;

  self = coco_transformed_allocate(problem, data, deprecated__logger_mo_free);
  self->evaluate_function = deprecated__logger_mo_evaluate;

  DEPRECATED__LOGGER_MO_CALLS++;

  return self;
}

/**
 * Checks for domination and updates the archive tree if the given node is not dominated by or equal to existing
 * nodes in the archive tree.
 * Returns 1 if the update was performed and 0 otherwise.
 */
static int logger_mo_tree_update(logger_mo_t *logger, logger_mo_avl_item_t *node_item) {

  avl_node_t *node, *next_node;
  int trigger_update = 0;
  int dominance;

  node = avl_item_search_right(logger->archive_tree, node_item, NULL);

  if (node == NULL) { /* The given node is an extremal point */
    trigger_update = 1;
    next_node = logger->archive_tree->head;
  } else {
    dominance = mo_get_dominance(node_item->y, ((logger_mo_avl_item_t*) node->item)->y, logger->number_of_objectives);
    if (dominance > -1) {
      trigger_update = 1;
      next_node = node->next;
      if (dominance == 1) {
        avl_item_delete(logger->buffer_tree, node->item);
        avl_node_delete(logger->archive_tree, node);
      }
    } else {
      trigger_update = 0;
    }
  }

  if (!trigger_update) {
    logger_mo_node_free(node_item, NULL);
  } else {
    /* Perform tree update */
    while (next_node != NULL) {
      /* Check the dominance relation between the given node and the next node. There are only two possibilities:
       * dominance = 0: the given node and the next node are nondominated
       * dominance = 1: the given node dominates the next node */
      node = next_node;
      dominance = mo_get_dominance(node_item->y, ((logger_mo_avl_item_t*) node->item)->y,
          logger->number_of_objectives);
      if (dominance == 1) {
        next_node = node->next;
        avl_item_delete(logger->buffer_tree, node->item);
        avl_node_delete(logger->archive_tree, node);
      } else {
        break;
      }
    }

    avl_item_insert(logger->archive_tree, node_item);
    avl_item_insert(logger->buffer_tree, node_item);
  }

  return trigger_update;
}

static logger_mo_indicator_t *logger_mo_indicator(char *indicator_name) {

  logger_mo_indicator_t *indicator;

  FILE *file;
  char *file_name;

  indicator = (logger_mo_indicator_t *) coco_allocate_memory(sizeof(logger_mo_indicator_t));

  indicator->name = coco_strdup(indicator_name);

  indicator->log_file = NULL; /* TODO! */
  indicator->info_file = NULL; /* TODO! */

  /* Store target_values (computed using reference values and LOGGER_MO_RELATIVE_TARGET_VALUES) */
  indicator->target_values = coco_allocate_vector(LOGGER_MO_NUMBER_OF_TARGETS);

  /* Open the file with reference values */
  file_name = coco_strdupf("reference_values_%s.txt", indicator_name);
  file = fopen(file_name, "r");
  if (file == NULL) {
    coco_error("logger_mo_indicator() failed to open file '%s'.", file_name);
    return NULL; /* Never reached */
  }

  indicator->next_target_id = 0;

  return indicator;
}

static void logger_mo_indicator_free(void *stuff) {

  logger_mo_indicator_t *indicator;

  if (stuff == NULL) /* TODO: Delete this! */
    return;

  assert(stuff != NULL);
  indicator = stuff;

  if (indicator->name != NULL) {
    coco_free_memory(indicator->name);
    indicator->name = NULL;
  }

  if (indicator->log_file != NULL) {
    fclose(indicator->log_file);
    indicator->log_file = NULL;
  }

  if (indicator->info_file != NULL) {
    fclose(indicator->info_file);
    indicator->info_file = NULL;
  }

  if (indicator->target_values != NULL) {
    coco_free_memory(indicator->target_values);
    indicator->target_values = NULL;
  }
}

static void logger_mo_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_mo_t *logger;
  coco_observer_t *observer;
  observer_mo_data_t *observer_mo;

  logger_mo_avl_item_t *node_item;
  int update_performed;

  logger = (logger_mo_t *) coco_transformed_get_data(problem);
  observer = logger->observer;
  observer_mo = (observer_mo_data_t *) observer->data;

  /* Evaluate function */
  coco_evaluate_function(coco_transformed_get_inner_problem(problem), x, y);
  logger->number_of_evaluations++;

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in the archive. */
  node_item = logger_mo_node_create(x, y, logger->number_of_evaluations, logger->number_of_variables,
      logger->number_of_objectives);
  update_performed = logger_mo_tree_update(logger, node_item);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to nondom_file */
  if (update_performed && (observer_mo->log_mode == ALL)) {
    logger_mo_tree_output(logger->nondom_file, logger->buffer_tree, logger->number_of_variables,
        logger->number_of_objectives, observer_mo->include_decision_variables);
    avl_tree_purge(logger->buffer_tree);
  }

  /* TODO If the archive was updated, compute and output a bunch of indicators */

  /* Flush output so that impatient users can see progress. */
  fflush(logger->nondom_file);
}

static void logger_mo_finalize(logger_mo_t *logger) {

  coco_observer_t *observer;
  observer_mo_data_t *observer_mo;

  observer = logger->observer;
  observer_mo = (observer_mo_data_t *) observer->data;

  if (observer_mo->log_mode == FINAL) {
    /* Resort archive_tree according to time stamp and then output it */

    avl_tree_t *resorted_tree;
    avl_node_t *solution;
    resorted_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

    if (logger->archive_tree->tail) {
      /* There is at least a solution in the tree to output */
      solution = logger->archive_tree->head;
      while (solution != NULL) {
        avl_item_insert(resorted_tree, solution->item);
        solution = solution->next;
      }
    }

    logger_mo_tree_output(logger->nondom_file, resorted_tree, logger->number_of_variables,
        logger->number_of_objectives, observer_mo->include_decision_variables);
  }

}

static void logger_mo_free(void *stuff) {

  logger_mo_t *data;

  assert(stuff != NULL);
  data = stuff;

  logger_mo_finalize(data);

  if (data->nondom_file != NULL) {
    fclose(data->nondom_file);
    data->nondom_file = NULL;
  }

  avl_tree_destruct(data->archive_tree);
  avl_tree_destruct(data->buffer_tree);

  logger_mo_indicator_free(data->hypervolume);
}

/**
 * Initializes the multiobjective logger.
 */
coco_problem_t *logger_mo(coco_observer_t *observer, coco_problem_t *problem) {

  coco_problem_t *self;
  logger_mo_t *logger;
  observer_mo_data_t *observer_mo;
  const char nondom_folder_name[] = "archive";
  char *path_name, *file_name, *prefix;

  logger = coco_allocate_memory(sizeof(*logger));

  logger->observer = observer;
  logger->number_of_evaluations = 0;
  logger->number_of_variables = problem->number_of_variables;
  logger->number_of_objectives = problem->number_of_objectives;

  observer_mo = (observer_mo_data_t *) observer->data;

  /* Initialize logging of nondominated solutions */
  if (observer_mo->log_mode != NONE) {

    /* Create the path to the file */
    path_name = (char *) coco_allocate_memory(COCO_PATH_MAX);
    memcpy(path_name, observer->output_folder, strlen(observer->output_folder) + 1);
    coco_join_path(path_name, COCO_PATH_MAX, nondom_folder_name, NULL);
    coco_create_path(path_name);

    /* Construct file name */
    prefix = coco_problem_get_id_without_instance(problem);
    if (observer_mo->log_mode == ALL)
      file_name = coco_strdupf("%s_nondom_all.dat", prefix);
    else if (observer_mo->log_mode == FINAL)
      file_name = coco_strdupf("%s_nondom_final.dat", prefix);
    coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
    coco_free_memory(prefix);

    /* Open and initialize the file */
    logger->nondom_file = fopen(path_name, "a");
    if (logger->nondom_file == NULL) {
      coco_error("logger_mo() failed to open file '%s'.", path_name);
      coco_free_memory(file_name);
      coco_free_memory(path_name);
      return NULL; /* Never reached */
    }

    coco_free_memory(file_name);
    coco_free_memory(path_name);

    /* Output header information */
    fprintf(logger->nondom_file, "%% instance = %ld\n", problem->suite_dep_instance_id + 1);
    if (observer_mo->include_decision_variables) {
      fprintf(logger->nondom_file, "%% function evaluation | %lu objectives | %lu variables\n",
          problem->number_of_objectives, problem->number_of_variables);
    } else {
      fprintf(logger->nondom_file, "%% function evaluation | %lu objectives \n", problem->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  logger->archive_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_last_objective,
      (avl_free_t) logger_mo_node_free);
  logger->buffer_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

  logger->hypervolume = NULL; /* TODO */

  self = coco_transformed_allocate(problem, logger, logger_mo_free);
  self->evaluate_function = logger_mo_evaluate;

  return self;
}
