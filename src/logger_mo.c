#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "coco.h"
#include "coco_archive.c"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

#include "mo_avl_tree.c"
#include "mo_generics.c"

/* Data for each indicator */
typedef struct {
  char *name;
  FILE *log_file;
  double *target_values;
  size_t number_of_targets;
  size_t next_target;
} logger_mo_indicator_t;

/* Mode of logging nondominated solutions */
typedef enum {NONE, FINAL, ALL} logger_mo_nondom_e;

/* Data for the multiobjective logger */
typedef struct {
  char *result_folder;

  /* File for logging nondominated solutions (either all or final) */
  FILE *nondom_file;
  /* File for logging indicator values at target hits */
  FILE *log_file;
  /* File for logging summary information on algorithm performance */
  FILE *info_file;
  logger_mo_nondom_e log_mode;

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

  int is_initialized;
} logger_mo_t;

/* Data contained in the node's ${item} in the AVL tree */
typedef struct {
  double *x;
  double *y;
  size_t time_stamp;
} logger_mo_avl_item_t;

/**
 * Creates and returns the information on the solution in the form of a node's ${item} in the AVL tree.
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
 * Frees the data of the given logger_mo_node_t.
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
static int compare_by_last_objective(const logger_mo_avl_item_t *item1, const logger_mo_avl_item_t *item2, void *userdata) {
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
static int compare_by_time_stamp(const logger_mo_avl_item_t *item1, const logger_mo_avl_item_t *item2, void *userdata) {
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
static int logger_mo_tree_update(logger_mo_t *data, logger_mo_avl_item_t *node_item) {

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


static void private_logger_mo_initialize(coco_problem_t *self) {
  /* TODO: Handle existing file names!*/
  logger_mo_t *data;

  const char nondom_folder_name[] = "archive";
  const char base_name[] = "problem_name";
  const char extension[] = ".dat";
  char path_name[COCO_PATH_MAX] = {0};
  char file_name[COCO_PATH_MAX] = {0};

  data = coco_transformed_get_data(self);

  coco_create_path(data->result_folder);

  if (data->log_mode != NONE) {

    /* Construct the name of the file (and create the path to it) */
    coco_join_path(path_name, sizeof(path_name), data->result_folder, nondom_folder_name, NULL);
    coco_create_path(path_name);
    /* TODO: Read base_name from the problem and use coco_strconcat() instead of strcat() */
    strcat(file_name, self->problem_id);
    strcat(file_name, "_");
    if (data->log_mode == ALL)
      strcat(file_name, "nondom_all");
    else if (data->log_mode == FINAL)
      strcat(file_name, "nondom_final");
    strcat(file_name, extension);
    coco_join_path(path_name, sizeof(path_name), file_name, NULL);

    /* Open and initialize the file */
    data->nondom_file = fopen(path_name, "a");
    if (data->nondom_file == NULL) {
      coco_error("private_logger_mo_initialize() failed to open file '%s'.", path_name);
      return; /* Never reached */
    }
    if (data->include_decision_variables) {
      fprintf(data->nondom_file, "%% function evaluation | %lu objectives | %lu variables\n",
          data->number_of_objectives, data->number_of_variables);
    } else {
      fprintf(data->nondom_file, "%% function evaluation | %lu objectives \n", data->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  data->archive_tree = avl_tree_construct((avl_compare_t) compare_by_last_objective,
      (avl_free_t) logger_mo_node_free);
  data->buffer_tree = avl_tree_construct((avl_compare_t) compare_by_time_stamp, NULL);

  data->is_initialized = 1;
}

static void private_logger_mo_evaluate(coco_problem_t *self, const double *x, double *y) {
  logger_mo_t *data;
  logger_mo_avl_item_t *node_item;
  int update_performed;

  data = coco_transformed_get_data(self);

  /* If not yet initialized, initialize it */
  if (!data->is_initialized)
    private_logger_mo_initialize(self);

  /* Evaluate function */
  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
  data->number_of_evaluations++;

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in the archive. */
  node_item = logger_mo_node_create(x, y, data->number_of_evaluations, data->number_of_variables,
      data->number_of_objectives);
  update_performed = logger_mo_tree_update(data, node_item);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to nondom_file */
  if (update_performed && (data->log_mode == ALL)) {
    logger_mo_tree_output(data->nondom_file, data->buffer_tree, data->number_of_variables,
        data->number_of_objectives, data->include_decision_variables);
    avl_tree_purge(data->buffer_tree);
  }

  /* TODO If the archive was updated, compute and output a bunch of indicators */

  /* Flush output so that impatient users can see progress. */
  fflush(data->nondom_file);
}

static void private_logger_mo_free(void *stuff) {

  size_t tree_size;
  logger_mo_t *data;
  assert(stuff != NULL);
  data = stuff;

  /* TODO: Resort according to time stamp before output!*/
  if (data->log_mode == FINAL) {
    tree_size = logger_mo_tree_output(data->nondom_file, data->archive_tree, data->number_of_variables,
        data->number_of_objectives, data->include_decision_variables);
    printf("%lu\n", tree_size);
  }

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
 * - reference_values_file : file_name (name of the file with reference values for all indicators; optional)
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
static coco_problem_t *logger_mo(coco_problem_t *problem, const char *options) {

  logger_mo_t *data;
  coco_problem_t *self;
  char string_value[COCO_PATH_MAX];

  data = coco_allocate_memory(sizeof(*data));

  data->number_of_evaluations = 0;
  data->number_of_variables = coco_problem_get_dimension(problem);
  data->number_of_objectives = coco_problem_get_number_of_objectives(problem);

  /* Read values from options */
  if (coco_options_read_string(options, "result_folder", string_value) > 0) {
    data->result_folder = coco_strdup(string_value);
  } else {
    coco_warning("logger_mo(): using results as the name of the result folder");
    data->result_folder = coco_strdup("results");
  }

  data->log_mode = FINAL;
  if (coco_options_read_string(options, "log_nondominated", string_value) > 0) {
    if (strcmp(string_value, "none") == 0)
        data->log_mode = NONE;
    else if (strcmp(string_value, "all") == 0)
      data->log_mode = ALL;
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
    data->log_mode = ALL;
  }

  if ((data->log_mode == NONE) && (!data->compute_indicators)) {
    /* No logging required, return NULL */
    return NULL;
  }

  data->nondom_file = NULL; /* Open lazily in private_logger_mo_evaluate(). */
  data->is_initialized = 0;

  self = coco_transformed_allocate(problem, data, private_logger_mo_free);
  self->evaluate_function = private_logger_mo_evaluate;

  return self;
}
