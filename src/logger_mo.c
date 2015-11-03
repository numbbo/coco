#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_archive.c"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

#include "mo_avl_tree.c"
#include "mo_generics.c"

/* Flags controlling the computation of indicators and logger output (set through observer options) */
static int bla;

/* Data for each indicator */
typedef struct {
  char *input_file_name;
  FILE *output_file;
  double *target_values;
  size_t number_of_targets;
  size_t next_target;
} logger_mo_indicator_t;

/* Data needed for the multiobjective logger */
typedef struct {
  char *path;
  FILE *logfile;
  size_t number_of_evaluations;
  size_t number_of_variables;
  size_t number_of_objectives;

  /* The tree keeping currently non-dominated solutions */
  avl_tree_t *archive_tree;
  /* The tree with pointers to nondominated solutions that haven't been logged yet */
  avl_tree_t *buffer_tree;
} logger_mo_t;

/* Data contained in the node's ${item} in the AVL tree */
typedef struct {
  double *x;
  double *y;
  size_t time_stamp;
} logger_mo_avl_item_t;

static int logger_mo_tree_update(logger_mo_t *data, logger_mo_avl_item_t *node_item);
static void logger_mo_output_nondominated(FILE *logfile, avl_tree_t *buffer_tree, const size_t dim,
    const size_t num_obj);
static void logger_mo_output_tree(const avl_tree_t *tree, FILE *fp, const size_t num_obj) ;

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

static void private_logger_mo_evaluate(coco_problem_t *self, const double *x, double *y) {
  logger_mo_t *data;
  logger_mo_avl_item_t *node_item;
  int update_performed;

  data = coco_transformed_get_data(self);

  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
  data->number_of_evaluations++;

  /* Open logfile if it is not already open */
  if (data->logfile == NULL) {
    data->logfile = fopen(data->path, "w");
    if (data->logfile == NULL) {
      char *buf;
      const char *error_format = "private_logger_mo_evaluate() failed to open log file '%s'.";
      size_t buffer_size = (size_t) snprintf(NULL, 0, error_format, data->path);
      buf = (char *) coco_allocate_memory(buffer_size);
      snprintf(buf, buffer_size, error_format, data->path);
      coco_error(buf);
      coco_free_memory(buf); /* Never reached */
    }
    fprintf(data->logfile, "# %lu variables  |  %lu objectives  |  func eval number\n",
        data->number_of_variables, data->number_of_objectives);

    /* Initialize the AVL trees */
    data->archive_tree = avl_tree_construct((avl_compare_t) compare_by_last_objective,
        (avl_free_t) logger_mo_node_free);
    data->buffer_tree = avl_tree_construct((avl_compare_t) compare_by_time_stamp, NULL);
  }

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in the archive. */
  node_item = logger_mo_node_create(x, y, data->number_of_evaluations, data->number_of_variables,
      data->number_of_objectives);
  update_performed = logger_mo_tree_update(data, node_item);

  /* If the archive was updated, output the new solution to logfile */
  if (update_performed) {
    logger_mo_output_nondominated(data->logfile, data->buffer_tree, data->number_of_variables,
        data->number_of_objectives);
  }

  /* TODO If the archive was updated, compute and output a bunch of indicators */

  /* Flush output so that impatient users can see progress. */
  fflush(data->logfile);
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

/**
 * Outputs the AVL tree to the given file.
 * TODO: This is used only for tests during coding... Should it be removed?
 */
static void logger_mo_output_tree(const avl_tree_t *tree, FILE *fp, const size_t num_obj) {

  avl_node_t *node;
  size_t i;

  node = tree->head;

  while (node) {
    for (i = 0; i < num_obj; i++) {
      fprintf(fp, "%13.10e\t", ((logger_mo_avl_item_t*) node->item)->y[i]);
    }
    fprintf(fp, "\n");

    node = node->next;
  }
}

/**
 * Output nondominated solutions that haven't been logged yet (pointed to by nodes of ${buffer_tree}).
 */
static void logger_mo_output_nondominated(FILE *logfile, avl_tree_t *buffer_tree, const size_t dim,
    const size_t num_obj) {

  avl_node_t *solution;
  size_t i;
  size_t j;

  if (buffer_tree->tail) { /* There is at least a solution in the buffer to log */
    /* Write out all solutions in the buffer and clean up the buffer */
    solution = buffer_tree->head;
    while (solution != NULL) {
      for (i = 0; i < dim; i++)
        fprintf(logfile, "%13.10e\t", ((logger_mo_avl_item_t*) solution->item)->x[i]);
      for (j = 0; j < num_obj; j++)
        fprintf(logfile, "%13.10e\t", ((logger_mo_avl_item_t*) solution->item)->y[j]);
      fprintf(logfile, "%lu", ((logger_mo_avl_item_t*) solution->item)->time_stamp);
      fprintf(logfile, "\n");
      solution = solution->next;
    }
    avl_tree_purge(buffer_tree);
  }
}

static void private_logger_mo_free(void *stuff) {

  logger_mo_t *data;
  assert(stuff != NULL);
  data = stuff;

  if (data->path != NULL) {
    coco_free_memory(data->path);
    data->path = NULL;
  }

  if (data->logfile != NULL) {
    fclose(data->logfile);
    data->logfile = NULL;
  }

  avl_tree_destruct(data->archive_tree);
  avl_tree_destruct(data->buffer_tree);
}

static coco_problem_t *logger_mo(coco_problem_t *inner_problem, const char *observer_options) {
  logger_mo_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->number_of_evaluations = 0;
  data->number_of_variables = coco_problem_get_dimension(inner_problem);
  data->number_of_objectives = coco_problem_get_number_of_objectives(inner_problem);
  data->path = coco_strdup(observer_options);
  data->logfile = NULL; /* Open lazily in private_logger_mo_evaluate(). */
  self = coco_transformed_allocate(inner_problem, data, private_logger_mo_free);
  self->evaluate_function = private_logger_mo_evaluate;
  return self;
}
