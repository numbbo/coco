#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"

#include "observer_biobj.c"

#include "logger_biobj_avl_tree.c"
#include "mo_generics.c"
#include "mo_targets.c"

#define READ_BEST_VALUES_FROM_FILE 0

/**
 * This is a biobjective logger that logs the values of some indicators and can output also nondominated
 * solutions.
 */

/* Data for each indicator */
typedef struct {
  /* Name of the indicator to be used for identification and in the output */
  char *name;

  /* File for logging indicator values at target hits */
  FILE *log_file;
  /* File for logging summary information on algorithm performance */
  FILE *info_file;

  /* The best known indicator value for this benchmark problem */
  double best_value;
  size_t next_target_id;
  /* The current overall indicator value */
  double current_value;

} logger_biobj_indicator_t;

/* Data for the biobjective logger */
typedef struct {
  coco_observer_t *observer;

  /* File for logging nondominated solutions (either all or final) */
  FILE *nondom_file;

  size_t number_of_evaluations;
  size_t number_of_variables;
  size_t number_of_objectives;
  long suite_dep_instance_id;

  /* The tree keeping currently non-dominated solutions */
  avl_tree_t *archive_tree;
  /* The tree with pointers to nondominated solutions that haven't been logged yet */
  avl_tree_t *buffer_tree;

  /* Indicators (TODO: Implement others!) */
  logger_biobj_indicator_t *indicators[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS];

} logger_biobj_t;

/* Data contained in the node's item in the AVL tree */
typedef struct {
  double *x;
  double *y;
  size_t time_stamp;

  /* The contribution of this solution to the overall indicator values */
  double indicator_contribution[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS];
  /* Whether the solution is within the region of interest (ROI) */
  int within_ROI;

} logger_biobj_avl_item_t;

/**
 * Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static logger_biobj_avl_item_t* logger_biobj_node_create(const double *x,
                                                         const double *y,
                                                         const size_t time_stamp,
                                                         const size_t dim,
                                                         const size_t num_obj) {

  size_t i;

  /* Allocate memory to hold the data structure logger_biobj_node_t */
  logger_biobj_avl_item_t *item = (logger_biobj_avl_item_t*) coco_allocate_memory(sizeof(*item));

  /* Allocate memory to store the (copied) data of the new node */
  item->x = coco_allocate_vector(dim);
  item->y = coco_allocate_vector(num_obj);

  /* Copy the data */
  for (i = 0; i < dim; i++)
    item->x[i] = x[i];
  for (i = 0; i < num_obj; i++)
    item->y[i] = y[i];
  item->time_stamp = time_stamp;
  for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++)
    item->indicator_contribution[i] = 0;
  item->within_ROI = 0;
  return item;
}

/**
 * Frees the data of the given logger_biobj_avl_item_t.
 */
static void logger_biobj_node_free(logger_biobj_avl_item_t *item, void *userdata) {

  coco_free_memory(item->x);
  coco_free_memory(item->y);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * Checks if the given node is smaller than the reference point, and stores this information in the node's
 * item->within_ROI field.
 */
static void logger_biobj_check_if_within_ROI(coco_problem_t *problem, avl_node_t *node) {

  logger_biobj_avl_item_t *node_item = (logger_biobj_avl_item_t *) node->item;
  mo_problem_data_t *data = (mo_problem_data_t *)((coco_stacked_problem_data_t *) problem->data)->data;
  size_t i;

  node_item->within_ROI = 1;
  for (i = 0; i < problem->number_of_objectives; i++)
    if (node_item->y[i] > data->reference_point[i]) {
      node_item->within_ROI = 0;
      break;
    }

  if (!node_item->within_ROI)
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++)
      node_item->indicator_contribution[i] = 0;

  return;
}

/**
 * Defines the ordering of AVL tree nodes based on the value of the last objective.
 */
static int avl_tree_compare_by_last_objective(const logger_biobj_avl_item_t *item1,
                                              const logger_biobj_avl_item_t *item2,
                                              void *userdata) {
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
static int avl_tree_compare_by_time_stamp(const logger_biobj_avl_item_t *item1,
                                          const logger_biobj_avl_item_t *item2,
                                          void *userdata) {
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
static size_t logger_biobj_tree_output(FILE *file,
                                       avl_tree_t *tree,
                                       const size_t dim,
                                       const size_t num_obj,
                                       const int output_x) {

  avl_node_t *solution;
  size_t i;
  size_t j;
  size_t number_of_nodes = 0;

  if (tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = tree->head;
    while (solution != NULL) {
      fprintf(file, "%lu\t", ((logger_biobj_avl_item_t*) solution->item)->time_stamp);
      for (j = 0; j < num_obj; j++)
        fprintf(file, "%22.15e\t", ((logger_biobj_avl_item_t*) solution->item)->y[j]);
      if (output_x) {
        for (i = 0; i < dim; i++)
          fprintf(file, "%22.15e\t", ((logger_biobj_avl_item_t*) solution->item)->x[i]);
      }
      fprintf(file, "\n");
      solution = solution->next;
      number_of_nodes++;
    }
  }

  return number_of_nodes;
}

/**
 * Checks for domination and updates the archive tree and the values of the indicators if the given node is
 * not weakly dominated by existing nodes in the archive tree.
 * Returns 1 if the update was performed and 0 otherwise.
 */
static int logger_biobj_tree_update(logger_biobj_t *logger,
                                    coco_problem_t *problem,
                                    logger_biobj_avl_item_t *node_item,
                                    int compute_indicators) {

  avl_node_t *node, *next_node, *new_node;
  int trigger_update = 0;
  int dominance;
  size_t i;

  mo_problem_data_t *data = (mo_problem_data_t *)((coco_stacked_problem_data_t *) problem->data)->data;

  /* Find the first point that is not worse than the new point (NULL if such point does not exist) */
  node = avl_item_search_right(logger->archive_tree, node_item, NULL);

  if (node == NULL) {
    /* The new point is an extremal point */
    trigger_update = 1;
    next_node = logger->archive_tree->head;
  } else {
    dominance = mo_get_dominance(node_item->y, ((logger_biobj_avl_item_t*) node->item)->y,
        logger->number_of_objectives);
    if (dominance > -1) {
      trigger_update = 1;
      next_node = node->next;
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        if (compute_indicators) {
          for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            logger->indicators[i]->current_value -= ((logger_biobj_avl_item_t*) node->item)->indicator_contribution[i];
          }
        }
        avl_item_delete(logger->buffer_tree, node->item);
        avl_node_delete(logger->archive_tree, node);
      }
    } else {
      /* The new point is dominated, nothing more to do */
      trigger_update = 0;
    }
  }

  if (!trigger_update) {
    logger_biobj_node_free(node_item, NULL);
  } else {
    /* Perform tree update */
    while (next_node != NULL) {
      /* Check the dominance relation between the new node and the next node. There are only two possibilities:
       * dominance = 0: the new node and the next node are nondominated
       * dominance = 1: the new node dominates the next node */
      node = next_node;
      dominance = mo_get_dominance(node_item->y, ((logger_biobj_avl_item_t*) node->item)->y,
          logger->number_of_objectives);
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        if (compute_indicators) {
          for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            logger->indicators[i]->current_value -= ((logger_biobj_avl_item_t*) node->item)->indicator_contribution[i];
          }
        }
        next_node = node->next;
        avl_item_delete(logger->buffer_tree, node->item);
        avl_node_delete(logger->archive_tree, node);
      } else {
        break;
      }
    }

    new_node = avl_item_insert(logger->archive_tree, node_item);
    avl_item_insert(logger->buffer_tree, node_item);

    if (compute_indicators) {
      logger_biobj_check_if_within_ROI(problem, new_node);
      if (node_item->within_ROI) {
        /* Compute indicator value for new node and update the indicator value of the affected nodes */
        logger_biobj_avl_item_t *next_item, *previous_item;

        if (new_node->next != NULL) {
          next_item = (logger_biobj_avl_item_t*) new_node->next->item;
          if (next_item->within_ROI) {
            for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              logger->indicators[i]->current_value -= next_item->indicator_contribution[i];
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                next_item->indicator_contribution[i] = (node_item->y[0] - next_item->y[0]) * data->normalization_factor[0]
                    * (data->reference_point[1] - next_item->y[1]) * data->normalization_factor[1];
              } else {
                coco_error(
                    "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                    logger->indicators[i]->name);
              }
              logger->indicators[i]->current_value += next_item->indicator_contribution[i];
            }
          }
        }

        if (new_node->prev != NULL) {
          previous_item = (logger_biobj_avl_item_t*) new_node->prev->item;
          if (previous_item->within_ROI) {
            for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                node_item->indicator_contribution[i] = (previous_item->y[0] - node_item->y[0]) * data->normalization_factor[0]
                    * (data->reference_point[1] - node_item->y[1]) * data->normalization_factor[1];
              } else {
                coco_error(
                    "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                    logger->indicators[i]->name);
              }
            }
          } else {
            /* Previous item does not exist or is out of ROI, use reference point instead */
            for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                node_item->indicator_contribution[i] = (data->reference_point[0] - node_item->y[0]) * data->normalization_factor[0]
                    * (data->reference_point[1] - node_item->y[1]) * data->normalization_factor[1];
              } else {
                coco_error(
                    "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                    logger->indicators[i]->name);
              }
            }
          }
        }
        for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
          logger->indicators[i]->current_value += node_item->indicator_contribution[i];
        }
      }
    }
  }

  return trigger_update;
}

/**
 * Initializes the indicator with name indicator_name.
 */
static logger_biobj_indicator_t *logger_biobj_indicator(logger_biobj_t *logger,
                                                        coco_problem_t *problem,
                                                        const char *indicator_name) {

  coco_observer_t *observer;
  observer_biobj_t *observer_biobj;
  logger_biobj_indicator_t *indicator;
  char *prefix, *file_name, *path_name;
  char *problem_type = ((mo_problem_data_t *) coco_stacked_problem_get_data(problem))->problem_type;
  int info_file_exists = 0;

  indicator = (logger_biobj_indicator_t *) coco_allocate_memory(sizeof(*indicator));
  observer = logger->observer;
  observer_biobj = (observer_biobj_t *) observer->data;

  indicator->name = coco_strdup(indicator_name);

#if READ_BEST_VALUES_FROM_FILE /* TODO: Remove these lines when you know the best values can be read from the file... */
  indicator->best_value = observer_biobj_read_best_value(observer_biobj, indicator->name, problem->problem_id);
#else
  indicator->best_value = 1;
#endif
  indicator->next_target_id = 0;
  indicator->current_value = 0;

  /* Prepare the info file */
  path_name = (char *) coco_allocate_memory(COCO_PATH_MAX);
  memcpy(path_name, observer->output_folder, strlen(observer->output_folder) + 1);
  coco_create_path(path_name);
  file_name = coco_strdupf("%s_%s.info", problem_type, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  info_file_exists = coco_file_exists(path_name);
  indicator->info_file = fopen(path_name, "a");
  if (indicator->info_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Prepare the log file */
  path_name = (char *) coco_allocate_memory(COCO_PATH_MAX);
  memcpy(path_name, observer->output_folder, strlen(observer->output_folder) + 1);
  coco_join_path(path_name, COCO_PATH_MAX, problem_type, NULL);
  coco_create_path(path_name);
  prefix = coco_remove_from_string(problem->problem_id, "_i", "");
  file_name = coco_strdupf("%s_%s.dat", prefix, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  indicator->log_file = fopen(path_name, "a");
  if (indicator->log_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }

  /* Output header information to the info file */
  if (!info_file_exists) {
    /* Output algorithm name */
    fprintf(indicator->info_file, "algId = '%s', indicator = '%s', folder = %s\n%% %s", observer->algorithm_name,
        indicator_name, problem_type, observer->algorithm_info);
  }
  if (observer_biobj->previous_function != problem->suite_dep_function_id) {
    fprintf(indicator->info_file, "\nfuncId = %03d, %s, ", problem->suite_dep_function_id + 1, file_name);
    fprintf(indicator->info_file, "DIM = %lu", problem->number_of_variables);
  }

  coco_free_memory(prefix);
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Output header information to the log file */
  fprintf(indicator->log_file, "%%\n%% index = %ld, name = %s\n", problem->suite_dep_index, problem->problem_name);
  fprintf(indicator->log_file, "DIM = %lu, instId = %ld, bestVal = %.15f\n", problem->number_of_variables,
      problem->suite_dep_instance_id + 1, indicator->best_value);
  fprintf(indicator->log_file, "%% function evaluation | indicator value | target value\n");

  return indicator;
}

/**
 * Outputs the final information about this indicator.
 */
static void logger_biobj_indicator_finalize(logger_biobj_indicator_t *indicator, logger_biobj_t *logger) {

  fprintf(indicator->info_file, ", %ld:%lu|%+.1e", logger->suite_dep_instance_id + 1, logger->number_of_evaluations,
      indicator->best_value - indicator->current_value);
}

/**
 * Frees the memory of the given indicator.
 */
static void logger_biobj_indicator_free(void *stuff) {

  logger_biobj_indicator_t *indicator;

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

}

/**
 * Evaluates the function, increases the number of evaluations and outputs information based on observer
 * options.
 */
static void logger_biobj_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_biobj_t *logger;
  coco_observer_t *observer;
  observer_biobj_t *observer_biobj;

  logger_biobj_avl_item_t *node_item;
  logger_biobj_indicator_t *indicator;
  int update_performed, target_hit;
  size_t i;

  logger = (logger_biobj_t *) coco_transformed_get_data(problem);
  observer = logger->observer;
  observer_biobj = (observer_biobj_t *) observer->data;

  /* Evaluate function */
  coco_evaluate_function(coco_transformed_get_inner_problem(problem), x, y);
  logger->number_of_evaluations++;

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in the archive */
  node_item = logger_biobj_node_create(x, y, logger->number_of_evaluations, logger->number_of_variables,
      logger->number_of_objectives);

  update_performed = logger_biobj_tree_update(logger, coco_transformed_get_inner_problem(problem), node_item, observer_biobj->compute_indicators);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to nondom_file */
  if (update_performed && (observer_biobj->log_mode == ALL)) {
    logger_biobj_tree_output(logger->nondom_file, logger->buffer_tree, logger->number_of_variables,
        logger->number_of_objectives, observer_biobj->include_decision_variables);
    avl_tree_purge(logger->buffer_tree);

    /* Flush output so that impatient users can see progress. */
    fflush(logger->nondom_file);
  }

  /* If the archive was updated and a new target was reached for an indicator, output indicator information.
   * Note that a target is reached when the (best_value - current_value) <= relative_target_value (the
   * relative_target_value is a target for indicator difference, not indicator value!)
   */
  if (update_performed && observer_biobj->compute_indicators)
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
      target_hit = 0;
      indicator = logger->indicators[i];
      while ((indicator->next_target_id < MO_NUMBER_OF_TARGETS)
          && (node_item->indicator_contribution[i] > 0)
          && (indicator->best_value - indicator->current_value <= MO_RELATIVE_TARGET_VALUES[indicator->next_target_id])) {
        /* A target was hit */
        target_hit = 1;
        if (indicator->next_target_id + 1 < MO_NUMBER_OF_TARGETS)
          indicator->next_target_id++;
        else
          break;
      }
      if (target_hit)
        fprintf(indicator->log_file, "%lu\t%22.15e\t%22.15e\n", logger->number_of_evaluations,
            indicator->best_value - indicator->current_value, MO_RELATIVE_TARGET_VALUES[indicator->next_target_id - 1]);

    }

}

/**
 * Outputs the final nondominated solutions.
 */
static void logger_biobj_finalize(logger_biobj_t *logger) {

  coco_observer_t *observer;
  observer_biobj_t *observer_biobj;
  avl_tree_t *resorted_tree;
  avl_node_t *solution;

  observer = logger->observer;
  observer_biobj = (observer_biobj_t *) observer->data;

  /* Resort archive_tree according to time stamp and then output it */
  resorted_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

  if (logger->archive_tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = logger->archive_tree->head;
    while (solution != NULL) {
      avl_item_insert(resorted_tree, solution->item);
      solution = solution->next;
    }
  }

  logger_biobj_tree_output(logger->nondom_file, resorted_tree, logger->number_of_variables,
      logger->number_of_objectives, observer_biobj->include_decision_variables);

}

/**
 * Frees the memory of the given biobjective logger.
 */
static void logger_biobj_free(void *stuff) {

  logger_biobj_t *logger;
  coco_observer_t *observer;
  observer_biobj_t *observer_biobj;
  size_t i;

  assert(stuff != NULL);
  logger = stuff;
  observer = logger->observer;
  observer_biobj = (observer_biobj_t *) observer->data;

  if (observer_biobj->log_mode == FINAL) {
     logger_biobj_finalize(logger);
  }

  if (observer_biobj->compute_indicators) {
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
      logger_biobj_indicator_finalize(logger->indicators[i], logger);
      logger_biobj_indicator_free(logger->indicators[i]);
    }
  }

  if ((observer_biobj->log_mode != NONE) && (logger->nondom_file != NULL)) {
    fclose(logger->nondom_file);
    logger->nondom_file = NULL;
  }

  avl_tree_destruct(logger->archive_tree);
  avl_tree_destruct(logger->buffer_tree);
}

/**
 * Initializes the biobjective logger.
 */
static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *problem) {

  coco_problem_t *self;
  logger_biobj_t *logger;
  observer_biobj_t *observer_biobj;
  const char nondom_folder_name[] = "archive";
  char *path_name, *file_name, *prefix;
  size_t i;

  if (problem->number_of_objectives != 2) {
    coco_error("logger_biobj(): The biobjective logger cannot log a problem with %d objective(s)", problem->number_of_objectives);
    return NULL; /* Never reached. */
  }

  logger = coco_allocate_memory(sizeof(*logger));

  logger->observer = observer;
  logger->number_of_evaluations = 0;
  logger->number_of_variables = problem->number_of_variables;
  logger->number_of_objectives = problem->number_of_objectives;
  logger->suite_dep_instance_id = problem->suite_dep_instance_id;

  observer_biobj = (observer_biobj_t *) observer->data;

  /* Initialize logging of nondominated solutions */
  if (observer_biobj->log_mode != NONE) {

    /* Create the path to the file */
    path_name = (char *) coco_allocate_memory(COCO_PATH_MAX);
    memcpy(path_name, observer->output_folder, strlen(observer->output_folder) + 1);
    coco_join_path(path_name, COCO_PATH_MAX, nondom_folder_name, NULL);
    coco_create_path(path_name);

    /* Construct file name */
    prefix = coco_remove_from_string(problem->problem_id, "_i", "_d");
    if (observer_biobj->log_mode == ALL)
      file_name = coco_strdupf("%s_nondom_all.dat", prefix);
    else if (observer_biobj->log_mode == FINAL)
      file_name = coco_strdupf("%s_nondom_final.dat", prefix);
    coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
    coco_free_memory(prefix);

    /* Open and initialize the file */
    logger->nondom_file = fopen(path_name, "a");
    if (logger->nondom_file == NULL) {
      coco_error("logger_biobj() failed to open file '%s'.", path_name);
      return NULL; /* Never reached */
    }

    coco_free_memory(file_name);
    coco_free_memory(path_name);

    /* Output header information */
    fprintf(logger->nondom_file, "%% instance = %ld\n", problem->suite_dep_instance_id + 1);
    if (observer_biobj->include_decision_variables) {
      fprintf(logger->nondom_file, "%% function evaluation | %lu objectives | %lu variables\n",
          problem->number_of_objectives, problem->number_of_variables);
    } else {
      fprintf(logger->nondom_file, "%% function evaluation | %lu objectives \n",
          problem->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  logger->archive_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_last_objective,
      (avl_free_t) logger_biobj_node_free);
  logger->buffer_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

  self = coco_transformed_allocate(problem, logger, logger_biobj_free);
  self->evaluate_function = logger_biobj_evaluate;

  /* Initialize the indicators */
  if (observer_biobj->compute_indicators) {
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++)
      logger->indicators[i] = logger_biobj_indicator(logger, problem, OBSERVER_BIOBJ_INDICATORS[i]);

    observer_biobj->previous_function = problem->suite_dep_function_id;
  }

  return self;
}
