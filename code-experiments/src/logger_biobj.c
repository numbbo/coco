/**
 * @file logger_biobj.c
 * @brief Implementation of the bbob-biobj logger.
 *
 * Logs the values of the implemented indicators and archives nondominated solutions.
 * Produces four kinds of files:
 * - The "info" files contain high-level information on the performed experiment. One .info file is created
 * for each problem group (and indicator type) and contains information on all the problems in that problem
 * group (and indicator type).
<<<<<<< HEAD
 * - The "dat" files contain function evaluations, indicator values and target hits for every target hit and
 * first and last evaluation. One .dat file is created for each problem function and dimension (and indicator
 * type) and contains information for all instances of that problem (and indicator type).
=======
 * - The "dat" files contain function evaluations, indicator values and target hits for every target hit as
 * well as for the last evaluation. One .dat file is created for each problem function and dimension (and
 * indicator type) and contains information for all instances of that problem (and indicator type).
>>>>>>> afb187777e873a4b00b100e8eb0f1233b2611e14
 * - The "tdat" files contain function evaluation and indicator values for every predefined evaluation
 * number as well as for the last evaluation. One .tdat file is created for each problem function and
 * dimension (and indicator type) and contains information for all instances of that problem (and indicator
 * type).
 * - The "adat" files are archive files that contain function evaluations, 2 objectives and dim variables
 * for every nondominated solution. Whether these files are created, at what point in time the logger writes
 * nondominated solutions to the archive and whether the decision variables are output or not depends on
 * the values of log_nondom_mode and log_nondom_mode. See the bi-objective observer constructor
 * observer_biobj() for more information. One .adat file is created for each problem function and dimension
 * and contains information for all instances of that problem.
 *
 * @note Whenever in this file a ROI is mentioned, it means the region of interest in the objective space.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_string.c"
#include "observer_biobj.c"

#include "logger_biobj_avl_tree.c"
#include "mo_targets.c"
#include "mo_utilities.c"

/**
 * @brief The indicator type.
 */
typedef struct {

  char *name;                /**< @brief Name of the indicator used for identification and the output. */

  FILE *dat_file;            /**< @brief File for logging indicator values at predefined values. */
  FILE *tdat_file;           /**< @brief File for logging indicator values at predefined evaluations. */
  FILE *info_file;           /**< @brief File for logging summary information on algorithm performance. */

  double best_value;         /**< @brief The best known indicator value for this problem. */
  size_t next_target_id;     /**< @brief The id of the target that hasn't been hit yet. */
  int target_hit;            /**< @brief Whether the target was hit in the latest evaluation. */
  double current_value;      /**< @brief The current indicator value. */

  double additional_penalty; /**< @brief Additional penalty for solutions outside the ROI. */
  double overall_value;      /**< @brief The overall value of the indicator tested for target hits. */

} logger_biobj_indicator_t;

/**
 * @brief The bi-objective logger data type.
 *
 * @note Some fields from the observers (coco_observer as well as observer_biobj) need to be copied here
 * because the observers can be deleted before the logger is finalized and we need these fields for
 * finalization.
 */
typedef struct {
  coco_observer_t *observer;     /**< @brief Pointer to the general observer to access its fields. */

  observer_biobj_log_nondom_e log_nondom_mode; /**< @brief Mode for archiving nondominated solutions. */
  FILE *archive_file;            /**< @brief File for archiving nondominated solutions (all or final). */

  int log_vars;                  /**< @brief Whether to log the decision values. */

  int precision_x;               /**< @brief Precision for outputting decision values. */
  int precision_f;               /**< @brief Precision for outputting objective values. */

  size_t number_of_evaluations;  /**< @brief The number of evaluations performed so far. */
  size_t number_of_variables;    /**< @brief Dimension of the problem. */
  size_t number_of_objectives;   /**< @brief Number of objectives (clearly equal to 2). */
  size_t suite_dep_instance;     /**< @brief Suite-dependent instance number of the observed problem. */

  avl_tree_t *archive_tree;      /**< @brief The tree keeping currently non-dominated solutions. */
  avl_tree_t *buffer_tree;       /**< @brief The tree with pointers to nondominated solutions that haven't
                                      been logged yet. */

  /* Indicators (TODO: Implement others!) */
  int compute_indicators;        /**< @brief Whether to compute the indicators. */
  logger_biobj_indicator_t *indicators[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS];
                                 /**< @brief The implemented indicators. */

  /* TODO: Check whether all this is really needed! */

} logger_biobj_data_t;

/**
 * @brief The type for the node's item in the AVL tree.
 */
typedef struct {
  double *x;          /**< @brief The decision values of this solution. */
  double *y;          /**< @brief The values of objectives of this solution. */
  size_t time_stamp;  /**< @brief The number of evaluations when the solution was created. */

  double indicator_contribution[OBSERVER_BIOBJ_NUMBER_OF_INDICATORS];
                      /**< @brief The contribution of this solution to the overall indicator values. */
  int within_ROI;     /**< @brief Whether the solution is within the region of interest (ROI). */

} logger_biobj_avl_item_t;

/**
 * @brief Creates and returns the information on the solution in the form of a node's item in the AVL tree.
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
 * @brief Frees the data of the given logger_biobj_avl_item_t.
 */
static void logger_biobj_node_free(logger_biobj_avl_item_t *item, void *userdata) {

  coco_free_memory(item->x);
  coco_free_memory(item->y);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Checks if the given node is smaller than the nadir point, and stores this information in the node's
 * item->within_ROI field.
 */
static void logger_biobj_check_if_within_ROI(const coco_problem_t *problem, avl_node_t *node) {

  logger_biobj_avl_item_t *node_item = (logger_biobj_avl_item_t *) node->item;
  size_t i;

  node_item->within_ROI = 1;
  for (i = 0; i < problem->number_of_objectives; i++)
    if (node_item->y[i] > problem->nadir_value[i]) {
      node_item->within_ROI = 0;
      break;
    }

  if (!node_item->within_ROI)
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++)
      node_item->indicator_contribution[i] = 0;

  return;
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the value of the last objective.
 *
 * @note This ordering is used by the archive_tree.
 */
static int avl_tree_compare_by_last_objective(const logger_biobj_avl_item_t *item1,
                                              const logger_biobj_avl_item_t *item2,
                                              void *userdata) {
  if (item1->y[1] < item2->y[1])
    return -1;
  else if (item1->y[1] > item2->y[1])
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the time stamp.
 *
 * @note This ordering is used by the buffer_tree.
 */
static int avl_tree_compare_by_time_stamp(const logger_biobj_avl_item_t *item1,
                                          const logger_biobj_avl_item_t *item2,
                                          void *userdata) {
  if (item1->time_stamp < item2->time_stamp)
    return -1;
  else if (item1->time_stamp > item2->time_stamp)
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Outputs the AVL tree to the given file. Returns the number of nodes in the tree.
 */
static size_t logger_biobj_tree_output(FILE *file,
                                       const avl_tree_t *tree,
                                       const size_t dim,
                                       const size_t num_obj,
                                       const int log_vars,
                                       const int precision_x,
                                       const int precision_f) {

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
        fprintf(file, "%.*e\t", precision_f, ((logger_biobj_avl_item_t*) solution->item)->y[j]);
      if (log_vars) {
        for (i = 0; i < dim; i++)
          fprintf(file, "%.*e\t", precision_x, ((logger_biobj_avl_item_t*) solution->item)->x[i]);
      }
      fprintf(file, "\n");
      solution = solution->next;
      number_of_nodes++;
    }
  }

  return number_of_nodes;
}

/**
 * @brief Updates the archive and buffer trees with the given node.
 *
 * Checks for domination and updates the archive tree and the values of the indicators if the given node is
 * not weakly dominated by existing nodes in the archive tree. This is where the main computation of
 * indicator values takes place.
 *
 * @return 1 if the update was performed and 0 otherwise.
 */
static int logger_biobj_tree_update(logger_biobj_data_t *logger,
                                    const coco_problem_t *problem,
                                    logger_biobj_avl_item_t *node_item) {

  avl_node_t *node, *next_node, *new_node;
  int trigger_update = 0;
  int dominance;
  size_t i;
  int previous_unavailable = 0;

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
        if (logger->compute_indicators) {
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
        if (logger->compute_indicators) {
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

    if (logger->compute_indicators) {
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
                next_item->indicator_contribution[i] = (node_item->y[0] - next_item->y[0])
                    / (problem->nadir_value[0] - problem->best_value[0])
                    * (problem->nadir_value[1] - next_item->y[1])
                    / (problem->nadir_value[1] - problem->best_value[1]);
              } else {
                coco_error(
                    "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                    logger->indicators[i]->name);
              }
              logger->indicators[i]->current_value += next_item->indicator_contribution[i];
            }
          }
        }

        previous_unavailable = 0;
        if (new_node->prev != NULL) {
          previous_item = (logger_biobj_avl_item_t*) new_node->prev->item;
          if (previous_item->within_ROI) {
            for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                node_item->indicator_contribution[i] = (previous_item->y[0] - node_item->y[0])
                    / (problem->nadir_value[0] - problem->best_value[0])
                    * (problem->nadir_value[1] - node_item->y[1])
                    / (problem->nadir_value[1] - problem->best_value[1]);
              } else {
                coco_error(
                    "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                    logger->indicators[i]->name);
              }
            }
          } else {
            previous_unavailable = 1;
          }
        } else {
          previous_unavailable = 1;
        }

        if (previous_unavailable) {
          /* Previous item does not exist or is out of ROI, use reference point instead */
          for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
              node_item->indicator_contribution[i] = (problem->nadir_value[0] - node_item->y[0])
                  / (problem->nadir_value[0] - problem->best_value[0])
                  * (problem->nadir_value[1] - node_item->y[1])
                  / (problem->nadir_value[1] - problem->best_value[1]);
            } else {
              coco_error(
                  "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                  logger->indicators[i]->name);
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
 * @brief Initializes the indicator with name indicator_name.
 *
 * Opens files for writing and resets counters.
 */
static logger_biobj_indicator_t *logger_biobj_indicator(const logger_biobj_data_t *logger,
                                                        const coco_problem_t *problem,
                                                        const char *indicator_name) {

  coco_observer_t *observer;
  observer_biobj_data_t *observer_biobj;
  logger_biobj_indicator_t *indicator;
  char *prefix, *file_name, *path_name;
  int info_file_exists = 0;

  indicator = (logger_biobj_indicator_t *) coco_allocate_memory(sizeof(*indicator));
  observer = logger->observer;
  observer_biobj = (observer_biobj_data_t *) observer->data;

  indicator->name = coco_strdup(indicator_name);

  indicator->best_value = suite_biobj_get_best_value(indicator->name, problem->problem_id);
  indicator->next_target_id = 0;
  indicator->target_hit = 0;
  indicator->current_value = 0;
  indicator->additional_penalty = 0;
  indicator->overall_value = 0;

  /* Prepare the info file */
  path_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_create_directory(path_name);
  file_name = coco_strdupf("%s_%s.info", problem->problem_type, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  info_file_exists = coco_file_exists(path_name);
  indicator->info_file = fopen(path_name, "a");
  if (indicator->info_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Prepare the tdat file */
  path_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_join_path(path_name, COCO_PATH_MAX, problem->problem_type, NULL);
  coco_create_directory(path_name);
  prefix = coco_remove_from_string(problem->problem_id, "_i", "_d");
  file_name = coco_strdupf("%s_%s.tdat", prefix, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  indicator->tdat_file = fopen(path_name, "a");
  if (indicator->tdat_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Prepare the dat file */
  path_name = coco_allocate_string(COCO_PATH_MAX);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_join_path(path_name, COCO_PATH_MAX, problem->problem_type, NULL);
  coco_create_directory(path_name);
  file_name = coco_strdupf("%s_%s.dat", prefix, indicator_name);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  indicator->dat_file = fopen(path_name, "a");
  if (indicator->dat_file == NULL) {
    coco_error("logger_biobj_indicator() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }

  /* Output header information to the info file */
  if (!info_file_exists) {
    /* Output algorithm name */
    fprintf(indicator->info_file, "algorithm = '%s', indicator = '%s', folder = '%s'\n%% %s", observer->algorithm_name,
        indicator_name, problem->problem_type, observer->algorithm_info);
  }
  if (observer_biobj->previous_function != problem->suite_dep_function) {
    fprintf(indicator->info_file, "\nfunction = %2lu, ", problem->suite_dep_function);
    fprintf(indicator->info_file, "dim = %2lu, ", problem->number_of_variables);
    fprintf(indicator->info_file, "%s", file_name);
  }

  coco_free_memory(prefix);
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Output header information to the dat file */
  fprintf(indicator->dat_file, "%%\n%% index = %ld, name = %s\n", problem->suite_dep_index, problem->problem_name);
  fprintf(indicator->dat_file, "%% instance = %ld, reference value = %.*e\n", problem->suite_dep_instance,
      logger->precision_f, indicator->best_value);
  fprintf(indicator->dat_file, "%% function evaluation | indicator value | target hit\n");

  /* Output header information to the tdat file */
  fprintf(indicator->tdat_file, "%%\n%% index = %ld, name = %s\n", problem->suite_dep_index, problem->problem_name);
  fprintf(indicator->tdat_file, "%% instance = %ld, reference value = %.*e\n", problem->suite_dep_instance,
      logger->precision_f, indicator->best_value);
  fprintf(indicator->tdat_file, "%% function evaluation | indicator value\n");

  return indicator;
}

/**
 * @brief Outputs the final information about this indicator.
 */
static void logger_biobj_indicator_finalize(logger_biobj_indicator_t *indicator, const logger_biobj_data_t *logger) {

  size_t target_index = 0;
  if (indicator->next_target_id > 0)
    target_index = indicator->next_target_id - 1;

  /* Log the last evaluation in the dat file if wasn't already logged */
  if (!indicator->target_hit) {
    fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
        indicator->overall_value, logger->precision_f, mo_relateive_target_values[target_index]);
  }

  /* Log the last evaluation in the tdat file if wasn't already logged */
  if (!coco_observer_evaluation_to_log(logger->number_of_evaluations, logger->number_of_variables)) {
    fprintf(indicator->tdat_file, "%lu\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
        indicator->overall_value);
  }

  /* Log the information in the info file */
  fprintf(indicator->info_file, ", %ld:%lu|%.1e", logger->suite_dep_instance, logger->number_of_evaluations,
      indicator->overall_value);
  fflush(indicator->info_file);
}

/**
 * @brief Frees the memory of the given indicator.
 */
static void logger_biobj_indicator_free(void *stuff) {

  logger_biobj_indicator_t *indicator;

  assert(stuff != NULL);
  indicator = (logger_biobj_indicator_t *) stuff;

  if (indicator->name != NULL) {
    coco_free_memory(indicator->name);
    indicator->name = NULL;
  }

  if (indicator->dat_file != NULL) {
    fclose(indicator->dat_file);
    indicator->dat_file = NULL;
  }

  if (indicator->tdat_file != NULL) {
    fclose(indicator->tdat_file);
    indicator->tdat_file = NULL;
  }

  if (indicator->info_file != NULL) {
    fclose(indicator->info_file);
    indicator->info_file = NULL;
  }

  coco_free_memory(stuff);

}

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information based on the
 * observer options.
 */
static void logger_biobj_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_biobj_data_t *logger;

  logger_biobj_avl_item_t *node_item;
  logger_biobj_indicator_t *indicator;
  avl_node_t *solution;
  int update_performed;
  size_t i;

  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);

  /* Evaluate function */
  coco_evaluate_function(coco_problem_transformed_get_inner_problem(problem), x, y);
  logger->number_of_evaluations++;

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in
   * the archive */
  node_item = logger_biobj_node_create(x, y, logger->number_of_evaluations, logger->number_of_variables,
      logger->number_of_objectives);

  update_performed = logger_biobj_tree_update(logger, coco_problem_transformed_get_inner_problem(problem),
      node_item);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to
   * nondom_file */
  if (update_performed && (logger->log_nondom_mode == LOG_NONDOM_ALL)) {
    logger_biobj_tree_output(logger->archive_file, logger->buffer_tree, logger->number_of_variables,
        logger->number_of_objectives, logger->log_vars, logger->precision_x, logger->precision_f);
    avl_tree_purge(logger->buffer_tree);

    /* Flush output so that impatient users can see progress. */
    fflush(logger->archive_file);
  }

  /* Perform output to the:
   * - dat file, if the archive was updated and a new target was reached for an indicator;
   * - tdat file, if the number of evaluations matches one of the predefined numbers.
   *
   * Note that a target is reached when the (best_value - current_value) <= relative_target_value
   * The relative_target_value is a target for indicator difference, not indicator value!
   */
  if (logger->compute_indicators) {
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {

      indicator = logger->indicators[i];
      indicator->target_hit = 0;

      /* If the update was performed, update the overall indicator value */
      if (update_performed) {
        /* Compute the overall_value of an indicator */
        if (strcmp(indicator->name, "hyp") == 0) {
          if (indicator->current_value == 0) {
            /* The additional penalty for hypervolume is the minimal distance from the nondominated set to the ROI */
            indicator->additional_penalty = DBL_MAX;
            if (logger->archive_tree->tail) {
              solution = logger->archive_tree->head;
              while (solution != NULL) {
                double distance = mo_get_distance_to_ROI(((logger_biobj_avl_item_t*) solution->item)->y,
                    problem->best_value, problem->nadir_value, problem->number_of_objectives);
                indicator->additional_penalty = coco_double_min(indicator->additional_penalty, distance);
                solution = solution->next;
              }
            }
            assert(indicator->additional_penalty >= 0);
          } else {
            indicator->additional_penalty = 0;
          }
          indicator->overall_value = indicator->best_value - indicator->current_value
              + indicator->additional_penalty;
        } else {
          coco_error("logger_biobj_evaluate(): Indicator computation not implemented yet for indicator %s",
              indicator->name);
        }

        /* Check whether a target was hit */
        while ((indicator->next_target_id < MO_NUMBER_OF_TARGETS)
            && (indicator->overall_value <= mo_relateive_target_values[indicator->next_target_id])) {
          /* A target was hit */
          indicator->target_hit = 1;
          if (indicator->next_target_id + 1 < MO_NUMBER_OF_TARGETS)
            indicator->next_target_id++;
          else
            break;
        }
      }

      /* Log to the dat file if a target was hit */
      if (indicator->target_hit) {
        fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
            indicator->overall_value, logger->precision_f,
            mo_relateive_target_values[indicator->next_target_id - 1]);
      }

      /* Log to the tdat file if the number of evaluations matches one of the predefined numbers */
      if (coco_observer_evaluation_to_log(logger->number_of_evaluations, logger->number_of_variables)) {
        fprintf(indicator->tdat_file, "%lu\t%.*e\n", logger->number_of_evaluations, logger->precision_f,
            indicator->overall_value);
      }

    }
  }
}

/**
 * @brief Outputs the final nondominated solutions to the archive file.
 */
static void logger_biobj_finalize(logger_biobj_data_t *logger) {

  avl_tree_t *resorted_tree;
  avl_node_t *solution;

  /* Re-sort archive_tree according to time stamp and then output it */
  resorted_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

  if (logger->archive_tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = logger->archive_tree->head;
    while (solution != NULL) {
      avl_item_insert(resorted_tree, solution->item);
      solution = solution->next;
    }
  }

  logger_biobj_tree_output(logger->archive_file, resorted_tree, logger->number_of_variables,
      logger->number_of_objectives, logger->log_vars, logger->precision_x, logger->precision_f);

  avl_tree_destruct(resorted_tree);
}

/**
 * @brief Frees the memory of the given biobjective logger.
 */
static void logger_biobj_free(void *stuff) {

  logger_biobj_data_t *logger;
  size_t i;

  assert(stuff != NULL);
  logger = (logger_biobj_data_t *) stuff;

  if (logger->log_nondom_mode == LOG_NONDOM_FINAL) {
     logger_biobj_finalize(logger);
  }

  if (logger->compute_indicators) {
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
      logger_biobj_indicator_finalize(logger->indicators[i], logger);
      logger_biobj_indicator_free(logger->indicators[i]);
    }
  }

  if ((logger->log_nondom_mode != LOG_NONDOM_NONE) && (logger->archive_file != NULL)) {
    fclose(logger->archive_file);
    logger->archive_file = NULL;
  }

  avl_tree_destruct(logger->archive_tree);
  avl_tree_destruct(logger->buffer_tree);

}

/**
 * @brief Initializes the biobjective logger.
 */
static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *inner_problem) {

  coco_problem_t *problem;
  logger_biobj_data_t *logger_biobj;
  observer_biobj_data_t *observer_biobj;
  const char nondom_folder_name[] = "archive";
  char *path_name, *file_name = NULL, *prefix;
  size_t i;

  if (inner_problem->number_of_objectives != 2) {
    coco_error("logger_biobj(): The bi-objective logger cannot log a problem with %d objective(s)", inner_problem->number_of_objectives);
    return NULL; /* Never reached. */
  }

  logger_biobj = (logger_biobj_data_t *) coco_allocate_memory(sizeof(*logger_biobj));

  logger_biobj->observer = observer;

  logger_biobj->number_of_evaluations = 0;
  logger_biobj->number_of_variables = inner_problem->number_of_variables;
  logger_biobj->number_of_objectives = inner_problem->number_of_objectives;
  logger_biobj->suite_dep_instance = inner_problem->suite_dep_instance;

  observer_biobj = (observer_biobj_data_t *) observer->data;
  /* Copy values from the observes that you might need even if they do not exist any more */
  logger_biobj->log_nondom_mode = observer_biobj->log_nondom_mode;
  logger_biobj->compute_indicators = observer_biobj->compute_indicators;
  logger_biobj->precision_x = observer->precision_x;
  logger_biobj->precision_f = observer->precision_f;

  if (((observer_biobj->log_vars_mode == LOG_VARS_LOW_DIM) && (inner_problem->number_of_variables > 5))
      || (observer_biobj->log_vars_mode == LOG_VARS_NEVER))
    logger_biobj->log_vars = 0;
  else
    logger_biobj->log_vars = 1;

  /* Initialize logging of nondominated solutions into the archive file */
  if (logger_biobj->log_nondom_mode != LOG_NONDOM_NONE) {

    /* Create the path to the file */
    path_name = coco_allocate_string(COCO_PATH_MAX);
    memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
    coco_join_path(path_name, COCO_PATH_MAX, nondom_folder_name, NULL);
    coco_create_directory(path_name);

    /* Construct file name */
    prefix = coco_remove_from_string(inner_problem->problem_id, "_i", "_d");
    if (logger_biobj->log_nondom_mode == LOG_NONDOM_ALL)
      file_name = coco_strdupf("%s_nondom_all.adat", prefix);
    else if (logger_biobj->log_nondom_mode == LOG_NONDOM_FINAL)
      file_name = coco_strdupf("%s_nondom_final.adat", prefix);
    coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
    if (logger_biobj->log_nondom_mode != LOG_NONDOM_NONE)
      coco_free_memory(file_name);
    coco_free_memory(prefix);

    /* Open and initialize the archive file */
    logger_biobj->archive_file = fopen(path_name, "a");
    if (logger_biobj->archive_file == NULL) {
      coco_error("logger_biobj() failed to open file '%s'.", path_name);
      return NULL; /* Never reached */
    }
    coco_free_memory(path_name);

    /* Output header information */
    fprintf(logger_biobj->archive_file, "%% instance = %ld, name = %s\n", inner_problem->suite_dep_instance, inner_problem->problem_name);
    if (logger_biobj->log_vars) {
      fprintf(logger_biobj->archive_file, "%% function evaluation | %lu objectives | %lu variables\n",
          inner_problem->number_of_objectives, inner_problem->number_of_variables);
    } else {
      fprintf(logger_biobj->archive_file, "%% function evaluation | %lu objectives \n",
          inner_problem->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  logger_biobj->archive_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_last_objective,
      (avl_free_t) logger_biobj_node_free);
  logger_biobj->buffer_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_time_stamp, NULL);

  problem = coco_problem_transformed_allocate(inner_problem, logger_biobj, logger_biobj_free);
  problem->evaluate_function = logger_biobj_evaluate;

  /* Initialize the indicators */
  if (logger_biobj->compute_indicators) {
    for (i = 0; i < OBSERVER_BIOBJ_NUMBER_OF_INDICATORS; i++)
      logger_biobj->indicators[i] = logger_biobj_indicator(logger_biobj, inner_problem, observer_biobj_indicators[i]);

    observer_biobj->previous_function = (long) inner_problem->suite_dep_function;
  }

  return problem;
}
