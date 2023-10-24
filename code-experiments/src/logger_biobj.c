/**
 * @file logger_biobj.c
 * @brief Implementation of the bbob-biobj logger.
 *
 * Logs the performance of an optimizer on bi-objective problems with or without constraints and with or
 * without knowing their true Pareto front (and set). Uses the hypervolume indicator, but supports
 * inclusion of other indicators. Archives nondominated solutions.
 *
 * In constrained problems only the feasible solutions are logged (except for the first one).
 *
 * Produces these files:
 * - The .info files contain high-level information on the performed experiment.
 * - The .dat files contain function evaluations, indicator values and target hits for every performance
 * target hit as well as for the last evaluation. Logarithmic targets are used only if an estimation of the
 * optimal indicator value is known. If it is not, linear targets are used instead (this is decided on the
 * suite level and passed to the logger during initialization.
 * - The .tdat files contain function evaluation and indicator values for every predefined evaluation
 * number as well as for the last evaluation.
 * - The .rdat files contain function evaluation and indicator values for each restart of the algorithm.
 * - The .mdat file contains function evaluation and indicator values for each recommended solution.
 * - The .adat files are archive files that contain function evaluations, 2 objectives and dim variables
 * for every nondominated solution. Whether these files are created, at what point in time the logger writes
 * nondominated solutions to the archive and whether the decision variables are output or not depends on
 * the values of log_nondom_mode and log_nondom_mode. See the bi-objective observer constructor
 * observer_biobj() for more information.
 *
 * One .info file is created for each problem group (and indicator type) and contains information on all
 * the problems in that problem group (and indicator type).
 * One .dat, .tdat and .rdat file is created for each problem function and dimension (and indicator type)
 * and contains information for all instances of that problem (and indicator type).
 * One .mdat file is created for each problem function and dimension and contains information for all
 * instances of that problem.
 * One .adat file is created for each problem function, dimension and instance.
 *
 * @note Whenever in this file a ROI is mentioned, it means the (normalized) region of interest in the
 * objective space. The non-normalized ROI is a rectangle with the ideal and nadir points as its two
 * opposite vertices, while the normalized ROI is the square [0, 1]^2. If not specifically mentioned, the
 * normalized ROI is assumed.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "coco.h"
#include "coco_internal.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_string.c"
#include "mo_avl_tree.c"
#include "observer_biobj.c"

#include "mo_utilities.c"

/** @brief Number of implemented indicators */
#define LOGGER_BIOBJ_NUMBER_OF_INDICATORS 1

/** @brief Names of implemented indicators
 *
 * "hyp" stands for the hypervolume indicator.
 * */
const char *logger_biobj_indicators[LOGGER_BIOBJ_NUMBER_OF_INDICATORS] = { "hyp" };

/**
 * @brief The indicator type.
 *
 * <B> The hypervolume indicator ("hyp") </B>
 *
 * The hypervolume indicator measures the volume of the portion of the ROI in the objective space that is
 * dominated by the current Pareto front approximation. Instead of logging the hypervolume indicator value,
 * this implementation logs the difference between the best know hypervolume indicator (a value stored in
 * best_value) and the hypervolume indicator of the current Pareto front approximation (current_value). The
 * current_value equals 0 if no solution is located in the ROI. In order to be able to register the
 * performance of an optimizer even before the ROI is reached, an additional value is computed when no
 * solutions are located inside the ROI. This value is stored in additional_penalty and equals the
 * normalized distance to the ROI of the solution closest to the ROI (additional_penalty is set to 0 as
 * soon as a solution reaches the ROI). The final value to be logged (overall_value) is therefore computed
 * in the following way:
 *
 * overall_value = best_value - current_value + additional_penalty
 *
 * If the suite does not provide an estimation for the best hypervolume value, best_value is set to 1.0.
 *
 * @note Other indicators are yet to be implemented.
 */
typedef struct {

  char *name;                /**< @brief Name of the indicator used for identification and the output. */

  FILE *info_file;           /**< @brief File for logging summary information on algorithm performance. */
  FILE *dat_file;            /**< @brief File for logging indicator values at predefined values */
  FILE *tdat_file;           /**< @brief File for logging indicator values at predefined evaluations. */
  FILE *rdat_file;           /**< @brief File for logging restart information */

  int target_hit;            /**< @brief Whether the performance target was hit in the latest evaluation. */
  coco_observer_targets_t *targets;
                             /**< @brief Triggers based on performance target values. */
  int evaluation_logged;     /**< @brief Whether the whether the latest evaluation was logged. */
  coco_observer_evaluations_t *evaluations;
                             /**< @brief Triggers based on the number of evaluations. */

  double best_value;         /**< @brief The best known indicator value for this problem. */
  double current_value;      /**< @brief The current indicator value. */
  double additional_penalty; /**< @brief Additional penalty for solutions outside the ROI. */
  double overall_value;      /**< @brief The overall value of the indicator tested for target hits. */
  double previous_value;     /**< @brief The previous overall value of the indicator. */

} logger_biobj_indicator_t;

/**
 * @brief The bi-objective logger data type.
 *
 * @note Some fields from the observers (coco_observer as well as observer_biobj) need to be copied here
 * because the observers can be deleted before the logger is finalized and we need these fields for
 * finalization.
 */
typedef struct {
  coco_observer_t *observer;          /**< @brief Pointer to the observer (might be NULL at the end) */

  observer_biobj_log_nondom_e log_nondom_mode;
                                      /**< @brief Mode for archiving nondominated solutions. */
  FILE *adat_file;                    /**< @brief File for archiving nondominated solutions (all or final). */
  FILE *mdat_file;                    /**< @brief File for logging recommended solutions */

  int log_vars;                       /**< @brief Whether to log the decision values. */
  int precision_x;                    /**< @brief Precision for outputting decision values. */
  int precision_f;                    /**< @brief Precision for outputting objective values. */
  int log_discrete_as_int;            /**< @brief Whether to output discrete variables in int or double format. */
  int algorithm_restarted;            /**< @brief Whether the algorithm has restarted (output information to .rdat file). */

  size_t num_func_evaluations;        /**< @brief The number of function evaluations performed so far. */
  size_t num_cons_evaluations;        /**< @brief The number of evaluations of constraints performed so far. */
  size_t number_of_variables;         /**< @brief Dimension of the problem. */
  size_t number_of_integer_variables; /**< @brief Number of integer variables. */
  size_t number_of_objectives;        /**< @brief Number of objectives (clearly equal to 2). */
  size_t suite_dep_instance;          /**< @brief Suite-dependent instance number of the observed problem. */

  size_t previous_evaluations;        /**< @brief The number of evaluations from the previous call to the logger. */

  avl_tree_t *archive_tree;           /**< @brief The tree keeping currently non-dominated solutions. */
  avl_tree_t *buffer_tree;            /**< @brief The tree with pointers to nondominated solutions that haven't
                                           been logged yet. */

  /* TODO: Implement other indicators */
  int compute_indicators;             /**< @brief Whether to compute the indicators. */
  logger_biobj_indicator_t *indicators[LOGGER_BIOBJ_NUMBER_OF_INDICATORS];
                                      /**< @brief The implemented indicators. */
} logger_biobj_data_t;

/**
 * @brief The type for the node's item in the AVL tree as used by the bi-objective logger.
 *
 * Contains information on the exact objective values (y) and their rounded normalized values (normalized_y).
 * The exact values are used for output, while archive update and indicator computation use the normalized
 * values.
 */
typedef struct {
  double *x;                 /**< @brief The decision values of this solution. */
  double *y;                 /**< @brief The values of objectives of this solution. */
  int is_feasible;           /**< @brief Whether the solution is feasible. */
  double *normalized_y;      /**< @brief The values of normalized objectives of this solution. */
  size_t evaluation_number;  /**< @brief The evaluation number of when the solution was created. */

  double indicator_contribution[LOGGER_BIOBJ_NUMBER_OF_INDICATORS];
                      /**< @brief The contribution of this solution to the overall indicator values. */
  int within_ROI;     /**< @brief Whether the solution is within the region of interest (ROI). */

} logger_biobj_avl_item_t;

/**
 * @brief Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static logger_biobj_avl_item_t* logger_biobj_node_create(const coco_problem_t *problem,
                                                         const double *x,
                                                         const double *y,
                                                         const double *constraints,
                                                         const size_t evaluation_number,
                                                         const size_t dim,
                                                         const size_t num_obj,
                                                         const size_t num_const) {

  size_t i;
  double sum_constraints = 0;

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

  /* Compute the normalized y */
  item->normalized_y = mo_normalize(item->y, problem->best_value, problem->nadir_value, num_obj);
  item->within_ROI = mo_is_within_ROI(item->normalized_y, num_obj);

  item->evaluation_number = evaluation_number;
  for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++)
    item->indicator_contribution[i] = 0;

  /* Determine whether the solution is feasible */
  item->is_feasible = 1;
  if (num_const > 0) {
    assert(constraints != NULL);
    for (i = 0; i < num_const; i++) {
      if (constraints[i] > 0)
        sum_constraints += constraints[i];
    }
    item->is_feasible = !(sum_constraints > 0);
  }

  return item;
}

/**
 * @brief Frees the data of the given logger_biobj_avl_item_t.
 */
static void logger_biobj_node_free(logger_biobj_avl_item_t *item, void *userdata) {

  coco_free_memory(item->x);
  coco_free_memory(item->y);
  coco_free_memory(item->normalized_y);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the value of the last objective.
 *
 * @note This ordering is used by the archive_tree.
 */
static int avl_tree_compare_by_last_objective(const logger_biobj_avl_item_t *item1,
                                              const logger_biobj_avl_item_t *item2,
                                              void *userdata) {
  (void) userdata; /* To silence the compiler */
  if (coco_double_almost_equal(item1->normalized_y[1], item2->normalized_y[1], mo_precision))
    return 0;
  else if (item1->normalized_y[1] < item2->normalized_y[1])
    return -1;
  else
    return 1;
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the evaluation number (the time when the nodes were
 * created).
 *
 * @note This ordering is used by the buffer_tree.
 */
static int avl_tree_compare_by_eval_number(const logger_biobj_avl_item_t *item1,
                                           const logger_biobj_avl_item_t *item2,
                                           void *userdata) {
  (void) userdata; /* To silence the compiler */
  if (item1->evaluation_number < item2->evaluation_number)
    return -1;
  else if (item1->evaluation_number > item2->evaluation_number)
    return 1;
  else
    return 0;
}

/**
 * @brief Outputs the AVL tree to the given file. Returns the number of nodes in the tree.
 */
static size_t logger_biobj_tree_output(FILE *file,
                                       const avl_tree_t *tree,
                                       const size_t dim,
                                       const size_t num_int_vars,
                                       const size_t num_obj,
                                       const int log_vars,
                                       const int precision_x,
                                       const int precision_f,
                                       const int log_discrete_as_int) {

  avl_node_t *solution;
  size_t i;
  size_t j;
  size_t number_of_nodes = 0;

  if (tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = tree->head;
    while (solution != NULL) {
      fprintf(file, "%lu\t", (unsigned long) ((logger_biobj_avl_item_t*) solution->item)->evaluation_number);
      for (j = 0; j < num_obj; j++)
        fprintf(file, "%.*e\t", precision_f, ((logger_biobj_avl_item_t*) solution->item)->y[j]);
      if (log_vars) {
        for (i = 0; i < dim; i++)
          if ((i < num_int_vars) && (log_discrete_as_int))
            fprintf(file, "%d\t", coco_double_to_int(((logger_biobj_avl_item_t*) solution->item)->x[i]));
          else
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
                                    logger_biobj_avl_item_t *node_item) {

  avl_node_t *node, *next_node, *new_node;
  int trigger_update = 0;
  int dominance;
  size_t i;
  int previous_unavailable = 0;

  /* If the node contains an infeasible solution, exit immediately (do not update the tree) */
  if (node_item->is_feasible == 0)
    return 0;

  /* Find the first point that is not worse than the new point (NULL if such point does not exist) */
  node = avl_item_search_right(logger->archive_tree, node_item, NULL);

  if (node == NULL) {
    /* The new point is an extreme point */
    trigger_update = 1;
    next_node = logger->archive_tree->head;
  } else {
    dominance = mo_get_dominance(node_item->normalized_y,
        ((logger_biobj_avl_item_t*) node->item)->normalized_y, logger->number_of_objectives);
    if (dominance > -1) {
      trigger_update = 1;
      next_node = node->next;
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        if (logger->compute_indicators) {
          for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            logger->indicators[i]->current_value -= ((logger_biobj_avl_item_t*) node->item)->indicator_contribution[i];
          }
        }
        avl_item_delete(logger->buffer_tree, node->item);
        avl_node_delete(logger->archive_tree, node);
      }
    } else {
      /* The new point is dominated or equal to an existing one, nothing more to do */
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
      dominance = mo_get_dominance(node_item->normalized_y,
          ((logger_biobj_avl_item_t*) node->item)->normalized_y, logger->number_of_objectives);
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point */
        if (logger->compute_indicators) {
          for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
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
    assert(new_node != NULL);
    avl_item_insert(logger->buffer_tree, node_item);

    if (logger->compute_indicators) {
      if (node_item->within_ROI) {
        /* Compute indicator value for new node and update the indicator value of the affected nodes */
        logger_biobj_avl_item_t *next_item, *previous_item;

        if (new_node->next != NULL) {
          next_item = (logger_biobj_avl_item_t*) new_node->next->item;
          if (next_item->within_ROI) {
            for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              logger->indicators[i]->current_value -= next_item->indicator_contribution[i];
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                next_item->indicator_contribution[i] = (node_item->normalized_y[0] - next_item->normalized_y[0])
                    * (1 - next_item->normalized_y[1]);
                assert(next_item->indicator_contribution[i] >= 0);
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
            for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
              if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
                node_item->indicator_contribution[i] = (previous_item->normalized_y[0] - node_item->normalized_y[0])
                    * (1 - node_item->normalized_y[1]);
                assert(node_item->indicator_contribution[i] >= 0);
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
          for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
            if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
              node_item->indicator_contribution[i] = (1 - node_item->normalized_y[0])
                  * (1 - node_item->normalized_y[1]);
              assert(node_item->indicator_contribution[i] >= 0);
            } else {
              coco_error(
                  "logger_biobj_tree_update(): Indicator computation not implemented yet for indicator %s",
                  logger->indicators[i]->name);
            }
          }
        }

        for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
          if (strcmp(logger->indicators[i]->name, "hyp") == 0) {
            assert(node_item->indicator_contribution[i] >= 0);
            logger->indicators[i]->current_value += node_item->indicator_contribution[i];
          }
        }
      }
    }
  }

  return trigger_update;
}

/**
 * @brief Creates and initializes one of the *dat files (.dat, .tdat or .rdat).
 */
static void logger_biobj_indicator_initialize_file(const logger_biobj_data_t *logger,
                                                   const coco_observer_t *observer,
                                                   const coco_problem_t *problem,
                                                   const logger_biobj_indicator_t *indicator,
                                                   FILE **f,
                                                   const char *file_ending,
                                                   const int output_targets) {
  char *prefix, *file_name, *path_name;
  static const char *header_w_targets = "%%\n"
      "%% index = %lu, name = %s\n"
      "%% instance = %lu, reference value = %.*e\n"
      "%% function evaluation | indicator value | target hit\n";
  static const char *header_wo_targets = "%%\n"
      "%% index = %lu, name = %s\n"
      "%% instance = %lu, reference value = %.*e\n"
      "%% function evaluation | indicator value\n";

  /* Create and open the file */
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_join_path(path_name, COCO_PATH_MAX, problem->problem_type, NULL);
  coco_create_directory(path_name);
  prefix = coco_remove_from_string(problem->problem_id, "_i", "_d");
  file_name = coco_strdupf("%s_%s.%s", prefix, indicator->name, file_ending);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  *f = fopen(path_name, "a");
  if (*f == NULL) {
    coco_error("logger_biobj_indicator_initialize_file() failed to open file '%s'.", path_name);
  }
  coco_free_memory(prefix);
  coco_free_memory(file_name);
  coco_free_memory(path_name);

  /* Output header */
  if (output_targets)
    fprintf(*f, header_w_targets, (unsigned long) problem->suite_dep_index, problem->problem_name,
        (unsigned long) problem->suite_dep_instance, logger->precision_f, indicator->best_value);
  else
    fprintf(*f, header_wo_targets, (unsigned long) problem->suite_dep_index, problem->problem_name,
        (unsigned long) problem->suite_dep_instance, logger->precision_f, indicator->best_value);

}

/**
 * @brief Initializes the indicator with name indicator_name.
 *
 * Opens files for writing and resets counters.
 */
static logger_biobj_indicator_t *logger_biobj_indicator(const logger_biobj_data_t *logger,
                                                        const coco_observer_t *observer,
                                                        const coco_problem_t *problem,
                                                        const char *indicator_name) {

  observer_biobj_data_t *observer_data;
  logger_biobj_indicator_t *indicator;
  coco_suite_t *suite;
  char *prefix, *file_name, *path_name;
  int info_file_exists = 0;

  coco_debug("Started logger_biobj_indicator()");

  indicator = (logger_biobj_indicator_t *) coco_allocate_memory(sizeof(*indicator));
  assert(observer);
  assert(observer->data);
  observer_data = (observer_biobj_data_t *) observer->data;

  indicator->name = coco_strdup(indicator_name);

  assert(problem->suite);
  suite = (coco_suite_t *)problem->suite;
  indicator->target_hit = 0;
  indicator->evaluation_logged = 0;
  indicator->current_value = 0;
  indicator->additional_penalty = DBL_MAX;
  indicator->overall_value = 0;
  indicator->previous_value = 0;

  coco_suite_get_best_indicator_value(suite->known_optima, problem, indicator->name, &(indicator->best_value));

  indicator->targets = coco_observer_targets(suite->known_optima, observer->lin_target_precision,
      observer->number_target_triggers, observer->log_target_precision);
  indicator->evaluations = coco_observer_evaluations(observer->base_evaluation_triggers, problem->number_of_variables);

  /* Prepare the info file */
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
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

  /* Output header information to the info file */
  prefix = coco_remove_from_string(problem->problem_id, "_i", "_d");
  if (!info_file_exists) {
    /* Output algorithm name */
    assert(problem->suite);
    fprintf(indicator->info_file,
        "suite = '%s', algorithm = '%s', indicator = '%s', folder = '%s', coco_version = '%s'\n%% %s",
        problem->suite->suite_name, observer->algorithm_name, indicator_name, problem->problem_type,
        coco_version, observer->algorithm_info);
    if (logger->log_nondom_mode == LOG_NONDOM_READ)
      fprintf(indicator->info_file, " (reconstructed)");
  }
  if ((observer_data->previous_function != (long) problem->suite_dep_function)
    || (observer_data->previous_dimension != (long) problem->number_of_variables)) {
    fprintf(indicator->info_file, "\nfunction = %2lu, ", (unsigned long) problem->suite_dep_function);
    fprintf(indicator->info_file, "dim = %2lu, ", (unsigned long) problem->number_of_variables);
    fprintf(indicator->info_file, "%s_%s.dat", prefix, indicator_name);
  }
  coco_free_memory(prefix);

  /* Initialize the .dat, .tdat and .rdat files */
  logger_biobj_indicator_initialize_file(logger, observer, problem, indicator, &(indicator->dat_file), "dat", 1);
  logger_biobj_indicator_initialize_file(logger, observer, problem, indicator, &(indicator->tdat_file), "tdat", 0);
  logger_biobj_indicator_initialize_file(logger, observer, problem, indicator, &(indicator->rdat_file), "rdat", 0);

  coco_debug("Ended   logger_biobj_indicator()");

  return indicator;
}

/**
 * @brief Outputs the final information about this indicator.
 */
static void logger_biobj_indicator_finalize(logger_biobj_indicator_t *indicator, const logger_biobj_data_t *logger) {

  /* Log the last eval_number in the dat file if wasn't already logged */
  if (!indicator->target_hit) {
    fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", (unsigned long) logger->num_func_evaluations,
        logger->precision_f, indicator->overall_value, logger->precision_f,
        coco_observer_targets_get_last_target(indicator->targets));
  }

  /* Log the last eval_number in the tdat file if wasn't already logged */
  if (!indicator->evaluation_logged) {
    fprintf(indicator->tdat_file, "%lu\t%.*e\n", (unsigned long) logger->num_func_evaluations,
        logger->precision_f, indicator->overall_value);
  }

  /* Log the information in the info file */
  fprintf(indicator->info_file, ", %lu:%lu|%.1e", (unsigned long) logger->suite_dep_instance,
      (unsigned long) logger->num_func_evaluations, indicator->overall_value);
  fflush(indicator->info_file);

}

/**
 * @brief Frees the memory of the given indicator.
 */
static void logger_biobj_indicator_free(void *stuff) {

  logger_biobj_indicator_t *indicator;

  assert(stuff != NULL);
  indicator = (logger_biobj_indicator_t *) stuff;

  coco_debug("Started logger_biobj_indicator_free()");

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

  if (indicator->rdat_file != NULL) {
    fclose(indicator->rdat_file);
    indicator->rdat_file = NULL;
  }

  if (indicator->info_file != NULL) {
    fclose(indicator->info_file);
    indicator->info_file = NULL;
  }

  if (indicator->targets != NULL){
    coco_observer_targets_free(indicator->targets);
    indicator->targets = NULL;
  }

  if (indicator->evaluations != NULL){
    coco_observer_evaluations_free(indicator->evaluations);
    indicator->evaluations = NULL;
  }

  coco_debug("Ended   logger_biobj_indicator_free()");

  coco_free_memory(stuff);

}

/*
 * @brief Outputs the information according to the observer options.
 *
 * Outputs to the:
 * - dat file, if the archive was updated and a new target was reached for an indicator;
 * - tdat file, if the number of evaluations matches one of the predefined numbers.
 *
 * Note that a target is reached when
 * best_value - current_value + additional_penalty <= relative_target_value
 *
 * The relative_target_value is a target for indicator difference, not the actual indicator value!
 */
static void logger_biobj_output(logger_biobj_data_t *logger,
                                const int update_performed,
                                const logger_biobj_avl_item_t *node_item) {

  size_t i, j;
  logger_biobj_indicator_t *indicator;

  coco_debug("Started logger_biobj_output()");

  if (logger->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {

      indicator = logger->indicators[i];
      indicator->target_hit = 0;
      indicator->previous_value = indicator->overall_value;

      /* If the update was performed, update the overall indicator value */
      if (update_performed) {
        /* Compute the overall_value of an indicator */
        if (strcmp(indicator->name, "hyp") == 0) {
          if (coco_double_almost_equal(indicator->current_value, 0, mo_precision)) {
            /* Update the additional penalty for hypervolume (the minimal distance from the nondominated set
             * to the ROI) */
            double new_distance = mo_get_distance_to_ROI(node_item->normalized_y, logger->number_of_objectives);
            indicator->additional_penalty = coco_double_min(indicator->additional_penalty, new_distance);
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
        indicator->target_hit = coco_observer_targets_trigger(indicator->targets, indicator->overall_value);
      }
      else if ((logger->num_func_evaluations == 1) && (node_item->is_feasible == 0)) {
        /* Special case if the solution is infeasible and this is the first evaluation */
        indicator->overall_value = INFINITY_FOR_LOGGING;
        indicator->target_hit = coco_observer_targets_trigger(indicator->targets, indicator->overall_value);
      }

      /* Log to the dat file if a performance target was hit */
      if (indicator->target_hit) {
        fprintf(indicator->dat_file, "%lu\t%.*e\t%.*e\n", (unsigned long) logger->num_func_evaluations,
            logger->precision_f, indicator->overall_value, logger->precision_f,
            coco_observer_targets_get_last_target(indicator->targets));
      }

      if (logger->log_nondom_mode == LOG_NONDOM_READ) {
        /* Log to the tdat file the previous indicator value if any evaluation number between the previous and
         * this one matches one of the predefined evaluation numbers. */
        for (j = logger->previous_evaluations + 1; j < logger->num_func_evaluations; j++) {
          indicator->evaluation_logged = coco_observer_evaluations_trigger(indicator->evaluations, j);
          if (indicator->evaluation_logged) {
            fprintf(indicator->tdat_file, "%lu\t%.*e\n", (unsigned long) j, logger->precision_f,
                indicator->previous_value);
          }
        }
      }

      /* Log to the tdat file if the number of evaluations matches one of the predefined numbers */
      indicator->evaluation_logged = coco_observer_evaluations_trigger(indicator->evaluations,
          logger->num_func_evaluations);
      if (indicator->evaluation_logged) {
        fprintf(indicator->tdat_file, "%lu\t%.*e\n", (unsigned long) logger->num_func_evaluations,
            logger->precision_f, indicator->overall_value);
      }

      /* Log to the rdat file if the algorithm was restarted */
      if (logger->algorithm_restarted) {
        fprintf(indicator->rdat_file, "%lu\t%.*e\n", (unsigned long) logger->num_func_evaluations,
            logger->precision_f, indicator->overall_value);
      }

    }
  }

  if (logger->algorithm_restarted)
    logger->algorithm_restarted = 0;

  coco_debug("Ended   logger_biobj_output()");
}

/**
 * @brief Evaluates the function, increases the number of evaluations and outputs information according to
 * observer options.
 */
static void logger_biobj_evaluate(coco_problem_t *problem, const double *x, double *y) {

  logger_biobj_data_t *logger;
  logger_biobj_avl_item_t *node_item;
  int update_performed;
  coco_problem_t *inner_problem;
  double *constraints = NULL;

  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* Evaluate the objectives */
  coco_evaluate_function(inner_problem, x, y);
  logger->num_func_evaluations++;

  /* Evaluate the constraints */
  if (problem->number_of_constraints > 0) {
    constraints = coco_allocate_vector(problem->number_of_constraints);
    inner_problem->evaluate_constraint(inner_problem, x, constraints, 0);
  }
  logger->num_cons_evaluations = problem->evaluations_constraints;

  node_item = logger_biobj_node_create(inner_problem, x, y, constraints, logger->num_func_evaluations,
      logger->number_of_variables, logger->number_of_objectives, problem->number_of_constraints);

  /* Update the archive with the new solution, if it is not dominated by or equal to existing solutions in
   * the archive */
  update_performed = logger_biobj_tree_update(logger, node_item);

  /* If the archive was updated and you need to log all nondominated solutions, output the new solution to
   * nondom_file */
  if (update_performed && (logger->log_nondom_mode == LOG_NONDOM_ALL)) {
    logger_biobj_tree_output(logger->adat_file, logger->buffer_tree, logger->number_of_variables,
        logger->number_of_integer_variables, logger->number_of_objectives, logger->log_vars,
        logger->precision_x, logger->precision_f, logger->log_discrete_as_int);
    avl_tree_purge(logger->buffer_tree);

    /* Flush output so that impatient users can see progress. */
    fflush(logger->adat_file);
  }

  /* Output according to observer options */
  logger_biobj_output(logger, update_performed, node_item);

  /* Free allocated memory */
  if (problem->number_of_constraints > 0)
    coco_free_memory(constraints);
}

/**
 * @brief Evaluates the function and outputs information according to observer options to the file with
 * recommendations. The evaluation result is not returned and the evaluation counter is not increased.
 */
static void logger_biobj_recommend(coco_problem_t *problem, const double *x) {

  logger_biobj_data_t *logger;
  coco_problem_t *inner_problem;
  size_t i, j;
  double *y = NULL;
  double *constraints = NULL;

  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);

  /* Evaluate the objectives */
  coco_evaluate_function(inner_problem, x, y);

  /* Evaluate the constraints */
  if (problem->number_of_constraints > 0) {
    constraints = coco_allocate_vector(problem->number_of_constraints);
    inner_problem->evaluate_constraint(inner_problem, x, constraints, 0);
  }
  logger->num_cons_evaluations = problem->evaluations_constraints;

  /* Log to the mdat file */
  fprintf(logger->mdat_file, "%lu\t", (unsigned long) logger->num_func_evaluations);
  for (j = 0; j < problem->number_of_objectives; j++)
    fprintf(logger->mdat_file, "%.*e\t", logger->precision_f, y[j]);
  for (j = 0; j < problem->number_of_constraints; j++)
    fprintf(logger->mdat_file, "%.*e\t", logger->precision_f, constraints[j]);
  if (logger->log_vars) {
    for (i = 0; i < logger->number_of_variables; i++)
      if ((i < logger->number_of_integer_variables) && (logger->log_discrete_as_int))
        fprintf(logger->mdat_file, "%d\t", coco_double_to_int(x[i]));
      else
        fprintf(logger->mdat_file, "%.*e\t", logger->precision_x, x[i]);
  }
  fprintf(logger->mdat_file, "\n");

  /* Free allocated memory */
  coco_free_memory(y);
  if (problem->number_of_constraints > 0)
    coco_free_memory(constraints);
}

/**
 * Sets the number of evaluations, adds the objective vector to the archive and outputs information according
 * to observer options (but does not output the archive).
 *
 * @note Vector y must point to a correctly sized allocated memory region and the given evaluation number must
 * be larger than the existing one.
 *
 * Constraints are not supported.
 *
 * @param problem The given COCO problem.
 * @param evaluation The number of evaluations.
 * @param y The objective vector.
 * @return 1 if archive was updated was done and 0 otherwise.
 */
int coco_logger_biobj_feed_solution(coco_problem_t *problem, const size_t evaluation, const double *y) {

  logger_biobj_data_t *logger;
  logger_biobj_avl_item_t *node_item;
  int update_performed;
  coco_problem_t *inner_problem;
  double *x;
  size_t i;

  assert(problem != NULL);
  logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
  inner_problem = coco_problem_transformed_get_inner_problem(problem);
  assert(logger->log_nondom_mode == LOG_NONDOM_READ);

  /* Set the number of evaluations */
  logger->previous_evaluations = logger->num_func_evaluations;
  if (logger->previous_evaluations >= evaluation)
    coco_error("coco_logger_biobj_reconstruct(): Evaluation %lu came before evaluation %lu. Note that "
        "the evaluations need to be always increasing.", logger->previous_evaluations, evaluation);
  logger->num_func_evaluations = evaluation;

  /* Update the archive with the new solution */
  x = coco_allocate_vector(problem->number_of_variables);
  for (i = 0; i < problem->number_of_variables; i++)
    x[i] = 0;
  node_item = logger_biobj_node_create(inner_problem, x, y, NULL, logger->num_func_evaluations,
      logger->number_of_variables, logger->number_of_objectives, 0);
  coco_free_memory(x);

  /* Update the archive */
  update_performed = logger_biobj_tree_update(logger, node_item);

  /* Output according to observer options */
  logger_biobj_output(logger, update_performed, node_item);

  return update_performed;
}

/**
 * @brief Outputs the final nondominated solutions to the archive file.
 */
static void logger_biobj_finalize(logger_biobj_data_t *logger) {

  avl_tree_t *resorted_tree;
  avl_node_t *solution;

  coco_debug("Started logger_biobj_finalize()");

  /* Re-sort archive_tree according to time stamp and then output it */
  resorted_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_eval_number, NULL);

  if (logger->archive_tree->tail) {
    /* There is at least a solution in the tree to output */
    solution = logger->archive_tree->head;
    while (solution != NULL) {
      avl_item_insert(resorted_tree, solution->item);
      solution = solution->next;
    }
  }

  logger_biobj_tree_output(logger->adat_file, resorted_tree, logger->number_of_variables,
      logger->number_of_integer_variables, logger->number_of_objectives, logger->log_vars,
      logger->precision_x, logger->precision_f, logger->log_discrete_as_int);

  avl_tree_destruct(resorted_tree);

  coco_debug("Ended   logger_biobj_finalize()");
}

/**
 * @brief Frees the memory of the given biobjective logger.
 */
static void logger_biobj_free(void *stuff) {

  logger_biobj_data_t *logger;
  coco_observer_t *observer; /* The observer might not exist at this point */
  size_t i;

  coco_debug("Started logger_biobj_free()");

  assert(stuff != NULL);
  logger = (logger_biobj_data_t *) stuff;

  if (logger->log_nondom_mode == LOG_NONDOM_FINAL) {
     logger_biobj_finalize(logger);
  }

  if (logger->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++) {
      logger_biobj_indicator_finalize(logger->indicators[i], logger);
      logger_biobj_indicator_free(logger->indicators[i]);
    }
  }

  if (((logger->log_nondom_mode == LOG_NONDOM_ALL) || (logger->log_nondom_mode == LOG_NONDOM_FINAL)) &&
      (logger->adat_file != NULL)) {
    fprintf(logger->adat_file, "%% evaluations = %lu\n", (unsigned long) logger->num_func_evaluations);
    fclose(logger->adat_file);
    logger->adat_file = NULL;
  }

  if (logger->mdat_file != NULL) {
    fclose(logger->mdat_file);
    logger->mdat_file = NULL;
  }

  avl_tree_destruct(logger->archive_tree);
  avl_tree_destruct(logger->buffer_tree);

  observer = logger->observer;
  if ((observer != NULL) && (observer->is_active == 1)) {
    if (observer->data != NULL) {
      /* If the observer still exists (if it does not, observed_problem does not matter any longer) */
      ((observer_biobj_data_t *)observer->data)->observed_problem = NULL;
    }
  }

  coco_debug("Ended   logger_biobj_free()");

}

/*
 * @brief Sets the pointer to the observer to NULL
 *
 * Has to be taken care of here due to
 */
static void logger_biobj_data_nullify_observer(void *stuff) {
  logger_biobj_data_t *logger = (logger_biobj_data_t *) stuff;
  logger->observer = NULL;
}

/**
 * @brief Saves the information that the algorithm has restarted
 */
static void logger_biobj_signal_restart(coco_problem_t *problem) {

  logger_biobj_data_t *logger = (logger_biobj_data_t *) coco_problem_transformed_get_data(problem);
  assert(logger);

  if (logger->num_func_evaluations > 0)
    logger->algorithm_restarted = 1;
}

/**
 * @brief Initializes the biobjective logger.
 *
 * Copies all observer field values that are needed after initialization into logger field values
 * because if the observer is deleted before the suite, the observer is not available anymore when
 * the logger is finalized.
 */
static coco_problem_t *logger_biobj(coco_observer_t *observer, coco_problem_t *inner_problem) {

  coco_problem_t *problem;
  logger_biobj_data_t *logger_data;
  observer_biobj_data_t *observer_data;
  const char nondom_folder_name[] = "archive";
  char *path_name, *prefix, *file_name = NULL;
  size_t i;

  coco_debug("Started logger_biobj()");

  if (inner_problem->number_of_objectives != 2) {
    coco_error("logger_biobj(): The bi-objective logger cannot log a problem with %d objective(s)",
        inner_problem->number_of_objectives);
    return NULL; /* Never reached. */
  }

  assert(observer != NULL);
  observer_data = (observer_biobj_data_t *) observer->data;
  assert(observer_data != NULL);

  if (observer_data->observed_problem != NULL) {
    coco_error("logger_biobj(): The observed problem must be closed before a new problem can be observed");
  }

  logger_data = (logger_biobj_data_t *) coco_allocate_memory(sizeof(*logger_data));

  logger_data->observer = observer;
  logger_data->num_func_evaluations = 0;
  logger_data->num_cons_evaluations = 0;
  logger_data->previous_evaluations = 0;
  logger_data->number_of_variables = inner_problem->number_of_variables;
  logger_data->number_of_integer_variables = inner_problem->number_of_integer_variables;
  logger_data->number_of_objectives = inner_problem->number_of_objectives;
  logger_data->suite_dep_instance = inner_problem->suite_dep_instance;
  logger_data->algorithm_restarted = 0;

  /* Copy values from the observer that you might need even if it does not exist any more */
  logger_data->log_nondom_mode = observer_data->log_nondom_mode;
  logger_data->compute_indicators = observer_data->compute_indicators;
  logger_data->precision_x = observer->precision_x;
  logger_data->precision_f = observer->precision_f;
  logger_data->log_discrete_as_int = observer->log_discrete_as_int;

  if (((observer_data->log_vars_mode == LOG_VARS_LOW_DIM) && (inner_problem->number_of_variables > 5))
      || (observer_data->log_vars_mode == LOG_VARS_NEVER))
    logger_data->log_vars = 0;
  else
    logger_data->log_vars = 1;

  /* Initialize logging of nondominated solutions into the archive file */
  if ((logger_data->log_nondom_mode == LOG_NONDOM_ALL) ||
      (logger_data->log_nondom_mode == LOG_NONDOM_FINAL)) {

    /* Create the path to the file */
    path_name = coco_allocate_string(COCO_PATH_MAX + 1);
    memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
    coco_join_path(path_name, COCO_PATH_MAX, nondom_folder_name, NULL);
    coco_create_directory(path_name);

    /* Construct file name */
    if (logger_data->log_nondom_mode == LOG_NONDOM_ALL)
      file_name = coco_strdupf("%s_nondom_all.adat", inner_problem->problem_id);
    else if (logger_data->log_nondom_mode == LOG_NONDOM_FINAL)
      file_name = coco_strdupf("%s_nondom_final.adat", inner_problem->problem_id);
    coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
    coco_free_memory(file_name);

    /* Open and initialize the archive file */
    logger_data->adat_file = fopen(path_name, "a");
    if (logger_data->adat_file == NULL) {
      coco_error("logger_biobj() failed to open file '%s'.", path_name);
      return NULL; /* Never reached */
    }
    coco_free_memory(path_name);

    /* Output header information */
    fprintf(logger_data->adat_file, "%% instance = %lu, name = %s\n",
        (unsigned long) inner_problem->suite_dep_instance, inner_problem->problem_name);
    if (logger_data->log_vars) {
      fprintf(logger_data->adat_file, "%% function evaluation | %lu objectives | %lu variables\n",
          (unsigned long) inner_problem->number_of_objectives,
          (unsigned long) inner_problem->number_of_variables);
    } else {
      fprintf(logger_data->adat_file, "%% function evaluation | %lu objectives \n",
          (unsigned long) inner_problem->number_of_objectives);
    }
  }

  /* Initialize the AVL trees */
  logger_data->archive_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_last_objective,
      (avl_free_t) logger_biobj_node_free);
  logger_data->buffer_tree = avl_tree_construct((avl_compare_t) avl_tree_compare_by_eval_number, NULL);

  /* Initialize the indicators */
  if (logger_data->compute_indicators) {
    for (i = 0; i < LOGGER_BIOBJ_NUMBER_OF_INDICATORS; i++)
      logger_data->indicators[i] = logger_biobj_indicator(logger_data, observer, inner_problem, logger_biobj_indicators[i]);

    observer_data->previous_function = (long) inner_problem->suite_dep_function;
    observer_data->previous_dimension = (long) inner_problem->number_of_variables;
  }

  problem = coco_problem_transformed_allocate(inner_problem, logger_data, logger_biobj_free, observer->observer_name);
  problem->evaluate_function = logger_biobj_evaluate;
  problem->recommend_solution = logger_biobj_recommend;

  /* Initialize the file with recommendations */

  /* Create the path to the file */
  path_name = coco_allocate_string(COCO_PATH_MAX + 1);
  memcpy(path_name, observer->result_folder, strlen(observer->result_folder) + 1);
  coco_join_path(path_name, COCO_PATH_MAX, problem->problem_type, NULL);
  coco_create_directory(path_name);

  /* Construct file name */
  prefix = coco_remove_from_string(problem->problem_id, "_i", "_d");
  file_name = coco_strdupf("%s.mdat", prefix);
  coco_join_path(path_name, COCO_PATH_MAX, file_name, NULL);
  coco_free_memory(file_name);
  coco_free_memory(prefix);

  /* Open and initialize the recommendation file */
  logger_data->mdat_file = fopen(path_name, "a");
  if (logger_data->mdat_file == NULL) {
    coco_error("logger_biobj() failed to open file '%s'.", path_name);
    return NULL; /* Never reached */
  }
  coco_free_memory(path_name);

  /* Output header information */
  fprintf(logger_data->mdat_file, "%% instance = %lu, name = %s\n%% function evaluation | %lu objectives",
      (unsigned long) inner_problem->suite_dep_instance,
      inner_problem->problem_name,
      (unsigned long) inner_problem->number_of_objectives);
  if (inner_problem->number_of_constraints > 0)
    fprintf(logger_data->mdat_file, " | %lu constraints", (unsigned long) inner_problem->number_of_constraints);
  if (logger_data->log_vars)
    fprintf(logger_data->mdat_file, " | %lu variables", (unsigned long) inner_problem->number_of_variables);
  fprintf(logger_data->mdat_file, "\n");

  coco_debug("Ended   logger_biobj()");

  observer_data->observed_problem = problem;
  return problem;
}
