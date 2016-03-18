/**
 * @file coco_acrhive.c
 * @brief Definitions of functions regarding COCO archives.
 *
 * COCO archives are used to do some pre-processing on the bi-objective archive files. Namely, through a
 * wrapper written in Python, these functions are used to merge archives and compute their hypervolumes.
 */

#include "coco.h"
#include "coco_utilities.c"
#include "mo_utilities.c"
#include "mo_avl_tree.c"

/**
 * @brief The COCO archive structure.
 *
 * The archive structure is used for pre-processing archives of non-dominated solutions.
 */
struct coco_archive_s {

  avl_tree_t *tree;              /**< @brief The AVL tree with non-dominated solutions. */
  double *ideal;                 /**< @brief The ideal point. */
  double *nadir;                 /**< @brief The nadir point. */

  size_t number_of_objectives;   /**< @brief Number of objectives (clearly equal to 2). */

  int is_up_to_date;             /**< @brief Whether archive fields have been updated since last addition. */
  size_t number_of_solutions;    /**< @brief Number of solutions in the archive. */
  double hypervolume;            /**< @brief Hypervolume of the solutions in the archive. */

  avl_node_t *current_solution;  /**< @brief Current solution (to return). */
  avl_node_t *extreme1;          /**< @brief Pointer to the first extreme solution. */
  avl_node_t *extreme2;          /**< @brief Pointer to the second extreme solution. */
  int extremes_already_returned; /**< @brief Whether the extreme solutions have already been returned. */
};

/**
 * @brief The type for the node's item in the AVL tree used by the archive.
 */
typedef struct {

  double f1;           /**< @brief The value of the first objective of this solution. */
  double f2;           /**< @brief The value of the second objective of this solution. */
  char *text;          /**< @brief The text describing the solution (the whole line of the archive). */

} coco_archive_avl_item_t;

/**
 * @brief Creates and returns the information on the solution in the form of a node's item in the AVL tree.
 */
static coco_archive_avl_item_t* coco_archive_node_item_create(const double f1,
                                                              const double f2,
                                                              const char *text) {

  /* Allocate memory to hold the data structure mo_preprocessing_avl_item_t */
  coco_archive_avl_item_t *item = (coco_archive_avl_item_t*) coco_allocate_memory(sizeof(*item));

  item->f1 = f1;
  item->f2 = f2;
  item->text = coco_strdup(text);

  return item;
}

/**
 * @brief Returns f1 and f2 in a vector of two doubles. Memory needs to be freed by the caller.
 */
static double *coco_archive_node_item_get_vector(coco_archive_avl_item_t *item) {
  double *result = coco_allocate_vector(2);
  result[0] = item->f1;
  result[1] = item->f2;
  return result;
}

/**
 * @brief Frees the data of the given coco_archive_avl_item_t.
 */
static void coco_archive_node_item_free(coco_archive_avl_item_t *item, void *userdata) {
  coco_free_memory(item->text);
  coco_free_memory(item);
  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Defines the ordering of AVL tree nodes based on the value of the last objective.
 */
static int coco_archive_compare_by_last_objective(const coco_archive_avl_item_t *item1,
                                                  const coco_archive_avl_item_t *item2,
                                                  void *userdata) {
  if (item1->f2 < item2->f2)
    return -1;
  else if (item1->f2 > item2->f2)
    return 1;
  else
    return 0;

  (void) userdata; /* To silence the compiler */
}

/**
 * @brief Allocates memory for the archive and initializes its fields.
 */
static coco_archive_t *coco_archive_allocate(void) {

  /* Allocate memory to hold the data structure coco_archive_t */
  coco_archive_t *archive = (coco_archive_t*) coco_allocate_memory(sizeof(*archive));

  /* Initialize the AVL tree */
  archive->tree = avl_tree_construct((avl_compare_t) coco_archive_compare_by_last_objective,
      (avl_free_t) coco_archive_node_item_free);

  archive->ideal = NULL;                /* To be allocated in coco_archive() */
  archive->nadir = NULL;                /* To be allocated in coco_archive() */
  archive->number_of_objectives = 2;
  archive->is_up_to_date = 0;
  archive->number_of_solutions = 0;
  archive->hypervolume = 0.0;

  archive->current_solution = NULL;
  archive->extreme1 = NULL;             /* To be set in coco_archive() */
  archive->extreme2 = NULL;             /* To be set in coco_archive() */
  archive->extremes_already_returned = 0;

  return archive;
}

/**
 * The archive always contains the two extreme solutions
 */
coco_archive_t *coco_archive(const char *suite_name,
                             const size_t function,
                             const size_t dimension,
                             const size_t instance) {

  coco_archive_t *archive = coco_archive_allocate();
  int output_precision = 15;
  coco_suite_t *suite;
  char *suite_instance = coco_strdupf("instances: %lu", (unsigned long) instance);
  char *suite_options = coco_strdupf("dimensions: %lu function_indices: %lu",
  		(unsigned long) dimension, (unsigned long) function);
  coco_problem_t *problem;
  char *text;
  int update;

  suite = coco_suite(suite_name, suite_instance, suite_options);
  if (suite == NULL) {
    coco_error("coco_archive(): cannot create suite '%s'", suite_name);
    return NULL; /* Never reached */
  }
  problem = coco_suite_get_next_problem(suite, NULL);
  if (problem == NULL) {
    coco_error("coco_archive(): cannot create problem f%02lu_i%02lu_d%02lu in suite '%s'",
    		(unsigned long) function, (unsigned long) instance, (unsigned long) dimension, suite_name);
    return NULL; /* Never reached */
  }

  /* Store the ideal and nadir points */
  archive->ideal = coco_duplicate_vector(problem->best_value, 2);
  archive->nadir = coco_duplicate_vector(problem->nadir_value, 2);

  /* Add the extreme points to the archive */
  text = coco_strdupf("0\t%.*e\t%.*e\n", output_precision, archive->nadir[0], output_precision, archive->ideal[1]);
  update = coco_archive_add_solution(archive, archive->nadir[0], archive->ideal[1], text);
  coco_free_memory(text);
  assert(update == 1);

  text = coco_strdupf("0\t%.*e\t%.*e\n", output_precision, archive->ideal[0], output_precision, archive->nadir[1]);
  update = coco_archive_add_solution(archive, archive->ideal[0], archive->nadir[1], text);
  coco_free_memory(text);
  assert(update == 1);

  archive->extreme1 = archive->tree->head;
  archive->extreme2 = archive->tree->tail;
  assert(archive->extreme1 != archive->extreme2);

  coco_free_memory(suite_instance);
  coco_free_memory(suite_options);
  coco_suite_free(suite);

  return archive;
}

int coco_archive_add_solution(coco_archive_t *archive, const double f1, const double f2, const char *text) {

  coco_archive_avl_item_t* insert_item = coco_archive_node_item_create(f1, f2, text);
  double *insert_objectives, *node_objectives;
  avl_node_t *node, *next_node;
  int update = 0;
  int dominance;

  insert_objectives = coco_archive_node_item_get_vector(insert_item);

  /* Find the first point that is not worse than the new point (NULL if such point does not exist) */
  node = avl_item_search_right(archive->tree, insert_item, NULL);

  if (node == NULL) {
    /* The new point is an extremal point */
    update = 1;
    next_node = archive->tree->head;
  } else {
    node_objectives = coco_archive_node_item_get_vector((coco_archive_avl_item_t*) node->item);
    dominance = mo_get_dominance(insert_objectives, node_objectives, archive->number_of_objectives);
    coco_free_memory(node_objectives);
    if (dominance > -1) {
      update = 1;
      next_node = node->next;
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point (as long as it's not an extreme point) */
      	if ((node == archive->extreme1) || (node == archive->extreme2))
      		update = 0;
      	else
      		avl_node_delete(archive->tree, node);
      }
    } else {
      /* The new point is dominated, nothing more to do */
      update = 0;
    }
  }

  if (!update) {
    coco_archive_node_item_free(insert_item, NULL);
  } else {
    /* Perform tree update */
    while (next_node != NULL) {
      /* Check the dominance relation between the new node and the next node. There are only two possibilities:
       * dominance = 0: the new node and the next node are nondominated
       * dominance = 1: the new node dominates the next node */
      node = next_node;
      node_objectives = coco_archive_node_item_get_vector((coco_archive_avl_item_t*) node->item);
      dominance = mo_get_dominance(insert_objectives, node_objectives, archive->number_of_objectives);
      coco_free_memory(node_objectives);
      if (dominance == 1) {
        /* The new point dominates the next point, remove the next point (as long as it's not an extreme point) */
        next_node = node->next;
      	if ((node == archive->extreme1) || (node == archive->extreme2)) {
      		update = 0;
      		break;
      	}
      	else
      		avl_node_delete(archive->tree, node);
      } else {
        break;
      }
    }

    avl_item_insert(archive->tree, insert_item);
    archive->is_up_to_date = 0;
  }

  coco_free_memory(insert_objectives);
  return update;
}

/**
 * @brief Updates the archive fields returned by the getters.
 */
static void coco_archive_update(coco_archive_t *archive) {

  if (!archive->is_up_to_date) {

    avl_node_t *node, *left_node;
    coco_archive_avl_item_t *node_item, *left_node_item;
    double *node_objectives, *left_node_objectives;

    /* Updates number_of_solutions */

    archive->number_of_solutions = avl_count(archive->tree);

    /* Updates hypervolume */

    node = archive->tree->head;
    archive->hypervolume = 0; /* Hypervolume of the extreme point equals 0 */
    while (node->next) {
      /* Add hypervolume contributions of the other points that are within ROI */
      left_node = node->next;
      node_item = (coco_archive_avl_item_t *) node->item;
      left_node_item = (coco_archive_avl_item_t *) left_node->item;
      node_objectives = coco_archive_node_item_get_vector(node_item);
      left_node_objectives = coco_archive_node_item_get_vector(left_node_item);
      if (mo_solution_is_within_ROI(left_node_objectives, archive->ideal, archive->nadir, archive->number_of_objectives)) {
        if (mo_solution_is_within_ROI(node_objectives, archive->ideal, archive->nadir, archive->number_of_objectives))
          archive->hypervolume += (node_item->f1 - left_node_item->f1) * (archive->nadir[1] - left_node_item->f2);
        else
          archive->hypervolume += (archive->nadir[0] - left_node_item->f1) * (archive->nadir[1] - left_node_item->f2);
      }
      coco_free_memory(node_objectives);
      coco_free_memory(left_node_objectives);
      node = left_node;
    }
    /* Performs normalization */
    archive->hypervolume /= ((archive->nadir[0] - archive->ideal[0]) * (archive->nadir[1] - archive->ideal[1]));

    archive->is_up_to_date = 1;
    archive->current_solution = NULL;
    archive->extremes_already_returned = 0;
  }

}

const char *coco_archive_get_next_solution_text(coco_archive_t *archive) {

  char *text;

  coco_archive_update(archive);

  if (!archive->extremes_already_returned) {

    if (archive->current_solution == NULL) {
      /* Return the first extreme */
      text = ((coco_archive_avl_item_t *) archive->extreme1->item)->text;
      archive->current_solution = archive->extreme2;
      return text;
    }

    if (archive->current_solution == archive->extreme2) {
      /* Return the second extreme */
      text = ((coco_archive_avl_item_t *) archive->extreme2->item)->text;
      archive->extremes_already_returned = 1;
      archive->current_solution = archive->tree->head;
      return text;
    }

  } else {

    if (archive->current_solution == NULL)
      return "";

    if ((archive->current_solution == archive->extreme1) || (archive->current_solution == archive->extreme2)) {
      /* Skip this one */
      archive->current_solution = archive->current_solution->next;
      return coco_archive_get_next_solution_text(archive);
    }

    /* Return the current solution and move to the next */
    text = ((coco_archive_avl_item_t *) archive->current_solution->item)->text;
    archive->current_solution = archive->current_solution->next;
    return text;
  }

  return NULL; /* This point should never be reached. */
}

size_t coco_archive_get_number_of_solutions(coco_archive_t *archive) {
  coco_archive_update(archive);
  return archive->number_of_solutions;
}

double coco_archive_get_hypervolume(coco_archive_t *archive) {
  coco_archive_update(archive);
  return archive->hypervolume;
}

void coco_archive_free(coco_archive_t *archive) {

  assert(archive != NULL);

  avl_tree_destruct(archive->tree);
  coco_free_memory(archive->ideal);
  coco_free_memory(archive->nadir);
  coco_free_memory(archive);

}
