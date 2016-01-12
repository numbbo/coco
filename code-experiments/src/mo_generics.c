#include <stdlib.h>
#include <stdio.h>
#include "coco.h"

/* A structure to hold information for the multiobjective optimization problems */
typedef struct {
  double *ideal_point;
  double *reference_point;
  double *normalization_factor;
} mo_problem_data_t;

static mo_problem_data_t *mo_problem_data_allocate(const size_t number_of_objectives) {

  mo_problem_data_t *data = coco_allocate_memory(sizeof(*data));

  data->ideal_point = coco_allocate_vector(number_of_objectives);
  data->reference_point = coco_allocate_vector(number_of_objectives);
  data->normalization_factor = coco_allocate_vector(number_of_objectives);

  return data;
}

static void mo_problem_data_compute_normalization_factor(mo_problem_data_t *data, const size_t number_of_objectives) {
  size_t i;
  double diff;

  for (i = 0; i < number_of_objectives; i++) {
    diff = data->reference_point[i] - data->ideal_point[i];
    if (diff == 0) {
      coco_warning("mo_problem_data_compute_normalization_factor(): ideal and reference points too close!");
      diff = 1.0;
    }
    else
      data->normalization_factor[i] = 1.0 / diff;
  }
}

static void mo_problem_data_free(void *stuff) {

  mo_problem_data_t *data;

  assert(stuff != NULL);
  data = stuff;

  if (data->ideal_point != NULL) {
    coco_free_memory(data->ideal_point);
    data->ideal_point = NULL;
  }

  if (data->reference_point != NULL) {
    coco_free_memory(data->reference_point);
    data->reference_point = NULL;
  }

  if (data->normalization_factor != NULL) {
    coco_free_memory(data->normalization_factor);
    data->normalization_factor = NULL;
  }

  coco_free_memory(data);

}

/**
 * Checks the dominance relation in the unconstrained minimization case between
 * objectives1 and objectives2 and returns:
 *  1 if objectives1 dominates objectives2
 *  0 if objectives1 and objectives2 are non-dominated
 * -1 if objectives2 dominates objectives1
 * -2 if objectives1 is identical to objectives2
 */
static int mo_get_dominance(const double *objectives1, const double *objectives2, const size_t num_obj) {
  /* TODO: Should we care about comparison precision? */
  size_t i;

  int flag1 = 0;
  int flag2 = 0;

  for (i = 0; i < num_obj; i++) {
    if (objectives1[i] < objectives2[i]) {
      flag1 = 1;
    } else if (objectives1[i] > objectives2[i]) {
      flag2 = 1;
    }
  }

  if (flag1 && !flag2) {
    return 1;
  } else if (!flag1 && flag2) {
    return -1;
  } else if (flag1 && flag2) {
    return 0;
  } else { /* (!flag1 && !flag2) */
    return -2;
  }
}

/**
 * Computes and returns the Euclidean norm of two dim-dimensional points first and second.
 */
static double mo_get_norm(const double *first, const double *second, const size_t dim) {

  size_t i;
  double norm = 0;

  for (i = 0; i < dim; i++) {
    norm += pow(first[i] - second[i], 2);
  }

  return sqrt(norm);
}
