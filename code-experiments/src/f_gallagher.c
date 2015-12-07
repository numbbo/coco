#include <assert.h>
#include <math.h>

#include "coco.h"
#include "coco_problem.c"
#include "coco_utilities.c"
#include "suite_bbob2009_legacy_code.c"

#define NB_PEAKS_21 101
#define NB_PEAKS_22 21
#define MAX_DIM SUITE_BBOB2009_MAX_DIM

static double *gallagher_peaks;
/* To make dimension free of restrictions (and save memory for large MAX_DIM),
 * these should be allocated in bbob_gallagher_problem */
static double gallagher_peaks21[NB_PEAKS_21 * MAX_DIM];
static double gallagher_peaks22[NB_PEAKS_22 * MAX_DIM];
static int f_gallagher_compare_doubles(const void *, const void *);

typedef struct {
  long rseed;
  unsigned int number_of_peaks;
  double *xopt;
  double **rotation, **Xlocal, **arrScales;
  double *peakvalues;
  coco_free_function_t old_free_problem;
} f_gallagher_data_t;

static void f_gallagher_evaluate(coco_problem_t *self, const double *x, double *y) {
  size_t i, j; /*Loop over dim*/
  double *tmx;
  f_gallagher_data_t *data = self->data;
  double a = 0.1;
  double tmp2, f = 0., Fadd, tmp, Fpen = 0., Ftrue = 0.;
  double fac = -0.5 / (double) self->number_of_variables;

  assert(self->number_of_objectives == 1);
  assert(self->number_of_variables > 0);
  /* Boundary handling */
  for (i = 0; i < self->number_of_variables; ++i) {
    tmp = fabs(x[i]) - 5.;
    if (tmp > 0.) {
      Fpen += tmp * tmp;
    }
  }
  Fadd = Fpen;
  /* Transformation in search space */
  /* FIXME: this should rather be done in bbob_gallagher_problem */
  tmx = (double *) calloc(self->number_of_variables, sizeof(double));
  for (i = 0; i < self->number_of_variables; i++) {
    for (j = 0; j < self->number_of_variables; ++j) {
      tmx[i] += data->rotation[i][j] * x[j];
    }
  }
  /* Computation core*/
  for (i = 0; i < data->number_of_peaks; ++i) {
    tmp2 = 0.;
    for (j = 0; j < self->number_of_variables; ++j) {
      tmp = (tmx[j] - data->Xlocal[j][i]);
      tmp2 += data->arrScales[i][j] * tmp * tmp;
    }
    tmp2 = data->peakvalues[i] * exp(fac * tmp2);
    f = coco_max_double(f, tmp2);
  }

  f = 10. - f;
  if (f > 0) {
    Ftrue = log(f) / a;
    Ftrue = pow(exp(Ftrue + 0.49 * (sin(Ftrue) + sin(0.79 * Ftrue))), a);
  } else if (f < 0) {
    Ftrue = log(-f) / a;
    Ftrue = -pow(exp(Ftrue + 0.49 * (sin(0.55 * Ftrue) + sin(0.31 * Ftrue))), a);
  } else
    Ftrue = f;

  Ftrue *= Ftrue;
  Ftrue += Fadd;
  y[0] = Ftrue;
  assert(y[0] >= self->best_value[0]);
  /* FIXME: tmx hasn't been allocated with coco_allocate... */
  coco_free_memory(tmx);
}

static void f_gallagher_free(coco_problem_t *self) {
  f_gallagher_data_t *data;
  data = self->data;
  coco_free_memory(data->xopt);
  coco_free_memory(data->peakvalues);
  bbob2009_free_matrix(data->rotation, self->number_of_variables);
  bbob2009_free_matrix(data->Xlocal, self->number_of_variables);
  bbob2009_free_matrix(data->arrScales, data->number_of_peaks);
  self->free_problem = NULL;
  coco_problem_free(self);
}

static coco_problem_t *f_gallagher(const size_t number_of_variables,
                                   const long instance_id,
                                   const unsigned int number_of_peaks) {
  size_t i, j, k, problem_id_length, *rperm;
  long rseed;
  coco_problem_t *problem;
  f_gallagher_data_t *data;
  double maxcondition = 1000., maxcondition1 = 1000., *arrCondition, fitvalues[2] = { 1.1, 9.1 }; /*maxcondition1 satisfies the old code and
   the doc but seems wrong in that it is,
   with very high probability, not the
   largest condition level!!!*/
  double b, c; /* Parameters for generating local optima. In the old code, they are
   different in f21 and f22 */

  assert(number_of_variables <= MAX_DIM);
  if (number_of_peaks == 101) {
    rseed = 21 + 10000 * instance_id;
    /* FIXME: rather use coco_allocate_vector here */
    gallagher_peaks = gallagher_peaks21;
    maxcondition1 = sqrt(maxcondition1);
  } else {
    rseed = 22 + 10000 * instance_id;
    gallagher_peaks = gallagher_peaks22;
  }

  data = coco_allocate_memory(sizeof(*data));
  /* Allocate temporary storage and space for the rotation matrices */
  data->rseed = rseed;
  data->number_of_peaks = number_of_peaks;
  data->xopt = coco_allocate_vector(number_of_variables);
  data->rotation = bbob2009_allocate_matrix(number_of_variables, number_of_variables);
  data->Xlocal = bbob2009_allocate_matrix(number_of_variables, number_of_peaks);
  data->arrScales = bbob2009_allocate_matrix(number_of_peaks, number_of_variables);
  bbob2009_compute_rotation(data->rotation, rseed, (long) number_of_variables);
  problem = coco_problem_allocate(number_of_variables, 1, 0);
  /* Construct a meaningful problem id */
  if (number_of_peaks == NB_PEAKS_21) {
    problem->problem_name = coco_strdup("BBOB f21");
    problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "bbob2009_f21", (long) number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "bbob2009_f21",
        (long) number_of_variables);
    b = 10.;
    c = 5.;
  } else if (number_of_peaks == NB_PEAKS_22) {
    problem->problem_name = coco_strdup("BBOB f22");
    problem_id_length = (size_t) snprintf(NULL, 0, "%s_%02lu", "bbob2009_f22", (long) number_of_variables);
    problem->problem_id = coco_allocate_memory(problem_id_length + 1);
    snprintf(problem->problem_id, problem_id_length + 1, "%s_%02lu", "bbob2009_f22",
        (long) number_of_variables);
    b = 9.8;
    c = 4.9;
  } else {
    b = 0.0;
    c = 0.0;
    coco_error("Bad number of peaks in f_gallagher");
  }

  problem->number_of_variables = number_of_variables;
  problem->number_of_objectives = 1;
  problem->number_of_constraints = 0;
  problem->data = data;
  problem->free_problem = f_gallagher_free;
  problem->evaluate_function = f_gallagher_evaluate;
  for (i = 0; i < number_of_variables; ++i) {
    problem->smallest_values_of_interest[i] = -5.0;
    problem->largest_values_of_interest[i] = 5.0;
  }

  /* Initialize all the data of the inner problem */
  bbob2009_unif(gallagher_peaks, number_of_peaks - 1, data->rseed);
  rperm = (size_t *) malloc((number_of_peaks - 1) * sizeof(size_t));
  for (i = 0; i < number_of_peaks - 1; ++i)
    rperm[i] = i;
  qsort(rperm, number_of_peaks - 1, sizeof(size_t), f_gallagher_compare_doubles);

  /* Random permutation */
  arrCondition = coco_allocate_vector(number_of_peaks);
  arrCondition[0] = maxcondition1;
  data->peakvalues = coco_allocate_vector(number_of_peaks);
  data->peakvalues[0] = 10;
  for (i = 1; i < number_of_peaks; ++i) {
    arrCondition[i] = pow(maxcondition, (double) (rperm[i - 1]) / ((double) (number_of_peaks - 2)));
    data->peakvalues[i] = (double) (i - 1) / (double) (number_of_peaks - 2) * (fitvalues[1] - fitvalues[0])
        + fitvalues[0];
  }
  coco_free_memory(rperm);

  rperm = (size_t *) malloc(number_of_variables * sizeof(size_t));
  for (i = 0; i < number_of_peaks; ++i) {
    bbob2009_unif(gallagher_peaks, (long) number_of_variables, data->rseed + 1000 * (long) i);
    for (j = 0; j < number_of_variables; ++j)
      rperm[j] = j;
    qsort(rperm, number_of_variables, sizeof(size_t), f_gallagher_compare_doubles);
    for (j = 0; j < number_of_variables; ++j) {
      data->arrScales[i][j] = pow(arrCondition[i],
          ((double) rperm[j]) / ((double) (number_of_variables - 1)) - 0.5);
    }
  }
  coco_free_memory(rperm);

  bbob2009_unif(gallagher_peaks, (long) (number_of_variables * number_of_peaks), data->rseed);
  for (i = 0; i < number_of_variables; ++i) {
    data->xopt[i] = 0.8 * (b * gallagher_peaks[i] - c);
    problem->best_parameter[i] = 0.8 * (b * gallagher_peaks[i] - c);
    for (j = 0; j < number_of_peaks; ++j) {
      data->Xlocal[i][j] = 0.;
      for (k = 0; k < number_of_variables; ++k) {
        data->Xlocal[i][j] += data->rotation[i][k] * (b * gallagher_peaks[j * number_of_variables + k] - c);
      }
      if (j == 0) {
        data->Xlocal[i][j] *= 0.8;
      }
    }
  }

  coco_free_memory(arrCondition);

  /* Calculate best parameter value */
  problem->evaluate_function(problem, problem->best_parameter, problem->best_value);
  return problem;
}

static int f_gallagher_compare_doubles(const void *a, const void *b) {
  double temp = gallagher_peaks[*(const int *) a] - gallagher_peaks[*(const int *) b]; /* TODO: replace int by size_t? */
  if (temp > 0)
    return 1;
  else if (temp < 0)
    return -1;
  else
    return 0;
}

/* Be nice and remove defines from amalgamation */
#undef NB_PEAKS_21
#undef NB_PEAKS_22
#undef MAX_DIM
