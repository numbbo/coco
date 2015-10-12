#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_archive.c"
#include "coco_archive.h"

#include "coco_utilities.c"
#include "coco_problem.c"
#include "coco_strdup.c"
#include "mo_pareto_filtering.c"

/* For making my multiobjective recorder work */

typedef struct {
  char *path;
  FILE *logfile;
  size_t max_size_of_archive;
  size_t number_of_evaluations;
} logger_nondominated_t;

static coco_archive_t *mo_archive;
static coco_archive_entry_t *entry;

static void private_logger_nondominated_evaluate(coco_problem_t *self, const double *x, double *y) {
  logger_nondominated_t *data;
  size_t i;
  size_t j;
  size_t k;
  data = coco_transformed_get_data(self);

  coco_evaluate_function(coco_transformed_get_inner_problem(self), x, y);
  data->number_of_evaluations++;

  /* Open logfile if it is not already open */
  if (data->logfile == NULL) {
    data->logfile = fopen(data->path, "w");
    if (data->logfile == NULL) {
      char *buf;
      const char *error_format =
          "private_logger_nondominated_evaluate_function() failed to open log file '%s'.";
      size_t buffer_size = (size_t) snprintf(NULL, 0, error_format, data->path);
      buf = (char *) coco_allocate_memory(buffer_size);
      snprintf(buf, buffer_size, error_format, data->path);
      coco_error(buf);
      coco_free_memory(buf); /* Never reached */
    }
    fprintf(data->logfile, "# %lu variables  |  %lu objectives  |  func eval number\n",
        coco_problem_get_dimension(coco_transformed_get_inner_problem(self)),
        coco_problem_get_number_of_objectives(coco_transformed_get_inner_problem(self)));

    /*********************************************************************/
    /* TODO: Temporary put it here, to check later */
    /* Allocate memory for the archive */
    mo_archive = (coco_archive_t *) malloc(1 * sizeof(coco_archive_t));
    coco_archive_allocate(mo_archive, data->max_size_of_archive,
        coco_problem_get_dimension(coco_transformed_get_inner_problem(self)),
        coco_problem_get_number_of_objectives(coco_transformed_get_inner_problem(self)), 1);
    /*********************************************************************/
  }

  /********************************************************************************/
  /* Finish evaluations of 1 single solution of the pop, with n_obj objectives,
   * now update the archive with this newly evaluated solution and check its nondomination. */
  coco_archive_push(mo_archive, &x, &y, 1, data->number_of_evaluations);
  mococo_pareto_filtering(mo_archive); /***** TODO: IMPROVE THIS ROUTINE *****/
  coco_archive_mark_updates(mo_archive, data->number_of_evaluations);

  /* Write out a line for this newly evaluated solution if it is nondominated */
  /* write main info to the log file for pfront*/
  for (i = 0; i < mo_archive->update_size; i++) {
    entry = mo_archive->update[i];
    for (j = 0; j < coco_problem_get_dimension(coco_transformed_get_inner_problem(self)); j++) /* all decision variables of a solution */
      fprintf(data->logfile, "%13.10e\t", entry->var[j]);
    for (k = 0; k < coco_problem_get_number_of_objectives(coco_transformed_get_inner_problem(self)); k++) /* all objective values of a solution */
      fprintf(data->logfile, "%13.10e\t", entry->obj[k]);
    fprintf(data->logfile, "%lu", entry->birth); /* its timestamp (FEval) */
    fprintf(data->logfile, "\n"); /* go to the next line for another solution */
  }
  /********************************************************************************/

  /* Flush output so that impatient users can see progress. */
  fflush(data->logfile);
}

static void private_logger_nondominated_free(void *stuff) {
  logger_nondominated_t *data;
  assert(stuff != NULL);
  data = stuff;

  if (data->path != NULL) {
    coco_free_memory(data->path);
    data->path = NULL;
  }
  /* if (data->target_values != NULL) {
   coco_free_memory(data->target_values);
   data->target_values = NULL;
   }
   */
  if (data->logfile != NULL) {
    fclose(data->logfile);
    data->logfile = NULL;

    /***************************************************************/
    /* TODO: Temporary put it here, to check later */
    coco_archive_free(mo_archive); /* free the archive */
    free(mo_archive);
    /***************************************************************/
  }
}

static coco_problem_t *logger_nondominated(coco_problem_t *inner_problem, const size_t max_size_of_archive,
    const char *path) {
  logger_nondominated_t *data;
  coco_problem_t *self;

  data = coco_allocate_memory(sizeof(*data));
  data->number_of_evaluations = 0;
  data->path = coco_strdup(path);
  data->logfile = NULL; /* Open lazily in private_logger_nondominated_evaluate_function(). */
  data->max_size_of_archive = max_size_of_archive;
  self = coco_transformed_allocate(inner_problem, data, private_logger_nondominated_free);
  self->evaluate_function = private_logger_nondominated_evaluate;
  return self;
}
