#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coco_archive.h"

void coco_archive_allocate(coco_archive_t *archive, size_t max_size, size_t size_var, size_t size_obj,
    size_t max_update) {
  size_t i;

  archive->max_size = max_size;
  archive->max_update_size = max_update;
  archive->size = 0;
  archive->update_size = 0;
  archive->num_var = size_var;
  archive->num_obj = size_obj;
  archive->entry = (coco_archive_entry_t *) malloc(max_size * sizeof(coco_archive_entry_t));
  archive->active = (coco_archive_entry_t **) malloc(max_size * sizeof(coco_archive_entry_t*));
  archive->update = (coco_archive_entry_t **) malloc(max_update * sizeof(coco_archive_entry_t*));
  if (archive->entry == NULL || archive->active == NULL || archive->update == NULL) {
    fprintf(stderr, "ERROR in allocating memory for the archive.\n");
    exit(EXIT_FAILURE);
  }

  /* Allocate memory for each solution entry */
  for (i = 0; i < max_size; i++) {
    archive->entry[i].status = 0; /* 0: inactive | 1: active */
    archive->entry[i].birth = 0;
    archive->entry[i].var = (double*) malloc(size_var * sizeof(double));
    archive->entry[i].obj = (double*) malloc(size_obj * sizeof(double));
    if (archive->entry[i].var == NULL || archive->entry[i].obj == NULL) {
      fprintf(stderr, "ERROR in allocating memory for some entry of the archive.\n");
      exit(EXIT_FAILURE);
    }
  }
}

void coco_archive_reset(coco_archive_t *archive) {
  size_t i;

  archive->size = 0;
  archive->update_size = 0;
  for (i = 0; i < archive->max_size; i++) {
    archive->entry[i].status = 0;
    archive->entry[i].birth = 0;
  }
}

void coco_archive_free(coco_archive_t *archive) {
  size_t i;
  for (i = 0; i < archive->max_size; i++) {
    free(archive->entry[i].var);
    free(archive->entry[i].obj);
  }
  free(archive->update);
  free(archive->active);
  free(archive->entry);
}

void coco_archive_push(coco_archive_t *archive, const double **var, double **obj, size_t num_var,
    size_t time_stamp) {
  coco_archive_entry_t *entry;
  size_t s = archive->size;
  size_t tnext = 0;
  size_t i;
  size_t t;
  size_t j;
  size_t k;

  for (i = 0; i < num_var; i++) {
    /* Find a non-active slot for the new i-th solution */
    for (t = tnext; t < archive->max_size; t++) {
      if (archive->entry[t].status == 0) {
        archive->active[s] = &(archive->entry[t]);
        tnext = t + 1;
        break;
      }
    }
    /* Keep the i-th solution in the slot found */
    entry = archive->active[s];
    entry->status = 1;
    entry->birth = time_stamp;
    for (j = 0; j < archive->num_var; j++) /* all decision variables of a solution */
      entry->var[j] = var[i][j];
    for (k = 0; k < archive->num_obj; k++) /* all objective values of a solution */
      entry->obj[k] = obj[i][k];
    s++;
  }
  archive->size = s;
}

void coco_archive_mark_updates(coco_archive_t *archive, size_t time_stamp) {
  size_t u = 0;
  size_t i;
  for (i = 0; i < archive->size; i++) {
    if (archive->active[i]->birth == time_stamp) {
      archive->update[u] = archive->active[i];
      u++;
    }
  }
  archive->update_size = u;
}

