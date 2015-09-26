#include <sys/types.h> /* for creating folder */
#include <sys/stat.h>  /* for creating folder */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mo_recorder.h"


void mococo_allocate_archive(struct mococo_solutions_archive *archive, size_t maxsize, size_t sizeVar, size_t sizeObj, size_t maxUpdate) {
    size_t i;
  
    archive->maxsize = maxsize;
    archive->maxupdatesize = maxUpdate;
    archive->size = 0;
    archive->updatesize = 0;
    archive->numvar = sizeVar;
    archive->numobj = sizeObj;
    archive->entry  = (struct mococo_solution_entry *) malloc(maxsize * sizeof(struct mococo_solution_entry));
    archive->active = (struct mococo_solution_entry **) malloc(maxsize * sizeof(struct mococo_solution_entry*));
    archive->update = (struct mococo_solution_entry **) malloc(maxUpdate * sizeof(struct mococo_solution_entry*));
    if (archive->entry == NULL || archive->active == NULL || archive->update == NULL) {
        fprintf(stderr, "ERROR in allocating memory for the archive.\n");
        exit(EXIT_FAILURE);
    }
    
    /* Allocate memory for each solution entry */
    for (i=0; i < maxsize; i++) {
        archive->entry[i].status = 0;  /* 0: inactive | 1: active */
        archive->entry[i].birth = 0;
        archive->entry[i].var = (double*) malloc(sizeVar * sizeof(double));
        archive->entry[i].obj = (double*) malloc(sizeObj * sizeof(double));
        if (archive->entry[i].var == NULL || archive->entry[i].obj == NULL) {
            fprintf(stderr, "ERROR in allocating memory for some entry of the archive.\n");
            exit(EXIT_FAILURE);
        }
    }
}

void mococo_reset_archive(struct mococo_solutions_archive *archive) {
    size_t i;
    
    archive->size = 0;
    archive->updatesize = 0;
    for (i=0; i < archive->maxsize; i++) {
        archive->entry[i].status = 0;
        archive->entry[i].birth = 0;
    }
}

void mococo_free_archive(struct mococo_solutions_archive *archive) {
    size_t i;  
    for (i=0; i < archive->maxsize; i++) {
        free(archive->entry[i].var);
        free(archive->entry[i].obj);
    }
    free(archive->update);
    free(archive->active);
    free(archive->entry);
}


void mococo_push_to_archive(const double **pop, double **obj, struct mococo_solutions_archive *archive, size_t nPop, size_t timestamp) {
    struct mococo_solution_entry *entry;
    size_t s = archive->size;
    size_t tnext = 0;
    size_t i;
    size_t t;
    size_t j;
    size_t k;
        
    for (i=0; i < nPop; i++) {
        /* Find a non-active slot for the new i-th solution */
        for (t = tnext; t < archive->maxsize; t++) {
            if (archive->entry[t].status == 0) {
                archive->active[s] = &(archive->entry[t]);
                tnext = t + 1;
                break;
            }
        }
        /* Keep the i-th solution in the slot found */
        entry = archive->active[s];
        entry->status = 1;
        entry->birth = timestamp;
        for (j=0; j < archive->numvar; j++)   /* all decision variables of a solution */
            entry->var[j] = pop[i][j];
        for (k=0; k < archive->numobj; k++)   /* all objective values of a solution */
            entry->obj[k] = obj[i][k];
        s++;
    }
    archive->size = s;
}


void mococo_mark_updates(struct mococo_solutions_archive *archive, size_t timestamp) {
    size_t u = 0;
    size_t i;
    for (i=0; i < archive->size; i++) {
        if (archive->active[i]->birth == timestamp) {
            archive->update[u] = archive->active[i];
            u++;
        }
    }
    archive->updatesize = u;
}

