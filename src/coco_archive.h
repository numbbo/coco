#ifndef __COCO_ARCHIVE__
#define	__COCO_ARCHIVE__

#ifdef	__cplusplus
extern "C" {
#endif

typedef struct {
  int status; /* 0: inactive | 1: active */
  size_t birth; /* time stamp to know which are newly created */
  double *var;
  double *obj;
} coco_archive_entry_t;

typedef struct {
  size_t max_size;
  size_t max_update_size;
  size_t size;
  size_t update_size;
  size_t num_var;
  size_t num_obj;
  coco_archive_entry_t *entry;
  coco_archive_entry_t **active;
  coco_archive_entry_t **update;
} coco_archive_t;

void coco_archive_allocate(coco_archive_t *archive,
                           size_t max_size,
                           size_t size_var,
                           size_t size_obj,
                           size_t max_update);
void coco_archive_free(coco_archive_t *archive);
void coco_archive_reset(coco_archive_t *archive);
void coco_archive_push(coco_archive_t *archive,
                       const double **var,
                       double **obj,
                       size_t num_var,
                       size_t time_stamp);
void coco_archive_mark_updates(coco_archive_t *archive, size_t time_stamp);

#ifdef	__cplusplus
}
#endif

#endif
