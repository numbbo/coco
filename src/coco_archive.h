#ifndef MO_RECORDER_H
#define	MO_RECORDER_H

#ifdef	__cplusplus
extern "C" {
#endif

struct coco_arhive_entry {
  int status;    /* 0: inactive | 1: active */
  size_t birth;  /* time stamp to know which are newly created */
  double *var;
  double *obj;
};

struct coco_archive {
  size_t max_size;
  size_t max_update_size;
  size_t size;
  size_t update_size;
  size_t num_var;
  size_t num_obj;
  struct coco_arhive_entry *entry;
  struct coco_arhive_entry **active;
  struct coco_arhive_entry **update;
};

void coco_allocate_archive(struct coco_archive *archive, size_t max_size, size_t size_var,
    size_t size_obj, size_t max_update);
void coco_free_archive(struct coco_archive *archive);
void coco_reset_archive(struct coco_archive *archive);
void coco_push_to_archive(const double **pop, double **obj, struct coco_archive *archive,
    size_t n_pop, size_t time_stamp);
void coco_mark_updates(struct coco_archive *archive, size_t time_stamp);

#ifdef	__cplusplus
}
#endif

#endif
