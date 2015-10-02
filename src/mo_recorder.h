#ifndef MO_RECORDER_H
#define	MO_RECORDER_H


#ifdef	__cplusplus
extern "C" {
#endif


struct mococo_solution_entry {
  int status;    /* 0: inactive | 1: active */
  size_t birth;  /* timestamp to know which are newly created */
  double *var;
  double *obj;
};

struct mococo_solutions_archive {
  size_t maxsize;
  size_t maxupdatesize;
  size_t size;
  size_t updatesize;
  size_t numvar;
  size_t numobj;
  struct mococo_solution_entry *entry;
  struct mococo_solution_entry **active;
  struct mococo_solution_entry **update;
};

void mococo_allocate_archive(struct mococo_solutions_archive *archive, size_t maxsize, size_t sizeVar, size_t sizeObj, size_t maxUpdate);
void mococo_free_archive(struct mococo_solutions_archive *archive);
void mococo_reset_archive(struct mococo_solutions_archive *archive);
void mococo_push_to_archive(const double **pop, double **obj, struct mococo_solutions_archive *archive, size_t nPop, size_t timestamp);
void mococo_mark_updates(struct mococo_solutions_archive *archive, size_t timestamp);
/* void mococo_recorder(const char *mode); */


#ifdef	__cplusplus
}
#endif

#endif	/* MO_RECORDER_H */
