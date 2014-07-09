#ifndef numbbo_xcode_logger_h
#define numbbo_xcode_logger_h

#endif

static const size_t nbpts_nbevals=20;
static const size_t nbpts_fval=5;

typedef struct {
    char *path;
    FILE *logfile;
    long idx_fval_trigger; // logging target = {10**(i/nbPtsF), i \in Z}
    double next_target;
    long idx_nbevals_trigger;
    long idx_dim_nbevals_trigger;
    double fTrigger;
    double evalsTrigger;
    long number_of_evaluations;
} logger_t;


void update_next_target(logger_t * state);