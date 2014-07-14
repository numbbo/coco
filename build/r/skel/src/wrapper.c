#include <R.h>
#include <Rinternals.h>

#include "numbbo.h"

#define CHECK_ARG_IS_INT_VECTOR(A)                           \
    if (!isInteger(A) || !isVector(A))                       \
        error("Argument '" #A "' is not an integer vector.");

#define UNPACK_INT(S, I)                        \
    CHECK_ARG_IS_INT_VECTOR(S);                 \
    int I = INTEGER(S)[0];                      \

#define CHECK_ARG_IS_REAL_VECTOR(A)                      \
    if (!isReal(A) || !isVector(A))                      \
        error("Argument '" #A "' is not a real vector.");

#define UNPACK_REAL_VECTOR(S, D, N)             \
    CHECK_ARG_IS_REAL_VECTOR(S);                \
    double *D = REAL(S);                        \
    const R_len_t N = length(S);                   

static void coco_problem_finalizer(SEXP s_problem) {
    coco_problem_t *problem;
    problem = R_ExternalPtrAddr(s_problem);
    coco_free_problem(problem);
}

SEXP do_lower_bounds(SEXP s_problem) {
    R_len_t i;
    coco_problem_t *problem;
    problem = R_ExternalPtrAddr(s_problem);
    SEXP s_lower_bounds = allocVector(REALSXP, problem->number_of_parameters);
    double *lower_bounds = REAL(s_lower_bounds);
    for (i = 0; i < LENGTH(s_lower_bounds); ++i) {
        lower_bounds[i] = problem->lower_bounds[i];
    }
    return s_lower_bounds;
}

SEXP do_upper_bounds(SEXP s_problem) {
    R_len_t i;
    coco_problem_t *problem;
    problem = R_ExternalPtrAddr(s_problem);
    SEXP s_upper_bounds = allocVector(REALSXP, problem->number_of_parameters);
    double *upper_bounds = REAL(s_upper_bounds);
    for (i = 0; i < LENGTH(s_upper_bounds); ++i) {
        upper_bounds[i] = problem->upper_bounds[i];
    }
    return s_upper_bounds;
}

SEXP do_evaluate_function(SEXP s_problem, SEXP s_x) {
    R_len_t i;
    coco_problem_t *problem;
    problem = R_ExternalPtrAddr(s_problem);
    SEXP s_y = allocVector(REALSXP, problem->number_of_objectives);
    coco_evaluate_function(problem, REAL(s_x), REAL(s_y));
    return s_y;
}

SEXP do_get_problem(SEXP s_benchmark_name, 
                    SEXP s_function_index) {
    coco_problem_t *problem;

    const char *benchmark_name = CHAR(STRING_ELT(s_benchmark_name, 0));
    UNPACK_INT(s_function_index, function_index);

    problem = coco_get_problem(benchmark_name, function_index);
    if (problem == NULL) {
        return R_NilValue;
    } else {
        SEXP s_ret = R_MakeExternalPtr((void *)problem, 
                                       R_NilValue, R_NilValue);
        R_RegisterCFinalizer(s_ret, coco_problem_finalizer);
        return s_ret;
    }
}
