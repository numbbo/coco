#include <assert.h>

#include "numbbo_generics.c"

#include "f_sphere.c"
#include "f_ellipsoid.c"
#include "f_rastrigin.c"
#include "f_rosenbrock.c"
#include "f_skewRastriginBueche.c"
#include "f_linearSlope.c"

#include "shift_objective.c"
#include "shift_variables.c"

/**
 * bbob2009_allocate_matrix(n, m):
 *
 * Allocate a ${n} by ${m} matrix structured as an array of pointers
 * to double arrays.
 */
static double** bbob2009_allocate_matrix(const size_t n, const size_t m) {
    double **matrix = NULL;
    matrix = (double **)numbbo_allocate_memory(sizeof(double *) * n);
    for (size_t i = 0; i < n; ++i) {
        matrix[i] = numbbo_allocate_vector(m);
    }
    return matrix;
}

static void bbob2009_free_matrix(double **matrix, const size_t n, const size_t m) {
    for (size_t i = 0; i < n; ++i) {
        numbbo_free_memory(matrix[i]);
        matrix[i] = NULL;
    }
    numbbo_free_memory(matrix);
}


/**
* bbob2009_unif(r, N, inseed):
 *
 * Generate N uniform random numbers using ${inseed} as the seed and
 * store them in ${r}.
 */
static void bbob2009_unif(double* r, int N, int inseed) {
    /* generates N uniform numbers with starting seed*/
    int aktseed;
    int tmp;
    int rgrand[32];
    int aktrand;
    int i;

    if (inseed < 0) inseed = -inseed;
    if (inseed < 1) inseed = 1;
    aktseed = inseed;
    for (i = 39; i >= 0; i--) {
        tmp = (int)floor((double)aktseed/(double)127773);
        aktseed = 16807  * (aktseed - tmp * 127773) - 2836 * tmp;
        if (aktseed < 0)
            aktseed = aktseed + 2147483647;
        if (i < 32)
           rgrand[i] = aktseed;
    }
    aktrand = rgrand[0];
    for (i = 0; i < N; i++) {
        tmp = (int)floor((double)aktseed/(double)127773);
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
        if (aktseed < 0)
            aktseed = aktseed + 2147483647;
        tmp = (int)floor((double)aktrand / (double)67108865);
        aktrand = rgrand[tmp];
        rgrand[tmp] = aktseed;
        r[i] = (double)aktrand/2.147483647e9;
        if (r[i] == 0.) {
            r[i] = 1e-99;
        }
    }
    return;
}

/**
 * bbob2009_reshape(B, vector, m, n):
 *
 * Convert from packed matrix storage to an array of array of double
 * representation.
 */
static double** bbob2009_reshape(double** B, double* vector, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[i][j] = vector[j * m + i];
        }
    }
    return B;
}

/**
 * bbob2009_gauss(g, N, seed)
 *
 * Generate ${N} Gaussian random numbers using the seed ${seed} and
 * store them in ${g}.
 */
static void bbob2009_gauss(double *g, int N, int seed) {
    assert(2*N < 3000);
    double uniftmp[3000];
    bbob2009_unif(uniftmp, 2*N, seed);

    for (int i = 0; i < N; i++) {
        g[i] = sqrt(-2*log(uniftmp[i])) * cos(2*numbbo_pi*uniftmp[N+i]);
        if (g[i] == 0.)
            g[i] = 1e-99;
    }
    return;
}

/**
 * bbob2009_compute_rotation(B, seed, DIM):
 *
 * Compute a ${DIM}x${DIM} rotation matrix based on ${seed} and store
 * it in ${B}.
 */
static void bbob2009_compute_rotation(double **B, int seed, int DIM) {
    /* To ensure temporary data fits into gvec */
    assert(DIM * DIM < 2000);
    double prod;
    double gvect[2000];
    int i, j, k; /*Loop over pairs of column vectors*/

    bbob2009_gauss(gvect, DIM * DIM, seed);
    bbob2009_reshape(B, gvect, DIM, DIM);
    /*1st coordinate is row, 2nd is column.*/

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < i; j++) {
            prod = 0;
            for (k = 0; k < DIM; k++)
                prod += B[k][i] * B[k][j];
            for (k = 0; k < DIM; k++) 
                B[k][i] -= prod * B[k][j];
        }
        prod = 0;
        for (k = 0; k < DIM; k++)
            prod += B[k][i] * B[k][i];
        for (k = 0; k < DIM; k++) 
            B[k][i] /= sqrt(prod);
    }
}

/**
 * bbob2009_compute_xopt(xopt, seed, DIM):
 *
 * Randomly compute the location of the global optimum. 
 */ 
static void bbob2009_compute_xopt(double *xopt, int seed, int DIM) {
    bbob2009_unif(xopt, DIM, seed);
    for (int i = 0; i < DIM; i++) {
        xopt[i] = 8 * floor(1e4 * xopt[i])/1e4 - 4;
        if (xopt[i] == 0.0)
            xopt[i] = -1e-5;
    }
}

/**
 * bbob2009_compute_fopt(function_id, instance_id):
 *
 * Randomly choose the objective offset for function ${function_id}
 * and instance ${instance_id}.
 */
double bbob2009_compute_fopt(int function_id, int instance_id) {
    int rseed, rrseed;
    if (function_id == 4)
        rseed = 3;
    else if (function_id == 18)
        rseed = 17;
    else if (function_id == 101 || function_id == 102 || function_id == 103 || 
             function_id == 107 ||  function_id == 108 || function_id == 109)
        rseed = 1;
    else if (function_id == 104 || function_id == 105 || function_id == 106 ||
             function_id == 110 || function_id == 111 || function_id == 112)
        rseed = 8;
    else if (function_id == 113 || function_id == 114 || function_id == 115)
        rseed = 7;
    else if (function_id == 116 || function_id == 117 || function_id == 118)
        rseed = 10;
    else if (function_id == 119 || function_id == 120 || function_id == 121)
        rseed = 14;
    else if (function_id == 122 || function_id == 123 || function_id == 124)
        rseed = 17;
    else if (function_id == 125 || function_id == 126 || function_id == 127)
        rseed = 19;
    else if (function_id == 128 || function_id == 129 || function_id == 130)
        rseed = 21;
    else
        rseed = function_id;

    rrseed = rseed + 10000 * instance_id;
    double gval, gval2;
    bbob2009_gauss(&gval, 1, rrseed);
    bbob2009_gauss(&gval2, 1, rrseed + 1);
    return fmin(1000., fmax(-1000., (round(100.*100.*gval/gval2)/100.)));
}

/**
 * bbob2009_decode_function_index(function_index, function_id, instance_id, dimension):
 * 
 * Decode the new function_index into the old convention of function,
 * instance and dimension. We have 24 functions in 6 different
 * dimensions so a total of 144 functions and any number of
 * instances. A natural thing would be to order them so that the
 * function varies faster than the dimension which is still faster
 * than the instance. For analysis reasons we want something
 * different. Our goal is to quickly produce 5 repetitions of a single
 * function in one dimension, then vary the function, then the
 * dimension.
 *
 *This gives us:
 *
 * function_index | function_id | instance_id | dimension
 * ---------------+-------------+-------------+-----------
 *              0 |           1 |           1 |         2
 *              1 |           1 |           2 |         2
 *              2 |           1 |           3 |         2
 *              3 |           1 |           4 |         2
 *              4 |           1 |           5 |         2
 *              5 |           2 |           1 |         2
 *              6 |           2 |           2 |         2
 *             ...           ...           ...         ... 
 *            119 |          24 |           5 |         2
 *            120 |           1 |           1 |         3
 *            121 |           1 |           2 |         3
 *             ...           ...           ...         ... 
 *           2157 |          24 |           13|        40
 *           2158 |          24 |           14|        40
 *           2159 |          24 |           15|        40
 *
 * The quickest way to decode this is using integer division and
 * remainders.
 */
void bbob2009_decode_function_index(const int function_index,
                                    int *function_id,
                                    int *instance_id,
                                    int *dimension) {
    static const int dims[] = {2, 3, 5, 10, 20, 40};
    static const int number_of_consecutive_instances = 5;
    static const int number_of_functions = 24;
    static const int number_of_dimensions = 6;
    const int high_instance_id = function_index / 
        (number_of_consecutive_instances * number_of_functions * number_of_dimensions);
    int rest = function_index %
        (number_of_consecutive_instances * number_of_functions * number_of_dimensions);
    *dimension = dims[rest / (number_of_consecutive_instances * number_of_functions)];
    rest = rest % (number_of_consecutive_instances * number_of_functions);
    *function_id = rest / number_of_consecutive_instances + 1;
    rest = rest % number_of_consecutive_instances;
    const int low_instance_id = rest + 1;
    *instance_id = low_instance_id + 5 * high_instance_id;
}

/**
 * bbob2009_suit(function_index):
 *
 * Return the ${function_index}-th benchmark problem from the BBOB2009
 * benchmark suit. If the function index is out of bounds, return *
 * NULL.
 */
numbbo_problem_t *bbob2009_suit(const int function_index) {
    int instance_id, function_id, dimension;
    numbbo_problem_t *problem = NULL;
    bbob2009_decode_function_index(function_index, &function_id, &instance_id, 
                                   &dimension);
    int rseed = function_id + 10000 * instance_id;
    
    /* Break if we are past our 15 instances. */
    if (instance_id > 15) return NULL;

    if (function_id == 1) {
        double offset[40];
        problem = sphere_problem(dimension);
        bbob2009_compute_xopt(offset, rseed, dimension);
        problem = shift_variables(problem, offset, false);
        problem = shift_objective(problem, 
                                  bbob2009_compute_fopt(function_id, instance_id));
    } else if (function_id == 2) {
        problem = ellipsoid_problem(dimension);
    } else if (function_id == 3) {
        problem = rastrigin_problem(dimension);
    } else if (function_id == 4) {
        problem = skewRastriginBueche_problem(dimension);
    } else if (function_id == 5) {
        problem = linearSlope_problem(dimension);
    } else if (function_id == 6) {
        double offset[40];
        double **rot1, **rot2;
        rot1 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot1, rseed + 1000000, dimension);

        rot2 = bbob2009_allocate_matrix(dimension, dimension);
        bbob2009_compute_rotation(rot2, rseed, dimension);

        problem = rosenbrock_problem(dimension);
        /* IMPORTANT: Penalize before any transformations of the variables! */
        #if 0
        problem = penalize_uninteresting_values(problem);
        #endif
        bbob2009_compute_xopt(offset, rseed, dimension);
        problem = shift_variables(problem, offset, false);

        problem = shift_objective(problem, 
                                  bbob2009_compute_fopt(function_id, instance_id));
    } else {
        return NULL;
    }

    return problem;
}
