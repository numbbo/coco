/*
 * Legacy code from BBOB2009 required to replicate the 2009 functions.
 *
 * All of this code should only be used by the bbob2009_suit functions
 * to provide compatibility to the legacy code. New test beds should
 * strive to use the new numbbo facilities for random number
 * generation etc.
 */

#include <math.h>
#include <stdio.h>
#include "coco.h"

static double bbob2009_fmin(double a, double b) {
    return (a < b) ? a : b;
}

static double bbob2009_fmax(double a, double b) {
    return (a > b) ? a : b;
}

static double bbob2009_round(double x) {
    return floor(x + 0.5);
}

/**
 * bbob2009_allocate_matrix(n, m):
 *
 * Allocate a ${n} by ${m} matrix structured as an array of pointers
 * to double arrays.
 */
static double** bbob2009_allocate_matrix(const size_t n, const size_t m) {
    double **matrix = NULL;
    size_t i;
    matrix = (double **)coco_allocate_memory(sizeof(double *) * n);
    for (i = 0; i < n; ++i) {
        matrix[i] = coco_allocate_vector(m);
    }
    return matrix;
}

static void bbob2009_free_matrix(double **matrix, const size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (matrix[i] != NULL) {
            coco_free_memory(matrix[i]);
            matrix[i] = NULL;
        }
    }
    coco_free_memory(matrix);
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
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
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
    int i;
    double uniftmp[6000];
    assert(2 * N < 6000);
    bbob2009_unif(uniftmp, 2*N, seed);

    for (i = 0; i < N; i++) {
        g[i] = sqrt(-2*log(uniftmp[i])) * cos(2*coco_pi*uniftmp[N+i]);
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
    double prod;
    double gvect[2000];
    int i, j, k; /*Loop over pairs of column vectors*/

    assert(DIM * DIM < 2000);

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
    int i;
    bbob2009_unif(xopt, DIM, seed);
    for (i = 0; i < DIM; i++) {
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
    double gval, gval2;

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
    bbob2009_gauss(&gval, 1, rrseed);
    bbob2009_gauss(&gval2, 1, rrseed + 1);
    return bbob2009_fmin(1000.,
                         bbob2009_fmax(-1000.,
                                       bbob2009_round(100.*100.*gval/gval2)/100.));
}
