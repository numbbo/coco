/*
 * 
 */

#include <alloca.h>
#include <math.h>

/**
 * bbob2009_unif(r, N, inseed):
 *
 * Generate ${N} uniform random numbers based on ${inseed} and store
 * them in ${r}.
 *
 * NOTE: This generator is included for historical reasons. For new
 * benchmarks / test functions, please use numbbo_random functions.
 */
static void bbob2009_unif(double* r, int N, int inseed) {
    /* generates N uniform numbers with starting seed */
    int aktseed, tmp, rgrand[32], aktrand, i;

    if (inseed < 0)
        inseed = -inseed;
    if (inseed < 1)
        inseed = 1;
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
 * bbob2009_gauss(g, N, seed):
 *
 * Sample ${N} N(0, 1) pseudo random numbers based on ${seed} and
 * store them in ${g}.
 *
 * NOTE: For new benchmarks or test functions, please use
 * numbbo_normal_random().
 */
static void bbob2009_gauss(double * g, int N, int seed) {
    double *u;
    int i;
    /* FIXME: For large N this could fail. */
    u = alloca(sizeof(double) * 2 * N);
    bbob2009_unif(u, 2*N, seed);
    for (i = 0; i < N; i++) {
        g[i] = sqrt(-2.0 * log(u[i])) * cos(2 * M_PI * u[N + i]);
        if (g[i] == 0.0)
            g[i] = 1e-99;
    }
    return;
}

static double bbob2009_computeFopt(int function_id, int trial_id) {
    int rseed, rrseed;
    switch (function_id) {
    case 4:
        rseed = 3;
        break;
    case 18:
        rseed = 3;
        break;
    case 101: case 102: case 103: case 107: case 108: case 109:
        rseed = 1;
        break;
    case 104: case 105: case 106: case 110: case 111: case 112:
        rseed = 8;
        break;
    case 113: case 114: case 115:
        rseed = 7;
        break;
    case 116: case 117: case 118:
        rseed = 10;
        break;
    case 119: case 120: case 121:
        rseed = 14;
        break;
    case 122: case 123: case 124:
        rseed = 17;
        break;
    case 125: case 126: case 127:
        rseed = 19;
        break;
    case 128: case 129: case 130:
        rseed = 21;
        break;
    default:
        rseed = function_id;
        break;
    }
    double g1, g2;
    rrseed = rseed + 10000 * trial_id;
    bbob2009_gauss(&g1, 1, rrseed);
    bbob2009_gauss(&g2, 1, rrseed + 1);
    return fmin(1000., fmax(-1000., (round(100.0 * 100.0 * g1 / g2) / 100.0)));
}

static void bbob2009_computeXopt(double *Xopt, 
                                 int seed, int number_of_dimenseions) {
    int i;
    double *tmpvect;
    unif(Xopt, number_of_dimensions, seed);
    for (i = 0; i < number_of_dimensions; ++i) {
        Xopt[i] = 8 * floor(1e4 * Xopt[i]) / 1e4 - 4;
        if (Xopt[i] == 0.0)
            Xopt[i] = -1e-5;
    }
}
