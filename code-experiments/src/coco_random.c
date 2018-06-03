/**
 * @file coco_random.c
 * @brief Definitions of functions regarding COCO random numbers.
 *
 * @note This file contains non-C89-standard types (such as uint32_t and uint64_t), which should
 * eventually be fixed.
 */

#include <math.h>

#include "coco.h"
#include <stdio.h>

#define COCO_NORMAL_POLAR /* Use polar transformation method */

#define COCO_SHORT_LAG 273
#define COCO_LONG_LAG 607

/**
 * @brief A structure containing the state of the COCO random generator.
 */
struct coco_random_state_s {
  double x[COCO_LONG_LAG];
  size_t index;
};

/**
 * @brief A lagged Fibonacci random number generator.
 *
 * This generator is nice because it is reasonably small and directly generates double values. The chosen
 * lags (607 and 273) lead to a generator with a period in excess of 2^607-1.
 */
static void coco_random_generate(coco_random_state_t *state) {
  size_t i;
  for (i = 0; i < COCO_SHORT_LAG; ++i) {
    double t = state->x[i] + state->x[i + (COCO_LONG_LAG - COCO_SHORT_LAG)];
    if (t >= 1.0)
      t -= 1.0;
    state->x[i] = t;
  }
  for (i = COCO_SHORT_LAG; i < COCO_LONG_LAG; ++i) {
    double t = state->x[i] + state->x[i - COCO_SHORT_LAG];
    if (t >= 1.0)
      t -= 1.0;
    state->x[i] = t;
  }
  state->index = 0;
}

coco_random_state_t *coco_random_new(uint32_t seed) {
  coco_random_state_t *state = (coco_random_state_t *) coco_allocate_memory(sizeof(*state));
  size_t i;
  /* printf("coco_random_new(): %u\n", seed); */
  /* Expand seed to fill initial state array. */
  for (i = 0; i < COCO_LONG_LAG; ++i) {
    /* Uses uint64_t to silence the compiler ("shift count negative or too big, undefined behavior" warning) */
    state->x[i] = ((double) seed) / (double) (((uint64_t) 1UL << 32) - 1);
    /* Advance seed based on simple RNG from TAOCP */
    seed = (uint32_t) 1812433253UL * (seed ^ (seed >> 30)) + ((uint32_t) i + 1);
  }
  state->index = 12;
  /* coco_random_generate(state); */
  return state;
}

void coco_random_free(coco_random_state_t *state) {
  coco_free_memory(state);
}

double coco_random_uniform(coco_random_state_t *state) {
  /* If we have consumed all random numbers in our archive, it is time to run the actual generator for one
   * iteration to refill the state with 'LONG_LAG' new values. */
  if (state->index >= COCO_LONG_LAG)
    coco_random_generate(state);
  return state->x[state->index++];
}

/**
 * Instead of using the (expensive) polar method, we may cheat and abuse the central limit theorem. The sum
 * of 12 uniform random values has mean 6, variance 1 and is approximately normal. Subtract 6 and you get
 * an approximately N(0, 1) random number.
 */
double coco_random_normal(coco_random_state_t *state) {
  double normal;
#ifdef COCO_NORMAL_POLAR
  const double u1 = coco_random_uniform(state);
  const double u2 = coco_random_uniform(state);
  normal = sqrt(-2 * log(u1)) * cos(2 * coco_pi * u2);
#else
  int i;
  normal = 0.0;
  for (i = 0; i < 12; ++i) {
    normal += coco_random_uniform(state);
  }
  normal -= 6.0;
#endif
  return normal;
}

/* Be hygienic (for amalgamation) and undef lags. */
#undef COCO_SHORT_LAG
#undef COCO_LONG_LAG
