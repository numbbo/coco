#include <stdlib.h>
#include <stdio.h>
#include "coco.h"

/**
 * Checks the dominance relation in the unconstrained minimization case between
 * objectives1 and objectives2 and returns:
 *  1 if objectives1 dominates objectives2
 *  0 if objectives1 and objectives2 are non-dominated
 * -1 if objectives2 dominates objectives1
 * -2 if objectives1 is identical to objectives2
 */
static int mo_get_dominance(const double *objectives1, const double *objectives2, const size_t num_obj) {
  /* TODO: Should we care about comparison precision? */
  size_t i;

  int flag1 = 0;
  int flag2 = 0;

  for (i = 0; i < num_obj; i++) {
    if (objectives1[i] < objectives2[i]) {
      flag1 = 1;
    } else if (objectives1[i] > objectives2[i]) {
      flag2 = 1;
    }
  }

  if (flag1 && !flag2) {
    return 1;
  } else if (!flag1 && flag2) {
    return -1;
  } else if (flag1 && flag2) {
    return 0;
  } else { /* (!flag1 && !flag2) */
    return -2;
  }
}
