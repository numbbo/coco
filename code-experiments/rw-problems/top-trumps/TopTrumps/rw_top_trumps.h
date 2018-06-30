/*
 * rw_top_trumps.h
 *
 *  Created on: 29. jun. 2018
 *      Author: Tea Tusar
 */
#ifndef RW_TOP_TRUMPS_H_
#define RW_TOP_TRUMPS_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void top_trumps_evaluate(size_t function, size_t instance, size_t size_x,
    double *x, size_t size_y, double *y);

void top_trumps_test(void);

#ifdef __cplusplus
}
#endif

#endif /* RW_TOP_TRUMPS_H_ */
