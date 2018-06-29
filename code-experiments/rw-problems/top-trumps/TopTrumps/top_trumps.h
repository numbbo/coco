/*
 * top_trumps.h
 *
 *  Created on: 29. jun. 2018
 *      Author: Tea Tusar
 */
#ifndef TOP_TRUMPS_H_
#define TOP_TRUMPS_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

double *top_trumps_evaluate(size_t function, size_t instance, size_t size_x,
    double *x, size_t size_y);

#ifdef __cplusplus
}
#endif

#endif /* TOP_TRUMPS_H_ */
