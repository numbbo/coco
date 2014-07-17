/*
 * R NUMBBO runtime implementation.
 */
#include <stdio.h>
#include <stdlib.h>

#include <R.h>
#include <Rinternals.h>

#include "coco.h"

/* After we have thrown an error, the framework is (by definition) in
 * an undefinded state and no further calls into the framework may by
 * made. We do not terminate immideatly because we want to give R a
 * chance to save any results or possibly recover.
 */
void coco_error(const char *message) {
    error(message);
}

void coco_warning(const char *message) {
    warning(message);
}

void *coco_allocate_memory(const size_t size) {
    void *data;
    data = (void *)Calloc(size, char);
    /* This should never happen, but better safe than sorry. */
    if (data == NULL)
        coco_error("coco_calloc() failed.");
    return data;
}

void coco_free_memory(void *data) {
    Free(data);
}
