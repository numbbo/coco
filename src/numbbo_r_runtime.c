/*
 * R NUMBBO runtime implementation.
 */
#include <stdio.h>
#include <stdlib.h>

#include <R.h>
#include <Rinternals.h>

#include "numbbo.h"

/* After we have thrown an error, the framework is (by definition) in
 * an undefinded state and no further calls into the framework may by
 * made. We do not terminate immideatly because we want to give R a
 * chance to save any results or possibly recover.
 */
void numbbo_error(const char *message) {
    error(message);
}

void numbbo_warning(const char *message) {
    warning(message);
}

void *numbbo_allocate_memory(size_t size) {
    void *data = (void *)Calloc(size, char);
    /* This should never happen, but better safe than sorry. */
    if (data == NULL)
        numbbo_error("numbbo_calloc() failed.");
    return data;
}

void numbbo_free_memory(void *data) {
    Free(data);
}
