/*
 * Generic NUMBBO runtime implementation. 
 *
 * Other language interfaces might want to replace this so that memory
 * allocation and error handling go through the respective language
 * runtime.
 */
#include <stdio.h>
#include <stdlib.h>

#include "numbbo.h"

void numbbo_error(const char *message) {
    fprintf(stderr, "FATAL ERROR: %s\n", message);
    exit(EXIT_FAILURE);    
}

void numbbo_warning(const char *message) {
    fprintf(stderr, "WARNING: %s\n", message);
}

void *numbbo_allocate_memory(const size_t size) {
    if (size == 0)  {
        numbbo_error("numbbo_allocate_memory() called with 0 size.");
        return NULL; /* never reached */
    }
    void *data = malloc(size);
    if (data == NULL)
        numbbo_error("numbbo_allocate_memory() failed.");
    return data;
}
 
void numbbo_free_memory(void *data) {
    free(data);
}
