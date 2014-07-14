#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "numbbo.h"

/**
 * numbbo_strdup(string):
 *
 * Create a duplicate copy of ${string} and return a pointer to
 * it. The caller is responsible for free()ing the memory allocated
 * using numbbo_free_memory().
 */
char *numbbo_strdup(const char *string) {    
    size_t len;
    char *duplicate;
    if (string == NULL)
        return NULL;
    len = strlen(string);
    duplicate = (char *)numbbo_allocate_memory(len + 1);
    memcpy(duplicate, string, len + 1);
    return duplicate;
}
