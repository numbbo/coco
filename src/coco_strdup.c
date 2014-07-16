#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "coco.h"

/**
 * coco_strdup(string):
 *
 * Create a duplicate copy of ${string} and return a pointer to
 * it. The caller is responsible for free()ing the memory allocated
 * using coco_free_memory().
 */
char *coco_strdup(const char *string) {    
    size_t len;
    char *duplicate;
    if (string == NULL)
        return NULL;
    len = strlen(string);
    duplicate = (char *)coco_allocate_memory(len + 1);
    memcpy(duplicate, string, len + 1);
    return duplicate;
}
