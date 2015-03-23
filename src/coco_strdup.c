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

/**
 * coco_strconcat(string1, string2):
 *
 * Return a concatenate copy of ${string1} + ${string1}. 
 * The caller is responsible for free()ing the memory allocated
 * using coco_free_memory().
 */
char *coco_strconcat(const char *s1, const char *s2) {
  size_t len1 = strlen(s1);
  size_t len2 = strlen(s2);
  char *s = (char *)coco_allocate_memory(len1 + len2 + 1);
  
  memcpy(s, s1, len1);
  memcpy(&s[len1], s2, len2 + 1);
  return s;
}
