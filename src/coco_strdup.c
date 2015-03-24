#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "coco.h"

char *coco_strdup(const char *string);
char * coco_strdupf(const char *str, ...);
char * coco_vstrdupf(const char *str, va_list args);
char *coco_strconcat(const char *s1, const char *s2); 
long coco_strfind(const char *base, const char *seq); 

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
char * coco_strdupf(const char *str, ...) {
  va_list args;
  char *s;

  va_start(args, str);
  s = coco_vstrdupf(str, args);
  va_end(args);
  return s;
}
char * coco_vstrdupf(const char *str, va_list args) {
  static char buf[444];
#if 1 /* this might not be defined on very old systems */
  vsnprintf(buf, 442, str, args); /* apparently args cannot be used another time */
#else /* less save alternative */
  assert(strlen(str) < 222);
  vsprintf(buf, str, args); /* apparently args cannot be used another time */
#endif
  return coco_strdup(buf);
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

/**
 * return index where ${seq} occurs in ${base}, -1 if it doesn't.
 *
 * If there is an equivalent standard C function, this should be removed. 
 */
long coco_strfind(const char *base, const char *seq) {
  const size_t strlen_seq = strlen(seq);
  const size_t last_first_idx = strlen(base) - strlen(seq);
  size_t i, j;
  
  for (i = 0; i <= last_first_idx; ++i) {
    if (base[i] == seq[0]) {
      for (j = 0; j < strlen_seq; ++j) {
        if (base[i+j] != seq[j])
          break;
      }
      if (j == strlen_seq) {
        if (i > 1e9)
          coco_error("coco_strfind(): strange values observed i=%lu, j=%lu, strlen(base)=%lu", i, j, strlen(base));
        return (long)i;
      }
    }
  }
  return -1;
}
