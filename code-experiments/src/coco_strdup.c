#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "coco.h"

char *coco_strdup(const char *string);
char *coco_strdupf(const char *str, ...);
char *coco_vstrdupf(const char *str, va_list args);
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
  duplicate = (char *) coco_allocate_memory(len + 1);
  memcpy(duplicate, string, len + 1);
  return duplicate;
}
/**
 * Formatted string duplication. Optional arguments are
 * used like in sprintf. 
 */
char *coco_strdupf(const char *str, ...) {
  va_list args;
  char *s;

  va_start(args, str);
  s = coco_vstrdupf(str, args);
  va_end(args);
  return s;
}

#define coco_vstrdupf_buflen 444
/**
 * va_list version of formatted string duplication coco_strdupf()
 */
char *coco_vstrdupf(const char *str, va_list args) {
  static char buf[coco_vstrdupf_buflen];
  long written;
  /* apparently args can only be used once, therefore
   * len = vsnprintf(NULL, 0, str, args) to find out the
   * length does not work. Therefore we use a buffer
   * which limits the max length. Longer strings should
   * never appear anyway, so this is rather a non-issue. */

#if 0
  written = vsnprintf(buf, coco_vstrdupf_buflen - 2, str, args);
  if (written < 0)
  coco_error("coco_vstrdupf(): vsnprintf failed on '%s'", str);
#else /* less safe alternative, if vsnprintf is not available */
  assert(strlen(str) < coco_vstrdupf_buflen / 2 - 2);
  if (strlen(str) >= coco_vstrdupf_buflen / 2 - 2)
    coco_error("coco_vstrdupf(): string is too long");
  written = vsprintf(buf, str, args);
  if (written < 0)
    coco_error("coco_vstrdupf(): vsprintf failed on '%s'", str);
#endif
  if (written > coco_vstrdupf_buflen - 3)
    coco_error("coco_vstrdupf(): A suspiciously long string is tried to being duplicated '%s'", buf);
  return coco_strdup(buf);
}
#undef coco_vstrdupf_buflen

/**
 * coco_strconcat(string1, string2):
 *
 * Return a concatenate copy of ${string1} + ${string2}. 
 * The caller is responsible for free()ing the memory allocated
 * using coco_free_memory().
 */
char *coco_strconcat(const char *s1, const char *s2) {
  size_t len1 = strlen(s1);
  size_t len2 = strlen(s2);
  char *s = (char *) coco_allocate_memory(len1 + len2 + 1);

  memcpy(s, s1, len1);
  memcpy(&s[len1], s2, len2 + 1);
  return s;
}

/**
 * return first index where ${seq} occurs in ${base}, -1 if it doesn't.
 *
 * If there is an equivalent standard C function, this can/should be removed. 
 */
long coco_strfind(const char *base, const char *seq) {
  const size_t strlen_seq = strlen(seq);
  const size_t last_first_idx = strlen(base) - strlen(seq);
  size_t i, j;

  for (i = 0; i <= last_first_idx; ++i) {
    if (base[i] == seq[0]) {
      for (j = 0; j < strlen_seq; ++j) {
        if (base[i + j] != seq[j])
          break;
      }
      if (j == strlen_seq) {
        if (i > 1e9)
          coco_error("coco_strfind(): strange values observed i=%lu, j=%lu, strlen(base)=%lu", i, j,
              strlen(base));
        return (long) i;
      }
    }
  }
  return -1;
}
