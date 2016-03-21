/**
 * @file coco_runtime_c.c
 * @brief Generic COCO runtime implementation for the C language.
 *
 * Other language interfaces might want to replace this so that memory allocation and error handling go
 * through the respective language runtime.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "coco.h"
#include "coco_utilities.c"

void coco_error(const char *message, ...) {
  va_list args;

  fprintf(stderr, "COCO FATAL ERROR: ");
  va_start(args, message);
  vfprintf(stderr, message, args);
  va_end(args);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

void coco_warning(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_WARNING) {
    fprintf(stderr, "COCO WARNING: ");
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fprintf(stderr, "\n");
  }
}

void coco_info(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_INFO) {
    fprintf(stdout, "COCO INFO: ");
    va_start(args, message);
    vfprintf(stdout, message, args);
    va_end(args);
    fprintf(stdout, "\n");
    fflush(stdout);
  }
}

/**
 * A function similar to coco_info that prints only the given message without any prefix and without
 * adding a new line.
 */
void coco_info_partial(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_INFO) {
    va_start(args, message);
    vfprintf(stdout, message, args);
    va_end(args);
    fflush(stdout);
  }
}

void coco_debug(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_DEBUG) {
    fprintf(stdout, "COCO DEBUG: ");
    va_start(args, message);
    vfprintf(stdout, message, args);
    va_end(args);
    fprintf(stdout, "\n");
    fflush(stdout);
  }
}

void *coco_allocate_memory(const size_t size) {
  void *data;
  if (size == 0) {
    coco_error("coco_allocate_memory() called with 0 size.");
    return NULL; /* never reached */
  }
  data = malloc(size);
  if (data == NULL)
    coco_error("coco_allocate_memory() failed.");
  return data;
}

void coco_free_memory(void *data) {
  free(data);
}
