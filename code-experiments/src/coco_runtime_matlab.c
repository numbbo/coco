/**
 * @file coco_runtime_matlab.c
 * @brief Specific COCO runtime implementation for the Matlab language
 * that replaces coco_runtime_c.c with the Matlab-specific counterparts.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <mex.h>

#include "coco.h"
#include "coco_utilities.c"



void coco_error(const char *message, ...) {
  va_list args;

  mexPrintf("COCO FATAL ERROR: ");
  mexPrintf(message, args);
  mexPrintf("\n");
  exit(EXIT_FAILURE);
}

void coco_warning(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_WARNING) {
    mexPrintf("COCO WARNING: ");
    mexPrintf(message, args);
    mexPrintf("\n");
  }
}

void coco_info(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_INFO) {
    mexPrintf("COCO INFO: ");
    mexPrintf(message, args);
    mexPrintf("\n");
  }
}

/**
 * A function similar to coco_info that prints only the given message without any prefix and without
 * adding a new line.
 */
void coco_info_partial(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_INFO) {
    mexPrintf(message, args);
  }
}

void coco_debug(const char *message, ...) {
  va_list args;

  if (coco_log_level >= COCO_DEBUG) {
    mexPrintf("COCO DEBUG: ");
    mexPrintf(message, args);
    mexPrintf("\n");
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
