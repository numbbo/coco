/*
 * Generic NUMBBO runtime implementation.
 *
 * Other language interfaces might want to replace this so that memory
 * allocation and error handling go through the respective language
 * runtime.
 */
#include <stdio.h>
#include <stdlib.h>

#include "coco.h"

void coco_error(const char *message) {
  fprintf(stderr, "FATAL ERROR: %s\n", message);
  exit(EXIT_FAILURE);
}

void coco_warning(const char *message) {
  fprintf(stderr, "WARNING: %s\n", message);
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

void coco_free_memory(void *data) { free(data); }
