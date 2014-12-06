#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_strdup.c"

/* Figure out if we are on a sane platform or on the dominant platform. */
#if defined(_WIN32) || defined(_WIN64)
  #include <windows.h>
  static const char *coco_path_separator = "\\";
  #define NUMBBO_PATH_MAX MAX_PATH
  #define HAVE_GFA 1
#elif defined(__gnu_linux__) 
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <linux/limits.h>
  static const char *coco_path_separator = "/";
  #define HAVE_STAT 1
  #define NUMBBO_PATH_MAX PATH_MAX
#elif defined(__APPLE__)
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <sys/syslimits.h>
  static const char *coco_path_separator = "/";
  #define HAVE_STAT 1
  #define NUMBBO_PATH_MAX PATH_MAX
#elif defined(__FreeBSD__)
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <limits.h>
  static const char *coco_path_separator = "/";
  #define HAVE_STAT 1
  #define NUMBBO_PATH_MAX PATH_MAX
#elif defined(__sun) || defined(sun)
  #if defined(__SVR4) || defined(__svr4__)
    /* Solaris */
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <limits.h>
    static const char *coco_path_separator = "/";
    #define HAVE_STAT 1
    #define NUMBBO_PATH_MAX PATH_MAX
  #endif
#else
  #error Unknown platform
#endif

#if !defined(NUMBBO_PATH_MAX)
  #error NUMBBO_PATH_MAX undefined
#endif

void coco_join_path(char *path, size_t path_max_length, ...) {
  const size_t path_separator_length = strlen(coco_path_separator);
  va_list args;
  char *path_component;
  size_t path_length = strlen(path);

  va_start(args, path_max_length);
  while (NULL != (path_component = va_arg(args, char *))) {
    size_t component_length = strlen(path_component);
    if (path_length + path_separator_length + component_length >=
        path_max_length) {
      coco_error("coco_file_path() failed because the ${path} is to short.");
      return; /* never reached */
    }
    /* Both should be safe because of the above check. */
    if (strlen(path) > 0)
      strncat(path, coco_path_separator, path_max_length - strlen(path) - 1);
    strncat(path, path_component, path_max_length - strlen(path) - 1);
  }
  va_end(args);
}

int coco_path_exists(const char *path) {
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributes(path);
  return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
          (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#elif defined(HAVE_STAT)
  struct stat buf;
  int res = stat(path, &buf);
  return res == 0;
#else
#error Ooops
#endif
}

void coco_create_path(const char *path) {
#if defined(HAVE_GFA)
  /* FIXME: Unimplemented for now. */
  /* Nothing to do if the path exists. */
  if (coco_path_exists(path))
    return;

#elif defined(HAVE_STAT)
  char *tmp = NULL;
  char buf[4096];
  char *p;
  size_t len = strlen(path);
  assert(strcmp(coco_path_separator, "/") == 0);

  /* Nothing to do if the path exists. */
  if (coco_path_exists(path))
    return;

  tmp = coco_strdup(path);
  /* Remove possible trailing slash */
  if (tmp[len - 1] == '/')
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      if (!coco_path_exists(tmp)) {
        if (0 != mkdir(tmp, S_IRWXU))
          goto error;
      }
      *p = '/';
    }
  }
  if (0 != mkdir(tmp, S_IRWXU))
    goto error;
  coco_free_memory(tmp);
  return;
error:
  snprintf(buf, sizeof(buf), "mkdir(\"%s\") failed.", tmp);
  coco_error(buf);
  return; /* never reached */
#else
#error Ooops
#endif
}

double *coco_allocate_vector(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(double);
  return (double *)coco_allocate_memory(block_size);
}

double *coco_duplicate_vector(const double *src,
                              const size_t number_of_elements) {
  size_t i;
  double *dst;

  assert(src != NULL);
  assert(number_of_elements > 0);

  dst = coco_allocate_vector(number_of_elements);
  for (i = 0; i < number_of_elements; ++i) {
    dst[i] = src[i];
  }
  return dst;
}
