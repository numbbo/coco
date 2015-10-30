#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_strdup.c"

/* Figure out if we are on a sane platform or on the dominant platform. */
#if defined(_WIN32) || defined(_WIN64) || defined(__MINGW64__) || defined(__CYGWIN__)
#include <windows.h>
static const char *coco_path_separator = "\\";
#define COCO_PATH_MAX MAX_PATH
#define HAVE_GFA 1
#elif defined(__gnu_linux__)
#include <sys/stat.h>
#include <sys/types.h>
#include <linux/limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#elif defined(__APPLE__)
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/syslimits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#elif defined(__FreeBSD__)
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#elif (defined(__sun) || defined(sun)) && (defined(__SVR4) || defined(__svr4__))
/* Solaris */
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
static const char *coco_path_separator = "/";
#define HAVE_STAT 1
#define COCO_PATH_MAX PATH_MAX
#else
#error Unknown platform
#endif

#if defined(HAVE_GFA)
#define S_IRWXU 0700
#endif

#if !defined(COCO_PATH_MAX)
#error COCO_PATH_MAX undefined
#endif

/* To get rid of the implicit-function-declaration warning. */
int mkdir(const char *pathname, mode_t mode);

/***********************************
 * Global definitions in this file
 * which are not in coco.h 
 ***********************************/
void coco_join_path(char *path, size_t path_max_length, ...);
int coco_path_exists(const char *path);
void coco_create_path(const char *path);
void coco_create_new_path(const char *path, size_t maxlen, char *new_path);
double *coco_duplicate_vector(const double *src, const size_t number_of_elements);
double coco_round_double(const double a);
double coco_max_double(const double a, const double b);
double coco_min_double(const double a, const double b);
/***********************************/

void coco_join_path(char *path, size_t path_max_length, ...) {
  const size_t path_separator_length = strlen(coco_path_separator);
  va_list args;
  char *path_component;
  size_t path_length = strlen(path);

  va_start(args, path_max_length);
  while (NULL != (path_component = va_arg(args, char *))) {
    size_t component_length = strlen(path_component);
    if (path_length + path_separator_length + component_length >= path_max_length) {
      coco_error("coco_file_path() failed because the ${path} is too short.");
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
  return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#elif defined(HAVE_STAT)
  struct stat buf;
  int res = stat(path, &buf);
  return res == 0;
#else
#error Ooops
#endif
}

void coco_create_path(const char *path) {
  /* current version should now work with Windows, Linux, and Mac */
  char *tmp = NULL;
  char *message;
  char *p;
  size_t len = strlen(path);
  char path_sep = coco_path_separator[0];

  /* Nothing to do if the path exists. */
  if (coco_path_exists(path))
    return;

  tmp = coco_strdup(path);
  /* Remove possible trailing slash */
  if (tmp[len - 1] == path_sep)
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++) {
    if (*p == path_sep) {
      *p = 0;
      if (!coco_path_exists(tmp)) {
        if (0 != mkdir(tmp, S_IRWXU))
          goto error;
      }
      *p = path_sep;
    }
  }
  if (0 != mkdir(tmp, S_IRWXU))
    goto error;
  coco_free_memory(tmp);
  return;
  error: message = "mkdir(\"%s\") failed";
  coco_error(message, tmp);
  return; /* never reached */
}

#if 0
/** path and new_path can be the same argument. 
 */
void coco_create_new_path(const char *path, size_t maxlen, char *new_path) {
  char sep = '_';
  size_t oldlen, len;
  time_t now;
  const char *snow;
  int i, tries;

  if (!coco_path_exists(path)) {
    coco_create_path(path);
    return;
  }

  maxlen -= 1; /* prevent failure from misinterpretation of what maxlen is */
  new_path[maxlen] = '\0';
  oldlen = strlen(path);
  assert(maxlen > oldlen);
  if (new_path != path)
  strncpy(new_path, path, maxlen);

  /* modify new_path name until path does not exist */
  for (tries = 0; tries <= (int)('z' - 'a'); ++tries) {
    /* create new name */
    now = time(NULL);
    snow = ctime(&now);
    /*                 012345678901234567890123
     * snow =         "Www Mmm dd hh:mm:ss yyyy"
     * new_path = "oldname_Mmm_dd_hh_mm_ss_yyyy[a-z]"
     *                    ^ oldlen
     */
    new_path[oldlen] = sep;
    strncpy(&new_path[oldlen+1], &snow[4], maxlen - oldlen - 1);
    for (i = oldlen; i < maxlen; ++i) {
      if (new_path[i] == ' ' || new_path[i] == ':')
      new_path[i] = sep;
      if (new_path[i] == '\n')
      new_path[i] = '\0';
      if (new_path[i] == '\0')
      break;
    }
    len = strlen(new_path);
    if (len > 6) {
      new_path[len - 5] = (char)(tries + 'a');
      new_path[len - 4] = '\0';
    }

    /* try new name */
    if (!coco_path_exists(new_path)) {
      /* not thread safe until path is created */
      coco_create_path(new_path);
      tries = -1;
      break;
    }
  }
  if (tries > 0) {
    char *message = "coco_create_new_path: could not create a new path from '%s' (%d attempts)";
    coco_warning(message, path, tries);
    coco_error(message);
  }
}
#endif

double *coco_allocate_vector(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(double);
  return (double *) coco_allocate_memory(block_size);
}

double *coco_duplicate_vector(const double *src, const size_t number_of_elements) {
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

/* some math functions which are not contained in C89 standard */
double coco_round_double(const double number) {
  return floor(number + 0.5);
}

double coco_max_double(const double a, const double b) {
  if (a >= b) {
    return a;
  } else {
    return b;
  }
}

double coco_min_double(const double a, const double b) {
  if (a <= b) {
    return a;
  } else {
    return b;
  }
}
