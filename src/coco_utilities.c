#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_strdup.c"

/* Figure out if we are on a sane platform or on the dominant platform */
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

/* Handle the special case of Microsoft Visual Studio 2008, which is not C89-compliant */
#if _MSC_VER
#include <direct.h>
#elif defined(__MINGW32__) || defined(__MINGW64__)
#include <dirent.h>
#else
#include <dirent.h>
/* To silence the compiler (implicit-function-declaration warning). */
int rmdir (const char *pathname);
int unlink (const char *filename);
int mkdir(const char *pathname, mode_t mode);
#endif

#if defined(HAVE_GFA)
#define S_IRWXU 0700
#endif

#if !defined(COCO_PATH_MAX)
#error COCO_PATH_MAX undefined
#endif

/***********************************
 * Global definitions in this file
 * which are not in coco.h 
 ***********************************/
void coco_join_path(char *path, size_t path_max_length, ...);
int coco_path_exists(const char *path);
void coco_create_path(const char *path);
void coco_create_new_path(const char *path, size_t maxlen, char *new_path);
int coco_create_directory(const char *path);
/* int coco_remove_directory(const char *path); Moved to coco.h */
int coco_remove_directory_msc(const char *path);
int coco_remove_directory_no_msc(const char *path);
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
  int res;
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributes(path);
  res = (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#elif defined(HAVE_STAT)
  struct stat buf;
  res = (!stat(path, &buf) && S_ISDIR(buf.st_mode));
#else
#error Ooops
#endif
  return res;
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
        if (0 != coco_create_directory(tmp))
          goto error;
      }
      *p = path_sep;
    }
  }
  if (0 != coco_create_directory(tmp))
    goto error;
  coco_free_memory(tmp);
  return;
  error: message = "coco_create_path(\"%s\") failed";
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

/**
 * Creates a directory with full privileges for the user (should work across different platforms/compilers).
 * Returns 0 on successful completion, and -1 on error.
 */
int coco_create_directory(const char *path) {
#if _MSC_VER
  return _mkdir(path);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  return mkdir(path);
#else
  return mkdir(path, S_IRWXU);
#endif
}

/**
 * Removes the given directory and all its contents (should work across different platforms/compilers).
 * Returns 0 on successful completion, and -1 on error.
 */
int coco_remove_directory(const char *path) {
#if _MSC_VER
  return coco_remove_directory_msc(path);
#else
  return coco_remove_directory_no_msc(path);
#endif
}

/**
 * Removes the given directory and all its contents (when using Microsoft Visual Studio's compiler).
 * Returns 0 on successful completion, and -1 on error.
 */
int coco_remove_directory_msc(const char *path) {
#if _MSC_VER
  WIN32_FIND_DATA find_data_file;
  HANDLE find_handle = NULL;
  char *buf;
  int r = -1;
  int r2 = -1;

  buf = coco_strdupf("%s\\*.*", path);
  /* Nothing to do if the folder does not exist */
  if ((find_handle = FindFirstFile(buf, &find_data_file)) == INVALID_HANDLE_VALUE) {
    coco_free_memory(buf);
    return 0;
  }
  coco_free_memory(buf);

  do {
    r = 0;

    /* Skip the names "." and ".." as we don't want to recurse on them */
    if (strcmp(find_data_file.cFileName, ".") != 0 && strcmp(find_data_file.cFileName, "..") != 0) {
      /* Build the new path using the argument path the file/folder name we just found */
      buf = coco_strdupf("%s\\%s", path, find_data_file.cFileName);

      if (find_data_file.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        /* Buf is a directory, recurse on it */
        r2 = coco_remove_directory_msc(buf);
      } else {
        /* Buf is a file, delete it */
        /* Careful, DeleteFile returns 0 if it fails and nonzero otherwise! */
        r2 = -(DeleteFile(buf) == 0);
      }

      coco_free_memory(buf);
    }

    r = r2;

  } while (FindNextFile(find_handle, &find_data_file)); /* Find the next file */

  FindClose(find_handle);

  if (!r) {
    /* Path is an empty directory, delete it */
    /* Careful, RemoveDirectory returns 0 if it fails and nonzero otherwise! */
    r = -(RemoveDirectory(path) == 0);
  }

  return r;
#else
  (void) path; /* To silence the compiler. */
  return -1; /* This should never be reached */
#endif
}

/**
 * Removes the given directory and all its contents (when NOT using Microsoft Visual Studio's compiler).
 * Returns 0 on successful completion, and -1 on error.
 */
int coco_remove_directory_no_msc(const char *path) {
#if !_MSC_VER
  DIR *d = opendir(path);
  int r = -1;
  int r2 = -1;
  char *buf;

  /* Nothing to do if the folder does not exist */
  if (!coco_path_exists(path))
    return 0;

  if (d) {
    struct dirent *p;

    r = 0;

    while (!r && (p = readdir(d))) {

      /* Skip the names "." and ".." as we don't want to recurse on them */
      if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
        continue;
      }

      buf = coco_strdupf("%s/%s", path, p->d_name);
      if (buf) {
        if (coco_path_exists(buf)) {
          /* Buf is a directory, recurse on it */
          r2 = coco_remove_directory(buf);
        } else {
          /* Buf is a file, delete it */
          r2 = unlink(buf);
        }
      }
      coco_free_memory(buf);

      r = r2;
    }

    closedir(d);
  }

  if (!r) {
    /* Path is an empty directory, delete it */
    r = rmdir(path);
  }

  return r;
#else
  (void) path; /* To silence the compiler. */
  return -1; /* This should never be reached */
#endif
}

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

/* Some math functions which are not contained in C89 standard */
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
