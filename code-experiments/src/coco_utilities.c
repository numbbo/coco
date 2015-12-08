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

/* Handle the special case of Microsoft Visual Studio 2008 and x86_64-w64-mingw32-gcc */
#if _MSC_VER
#include <direct.h>
#elif defined(__MINGW32__) || defined(__MINGW64__)
#include <dirent.h>
#else
#include <dirent.h>
/* To silence the compiler (implicit-function-declaration warning). */
int rmdir(const char *pathname);
int unlink(const char *file_name);
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
void coco_create_unique_filename(char **file_name);
void coco_create_unique_path(char **path);
int coco_create_directory(const char *path);
int coco_remove_directory_msc(const char *path);
int coco_remove_directory_no_msc(const char *path);
double *coco_duplicate_vector(const double *src, const size_t number_of_elements);
int coco_options_read_int(const char *options, const char *name, int *pointer);
int coco_options_read_long(const char *options, const char *name, long *pointer);
int coco_options_read_double(const char *options, const char *name, double *pointer);
int coco_options_read_string(const char *options, const char *name, char *pointer);
static int coco_options_read(const char *options, const char *name, const char *format, void *pointer);
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

int coco_file_exists(const char *path) {
  int res;
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributes(path);
  res = (dwAttrib != INVALID_FILE_ATTRIBUTES) && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY);
#elif defined(HAVE_STAT)
  struct stat buf;
  res = (!stat(path, &buf) && !S_ISDIR(buf.st_mode));
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

/**
 * Creates a unique file name from the given file_name. If the file_name does not yet exit, it is left as
 * is, otherwise it is changed(!) by prepending a number to it.
 *
 * If filename.ext already exists, 01-filename.ext will be tried. If this one exists as well,
 * 02-filename.ext will be tried, and so on. If 99-filename.ext exists as well, the function returns
 * an error.
 */
void coco_create_unique_filename(char **file_name) {

  int counter = 1;
  char *new_file_name;

  /* Do not change the file_name if it does not yet exist */
  if (!coco_file_exists(*file_name)) {
    return;
  }

  while (counter < 99) {

    new_file_name = coco_strdupf("%02d-%s", counter, *file_name);

    if (!coco_file_exists(new_file_name)) {
      coco_free_memory(*file_name);
      *file_name = new_file_name;
      return;
    } else {
      counter++;
      coco_free_memory(new_file_name);
    }

  }

  coco_free_memory(new_file_name);
  coco_error("coco_create_unique_filename(): could not create a unique file name");
  return; /* Never reached */
}

/**
 * Creates a unique path from the given path. If the given path does not yet exit, it is left as
 * is, otherwise it is changed(!) by appending a number to it.
 *
 * If path already exists, path-01 will be tried. If this one exists as well, path-02 will be tried,
 * and so on. If path-99 exists as well, the function returns an error.
 */
void coco_create_unique_path(char **path) {

  int counter = 1;
  char *new_path;

  /* Create the path if it does not yet exist */
  if (!coco_path_exists(*path)) {
    coco_create_path(*path);
    return;
  }

  while (counter < 99) {

    new_path = coco_strdupf("%s-%02d", *path, counter);

    if (!coco_path_exists(new_path)) {
      coco_free_memory(*path);
      *path = new_path;
      coco_create_path(*path);
      return;
    } else {
      counter++;
      coco_free_memory(new_path);
    }

  }

  coco_free_memory(new_path);
  coco_error("coco_create_unique_path(): could not create a unique path");
  return; /* Never reached */
}

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

  }while (FindNextFile(find_handle, &find_data_file)); /* Find the next file */

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

/**
 * Reads an integer from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon AND a space (spaces between name and the
 * semicolon are optional)
 * - the value corresponding to the given name needs to be an integer
 * Returns the number of successful assignments.
 */
int coco_options_read_int(const char *options, const char *name, int *pointer) {
  return coco_options_read(options, name, " %i", pointer);
}

/**
 * Reads a size_t from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon AND a space (spaces between name and the
 * semicolon are optional)
 * - the value corresponding to the given name needs to be a size_t
 * Returns the number of successful assignments.
 */
int coco_options_read_size_t(const char *options, const char *name, size_t *pointer) {
  return coco_options_read(options, name, " %lu", pointer);
}

/**
 * Reads a long integer from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon AND a space (spaces between name and the
 * semicolon are optional)
 * - the value corresponding to the given name needs to be a long integer
 * Returns the number of successful assignments.
 */
int coco_options_read_long(const char *options, const char *name, long *pointer) {
  return coco_options_read(options, name, " %lu", pointer);
}

/**
 * Reads a double value from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon AND a space (spaces between name and the
 * semicolon are optional)
 * - the value corresponding to the given name needs to be a double
 * Returns the number of successful assignments.
 */
int coco_options_read_double(const char *options, const char *name, double *pointer) {
  return coco_options_read(options, name, " %f", pointer);
}

/**
 * Reads a string from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon AND a space (spaces between name and the
 * semicolon are optional)
 * - the value corresponding to the given name needs to be a string - either a single word or multiple words
 * in double quotes
 * Returns the number of successful assignments.
 */
int coco_options_read_string(const char *options, const char *name, char *pointer) {

  long i1 = coco_strfind(options, name);
  long i2;

  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;
  if (i2 <= i1)
    return 0;

  if (options[i2 + 1] == '\"') {
    /* The value starts with a quote: read everything between two quotes into a string */
    return sscanf(&options[i2], " \"%[^\"]\"", pointer);
  } else
    return sscanf(&options[i2], " %s", pointer);
}

/**
 * Parse options in the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon
 * - value needs to be a single string (no spaces allowed)
 * Returns the number of successful assignments.
 */
static int coco_options_read(const char *options, const char *name, const char *format, void *pointer) {

  long i1 = coco_strfind(options, name);
  long i2;

  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;
  if (i2 <= i1)
    return 0;

  return sscanf(&options[i2], format, pointer);
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

/**
 * Returns 1 if |a - b| < accuracy and 0 otherwise
 */
int coco_doubles_almost_equal(const double a, const double b, const double accuracy) {
  return ((fabs(a - b) < accuracy) == 0);
}

/* Creates and returns a string with removed characters from @{from} to @{to}.
 * The caller is responsible for freeing the allocated memory. */
static char *coco_remove_from_string(char *string, char *from, char *to) {

  char *result, *start, *stop;

  result = coco_strdup(string);

  if (strcmp(from, "") == 0) {
    /* Remove from the start */
    start = result;
  } else
    start = strstr(result, from);

  if (strcmp(to, "") == 0) {
    /* Remove until the end */
    stop = result + strlen(result);
  } else
    stop = strstr(result, to);

  if ((start == NULL) || (stop == NULL) || (stop < start)) {
    coco_error("coco_remove_from_string(): failed to remove characters between %s and %s from string %s",
        from, to, string);
    return NULL; /* Never reached */
  }

  memmove(start, stop, strlen(stop) + 1);

  return result;
}
