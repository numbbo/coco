#include "coco_platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <ctype.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_strdup.c"

/**
 * Initialize the logging level to COCO_INFO.
 */
static coco_log_level_type_e coco_log_level = COCO_INFO;

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
static int coco_options_read_int(const char *options, const char *name, int *pointer);
static int coco_options_read_string(const char *options, const char *name, char *pointer);
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

  while (counter < 999) {

    new_path = coco_strdupf("%s-%03d", *path, counter);

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

  coco_error("coco_create_unique_path(): could not create a unique path with name %s", *path);
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
 * - name and value need to be separated by a semicolon (spaces are optional)
 * - the value corresponding to the given name needs to be an integer
 * Returns the number of successful assignments.
 */
static int coco_options_read_int(const char *options, const char *name, int *pointer) {
  return coco_options_read(options, name, " %i", pointer);
}

/**
 * Reads a size_t from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon (spaces are optional)
 * - the value corresponding to the given name needs to be a size_t
 * Returns the number of successful assignments.
 */
static int coco_options_read_size_t(const char *options, const char *name, size_t *pointer) {
  return coco_options_read(options, name, "%lu", pointer);
}

/**
 * Reads a double value from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon (spaces are optional)
 * - the value corresponding to the given name needs to be a double
 * Returns the number of successful assignments.
 */
/* Commented to silence the compiler
static int coco_options_read_double(const char *options, const char *name, double *pointer) {
  return coco_options_read(options, name, "%f", pointer);
}
*/

/**
 * Reads a string from options using the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon (spaces are optional)
 * - the value corresponding to the given name needs to be a string - either a single word or multiple words
 * in double quotes
 * Returns the number of successful assignments.
 */
static int coco_options_read_string(const char *options, const char *name, char *pointer) {

  long i1, i2;

  if ((!options) || (strlen(options) == 0))
    return 0;

  i1 = coco_strfind(options, name);
  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;

  /* Remove trailing whitespaces */
  while (isspace((unsigned char) options[i2]))
    i2++;

  if (i2 <= i1){
    return 0;
  }

  if (options[i2] == '\"') {
    /* The value starts with a quote: read everything between two quotes into a string */
    return sscanf(&options[i2], "\"%[^\"]\"", pointer);
  } else
    return sscanf(&options[i2], "%s", pointer);
}

/**
 * Reads (possibly delimited) values from options using the form "name1 : value1,value2,value3 name2: value4",
 * i.e. reads all characters from the corresponding name up to the next whitespace or end of string.
 * Formatting requirements:
 * - name and value need to be separated by a semicolon (spaces are optional)
 * Returns the number of successful assignments.
 */
static int coco_options_read_values(const char *options, const char *name, char *pointer) {

  long i1, i2;
  int i;

  if ((!options) || (strlen(options) == 0))
    return 0;

  i1 = coco_strfind(options, name);
  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;

  /* Remove trailing whitespaces */
  while (isspace((unsigned char) options[i2]))
    i2++;

  if (i2 <= i1) {
    return 0;
  }

  i = 0;
  while (!isspace((unsigned char) options[i2 + i]) && (options[i2 + i] != '\0')) {
    pointer[i] = options[i2 + i];
    i++;
  }
  pointer[i] = '\0';
  return i;
}

/**
 * Parse options in the form "name1 : value1 name2: value2". Formatting requirements:
 * - name and value need to be separated by a semicolon (spaces are optional)
 * - value needs to be a single string (no spaces allowed)
 * Returns the number of successful assignments.
 */
static int coco_options_read(const char *options, const char *name, const char *format, void *pointer) {

  long i1, i2;

  if ((!options) || (strlen(options) == 0))
    return 0;

  i1 = coco_strfind(options, name);
  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;

  /* Remove trailing whitespaces */
  while (isspace((unsigned char) options[i2]))
    i2++;

  if (i2 <= i1){
    return 0;
  }

  return sscanf(&options[i2], format, pointer);
}

/**
 * Splits a string based on the given delimiter. Returns a pointer to the resulting substrings with NULL
 * as the last one. The memory of the result needs to be freed by the caller.
 */
static char **coco_string_split(const char *string, const char delimiter) {

  char **result;
  char *str_copy, *ptr, *token;
  char str_delimiter[2];
  size_t i;
  size_t count = 1;

  str_copy = coco_strdup(string);

  /* Counts the parts between delimiters */
  ptr = str_copy;
  while (*ptr != '\0') {
    if (*ptr == delimiter) {
      count++;
    }
    ptr++;
  }
  /* Makes room for an empty string that will be appended at the end */
  count++;

  result = coco_allocate_memory(count * sizeof(char*));

  /* Iterates through tokens
   * NOTE: strtok() ignores multiple delimiters, therefore the final number of detected substrings might be
   * lower than the count. This is OK. */
  i = 0;
  /* A char* delimiter needs to be used, otherwise strtok() can surprise */
  str_delimiter[0] = delimiter;
  str_delimiter[1] = '\0';
  token = strtok(str_copy, str_delimiter);
  while (token)
  {
      assert(i < count);
      *(result + i++) = coco_strdup(token);
      token = strtok(NULL, str_delimiter);
  }
  *(result + i) = NULL;

  coco_free_memory(str_copy);

  return result;
}

static size_t coco_numbers_count(const size_t *numbers, const char *name) {

  const size_t count_limit = 100;

  size_t count = 0;
  while ((count < count_limit) && (numbers[count] != 0)) {
    count++;
  }
  if (count == count_limit) {
    coco_error("coco_numbers_count(): over %lu numbers in %s", count_limit, name);
    return 0; /* Never reached*/
  }

  return count;
}
/**
 * Reads ranges from a string of positive ranges separated by commas. For example: "-3,5-6,8-". Returns the
 * numbers that are defined by the ranges if min and max are used as their extremes. If the ranges with open
 * beginning/end are not allowed, use 0 as min/max. The returned string has an appended 0 to mark its end.
 * A maximum of 100 values is returned. If there is a problem with one of the ranges, the parsing stops and
 * the current result is returned. The memory of the returned object needs to be freed by the caller.
 */
static size_t *coco_string_get_numbers_from_ranges(char *string, const char *name, size_t min, size_t max) {

  char *ptr, *dash = NULL;
  char **ranges, **numbers;
  size_t i, j, count;
  size_t num[2];

  /* Don't allow ranges that are too long */
  const size_t length_limit = 100;
  size_t *result;
  size_t i_result = 0;

  /* Check for empty string */
  if ((string == NULL) || (strlen(string) == 0)) {
    coco_warning("coco_options_read_ranges(): cannot parse empty ranges");
    return NULL;
  }

  ptr = string;
  /* Check for disallowed characters */
  while (*ptr != '\0') {
    if ((*ptr != '-') && (*ptr != ',') && !isdigit((unsigned char )*ptr)) {
      coco_warning("coco_options_read_ranges(): problem parsing '%s' - cannot parse ranges with '%c'", string,
          *ptr);
      return NULL;
    } else
      ptr++;
  }

  /* Check for incorrect boundaries */
  if ((max > 0) && (min > max)) {
    coco_warning("coco_options_read_ranges(): incorrect boundaries");
    return NULL;
  }

  result = coco_allocate_memory((length_limit + 1) * sizeof(size_t));

  /* Split string to ranges w.r.t commas */
  ranges = coco_string_split(string, ',');

  if (ranges) {
    /* Go over the current range */
    for (i = 0; *(ranges + i); i++) {

      ptr = *(ranges + i);
      /* Count the number of '-' */
      count = 0;
      while (*ptr != '\0') {
        if (*ptr == '-') {
          if (count == 0)
            /* Remember the position of the first '-' */
            dash = ptr;
          count++;
        }
        ptr++;
      }
      /* Point again to the start of the range */
      ptr = *(ranges + i);

      /* Check for incorrect number of '-' */
      if (count > 1) {
        coco_warning("coco_options_read_ranges(): problem parsing '%s' - too many '-'s", string);
        /* Cleanup */
        for (j = i; *(ranges + j); j++)
          coco_free_memory(*(ranges + j));
        coco_free_memory(ranges);
        if (i_result == 0) {
          coco_free_memory(result);
          return NULL;
        }
        result[i_result] = 0;
        return result;
      } else if (count == 0) {
        /* Range is in the format: n (no range) */
        num[0] = (size_t) strtol(ptr, NULL, 10);
        num[1] = num[0];
      } else {
        /* Range is in one of the following formats: n-m / -n / n- / - */

        /* Split current range to numbers w.r.t '-' */
        numbers = coco_string_split(ptr, '-');
        j = 0;
        if (numbers) {
          /* Read the numbers */
          for (j = 0; *(numbers + j); j++) {
            assert(j < 2);
            num[j] = (size_t) strtol(*(numbers + j), NULL, 10);
            coco_free_memory(*(numbers + j));
          }
        }
        coco_free_memory(numbers);

        if (j == 0) {
          /* Range is in the format - (open ends) */
          if ((min == 0) || (max == 0)) {
            coco_warning("coco_options_read_ranges(): '%s' ranges cannot have an open ends; some ranges ignored", name);
            /* Cleanup */
            for (j = i; *(ranges + j); j++)
              coco_free_memory(*(ranges + j));
            coco_free_memory(ranges);
            if (i_result == 0) {
              coco_free_memory(result);
              return NULL;
            }
            result[i_result] = 0;
            return result;
          }
          num[0] = min;
          num[1] = max;
        } else if (j == 1) {
          if (dash - *(ranges + i) == 0) {
            /* Range is in the format -n */
            if (min == 0) {
              coco_warning("coco_options_read_ranges(): '%s' ranges cannot have an open beginning; some ranges ignored", name);
              /* Cleanup */
              for (j = i; *(ranges + j); j++)
                coco_free_memory(*(ranges + j));
              coco_free_memory(ranges);
              if (i_result == 0) {
                coco_free_memory(result);
                return NULL;
              }
              result[i_result] = 0;
              return result;
            }
            num[1] = num[0];
            num[0] = min;
          } else {
            /* Range is in the format n- */
            if (max == 0) {
              coco_warning("coco_options_read_ranges(): '%s' ranges cannot have an open end; some ranges ignored", name);
              /* Cleanup */
              for (j = i; *(ranges + j); j++)
                coco_free_memory(*(ranges + j));
              coco_free_memory(ranges);
              if (i_result == 0) {
                coco_free_memory(result);
                return NULL;
              }
              result[i_result] = 0;
              return result;
            }
            num[1] = max;
          }
        }
        /* if (j == 2), range is in the format n-m and there is nothing to do */
      }

      /* Make sure the boundaries are taken into account */
      if ((min > 0) && (num[0] < min)) {
        num[0] = min;
        coco_warning("coco_options_read_ranges(): '%s' ranges adjusted to be >= %lu", name, min);
      }
      if ((max > 0) && (num[1] > max)) {
        num[1] = max;
        coco_warning("coco_options_read_ranges(): '%s' ranges adjusted to be <= %lu", name, max);
      }
      if (num[0] > num[1]) {
        coco_warning("coco_options_read_ranges(): '%s' ranges not within boundaries; some ranges ignored", name);
        /* Cleanup */
        for (j = i; *(ranges + j); j++)
          coco_free_memory(*(ranges + j));
        coco_free_memory(ranges);
        if (i_result == 0) {
          coco_free_memory(result);
          return NULL;
        }
        result[i_result] = 0;
        return result;
      }

      /* Write in result */
      for (j = num[0]; j <= num[1]; j++) {
        if (i_result > length_limit - 1)
          break;
        result[i_result++] = j;
      }

      coco_free_memory(*(ranges + i));
      *(ranges + i) = NULL;
    }
  }

  coco_free_memory(ranges);

  if (i_result == 0) {
    coco_free_memory(result);
    return NULL;
  }

  result[i_result] = 0;
  return result;
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
