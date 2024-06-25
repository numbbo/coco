/**
 * @file coco_utilities.c
 * @brief Definitions of miscellaneous functions used throughout the COCO framework.
 */

#include "coco_platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>

#include "coco.h"
#include "coco_internal.h"
#include "coco_string.c"


/***********************************************************************************************************/

/**
 * @brief Sets the constant chosen_precision to 1e-9.
 */
static const double chosen_precision = 1e-9;

/***********************************************************************************************************/

/**
 * @brief Initializes the logging level to COCO_INFO.
 */
static coco_log_level_type_e coco_log_level = COCO_INFO;

/**
 * @param log_level Denotes the level of information given to the user through the standard output and
 * error streams. Can take on the values:
 * - "error" (only error messages are output),
 * - "warning" (only error and warning messages are output),
 * - "info" (only error, warning and info messages are output) and
 * - "debug" (all messages are output).
 * - "" does not set a new value
 * The default value is info.
 *
 * @return The previous coco_log_level value as an immutable string.
 */
const char *coco_set_log_level(const char *log_level) {

  coco_log_level_type_e previous_log_level = coco_log_level;

  if (strcmp(log_level, "error") == 0)
    coco_log_level = COCO_ERROR;
  else if (strcmp(log_level, "warning") == 0)
    coco_log_level = COCO_WARNING;
  else if (strcmp(log_level, "info") == 0)
    coco_log_level = COCO_INFO;
  else if (strcmp(log_level, "debug") == 0)
    coco_log_level = COCO_DEBUG;
  else if (strcmp(log_level, "") == 0) {
    /* Do nothing */
  } else {
    coco_warning("coco_set_log_level(): unknown level %s", log_level);
  }

  if (previous_log_level == COCO_ERROR)
    return "error";
  else if (previous_log_level == COCO_WARNING)
    return "warning";
  else if (previous_log_level == COCO_INFO)
    return "info";
  else if (previous_log_level == COCO_DEBUG)
    return "debug";
  else {
    coco_error("coco_set_log_level(): unknown previous log level");
    return "";
  }
}

/***********************************************************************************************************/

/**
 * @name Methods regarding file, directory and path manipulations
 */
/**@{*/
/**
 * @brief Creates a platform-dependent path from the given strings.
 *
 * @note The last argument must be NULL.
 * @note The first parameter must be able to accommodate path_max_length characters and the length
 * of the joined path must not exceed path_max_length characters.
 * @note Should work cross-platform.
 *
 * Usage examples:
 * - coco_join_path(base_path, 100, folder1, folder2, folder3, NULL) creates base_path/folder1/folder2/folder3
 * - coco_join_path(base_path, 100, folder1, file_name, NULL) creates base_path/folder1/file_name
 * @param path The base path; it's also where the joined path is stored to.
 * @param path_max_length The maximum length of the path.
 * @param ... Additional strings, must end with NULL
 */
static void coco_join_path(char *path, const size_t path_max_length, ...) {
  const size_t path_separator_length = strlen(coco_path_separator);
  va_list args;
  char *path_component;
  size_t path_length = strlen(path);

  va_start(args, path_max_length);
  while (NULL != (path_component = va_arg(args, char *))) {
    size_t component_length = strlen(path_component);
    if (path_length + path_separator_length + component_length >= path_max_length) {
      coco_error("coco_join_path() failed because the ${path} is too short.");
      return; /* never reached */
    }
    /* Both should be safe because of the above check. */
    if (strlen(path) > 0)
      strncat(path, coco_path_separator, path_max_length - strlen(path) - 1);
    strncat(path, path_component, path_max_length - strlen(path) - 1);
  }
  va_end(args);
}

/**
 * @brief Checks if the given directory exists.
 *
 * @note Should work cross-platform.
 *
 * @param path The given path.
 *
 * @return 1 if the path exists and corresponds to a directory and 0 otherwise.
 */
static int coco_directory_exists(const char *path) {
  int res;
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributesA(path);
  res = (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#elif defined(HAVE_STAT)
  struct stat buf;
  res = (!stat(path, &buf) && S_ISDIR(buf.st_mode));
#else
#error Ooops
#endif
  return res;
}

/**
 * @brief Checks if the given file exists.
 *
 * @note Should work cross-platform.
 *
 * @param path The given path.
 *
 * @return 1 if the path exists and corresponds to a file and 0 otherwise.
 */
static int coco_file_exists(const char *path) {
  int res;
#if defined(HAVE_GFA)
  DWORD dwAttrib = GetFileAttributesA(path);
  res = (dwAttrib != INVALID_FILE_ATTRIBUTES) && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY);
#elif defined(HAVE_STAT)
  struct stat buf;
  res = (!stat(path, &buf) && !S_ISDIR(buf.st_mode));
#else
#error Ooops
#endif
  return res;
}

/**
 * @brief Calls the right mkdir() method (depending on the platform) with full privileges for the user. 
 * If the created directory has not existed before, returns 0, otherwise returns 1. If the directory has 
 * not been created, a coco_error is raised. 
 *
 * @param path The directory path.
 *
 * @return 0 if the created directory has not existed before and 1 otherwise.
 */
static int coco_mkdir(const char *path) {
  int result = 0;

  /* Do not create the path if is of the form "C:" (two letters, of which the second is a colon)*/
  if ((strlen(path) == 2) && (path[1] == ':'))
    return 1;

#if _MSC_VER
  result = _mkdir(path);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  result = mkdir(path);
#else
  result = mkdir(path, S_IRWXU);
#endif

  if (result == 0)
    return 0;
  else if (errno == EEXIST)
    return 1;
  else  {
    coco_error("coco_mkdir(): unable to create %s, mkdir error: %s", path, strerror(errno));
    return 1; /* Never reached */
  }
}

/**
 * @brief Creates a directory (possibly having to create nested directories). If the last created directory 
 * has not existed before, returns 0, otherwise returns 1.
 *
 * @param path The directory path.
 *
 * @return 0 if the created directory has not existed before and 1 otherwise.
 */
static int coco_create_directory(const char *path) {
  char *path_copy = NULL;
  char *tmp, *p;
  char path_sep = coco_path_separator[0];
  size_t len = strlen(path);

  int result = 0;

  path_copy = coco_strdup(path);
  tmp = path_copy;

  /* Remove possible trailing (back)slash */
  if (tmp[len - 1] == path_sep)
    tmp[len - 1] = 0;

  /* Iterate through nested directories (does nothing if directories are not nested) */
  for (p = tmp; *p; p++) {
    if (*p == path_sep) {
      *p = 0;
      if (strlen(tmp) == 0) {
        *p = path_sep;
        continue;
      }
      coco_mkdir(tmp);
      *p = path_sep;
    }
  }
  
  /* Create the last nested or only directory */
  result = coco_mkdir(tmp);
  coco_free_memory(path_copy);
  return result;
}

/**
 * @brief Creates a unique file name from the given file_name.
 *
 * If path/file_name.ext does not yet exit, it is left as is, otherwise it is changed(!) by appending a number
 * to it. If path/file_name.ext already exists, path/filename-0001.ext will be tried. If this one exists as well,
 * path/filename-0002.ext will be tried, and so on. If path/filename-9999.ext exists as well, the function throws
 * an error. Every 1000 trials a warning is issued.
 */
static void coco_create_unique_filename(const char *path,
                                        char **file_name,
                                        const char *ext) {
  int counter = 1;
  char file_path[COCO_PATH_MAX + 2] = { 0 };
  char relative_file_path[COCO_PATH_MAX + 2] = { 0 };
  char *new_file_name;

  strncpy(relative_file_path, *file_name, COCO_PATH_MAX - strlen(relative_file_path) - 1);
  strncat(relative_file_path, ext, COCO_PATH_MAX - strlen(relative_file_path) - 1);
  coco_join_path(file_path, sizeof(file_path), path, relative_file_path, NULL);
  
  if (!coco_file_exists(file_path)) 
    return;

  while (counter < 9999) {

    new_file_name = coco_strdupf("%s-%04d", *file_name, counter);

    memset(relative_file_path, 0, sizeof(path));
    memset(file_path, 0, sizeof(path));
    strncpy(relative_file_path, new_file_name, COCO_PATH_MAX - strlen(relative_file_path) - 1);
    strncat(relative_file_path, ext, COCO_PATH_MAX - strlen(relative_file_path) - 1);
    coco_join_path(file_path, sizeof(file_path), path, relative_file_path, NULL);

    if (!coco_file_exists(file_path)) {
      coco_free_memory(*file_name);
      *file_name = new_file_name;
      return;
    } 
    
    counter++;
    if (counter % 1000 == 0)
      coco_warning("coco_create_unique_filename(): trying to create a unique file name %s (%d trials)", *file_name, counter);      
    coco_free_memory(new_file_name);
    
  }

  coco_error("coco_create_unique_filename(): could not create a unique file name %s", *file_name);
  return; /* Never reached */
}

/**
 * @brief Creates a directory that has not existed before.
 *
 * If the given path does not yet exit, it is left as is, otherwise it is changed(!) by appending a number
 * to it. If path already exists, path-0001 will be tried. If this one exists as well, path-0002 will be tried,
 * and so on. If path-9999 exists as well, an error is raised. Every 1000 trials a warning is issued.
 */
static void coco_create_unique_directory(char **path) {

  int counter = 1;
  char *new_path;

  if (coco_create_directory(*path) == 0) {
	/* Directory created */
    return;
  }

  while (counter < 9999) {

    new_path = coco_strdupf("%s-%04d", *path, counter);

    if (coco_create_directory(new_path) == 0) {
      /* Directory created */
      coco_free_memory(*path);
      *path = new_path;
      return;
    } else {
      counter++;
      if (counter % 1000 == 0)
        coco_warning("coco_create_unique_directory(): creating a unique directory %s (%d trials)", *path, counter);
      coco_free_memory(new_path);
    }

  }

  coco_error("coco_create_unique_directory(): unable to create unique directory %s", *path);
  return; /* Never reached */
}

/**
 * The method should work across different platforms/compilers.
 *
 * @path The path to the directory
 *
 * @return 0 on successful completion, and -1 on error.
 */
int coco_remove_directory(const char *path) {
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
        r2 = coco_remove_directory(buf);
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
  DIR *d = opendir(path);
  int r = -1;
  int r2 = -1;
  char *buf;

  /* Nothing to do if the folder does not exist */
  if (!coco_directory_exists(path))
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
        if (coco_directory_exists(buf)) {
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
#endif
}



/**
 * The method should work across different platforms/compilers.
 *
 * @file_name The path to the file
 *
 * @return 0 on successful completion, and -1 on error.
 */
int coco_remove_file(const char *file_name) {
#if _MSC_VER
  int r = -1;
  /* Try to delete the file */
  /* Careful, DeleteFile returns 0 if it fails and nonzero otherwise! */
  r = -(DeleteFile(file_name) == 0);
  return r;
#else
  int r = -1;
  /* Try to delete the file */
  r = unlink(file_name);
  return r;
#endif
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding memory allocations
 */
/**@{*/
double *coco_allocate_vector(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(double);
  return (double *) coco_allocate_memory(block_size);
}

/**
 * @brief Allocates memory for a vector and sets all its elements to value.
 */
static double *coco_allocate_vector_with_value(const size_t number_of_elements, double value) {
  const size_t block_size = number_of_elements * sizeof(double);
  double *vector = (double *) coco_allocate_memory(block_size);
  size_t i;

  for (i = 0; i < number_of_elements; i++)
  	vector[i] = value;

  return vector;
}

/**
 * @brief Safe memory allocation for a vector with size_t elements that either succeeds or triggers a
 * coco_error.
 */
static size_t *coco_allocate_vector_size_t(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(size_t);
  return (size_t *) coco_allocate_memory(block_size);
}

static char *coco_allocate_string(const size_t number_of_elements) {
  const size_t block_size = number_of_elements * sizeof(char);
  return (char *) coco_allocate_memory(block_size);
}

static double *coco_duplicate_vector(const double *src, const size_t number_of_elements) {
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
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods regarding string options
 */
/**@{*/

/**
 * @brief Allocates an option keys structure holding the given number of option keys.
 */
static coco_option_keys_t *coco_option_keys_allocate(const size_t count, const char **keys) {

  size_t i;
  coco_option_keys_t *option_keys;

  if ((count == 0) || (keys == NULL))
    return NULL;

  option_keys = (coco_option_keys_t *) coco_allocate_memory(sizeof(*option_keys));

  option_keys->keys = (char **) coco_allocate_memory(count * sizeof(char *));
  for (i = 0; i < count; i++) {
    assert(keys[i]);
    option_keys->keys[i] = coco_strdup(keys[i]);
  }
  option_keys->count = count;

  return option_keys;
}

/**
 * @brief Frees the given option keys structure.
 */
static void coco_option_keys_free(coco_option_keys_t *option_keys) {

  size_t i;

  if (option_keys) {
    for (i = 0; i < option_keys->count; i++) {
      coco_free_memory(option_keys->keys[i]);
    }
    coco_free_memory(option_keys->keys);
    coco_free_memory(option_keys);
  }
}

/**
 * @brief Returns redundant option keys (the ones present in given_option_keys but not in known_option_keys).
 */
static coco_option_keys_t *coco_option_keys_get_redundant(const coco_option_keys_t *known_option_keys,
                                                          const coco_option_keys_t *given_option_keys) {

  size_t i, j, count = 0;
  int found;
  char **redundant_keys;
  coco_option_keys_t *redundant_option_keys;

  assert(known_option_keys != NULL);
  assert(given_option_keys != NULL);

  /* Find the redundant keys */
  redundant_keys = (char **) coco_allocate_memory(given_option_keys->count * sizeof(char *));
  for (i = 0; i < given_option_keys->count; i++) {
    found = 0;
    for (j = 0; j < known_option_keys->count; j++) {
      if (strcmp(given_option_keys->keys[i], known_option_keys->keys[j]) == 0) {
        found = 1;
        break;
      }
    }
    if (!found) {
      redundant_keys[count++] = coco_strdup(given_option_keys->keys[i]);
    }
  }
  redundant_option_keys = coco_option_keys_allocate(count, (const char**) redundant_keys);

  /* Free memory */
  for (i = 0; i < count; i++) {
    coco_free_memory(redundant_keys[i]);
  }
  coco_free_memory(redundant_keys);

  return redundant_option_keys;
}

/**
 * @brief Adds additional option keys to the given basic option keys (changes the basic keys).
 */
static void coco_option_keys_add(coco_option_keys_t **basic_option_keys,
                                 const coco_option_keys_t *additional_option_keys) {

  size_t i, j;
  size_t new_count;
  char **new_keys;
  coco_option_keys_t *new_option_keys;

  assert(*basic_option_keys != NULL);
  if (additional_option_keys == NULL)
    return;

  /* Construct the union of both keys */
  new_count = (*basic_option_keys)->count + additional_option_keys->count;
  new_keys = (char **) coco_allocate_memory(new_count * sizeof(char *));
  for (i = 0; i < (*basic_option_keys)->count; i++) {
    new_keys[i] = coco_strdup((*basic_option_keys)->keys[i]);
  }
  for (j = 0; j < additional_option_keys->count; j++) {
    new_keys[(*basic_option_keys)->count + j] = coco_strdup(additional_option_keys->keys[j]);
  }
  new_option_keys = coco_option_keys_allocate(new_count, (const char**) new_keys);

  /* Free the old basic keys */
  coco_option_keys_free(*basic_option_keys);
  *basic_option_keys = new_option_keys;
  for (i = 0; i < new_count; i++) {
    coco_free_memory(new_keys[i]);
  }
  coco_free_memory(new_keys);
}

/**
 * @brief Creates an instance of option keys from the given string of options containing keys and values
 * separated by colons.
 *
 * @note Relies heavily on the "key: value" format and might fail if the number of colons doesn't match the
 * number of keys. Values that are strings surrounded by quotation marks should work as long as they come
 * in pairs.
 */
static coco_option_keys_t *coco_option_keys(const char *option_string) {

  size_t i;
  char **keys;
  coco_option_keys_t *option_keys = NULL;
  char *string_to_parse, *key, *string_pointer;
  char *cleaned_option_string = NULL;
  const char *replacement_string = "STR";

  /* Check for empty string */
  if ((option_string == NULL) || (strlen(option_string) == 0)) {
	    return NULL;
  }

  /* Construct the cleaned_option_string by replacing any string between two quotation marks with "STR" */
  keys = coco_string_split(option_string, '\"');
  if (keys) {
    for (i = 0; *(keys + i); i++) {
      if (i == 0)
        cleaned_option_string = coco_strdupf(*(keys + i));
      else {
        string_pointer = cleaned_option_string;
        if (i % 2 == 0) {
          /* This is outside of a pair of quotation marks */
          cleaned_option_string = coco_strconcat(string_pointer, *(keys + i));
        }
        else {
          /* This is inside of a pair of quotation marks */
          cleaned_option_string = coco_strconcat(string_pointer, replacement_string);
        }
        coco_free_memory(string_pointer);
      }
    }
  }
  /* Free the keys */
  for (i = 0; *(keys + i); i++) {
    coco_free_memory(*(keys + i));
  }
  coco_free_memory(keys);

  /* Split the options w.r.t ':' */
  keys = coco_string_split(cleaned_option_string, ':');

  if (keys) {
    /* Keys now contain something like this: "values_of_previous_key this_key" except for the first, which
     * contains only the key and the last, which contains only the previous values */
    for (i = 0; *(keys + i); i++) {
      string_to_parse = coco_strdup(*(keys + i));

      /* Remove any leading and trailing spaces */
      string_to_parse = coco_string_trim(string_to_parse);

      /* Stop if this is the last substring (contains a value and no key) */
      if ((i > 0) && (*(keys + i + 1) == NULL)) {
        coco_free_memory(string_to_parse);
        break;
      }

      /* Disregard everything before the last space */
      key = strrchr(string_to_parse, ' ');
      if ((key == NULL) || (i == 0)) {
        /* No spaces left (or this is the first key), everything is the key */
        key = string_to_parse;
      } else {
        /* Move to the start of the key (one char after the space) */
        key++;
      }

      /* Put the key in keys */
      coco_free_memory(*(keys + i));
      *(keys + i) = coco_strdup(key);
      coco_free_memory(string_to_parse);
    }

    option_keys = coco_option_keys_allocate(i, (const char**) keys);

    /* Free the keys */
    for (i = 0; *(keys + i); i++) {
      coco_free_memory(*(keys + i));
    }
    coco_free_memory(keys);
  }

  coco_free_memory(cleaned_option_string);

  return option_keys;
}

/**
 * @brief Creates and returns a string containing the info_string and all keys from option_keys.
 *
 * Can be used to output information about the given option_keys.
 */
static char *coco_option_keys_get_output_string(const coco_option_keys_t *option_keys,
                                                const char *info_string) {
  size_t i;
  char *string = NULL, *new_string;

  if ((option_keys != NULL) && (option_keys->count > 0)) {

    string = coco_strdup(info_string);
    for (i = 0; i < option_keys->count; i++) {
      new_string = coco_strdupf("%s %s\n", string, option_keys->keys[i]);
      coco_free_memory(string);
      string = new_string;
    }
  }

  return string;
}

/**
 * @brief Parses options in the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - value needs to be a single string (no spaces allowed)
 *
 * @return The number of successful assignments.
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

  if (i2 <= i1) {
    return 0;
  }

  return sscanf(&options[i2], format, pointer);
}

/**
 * @brief Reads an integer from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be an integer
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_int(const char *options, const char *name, int *pointer) {
  return coco_options_read(options, name, " %i", pointer);
}

/**
 * @brief Reads a size_t from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be a size_t
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_size_t(const char *options, const char *name, size_t *pointer) {
  return coco_options_read(options, name, "%lu", pointer);
}

/**
 * @brief Reads a double value from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be a double
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_double(const char *options, const char *name, double *pointer) {
  return coco_options_read(options, name, "%lf", pointer);
}

/**
 * @brief Reads a string from options using the form "name1: value1 name2: value2".
 *
 * Formatting requirements:
 * - name and value need to be separated by a colon (spaces are optional)
 * - the value corresponding to the given name needs to be a string - either a single word or multiple words
 * in double quotes
 *
 * @return The number of successful assignments.
 */
static int coco_options_read_string(const char *options, const char *name, char *pointer) {

  long i1, i2;

  if ((!options) || (strlen(options) == 0))
    return 0;

  i1 = coco_strfind(options, name);
  if (i1 < 0)
    return 0;
  i2 = i1 + coco_strfind(&options[i1], ":") + 1;

  /* Remove trailing white spaces */
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
 * @brief Reads (possibly delimited) values from options using the form "name1: value1,value2,value3 name2: value4",
 * i.e. reads all characters from the corresponding name up to the next alphabetic character or end of string,
 * ignoring white-space characters.
 *
 * Formatting requirements:
 * - names have to start with alphabetic characters
 * - values cannot include alphabetic characters
 * - name and value need to be separated by a colon (spaces are optional)
 *
 * @return The number of successful assignments.
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

  if (i2 <= i1) {
    return 0;
  }

  i = 0;
  while (!isalpha((unsigned char) options[i2 + i]) && (options[i2 + i] != '\0')) {
    if(isspace((unsigned char) options[i2 + i])) {
        i2++;
    } else {
        pointer[i] = options[i2 + i];
        i++;
    }
  }
  pointer[i] = '\0';
  return i;
}
/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods implementing functions on double values not contained in C89 standard
 */
/**@{*/

/**
 * @brief  Returns 1 if |a - b| < precision and 0 otherwise.
 */
static int coco_double_almost_equal(const double a, const double b, const double precision) {
  return (fabs(a - b) < precision);
}

/**
 * @brief Rounds the given double to the nearest integer.
 */
static double coco_double_round(const double number) {
  return floor(number + 0.5);
}

/**
 * @brief Rounds the given double up to the nearest double value with the given precision.
 *
 * @note The implementation is (probably unnecessarily) complex, but this was the only way to make
 * sure it works also for edge cases due to float precision issue.
 */
static double coco_double_round_up_with_precision(const double number, const double precision) {
  double rounded_up, rounded;
  double min_precision = 1e-12;
  assert(precision > min_precision);
  rounded_up = ceil(number / precision) * precision;
  rounded = coco_double_round(number / precision) * precision;
  if (coco_double_almost_equal(rounded, rounded_up, precision))
    return rounded_up;
  else {
    if (coco_double_almost_equal(number - rounded, 0, min_precision))
      return rounded;
    else
      return rounded_up;
  }
}

/**
 * @brief Returns the maximum of a and b.
 */
static double coco_double_max(const double a, const double b) {
  if (a >= b) {
    return a;
  } else {
    return b;
  }
}

/**
 * @brief Returns the minimum of a and b.
 */
static double coco_double_min(const double a, const double b) {
  if (a <= b) {
    return a;
  } else {
    return b;
  }
}

/**
 * @brief Performs a "safer" double to size_t conversion.
 *
 * TODO: This method could (should?) check for overflow when casting (similarly as is done in
 * coco_double_to_int()).
 */
static size_t coco_double_to_size_t(const double number) {
  return (size_t) coco_double_round(number);
}

/**
 * @brief Rounds the given double to the nearest integer (returns the number in int type)
 */
static int coco_double_to_int(const double number) {
  if (number > (double)INT_MAX) {
    coco_error("coco_double_to_int(): Cannot cast %f to the nearest integer, max %d allowed",
        number, INT_MAX);
    return -1; /* Never reached */
  }
  else if (number < (double)INT_MIN) {
    coco_error("coco_double_to_int(): Cannot cast %f to the nearest integer, min %d allowed",
        number, INT_MIN);
    return -1; /* Never reached */
  }
  else {
    return (int)(number + 0.5);
  }
}

/**@}*/

/***********************************************************************************************************/

/**
 * @name Methods handling NAN and INFINITY
 */
/**@{*/

/**
 * @brief Returns 1 if x is NAN and 0 otherwise.
 */
static int coco_is_nan(const double x) {
  return (isnan(x) || (x != x) || !(x == x) || ((x >= NAN / (1 + chosen_precision)) && (x <= NAN * (1 + chosen_precision))));
}

/**
 * @brief Returns 1 if the input vector of dimension dim contains any NAN values and 0 otherwise.
 */
static int coco_vector_contains_nan(const double *x, const size_t dim) {
	size_t i;
	for (i = 0; i < dim; i++) {
		if (COCO_UNLIKELY(coco_is_nan(x[i])))
		  return 1;
	}
	return 0;
}

/**
 * @brief Sets all dim values of y to NAN.
 */
static void coco_vector_set_to_nan(double *y, const size_t dim) {
	size_t i;
	for (i = 0; i < dim; i++) {
		y[i] = NAN;
	}
}

/**
 * @brief Returns 1 if x is INFINITY and 0 otherwise.
 */
static int coco_is_inf(const double x) {
	if (coco_is_nan(x))
		return 0;
	return (isinf(x) || (x <= -INFINITY) || (x >= INFINITY));
}

/**
 * @brief Returns 1 if the input vector of dimension dim contains no NaN of inf values, and 0 otherwise.
 */
static int coco_vector_isfinite(const double *x, const size_t dim) {
	size_t i;
	for (i = 0; i < dim; i++) {
		if (COCO_UNLIKELY(coco_is_nan(x[i])) || COCO_UNLIKELY(coco_is_inf(x[i])))
		  return 0;
	}
	return 1;
}

/**
 * @brief Returns 1 if the point x is feasible, and 0 otherwise.
 *
 * Allows constraint_values == NULL, otherwise constraint_values
 * must be a valid double* pointer and contains the g-values of x
 * on "return".
 * 
 * Any point x containing NaN or inf values is considered infeasible.
 *
 * This function is (and should be) used internally only, and does not
 * increase the counter of constraint function evaluations.
 *
 * @param problem The given COCO problem.
 * @param x Decision vector.
 * @param constraint_values Vector of contraints values resulting from evaluation.
 */
static int coco_is_feasible(coco_problem_t *problem,
                            const double *x,
                            double *constraint_values) {

  size_t i;
  double *cons_values = constraint_values;
  int ret_val = 1;

  /* Return 0 if the decision vector contains any INFINITY or NaN values */
  if (!coco_vector_isfinite(x, coco_problem_get_dimension(problem)))
    return 0;

  if (coco_problem_get_number_of_constraints(problem) <= 0)
    return 1;

  assert(problem != NULL);
  assert(problem->evaluate_constraint != NULL);
  
  if (constraint_values == NULL)
     cons_values = coco_allocate_vector(problem->number_of_constraints);

  problem->evaluate_constraint(problem, x, cons_values, 0);

  for(i = 0; i < coco_problem_get_number_of_constraints(problem); ++i) {
    if (cons_values[i] > 0.0) {
      ret_val = 0;
      break;
    }
  }

  if (constraint_values == NULL)
    coco_free_memory(cons_values);
  return ret_val;
}

/**@}*/

/***********************************************************************************************************/

/**
 * @name Miscellaneous methods
 */
/**@{*/

/**
 * @brief Returns the current time as a string.
 *
 * The caller is responsible for freeing the allocated memory using coco_free_memory().
 */
static char *coco_current_time_get_string(void) {
  time_t timer;
  char *time_string = coco_allocate_string(30);
  struct tm* tm_info;
  time(&timer);
  tm_info = localtime(&timer);
  assert(tm_info != NULL);
  strftime(time_string, 30, "%d.%m.%y %H:%M:%S", tm_info);
  return time_string;
}

/**
 * @brief Returns the number of positive numbers pointed to by numbers (the count stops when the first
 * 0 is encountered of max_count numbers have been read).
 *
 * If there are more than max_count numbers, a coco_error is raised. The name argument is used
 * only to provide more informative output in case of any problems.
 */
static size_t coco_count_numbers(const size_t *numbers, const size_t max_count, const char *name) {

  size_t count = 0;
  while ((count < max_count) && (numbers[count] != 0)) {
    count++;
  }
  if (count == max_count) {
    coco_error("coco_count_numbers(): over %lu numbers in %s", (unsigned long) max_count, name);
    return 0; /* Never reached*/
  }

  return count;
}

/**
 * @brief multiply each componenent by nom/denom or by nom if denom == 0.
 *
 * return used scaling factor, usually nom/denom.
 *
 * Example: coco_vector_scale(x, dimension, 1, coco_vector_norm(x, dimension));
 */
static double coco_vector_scale(double *x, size_t dimension, double nom, double denom) {

  size_t i;

  assert(x);

  if (denom != 0)
    nom /= denom;

  for (i = 0; i < dimension; ++i)
      x[i] *= nom;
  return nom;
}

/**
 * @brief return norm of vector x.
 *
 */
static double coco_vector_norm(const double *x, size_t dimension) {

  size_t i;
  double ssum = 0.0;

  assert(x);

  for (i = 0; i < dimension; ++i)
    ssum += x[i] * x[i];

  return sqrt(ssum);
}

/**
 * @brief return scalar product between vectors x and y.
 *
 */
static double coco_vector_scalar_product(const double *x, const double *y, size_t dimension) {

  size_t i;
  double ssum = 0.0;

  assert(x);
  assert(y);

  for (i = 0; i < dimension; ++i)
    ssum += x[i] * y[i];

  return ssum;
}

/**
 * @brief Checks if a given matrix M is orthogonal by (partially) computing M * M^T.
 * If M is a square matrix and M * M^T is close enough to the identity matrix
 * (up to a chosen precision), the function returns 1. Otherwise, it returns 0.
 * The matrix M must be represented as an array of doubles.
 */
static int coco_is_orthogonal(const double *M, const size_t nb_rows, const size_t nb_columns) {

  size_t i, j, z;
  double sum;

  if (nb_rows != nb_columns)
    return 0;

  for (i = 0; i < nb_rows; ++i) {
    for (j = 0; j < nb_rows; ++j) {
        /* Compute the dot product of the ith row of M
         * and the jth column of M^T (i.e. jth row of M)
         */
        sum = 0.0;
        for (z = 0; z < nb_rows; ++z) {
            sum += M[i * nb_rows + z] * M[j * nb_rows + z];
        }

        /* Check if the dot product is 1 (resp. 0) when the row and the column
         * indices are the same (resp. different)
         */
        if (((i == j) && !coco_double_almost_equal(sum, 1, chosen_precision)) ||
            ((i != j) && !coco_double_almost_equal(sum, 0, chosen_precision)))
                return 0;

    }
  }
  return 1;
}

/**
 * @brief Returns 1 if the input vector x is (close to) zero and 0 otherwise.
 */
static int coco_vector_is_zero(const double *x, const size_t dim) {
  size_t i = 0;
  int is_zero = 1;

  if (coco_vector_contains_nan(x, dim))
    return 0;

  while (i < dim && is_zero) {
    is_zero = coco_double_almost_equal(x[i], 0, chosen_precision);
    i++;
  }

  return is_zero;
}
/**@}*/

/***********************************************************************************************************/
