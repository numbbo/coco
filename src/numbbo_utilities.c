#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "numbbo.h"
#include "numbbo_strdup.c"

/* Figure out if we are on a sane platform or on the dominant platform. */
#if defined(_WIN32) || defined(_WIN64)
  #include <windows.h>
  static const char *numbbo_path_separator = "\\";
  #define NUMBBO_PATH_MAX PATH_MAX
  #define HAVE_GFA
#elif defined(_POSIX_VERSION) || defined(__gnu_linux__) || defined(__APPLE__)
  #include <sys/stat.h>
  #include <sys/types.h>
  static const char *numbbo_path_separator = "/";
  #define HAVE_STAT

  #if defined(__linux__)
    #include <linux/limits.h>
  #elif defined(__FreeBSD__)
    #include <limits.h>
  #endif
  #if !defined(PATH_MAX)
    #error PATH_MAX undefined
  #endif
  #define NUMBBO_PATH_MAX PATH_MAX
#else
  #error Unknown platform
#endif

void numbbo_join_path(char *path, size_t path_max_length, ...) {
    const size_t path_separator_length = strlen(numbbo_path_separator);
    va_list args;
    char *path_component;
    size_t path_length = strlen(path);

    va_start(args, path_max_length);
    while (NULL != (path_component = va_arg(args, char *))) {
        size_t component_length = strlen(path_component);
        if (path_length + path_separator_length + component_length >= path_max_length) {
            numbbo_error("numbbo_file_path() failed because the ${path} is to short.");
            return; /* never reached */
        }
        /* Both should be safe because of the above check. */
        if (strlen(path) > 0) 
            strcat(path, numbbo_path_separator);
        strcat(path, path_component);        
    }
    va_end(args);
}

int numbbo_path_exists(const char *path) {
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

void numbbo_create_path(const char *path) {
    /* Nothing to do if the path exists. */
    if (numbbo_path_exists(path))
        return;
#if defined(HAVE_GFD)
#error Unimplemented
#elif defined(HAVE_STAT)
    assert(strcmp(numbbo_path_separator, "/") == 0);
    char *tmp = NULL;
    char buf[4096];
    size_t len = strlen(path);
    tmp = numbbo_strdup(path);
    /* Remove possible trailing slash */
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (!numbbo_path_exists(tmp)) {
                if (0 != mkdir(tmp, S_IRWXU))
                    goto error;
            }
            *p = '/';
        }
    }
    if (0 != mkdir(tmp, S_IRWXU))
        goto error;
    numbbo_free_memory(tmp);
    return;
error:
    snprintf(buf, sizeof(buf), "mkdir(\"%s\") failed.", tmp);
    numbbo_error(buf);
    return; /* never reached */    
#else
#error Ooops
#endif
}

double *numbbo_allocate_vector(const size_t number_of_elements) {
    return (double *)numbbo_allocate_memory(number_of_elements * sizeof(double));
}

double *numbbo_duplicate_vector(const double *src,
                                const size_t number_of_elements) {
    size_t i;
    double *dst = numbbo_allocate_vector(number_of_elements);

    for (i = 0; i < number_of_elements; ++i) {
        dst[i] = src[i];
    }
    return dst;
}

