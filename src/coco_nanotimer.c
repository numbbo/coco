#include "coco.h"

#if defined(WIN32)
#include <windows.h>

coco_nanotime_t coco_get_nanotime(void) {
  LARGE_INTEGER time_var, frequency;
  QueryPerformanceCounter(&time_var);
  QueryPerformanceFrequency(&frequency);

  /* Convert to nanoseconds */
  return 1.0e9 * time_var.QuadPart / frequency.QuadPart;
}

#elif defined(__MACH__) || defined(__APPLE__)
#include <mach/mach_time.h>

/* see http://developer.apple.com/library/mac/#qa/qa2004/qa1398.html */
coco_nanotime_t coco_get_nanotime(void) {
  uint64_t time;
  mach_timebase_info_data_t info;

  time = mach_absolute_time();
  mach_timebase_info(&info);

  /* Convert to nanoseconds */
  return time * (info.numer / info.denom);
}

#elif defined(linux) || defined(__linux) || defined(__FreeBSD__) ||            \
    defined(__OpenBSD__)
#include <time.h>

static const coco_nanotime_t nanoseconds_in_second = 1000000000LL;

coco_nanotime_t coco_get_nanotime(void) {
  struct timespec time_var;

  clock_gettime(CLOCK_MONOTONIC, &time_var);

  coco_nanotime_t sec = time_var.tv_sec;
  coco_nanotime_t nsec = time_var.tv_nsec;

  /* Combine both values to one nanoseconds value */
  return (nanoseconds_in_second * sec) + nsec;
}

#elif defined(sun) || defined(__sun) || defined(_AIX)
#include <sys/time.h>

/* short an sweet! */
coco_nanotime_t coco_get_nanotime(void) { return gethrtime(); }

#else
#error "Unsupported OS."
#endif
