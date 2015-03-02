/*
 * Automatic platform dependent "configuration" of COCO framework
 *
 * Some platforms and standard conforming compilers require extra defines or
 * includes to provide some functionality. 
 *
 * Because most feature defines need to be set before the first system header
 * is included and we do not know when a system header is included for the
 * first time in the amalgamation, all internal files should include this file
 * before any system headers.
 *
 */
#ifndef __COCO_PLATFORM__ 
#define __COCO_PLATFORM__

/* Because C89 does not have a round() function, dance around and try to force
 * a definition.
 */
#if defined(unix) || defined(__unix__) || defined(__unix)
/* On Unix like platforms, force POSIX 2008 behaviour which gives us fmin(),
 * fmax(), round() and snprintf() even if we do not have a C99 compiler.
 */
#define _POSIX_C_SOURCE 200809L
#endif

#endif
