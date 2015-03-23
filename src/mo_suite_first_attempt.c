#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco.h"
#include "coco_problem.c"
#include "bbob2009_suite.c"


char * coco_strdup2_v(const char *str, va_list args) {
  static char buf[444];
#if 1 /* this might not be defined on very old systems */
  vsnprintf(buf, 442, str, args); /* apparently args cannot be used another time */
#else /* less save alternative */
  vsprintf(buf, str, args); /* apparently args cannot be used another time */
#endif
  return coco_strdup(buf);
}
char * coco_strdup2(const char *str, ...) {
  va_list args;
  char *s;

  va_start(args, str);
  s = coco_strdup2_v(str, args);
  va_end(args);
  return s;
}

void
coco_problem_set_str_v(char **target, const char *value, va_list args) {
  if (*target)
    coco_free_memory(*target);
  *target = coco_strdup2_v(value, args);
}
void
coco_problem_set_str(char **target, const char *value, ...) {
  va_list args;
  va_start(args, value);
  coco_problem_set_str_v(target, value, args);
  va_end(args);
}

int coco_problem_id_is_fine(const char *id, ...) {
  va_list args;
  const int reject = 0;
  const int OK = 1;
  const char *cp;
  char *s;
  int result = OK;
  
  va_start(args, id);
  s = coco_strdup2_v(id, args);
  va_end(args);
  for (cp = s; *cp != '\0'; ++cp) {
    if (('A' <= *cp) && (*cp <= 'Z'))
      continue;
    if (('a' <= *cp) && (*cp <= 'z'))
      continue;
    if ((*cp == '_') || (*cp == '-'))
      continue;
    if (('0' <= *cp) && (*cp <= '9'))
      continue;
    result = reject;
  }
  coco_free_memory(s);
  return result;  
}
/**
 * Do sprintf(coco_get_problem_id(problem), id, ...) in the right way. 
 *
 */
void coco_problem_set_id(coco_problem_t *problem, const char *id, ...) {
  va_list args;

  va_start(args, id);
  coco_problem_set_str_v(&(problem->problem_id), id, args);
  va_end(args);
  if (!coco_problem_id_is_fine(coco_get_problem_id(problem)))
    coco_error("Problem id should only contain standard chars, not like '%s'", coco_get_problem_id(problem));
}
/**
 * Do sprintf(coco_get_problem_id(problem), id, ...) in the right way, tentative, 
 * needs at the minimum some (more) testing. 
 *
 */
void coco_problem_set_name(coco_problem_t *problem, const char *name, ...) {
  va_list args;
  
  printf("in set name\n");
  va_start(args, name);
  coco_problem_set_str_v(&(problem->problem_name), name, args);
  va_end(args);
}


/**
 * mo_suit...(problem_index):
 *
 * Return the ${problem_index}-th benchmark problem from...
 * If the function index is out of bounds, return
 * NULL.
 */
static coco_problem_t *mo_suite_first_attempt(const int problem_index) {
  coco_problem_t *problem, *problem2;
  long dimension, instance, instance2;
  int f, f2;
  
  if (problem_index < 0) 
    return NULL;

  if (problem_index < 24) { 
  
    /* here we compute the mapping from problem index to the following five values */
    
    dimension = 10;
    f = 1;
    f2 = 1 + (problem_index % 24);
    instance = 0;
    instance2 = 1;
    
    problem = bbob2009_problem(f, dimension, instance);

    problem2 = bbob2009_problem(f2, dimension, instance2);
    problem = coco_stacked_problem_allocate(problem, problem2);
    /* repeat the last two lines to add more objectives */
#if 0
    coco_problem_set_id(problem, "ID-F%03d-F%03d-d03%ld-%06ld", f, f2, dimension, problem_index);
    coco_problem_set_name(problem, "%s + %s",
                          coco_get_problem_name(problem), coco_get_problem_name(problem2));
#endif
    problem->index = problem_index;
    
    return problem; 
  } /* else if ... */
  return NULL;
}
