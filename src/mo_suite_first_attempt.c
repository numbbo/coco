#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "coco.h"
#include "coco_strdup.c"
#include "coco_problem.c"
#include "bbob2009_suite.c"


int coco_problem_id_is_fine(const char *id, ...) {
  va_list args;
  const int reject = 0;
  const int OK = 1;
  const char *cp;
  char *s;
  int result = OK;
  
  va_start(args, id);
  s = coco_vstrdupf(id, args);
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
  coco_free_memory(problem->problem_id);
  problem->problem_id = coco_vstrdupf(id, args);
  va_end(args);
  if (!coco_problem_id_is_fine(problem->problem_id)) {
    coco_error("Problem id should only contain standard chars, not like '%s'",
               coco_get_problem_id(problem));
  }
}
/**
 * Do sprintf(coco_get_problem_id(problem), id, ...) in the right way, tentative, 
 * needs at the minimum some (more) testing. 
 *
 */
void coco_problem_set_name(coco_problem_t *problem, const char *name, ...) {
  va_list args;
  
  va_start(args, name);
  coco_free_memory(problem->problem_name);
  problem->problem_name = coco_vstrdupf(name, args);
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
    coco_problem_set_idf(problem, "ID-F%03d-F%03d-d03%ld-%06ld", f, f2, dimension, problem_index);
    coco_problem_set_namef(problem, "%s + %s",
                          coco_get_problem_name(problem), coco_get_problem_name(problem2));
#endif
    problem->index = problem_index;
    
    return problem; 
  } /* else if ... */
  return NULL;
}
