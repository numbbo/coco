/*
 * Test function and instance ID extraction for BBOB2009 problem suit.
 */

#include <stdio.h>
#include <float.h>
#include <math.h>

#include "coco.h"

int main(int argc, char **argv) {
    int function_id = 0;
    coco_problem_t *problem = NULL;
    
    while (NULL != (problem = coco_get_problem("bbob2009", function_id++))) {
        printf("%4i: %s: f=%02i i=%02i\n", 
               function_id, coco_get_problem_id(problem),
               bbob2009_get_function_id(problem),
               bbob2009_get_instance_id(problem));
    }
    return 0;
}
