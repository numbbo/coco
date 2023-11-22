#include "vendor/coco.h"

/** Makes the problem returned by coco_suite_get_next_problem owned **/
void coco_suite_forget_current_problem(coco_suite_t *suite);

/** Returns the optimal function value + delta of the problem **/
double coco_problem_get_final_target_fvalue1(const coco_problem_t *problem);
