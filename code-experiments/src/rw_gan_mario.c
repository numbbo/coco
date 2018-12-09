/**
 * @file rw_gan_mario.c
 *
 * @brief Implementation of the real-world problems of unsupervised learning of a Generative
 * Adversarial Network (GAN) that understands the structure of Super Mario Bros. levels.
 */

#include <stdio.h>
#include <assert.h>

#include "coco.h"
#include "coco_problem.c"
#include "rw_problem.c"

/**
 * @brief Creates a single- or bi-objective rw_gan_mario problem.
 */
static coco_problem_t *rw_gan_mario_problem_allocate(const char *suite_name,
                                                     const size_t objectives,
                                                     const size_t function,
                                                     const size_t dimension,
                                                     const size_t instance) {

  coco_problem_t *problem = NULL;
  char *str1 = NULL, *str2 = NULL, *str3 = NULL;
  size_t i, num;

  if ((objectives != 1) && (objectives != 2))
    coco_error("rw_gan_mario_problem_allocate(): %lu objectives are not supported (only 1 or 2)",
        (unsigned long)objectives);

  problem = coco_problem_allocate(dimension, objectives, 0);
  for (i = 0; i < dimension; ++i) {
    problem->smallest_values_of_interest[i] = -1;
    problem->largest_values_of_interest[i] = 1;
  }
  problem->number_of_integer_variables = 0;
  problem->evaluate_function = rw_problem_evaluate;
  problem->problem_free_function = rw_problem_data_free;

  coco_problem_set_id(problem, "%s_f%03lu_i%02lu_d%02lu", suite_name, (unsigned long) function,
      (unsigned long) instance, (unsigned long) dimension);

  /*coco_error("test objectives %lu problem f%lu instance %lu in %luD",
        objectives, function, instance, dimension);*/


  if (objectives == 1) {
    coco_problem_set_name(problem, "real-world GAN Mario single-objective problem f%lu instance %lu in %luD",
        function, instance, dimension);
    /* TODO Add realistic best values */
    if (((function >= 1) && (function <=3)) || ((function >=43) && (function <=45))){
         /*enemy Distribution*/
         problem->best_value[0] = -13.5;
    }else if (((function >= 4) && (function <=6)) || ((function >=46) && (function <=48))){
         /*position Distribution*/
         problem->best_value[0] = -7;
    }else if (((function >= 22) && (function <=24)) || ((function >=64) && (function <=66))){
         /*basic fitness agent 0 */
         problem->best_value[0] = 1;
    }else if (((function >= 34) && (function <=36)) || ((function >=76) && (function <=78))){
         /*basic fitness agent 1*/
         problem->best_value[0] = 1;
    }else{
         problem->best_value[0] = 0;
    }



    if ((function >= 1) && (function <= 84)) {
      if (function <= 42)
        str3 = coco_strdup("non-concatenated");
      else
        str3 = coco_strdup("concatenated");
        
      num = (function - 1) % 3;
      if (num == 0)
        str2 = coco_strdup("overworld");
      else if (num == 1)
        str2 = coco_strdup("underground");
      else if (num == 2)
        str2 = coco_strdup("overworlds");

      num = ((function - 1) / 3) % 14;
      str1 = coco_strdupf("%lu", (unsigned long) num + 1);
    }
    else {
      coco_error("rw_gan_mario_problem_allocate(): cannot allocate problem with function %lu",
          (unsigned long)function);
    }
    coco_problem_set_type(problem,
        "rw_gan_mario_eval = '%s', rw_gan_mario_set = '%s', rw_gan_mario_concat = '%s'",
        str1, str2, str3);
    coco_free_memory(str1);
    coco_free_memory(str2);
    coco_free_memory(str3);
  }
  else if (objectives == 2) {
    coco_problem_set_name(problem, "real-world GAN Mario bi-objective problem f%lu instance %lu in %luD",
        function, instance, dimension);
    coco_problem_set_type(problem, "bi-objective");
    /* TODO Add realistic values */
    problem->best_value[0] = -1000;
    problem->best_value[1] = -1000;
    problem->nadir_value[0] = 1000;
    problem->nadir_value[1] = 1000;
  }

  if (problem->best_parameter != NULL) {
    coco_free_memory(problem->best_parameter);
    problem->best_parameter = NULL;
  }

  problem->data = get_rw_problem_data("gan-mario", objectives, function, dimension, instance);

  return problem;
}
