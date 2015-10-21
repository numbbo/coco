#include <stdlib.h>
#include "coco.h"
#include "coco_problem.c"

typedef struct {
  double **constraints_matrix;
  size_t number_of_linear_constraints;
  coco_free_function_t old_free_problem;
} _addcons_data_t;

static void linear_constraint(coco_problem_t *self, const double *x, double *y){
  
  size_t i, j;
  
  _addcons_data_t *data;

  data = coco_get_transform_data(self);
  
  for (j = 0; j < data->number_of_linear_constraints; ++j) { y[j] = 0.0; }
  
  for (j = 0; j < data->number_of_linear_constraints; ++j) {
      for (i = 0; i < self->number_of_variables; ++i) {
	  y[j] += data->constraints_matrix[j][i] * (x[i] - self->best_parameter[i]);
      }
  }
}

static void _addcons_free_data(void *thing) {
  
  size_t i;
  _addcons_data_t *data = thing;
  
  for (i = 0; i < data->number_of_linear_constraints; ++i) {
      coco_free_memory(data->constraints_matrix[i]);
  }
  coco_free_memory(data->constraints_matrix);
}

static coco_problem_t *coco_add_constraints(coco_problem_t *inner_problem, 
					    const char *constraint_type, 
					    size_t number_of_constraints) {
  
  if (strcmp(constraint_type, "linear_constraint") == 0) {
    
      int random_sign;
      double random_number;
      size_t i, j;
      coco_problem_t *self;
      _addcons_data_t *data;
      
      inner_problem->number_of_constraints = coco_get_number_of_constraints(inner_problem) + number_of_constraints;
      
      data = coco_allocate_memory(sizeof(*data));
      data->number_of_linear_constraints = number_of_constraints;
      
      double **constraints_matrix = (double **)malloc(number_of_constraints*sizeof(double *));
      
      for (j = 0; j < number_of_constraints; j++) {
	  constraints_matrix[j] = coco_allocate_vector(inner_problem->number_of_variables);
      }
      
      srand(time(NULL));
      
      /*
       * Yet to find a good way of generating the constraints matrix
       */
      
      for (j = 0; j < number_of_constraints; j++) {
	  for (i = 0; i < inner_problem->number_of_variables; i++) {
	      random_number = (double)rand();
	      random_sign = rand() % 2;
	      if (random_sign == 0) {
		  constraints_matrix[j][i] = random_number;
	      } else {
		  constraints_matrix[j][i] = -random_number;
	      }
	  }
      }
      /* The vector grad_opt should be passed from somewhere
       * 
      for (i = 0; i < inner_problem->number_of_variables; i++) {
	  constraints_matrix[inner_problem->number_of_constraints-1][i] = -grad_opt[i];
      }
      */
      data->constraints_matrix = constraints_matrix;
      
      self = coco_allocate_transformed_problem(inner_problem, data, _addcons_free_data);
      self->evaluate_constraint = linear_constraint;
      
      return self;
  }
}
