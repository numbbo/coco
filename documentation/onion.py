##
## onion.py - example to show how the wrapping of functions in the C
##   code works. 
##
## This is a simplified version of the coco_problem_t /
## coco_transformed_problem_t structure of the C code for
## illustation purposes. The idea behind the C code is the same but we
## need much more boilerplate code to get there. 
class sphere_function(object):
    def __init__(self):
        self._lower_bounds = -5
        self._upper_bounds = 5
        self._best_parameter = 0

    def __call__(self, x):
        return x * x

    def best_parameter(self):
        return self._best_parameter

    def best_value(self):
        return self(self._best_parameter)
        
    def lower_bounds(self):
        return self._lower_bounds
    
    def upper_bounds(self):
        return self._upper_bounds

class transformed_problem(object):
    def __init__(self, inner_problem):
        self._inner_problem = inner_problem

    def __call__(self, x):
        return self._inner_problem(x)

    def lower_bounds(self):
        return self._inner_problem.lower_bounds()

    def upper_bounds(self):
        return self._inner_problem.upper_bounds()

    def best_parameter(self):
        return self._inner_problem.best_parameter()

    def best_value(self):
        return self._inner_problem.best_value()

class tran_var_shift(transformed_problem):
    def __init__(self, inner_problem, shift):
        super(tran_var_shift, self).__init__(inner_problem)
        self.shift = shift

    def __call__(self, x):
        return self._inner_problem(x + self.shift)

    def best_parameter(self):
        return self._inner_problem.best_parameter() + self.shift

class tran_obj_offset(transformed_problem):
    def __init__(self, inner_problem, offset):
        super(tran_obj_offset, self).__init__(inner_problem)
        self.offset = offset

    def __call__(self, x):
        y = self._inner_problem(x)
        return y + self.offset

    def best_value(self):
        return self._inner_problem.best_value() + self.offset

## Example usage:
sphere = sphere_function()
offset_sphere = tran_obj_offset(sphere, 2)
shifted_sphere = tran_var_shift(sphere, 2)
shifted_offset_sphere = tran_var_shift(offset_sphere, 3)

sphere(2)
sphere.best_parameter()
sphere.best_value()

offset_sphere(2)
offset_sphere.best_parameter()
offset_sphere.best_value()

shifted_sphere(2)
shifted_sphere.best_parameter()
shifted_sphere.best_value()

shifted_offset_sphere(2)
shifted_offset_sphere.best_parameter()
shifted_offset_sphere.best_value()
