##
## onion.py - example to show how the wrapping of functions in the C
##   code works. 
##
## This is a simplified version of the numbbo_problem_t /
## numbbo_transformed_problem_t structure of the C code for
## illustation purposes. The idea behind the C code is the same but we
## need much more boilerplate code to get there. 
class sphere_function(object):
    def __init__(self):
        self.lower_bounds = -5
        self.upper_bounds = 5
        
    def evaluate(self, x):
        return x * x

    def lower(self):
        return self.lower_bounds
    
    def upper(self):
        return self.upper_bounds

class transformed_problem(object):
    def __init__(self, inner_problem):
        self.inner_problem = inner_problem

    def evaluate(self, x):
        return self.inner_problem.evaluate(x)

    def lower(self):
        return self.inner_problem.lower()

    def upper(self):
        return self.inner_problem.upper()

class shift_objective(transformed_problem):
    def __init__(self, inner_problem, offset):
        super(shift_objective, self).__init__(inner_problem)
        self.offset = offset

    def evaluate(self, x):
        y = self.inner_problem.evaluate(x)
        return y + self.offset

a = sphere_function()
b = shift_objective(a, 2)
