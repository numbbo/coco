# -*- mode: cython -*-
import numpy as np
cimport numpy as np

# Must initialize numpy or risk segfaults
np.import_array()

cdef extern from "numbbo.h":
    ctypedef struct numbbo_problem_t:
        size_t number_of_parameters
        size_t number_of_objectives
        size_t number_of_constraints
        double *lower_bounds
        double *upper_bounds
        char *problem_id
    
    numbbo_problem_t *numbbo_get_problem(const char *benchmark, 
                                         int function_index)

    numbbo_problem_t *numbbo_observe_problem(const char *observer_name,
                                             numbbo_problem_t *problem,
                                             const char *options)
    
    void numbbo_free_problem(numbbo_problem_t *problem)
    
    void numbbo_evaluate_function(numbbo_problem_t *problem, double *x, double *y)

cdef class Problem:
    cdef numbbo_problem_t* problem
    cdef np.ndarray y
    cdef public np.ndarray lower_bounds
    cdef public np.ndarray upper_bounds
    
    def __cinit__(self, char *problem_suit, int function_index,
                  char *observer, char *options):
        cdef np.npy_intp shape[1]
        self.problem = numbbo_get_problem(problem_suit, function_index)
        if self.problem is NULL:
            raise Exception("No such function")
        self.problem = numbbo_observe_problem(observer, self.problem, options)
        self.y = np.zeros(self.problem.number_of_objectives)
        ## FIXME: Inefficient because we copy the bounds instead of
        ## sharing the data.
        self.lower_bounds = np.zeros(self.problem.number_of_parameters)
        self.upper_bounds = np.zeros(self.problem.number_of_parameters)
        for i in range(self.problem.number_of_parameters):
            self.lower_bounds[i] = self.problem.lower_bounds[i]
            self.upper_bounds[i] = self.problem.upper_bounds[i]

    def free(self):
        """
        Free the given test problem. Not strictly necessary but it will
        ensure that all files associated with the problem are closed as
        soon as possible and any memory is freed. After free()ing the
        problem, all other operations are invalid and will raise an
        exception.
        """
        if self.problem is not NULL:
            numbbo_free_problem(self.problem)
            self.problem = NULL

    def __dealloc__(self):
        if self.problem is not NULL:
            numbbo_free_problem(self.problem)
            self.problem = NULL

    def __call__(self, np.ndarray[double, ndim=1, mode="c"] x):
        if self.problem is NULL:
            raise Exception("Invalid problem.")        
        numbbo_evaluate_function(self.problem,
                                 <double *>np.PyArray_DATA(x),
                                 <double *>np.PyArray_DATA(self.y))
        return self.y

    def __str__(self):
        if self.problem is not NULL:
            return self.problem.problem_id
        else:
            return "finalized/invalid problem"

cdef class Benchmark:
    cdef char *problem_suit
    cdef char *observer
    cdef char *options
    cdef int _function_index

    def __cinit__(self, problem_suit, observer, options):
        self.problem_suit = problem_suit
        self.observer = observer
        self.options = options
        self._function_index = 0

    def __iter__(self):        
        return self
    
    def __next__(self):
        try:
            problem = Problem(self.problem_suit, self._function_index,
                              self.observer, self.options)
            self._function_index = self._function_index + 1
        except Exception, e:
            raise StopIteration()
        return problem
