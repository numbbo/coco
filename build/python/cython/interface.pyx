# -*- mode: cython -*-
#cython: c_string_type=str, c_string_encoding=ascii
import numpy as np
cimport numpy as np

from coco.exceptions import NoSuchProblemException
from coco.exceptions import InvalidProblemException

__all__ = ['Problem', 'Benchmark']

# Must initialize numpy or risk segfaults
np.import_array()

cdef extern from "coco.h":
    ctypedef struct coco_problem_t:
        pass

    coco_problem_t *coco_get_problem(const char *benchmark,
                                     int function_index)

    coco_problem_t *coco_observe_problem(const char *observer_name,
                                         coco_problem_t *problem,
                                         const char *options)

    void coco_free_problem(coco_problem_t *problem)

    void coco_evaluate_function(coco_problem_t *problem, double *x, double *y)

    size_t coco_get_number_of_variables(coco_problem_t *problem)
    size_t coco_get_number_of_objectives(coco_problem_t *problem)
    const char *coco_get_problem_id(coco_problem_t *problem)
    const double *coco_get_smallest_values_of_interest(coco_problem_t *problem)
    const double *coco_get_largest_values_of_interest(coco_problem_t *problem)

cdef bytes _bstring(s):
    if type(s) is bytes:
        return <bytes>s
    elif isinstance(s, unicode):
        return s.encode('ascii')
    else:
        raise TypeError(...)

cdef class Problem:
    cdef coco_problem_t* problem
    cdef np.ndarray y
    cdef public np.ndarray lower_bounds
    cdef public np.ndarray upper_bounds

    def __cinit__(self, problem_suit, int function_index):
        cdef np.npy_intp shape[1]
        _problem_suit = _bstring(problem_suit)
        self.problem = coco_get_problem(_problem_suit, function_index)
        if self.problem is NULL:
            raise NoSuchProblemException(problem_suit, function_index)
        self.y = np.zeros(coco_get_number_of_objectives(self.problem))
        ## FIXME: Inefficient because we copy the bounds instead of
        ## sharing the data.
        self.lower_bounds = np.zeros(coco_get_number_of_variables(self.problem))
        self.upper_bounds = np.zeros(coco_get_number_of_variables(self.problem))
        for i in range(coco_get_number_of_variables(self.problem)):
            self.lower_bounds[i] = coco_get_smallest_values_of_interest(self.problem)[i]
            self.upper_bounds[i] = coco_get_largest_values_of_interest(self.problem)[i]

    def add_observer(self, char *observer, char *options):
        self.problem = coco_observe_problem(observer, self.problem, options)

    property number_of_variables:
        """
        The number of variables this problem instance expects as input.
        """
        def __get__(self):
            return len(self.lower_bounds)

    def free(self):
        """
        Free the given test problem. Not strictly necessary but it will
        ensure that all files associated with the problem are closed as
        soon as possible and any memory is freed. After free()ing the
        problem, all other operations are invalid and will raise an
        exception.
        """
        if self.problem is not NULL:
            coco_free_problem(self.problem)
            self.problem = NULL

    def __dealloc__(self):
        if self.problem is not NULL:
            coco_free_problem(self.problem)
            self.problem = NULL

    def __call__(self, np.ndarray[double, ndim=1, mode="c"] x):
        if self.problem is NULL:
            raise InvalidProblemException()
        coco_evaluate_function(self.problem,
		               <double *>np.PyArray_DATA(x),
                               <double *>np.PyArray_DATA(self.y))
        return self.y

    def __str__(self):
        if self.problem is not NULL:
            return coco_get_problem_id(self.problem)
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
            problem = Problem(self.problem_suit, self._function_index)
            problem.add_observer(self.observer, self.options)
            self._function_index = self._function_index + 1
        except NoSuchProblemException, e:
            raise StopIteration()
        return problem
