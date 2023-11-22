# -*- mode: cython -*-
#cython: language_level=3, boundscheck=False, c_string_type=str, c_string_encoding=ascii

import numpy as np
cimport numpy as np

from .exceptions import NoSuchProblemException, NoSuchSuiteException

np.import_array()


cdef extern from 'coco.h':
    ctypedef struct coco_problem_t:
        pass

    char *coco_problem_get_id(coco_problem_t *p)
    void coco_evaluate_function(coco_problem_t *p, double *x, double *y)
    void coco_problem_free(coco_problem_t *p)

# IMPORTANT: These functions are *not* declared public in coco.h so we have to
# explicitly declare it as an external function. Otherwise Cython will *not*
# add a declaration to the generated source files.
cdef extern coco_problem_t *coco_get_bbob_problem(size_t function, size_t
                                                  dimension, size_t instance)
                                                  
cdef class BenchmarkFunction:
    """A bare benchmark function from one of the available suites.

    Examples
    --------

    >>> import numpy as np

    Create a 13 dimensional sphere function

    >>> fn = BenchmarkFunction("bbob", 1, 13, 1)
    >>> fn
    BenchmarkFunction('bbob', 1, 13, 1)

    We can also get a short mnemonic name for the function

    >>> str(fn)
    'bbob_f001_i01_d13'

    And of course evaluate it

    >>> x = np.zeros(fn.dimension)
    >>> fn(x)
    124.61122368000001
    """

    cdef coco_problem_t *_problem
    cdef readonly char* suite
    cdef readonly char* id
    cdef readonly int function
    cdef readonly int dimension
    cdef readonly int instance

    def __init__(self, suite: str, function: int, dimension: int, instance: int):
        """
        Create a bare benchmark function from one of the COCO suites.

        Parameters
        ----------
        suite
            Name of benchmark suite ("bbob" only currently)

        function
            ID of function from suite

        dimension
            Dimension of instantiated function

        instance
            Instance ID of instantiated function


        Raises
        ------
        NoSuchSuiteException
           If the `suite` is not known or not yet supported

        NoSuchProblemException
          If no problem with the given `function`, `dimension` and `instance` exists in
          the given `suite`.
        """
        self.suite = suite
        self.function = function
        self.dimension = dimension
        self.instance = instance
        self._problem = NULL
        if suite == "bbob":
            self._problem = coco_get_bbob_problem(function, dimension, instance)
            if self._problem == NULL:
                # FIXME: Possibly extend Exception to include dimension and instance?
                raise NoSuchProblemException(suite, function)
        else:
            raise NoSuchSuiteException(suite)

        self.id = coco_problem_get_id(self._problem)

    def __del__(self):
        if self._problem != NULL:
            coco_problem_free(self._problem)
 
    def __str__(self):
        return self.id

    def __repr__(self):
        return f"BenchmarkFunction('{self.suite}', {self.function}, {self.dimension}, {self.instance})"

    def __call__(self, x):
        cdef double[::1] xi
        cdef double[:, ::1] X
        cdef Py_ssize_t N, D, i
        cdef double[::1] y_view
        cdef double y
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        if x.ndim == 1:
            # Evaluate a single parameter
            xi = np.array(x, dtype=float)
            coco_evaluate_function(self._problem, &xi[0], &y)
            return y
        elif x.ndim == 2:
            # Evaluate several parameters at once
            X = x
            N = X.shape[0]
            D = X.shape[1]
            Y = np.zeros(N, dtype=np.float64)
            y_view = Y
            for i in range(N):
                xi = X[i, :]
                coco_evaluate_function(self._problem, &xi[0], &y)
                y_view[i] = y
            return Y
        else:
          return None


__all__ = ["BenchmarkFunction"]
