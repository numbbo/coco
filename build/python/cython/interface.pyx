# -*- mode: cython -*-
#cython: c_string_type=str, c_string_encoding=ascii
import numpy as np
cimport numpy as np

from cocoex.exceptions import InvalidProblemException, NoSuchProblemException

# __all__ = ['Problem', 'Benchmark']

# Must initialize numpy or risk segfaults
np.import_array()

cdef extern from "coco.h":
    ctypedef struct coco_problem_t:
        pass
    coco_problem_t *coco_get_problem(const char *benchmark,
                                     int problem_index)
    coco_problem_t *coco_observe_problem(const char *observer_name,
                                         coco_problem_t *problem,
                                         const char *options)
    int coco_next_problem_index(const char *benchmark, 
                                const int problem_index,
                                const char *benchmark_options)
    void coco_free_problem(coco_problem_t *problem)
    void coco_evaluate_function(coco_problem_t *problem, double *x, double *y)
    size_t coco_get_number_of_variables(coco_problem_t *problem)
    size_t coco_get_number_of_objectives(coco_problem_t *problem)
    const char *coco_get_problem_id(coco_problem_t *problem)
    const char *coco_get_problem_name(coco_problem_t *problem)
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
    """Problem(problem_suite: str, problem_index: int)"""
    cdef coco_problem_t* problem
    cdef np.ndarray y
    cdef public np.ndarray lower_bounds
    cdef public np.ndarray upper_bounds
    # cdef public double[:] lower_vals_oi
    cdef size_t _number_of_objectives
    cdef problem_suite  # for the record
    cdef problem_index  # for the record

    def __cinit__(self, problem_suite, int problem_index):
        # see http://docs.cython.org/src/userguide/special_methods.html
        cdef np.npy_intp shape[1]
        _problem_suite = _bstring(problem_suite)
        self.problem_suite = _problem_suite
        self.problem_index = problem_index
        # Implicit type conversion via passing safe, 
        # see http://docs.cython.org/src/userguide/language_basics.html
        self.problem = coco_get_problem(_problem_suite, problem_index)
        if self.problem is NULL:
            raise NoSuchProblemException(problem_suite, problem_index)
        self.y = np.zeros(coco_get_number_of_objectives(self.problem))
        ## FIXME: Inefficient because we copy the bounds instead of
        ## sharing the data.
        self.lower_bounds = np.zeros(coco_get_number_of_variables(self.problem))
        self.upper_bounds = np.zeros(coco_get_number_of_variables(self.problem))
        for i in range(coco_get_number_of_variables(self.problem)):
            self.lower_bounds[i] = coco_get_smallest_values_of_interest(self.problem)[i]
            self.upper_bounds[i] = coco_get_largest_values_of_interest(self.problem)[i]
        # self.lower_vals_oi = coco_get_smallest_values_of_interest(self.problem)
        self._number_of_objectives = coco_get_number_of_objectives(self.problem)

    def add_observer(self, observer, options):
        """`add_observer(observer: str, options: str)`
        
        A valid observer is "bbob2009_observer" and `options`
        give the folder to be written into.             
            
        """
        _observer, _options = _bstring(observer), _bstring(options)
        self.problem = coco_observe_problem(_observer, self.problem, _options)
    
    # shouldn't this be part of init?
    property number_of_variables:
        """Number of variables this problem instance expects as input.
        """
        def __get__(self):
            return coco_get_number_of_variables(self.problem)
            # this was somewhat a hack, as a problem might not have bounds
            # return len(self.lower_bounds)
    
    @property
    def number_of_objectives(self):
        "number of objectives, if equal to 1, call returns a scalar"
        return self._number_of_objectives
    # number_of_objectives = property(_get_number_of_objectives, None, None, 
    # "number of objectives, if 1 call returns a scalar")
    
    def free(self):
        """Free the given test problem. 
        
        Not strictly necessary (unless for the observer), but it will  
        ensure that all files associated with the problem are closed as
        soon as possible and any memory is freed. After free()ing the
        problem, all other operations are invalid and will raise an
        exception.
        """
        if self.problem is not NULL:
            coco_free_problem(self.problem)
            self.problem = NULL

    def __dealloc__(self):
        # see http://docs.cython.org/src/userguide/special_methods.html
        if self.problem is not NULL:
            coco_free_problem(self.problem)
            self.problem = NULL

    # def __call__(self, np.ndarray[double, ndim=1, mode="c"] x):
    def __call__(self, x):
        cdef np.ndarray[double, ndim=1, mode="c"] _x
        x = np.array(x, copy=False, dtype=np.double, order='C')
        if np.size(x) != self.number_of_variables:
            raise ValueError(
                "Dimension (`npsize(x)==%d`) of input `x` does" % np.size(x) +
                " not match the problem dimension `number_of_variables==%d`." 
                             % self.number_of_variables)
        _x = x  # type conversion
        if self.problem is NULL:
            raise InvalidProblemException()
        coco_evaluate_function(self.problem,
                               <double *>np.PyArray_DATA(_x),
                               <double *>np.PyArray_DATA(self.y))
        return self.y[0] if self._number_of_objectives == 1 else self.y
        
    @property
    def id(self): 
        "id as string without spaces or weird characters"
        return coco_get_problem_id(self.problem)
    
    @property    
    def name(self):
        return coco_get_problem_name(self.problem)
    
    @property
    def info(self):
        return str(self)

    def __str__(self):
        if self.problem is not NULL:
            objective = "%s-objective" % ('single' 
                    if self.number_of_objectives == 1 
                    else str(self.number_of_objectives))
            return "%s %s problem (%s)" % (self.id, objective,  
                self.name.replace(self.name.split()[1], 
                                  self.name.split()[1] + "(%d)" 
                                  % self.problem_index))
        else:
            return "finalized/invalid problem"
    
    def __repr__(self):
        if self.problem is not NULL:
            return "<%s(%r, %d), id=%r>" % (
                    repr(self.__class__).split()[1][1:-2], self.problem_suite,
                    self.problem_index, self.id)
        else:
            return "<finalized/invalid problem>"
        
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        try:
            self.free()
        except:
            pass

cdef class Benchmark:
    """Benchmark(problem_suite: str, suite_options: str, 
                 observer: str, observer_options: str)
    
    Example::
    
        from cocoex import Benchmark
        bm = Benchmark("bbob2009", "", "bbob2009_observer", "random_search")
        fun = bm.get_problem(0)  # first problem in *this* suite
        0 == bm.first_problem_index
        
    where the latter name defines the data folder. 
    
    """
    cdef bytes problem_suite
    cdef bytes problem_suite_options
    cdef bytes observer
    cdef bytes observer_options
    cdef int _current_problem_index
    cdef Problem _current_problem
    cdef _len
    cdef _dimensions
    cdef _objectives

    def __init__(self, problem_suite, problem_suite_options, 
                  observer, observer_options):
        self.problem_suite = _bstring(problem_suite)
        self.problem_suite_options = _bstring(problem_suite_options)
        self.observer = _bstring(observer)
        self.observer_options = _bstring(observer_options)
        self._current_problem_index = -1  # depreciated
        self._current_problem = None  # depreciated 
        self._len = None
        self._dimensions = None
        self._objectives = None
        
    def __len__(self):
        if self._len is None:
            self._len = len(list(self.problem_indices))
        return self._len

    def get_problem(self, problem_index):
        """return callable for benchmarking. 
        
        get_problem(problem_index: int) -> Problem, where Problem is callable,
        taking an array of length Problem.number_of_variables as input and 
        return an array as output. 
        """
        try:
            problem = Problem(self.problem_suite, problem_index)
            problem.add_observer(self.observer, self.observer_options)
        except NoSuchProblemException, e:
            return None
        return problem
        
    def get_problem_unobserved(self, problem_index):
        """return problem without observer (problem_index: int).
        
        Useful if writing of data is not necessary. Unobserved problems
        do not need to be free'd. 
        """
        try:
            problem = Problem(self.problem_suite, problem_index)
        except NoSuchProblemException, e:
            return None
        return problem
    
    @property    
    def first_problem_index(self):
        "is `self.next_problem_index(-1)`"
        return self.next_problem_index(-1)
    def next_problem_index(self, problem_index):
        return coco_next_problem_index(self.problem_suite, problem_index, 
                                       self.problem_suite_options)
    @property
    def problem_indices(self):
        """is an iterator over all problem indices. 
        
        Example::
            
            for i in bm.problem_indices:
                print("There exists a problem with index %d" % i)
                # do something interesting, e.g.
                # p = bm.get_problem(i)
                # ...
            
        """
        index = self.first_problem_index
        while index >= 0:
            yield index
            index = self.next_problem_index(index)
        # raise StopIteration()  # not necessary
                
    @property
    def dimensions(self):
        if self._dimensions is None:
            s, o = set(), set()
            for i in self.problem_indices:
                with self.get_problem_unobserved(i) as p:
                    s.update([p.number_of_variables])
                    o.update([p.number_of_objectives])
            self._dimensions = sorted(s)
            self._objectives = sorted(o)
        return self._dimensions
    
    @property
    def objectives(self):
        if self._objectives is None:
            self.dimensions  # purely for the side effect
        return self._objectives

    @property
    def info(self):
        return str(self)

    def __str__(self):
        if self.objectives == [1]:
            o = 'single-objective'
        else:
            o = str(self.objectives)
        return 'Suite with %d %s problems in dimensions %s' % (
            len(self), o, str(self.dimensions))
    
    def __repr__(self):
        return '<%s(%r, %r, %r, %r)>' % (str(type(self)).split()[1][1:-2],
            self.problem_suite, self.problem_suite_options, self.observer, 
            self.observer_options)
        
    def __iter__(self):
        for index in self.problem_indices:
            try:
                problem = Problem(self.problem_suite, index)
                if not problem:
                    raise NoSuchProblemException
            except Exception as e:  # as requires Python >= 2.6
                print("problem %d of suite %s failed with exception %s" 
                    % (index, self.problem_suite), str(e))
                continue
            try:
                problem.add_observer(self.observer, self.observer_options)
            except Exception as e:
                print("adding observer %s with options %s on problem %d of suite %s \n failed with exception %s" 
                    % (self.observer, self.observer_options, index, self.problem_suite), str(e))
                continue
            else:
                yield problem
            finally:
                # now it is ctrl-C save
                problem.free()
                