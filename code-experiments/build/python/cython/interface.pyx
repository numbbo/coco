# -*- mode: cython -*-
#cython: c_string_type=str, c_string_encoding=ascii
from __future__ import division, print_function
import sys
import numpy as np
cimport numpy as np

from cocoex.exceptions import InvalidProblemException, NoSuchProblemException, NoSuchSuiteException

known_suite_names = ["suite_bbob", "suite_biobj"]

# _test_assignment = "seems to prevent an 'export' error (i.e. induce export) to make this module known under Linux and Windows (possibly because of the leading underscore of _interface)"
# __all__ = ['Problem', 'Benchmark']

# Must initialize numpy or risk segfaults
np.import_array()

cdef extern from "coco.h":
    ctypedef struct coco_problem_t: 
        pass
    ctypedef struct coco_observer_t: 
        pass
    ctypedef struct coco_suite_t: 
        pass
    coco_problem_t *deprecated__coco_suite_get_problem(const char *problem_suite,
                                           const long problem_index)
    int deprecated__coco_suite_get_next_problem_index(const char *problem_suite, 
                                          const long problem_index,
                                          const char *select_options)
    
    coco_problem_t *deprecated__coco_problem_add_observer(coco_problem_t *problem,
                                              const char *observer_name,
                                              const char *options)

    void coco_problem_free(coco_problem_t *problem)
    coco_observer_t *coco_observer(const char *observer_name, const char *options)
    void coco_observer_free(coco_observer_t *self)
    coco_problem_t *coco_problem_add_observer(coco_problem_t *problem, 
                                              coco_observer_t *observer)    
    
    coco_suite_t *coco_suite(const char *suite_name, const char *suite_instance, 
                             const char *suite_options)
    void coco_suite_free(coco_suite_t *suite)
    
    void coco_evaluate_function(coco_problem_t *problem, double *x, double *y)
    void coco_evaluate_constraint(coco_problem_t *problem, const double *x, double *y)
    void coco_recommend_solutions(coco_problem_t *problem, 
                                  const double *x,
                                  size_t number_of_solutions)
                                  
    coco_problem_t* coco_suite_get_next_problem(coco_suite_t*, coco_observer_t*)
    coco_problem_t* coco_suite_get_problem(coco_suite_t *, size_t)

    size_t coco_problem_get_suite_dep_index(coco_problem_t* )
    size_t coco_problem_get_dimension(coco_problem_t *problem)
    size_t coco_problem_get_number_of_objectives(coco_problem_t *problem)
    size_t coco_problem_get_number_of_constraints(coco_problem_t *problem)
    const char *coco_problem_get_id(coco_problem_t *problem)
    const char *coco_problem_get_name(coco_problem_t *problem)
    const double *coco_problem_get_smallest_values_of_interest(coco_problem_t *problem)
    const double *coco_problem_get_largest_values_of_interest(coco_problem_t *problem)
    double coco_problem_get_final_target_fvalue1(coco_problem_t *problem)
    size_t coco_problem_get_evaluations(coco_problem_t *problem)
    double coco_problem_get_best_observed_fvalue1(coco_problem_t *problem)

cdef bytes _bstring(s):
    if type(s) is bytes:
        return <bytes>s
    elif isinstance(s, unicode):
        return s.encode('ascii')
    else:
        raise TypeError()

cdef coco_observer_t* _current_observer

cdef class Suite:
    """Suite of benchmark problems.
    
    Input arguments to `Suite` are `name: str`, `instance: str`, `options: str`,
    and passed to the respective C code (see `coco.h`).

    >>> import cocoex as co
    >>> suite = co.Suite("suite_bbob", "", "")
    >>> f = suite.next_problem()
    >>> print("f([1,2]) = %.11f" % f([1,2]))
    f([1,2]) = 90.00369408000
    
    Sweeping through all functions is as simple as::
    
    >>> suite = co.Suite("suite_biobj", "", "")
    >>> observer = co.Observer("observer_biobj", "")
    >>> for f in suite:
    >>>     observer.observer(f)
    >>>     # run run run using f
    
    In this example, an observer was added to produce output data for the
    COCO post-processing.
    
    The following example runs the entire bbob2009 benchmark suite
    on random search::
    
    >>> import numpy as np
    >>> from cocoex import Suite, Observer
    ...
    >>> MAX_FE = 22  # max f-evaluations
    >>> def random_search(f, lb, ub, m):  # don't use m >> 1e5 with this implementation
    ...     candidates = lb + (ub - lb) * np.random.rand(m, len(lb))
    ...     return candidates[np.argmin([f(x) for x in candidates])]
    ...
    >>> solver = random_search
    >>> suite = Suite("suite_bbob", "year:2009", "")
    >>> observer = Observer("observer_bbob",
    ...                "result_folder: %s_on_%s" % (solver.__name__, "bbob2009"))
    >>> for fun in suite:
    ...     observer.observe(fun)
    ...     solver(fun, fun.lower_bounds, fun.upper_bounds, MAX_FE)
    >>> # data should be now in "random_search_on_bbob2009" folder 
    >>>
    >>> # Exactly the same using another looping technique:
    >>> for id in suite.ids:
    ...     fun = suite.get_problem(id, observer)
    ...     solver(fun, fun.lower_bounds, fun.upper_bounds, MAX_FE)
    ...     fun.free()  # this is absolutely necessary here
    
    See module attribute `cocoex.known_suite_names` for known suite names and file
    `example_experiment.py` for a full example use case.
    
    """
    cdef coco_suite_t* suite  # AKA _self
    cdef coco_problem_t* _current_problem
    cdef current_problem
    cdef size_t _current_index
    cdef bytes _name  # used in @property name
    cdef bytes instance
    cdef bytes options
    cdef _ids
    cdef _indices
    cdef _names
    cdef _dimensions
    cdef initialized
    def __cinit__(self, suite_name, suite_instance, suite_options):
        cdef np.npy_intp shape[1]  # probably completely useless
        self._name = _bstring(suite_name)
        self.instance = _bstring(suite_instance)
        self.options = _bstring(suite_options)
        self._current_problem = NULL
        self.current_problem = None
        self._current_index = -1  # in self._indices
        self.initialized = False
        self._initialize()
        assert self.initialized
    cdef _initialize(self):
        """sweeps through `suite` to collect indices and id's to operate by
        direct access in the remainder"""
        cdef np.npy_intp shape[1]  # probably completely useless
        cdef coco_suite_t* suite
        cdef coco_problem_t* p
        if self.initialized:
            self.reset()
        self._ids = []
        self._indices = []
        self._names = []
        self._dimensions = []
        try:
            suite = coco_suite(self._name, self.instance, self.options)
        except:
            raise NoSuchSuiteException("No suite with name '%s' found" % self._name)
        if suite == NULL:
            raise NoSuchSuiteException("No suite with name '%s' found" % self._name)
        while True:
            p = coco_suite_get_next_problem(suite, NULL)
            if not p:
                break
            self._indices.append(coco_problem_get_suite_dep_index(p))
            self._ids.append(coco_problem_get_id(p))
            self._names.append(coco_problem_get_name(p))
            self._dimensions.append(coco_problem_get_dimension(p))
        coco_suite_free(suite)
        self.suite = coco_suite(self._name, self.instance, self.options)
        self.initialized = True
        return self
    def reset(self):
        """reset to original state, affecting `next_problem()`"""
        self._current_index = -1
        if self.current_problem:
            self.current_problem.free()
        self.current_problem = None
        self._current_problem = NULL    
    def next_problem(self, observer=None):
        """`next_problem(observer=None)` serves to sweep through the `Suite`.
        """
        cdef size_t index 
        global _current_observer
        if not self.initialized:
            raise ValueError("Suite has been finalized/free'ed")
        if self.current_problem:
            self.current_problem.free()
        self._current_index += 1
        if self._current_index >= len(self):
            self._current_problem = NULL
            self.current_problem = None
            # self._current_index = -1  # or use reset?
        else:
            self._current_problem = coco_suite_get_problem( 
                                        self.suite, self._current_index)
            self.current_problem = Problem_init(self._current_problem, 
                                                True, self._name)
            if observer:
                observer.observe(self.current_problem)
        return self.current_problem
    def get_problem(self, id, observer=None):
        """`get_problem(self, id, observer=None)` returns a `Problem` instance,
        by default unobserved, using `id: str` or index (`id: int`) to 
        identify the desired problem.
        
        Details:
        - Here an `index` takes values between 0 and `len(self)-1` and can in
          principle be different from the problem index in the benchmark suite.
        
        - For some suites and/or observers, the `free()` method of the problem
          must be called before the next call of `get_problem`. Otherwise Python
          might just silently die.
          
        """
        if not self.initialized:
            raise ValueError("Suite has been finalized/free'ed")
        index = id
        try:
            1 / (id == int(id))  # int(id) might raise an exception
        except:
            index = self._ids.index(id)
        try:
            return Problem_init(coco_suite_get_problem(self.suite, self._indices[index]), 
                                True, self._name).add_observer(observer)
        except:
            raise NoSuchProblemException(self.name, str(id))
    
    def free(self):
        """free underlying C structures"""
        self.__dealloc__()
        self.suite = NULL
        self.initialized = False  # not (yet) visible from outside
    def __dealloc__(self):
        if self.suite:
            coco_suite_free(self.suite)

    def find_problem_ids(self, *id_snippets, get_problem=False, verbose=False):
        """`find_problem_ids(*id_snippets, verbose=False)`
        returns all problem IDs that contain each of the `id_snippets`.

        If `get_problem is True`, the problem for the first matching ID is
        returned.
        
        >>> import cocoex as co
        >>> s = co.Suite("suite_bbob", "", "")
        >>> s.find_problem_ids("f001", "d10", "i01")
        ['suite_bbob_f001_i01_d10']
        
        """
        res = []
        for idx, id in enumerate(self._ids):
            if all([id.find(i) >= 0 for i in id_snippets]):
                if verbose:
                    print("  id=%s, index=%d" % (id, idx))
                res.append(id)
        if get_problem:
            return self.get_problem(res[0])
        return res

    @property
    def current_index(self):
        """index in the enumerator of all problems in this suite.
        
        To get the index in the underlying C implementation use::
        
            import cocoex as ex
            suite = ex.Suite("suite_bbob", "", "")
            suite.indices[suite.current_index]  # raises an exception, as current_index is None
        """
        return self._current_index
    @property
    def names(self):
        """list of problem names in this `Suite`, see also `ids`"""
        return list(self._names)
    @property
    def dimensions(self):
        """list of problem dimensions occuring at least once in this `Suite`"""
        return sorted(set(self._dimensions))
    @property
    def ids(self):
        """list of all problem IDs.

        An ID can be used when calling the method `get_problem(id)`.

        See also `find_problem_ids`.
        """
        return list(self._ids)
    @property
    def indices(self):
        """list of all problem indices, depreciated.
        
        These values are (only) used to call the underlying C structures.
        Indices used in the Python interface run between 0 and `len(self)`.
        """
        return list(self._indices)
    @property
    def name(self):
        """name of this suite as used to instantiate the suite via `Suite(name, ...)`"""
        return self._name

    def __repr___(self):
        return '<Suite("%r", "%r", "%r")>'  % (self.name, self.instance, self.options)
    def __str__(self):
        return 'Suite("%s", "%s", "%s") with %d problems'  % (self.name, self.instance, self.options, len(self))
    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        """iterator over self
        CAVEAT: side effect on current problem
        """
        if 1 < 3:
            s = self
            s.reset()
        else:
            s = Suite(self.name, self.instance, self.options)
        try:
            while True:
                try:
                    problem = s.next_problem()
                    if problem is None:
                        raise StopIteration
                except NoSuchProblemException:
                    raise StopIteration
                yield problem
        except:
            raise
        finally:  # makes this ctrl-c safe, at least it should
            s is self or s.free()

cdef class Observer:
    """Observer which can be "attached to" one or several problems"""
    cdef coco_observer_t* _observer
    cdef bytes _name
    cdef bytes _options
    cdef _state
    def __cinit__(self, name, options):
        self._name = _bstring(name)
        self._options = _bstring(options)
        self._observer = coco_observer(self._name, self._options)
        self._state = 'initialized'

    def _update_current_observer_global(self):
        """assign the global _current_observer variable to self._observer, 
        for purely technical reasons"""
        global _current_observer
        _current_observer = self._observer
        
    def observe(self, problem):
        """`observe(problem)` let `self` observe the `problem` by calling 
        `problem.add_observer(self)`.

        The typical observer records data to be used in the COCO post-processing
        afterwards.

        >>> import cocoex as co
        >>> suite = co.Suite("suite_bbob", "", "")
        >>> f = suite.get_problem(33)
        >>> observer = co.Observer("observer_bbob", "").observe(f)
        >>> # work work work with f

        """
        self._update_current_observer_global()
        # problem.problem = coco_problem_add_observer(problem.problem, self._observer)
        # cannot be done here, because problem.problem is not recognized as C variable here
        problem.add_observer(self)  
        return self
        
    @property
    def name(self):
        """name of the observer as used with `Observer(name, ...)` to instantiate
        `self` before.
        """
        return self._name
    @property
    def options(self):
        return self._options
    @property
    def state(self):
        return self._state

    def free(self):
        self.__dealloc__()
        self._observer = NULL
        self._state = 'deactivated'
    def __dealloc__(self):
        if self._observer !=  NULL:
            coco_observer_free(self._observer)

cdef Problem_init(coco_problem_t* problem, free=True, suite_name=None):
    """`Problem` class instance initialization wrapper passing 
    a `problem_t*` C-variable to `__init__`. 
    
    This is necessary because __cinit__ cannot be defined as cdef, only as def. 
    """
    res = Problem()
    res._suite_name = suite_name
    return res._initialize(problem, free)
cdef class Problem:
    """`Problem` instances are usually generated using `Suite`.
    
    The main feature of a problem instance is that it is callable, returning the
    objective function value when called with a candidate solution as input.
    """
    cdef coco_problem_t* problem
    cdef np.ndarray y  # argument for coco_evaluate
    # cdef public const double[:] test_bounds
    # cdef public np.ndarray lower_bounds
    # cdef public np.ndarray upper_bounds
    cdef public np.ndarray _lower_bounds
    cdef public np.ndarray _upper_bounds
    cdef size_t _number_of_variables
    cdef size_t _number_of_objectives
    cdef size_t _number_of_constraints
    cdef _suite_name  # for the record
    cdef _list_of_observers  # for the record
    cdef _problem_index  # for the record, this is not public but used in index property
    cdef _do_free
    cdef initialized
    def __cinit__(self):
        cdef np.npy_intp shape[1]
        self.initialized = False  # all done in _initialize
    cdef _initialize(self, coco_problem_t* problem, free=True):
        cdef np.npy_intp shape[1]
        if self.initialized:
            raise RuntimeError("Problem already initialized")
        if problem == NULL:
            raise ValueError("in Problem._initialize(problem,...): problem is NULL")
        self.problem = problem
        self._problem_index = coco_problem_get_suite_dep_index(self.problem)
        self._do_free = free
        self._list_of_observers = []
        # _problem_suite = _bstring(problem_suite)
        # self.problem_suite = _problem_suite
        # Implicit type conversion via passing safe, 
        # see http://docs.cython.org/src/userguide/language_basics.html
        self._number_of_variables = coco_problem_get_dimension(self.problem)
        self._number_of_objectives = coco_problem_get_number_of_objectives(self.problem)
        self._number_of_constraints = coco_problem_get_number_of_constraints(self.problem)
        self.y = np.zeros(self._number_of_objectives)
        ## FIXME: Inefficient because we copy the bounds instead of
        ## sharing the data.
        self._lower_bounds = -np.inf * np.ones(self._number_of_variables)
        self._upper_bounds = np.inf * np.ones(self._number_of_variables)
        # self.test_bounds = coco_problem_get_smallest_values_of_interest(self.problem)  # fails
        for i in range(self._number_of_variables):
            if coco_problem_get_smallest_values_of_interest(self.problem) is not NULL:
                self._lower_bounds[i] = coco_problem_get_smallest_values_of_interest(self.problem)[i]
            if coco_problem_get_largest_values_of_interest(self.problem) is not NULL:
                self._upper_bounds[i] = coco_problem_get_largest_values_of_interest(self.problem)[i]
        self.initialized = True
        return self
    def constraint(self, x):
        """return constraint values for `x`. 

        By convention, constraints with values >= 0 are satisfied.
        """
        raise NotImplementedError("has never been tested, incomment this to start testing")
        cdef np.ndarray[double, ndim=1, mode="c"] _x
        x = np.array(x, copy=False, dtype=np.double, order='C')
        if np.size(x) != self.number_of_variables:
            raise ValueError(
                "Dimension, `np.size(x)==%d`, of input `x` does " % np.size(x) +
                "not match the problem dimension `number_of_variables==%d`." 
                             % self.number_of_variables)
        _x = x  # this is the final type conversion
        if self.problem is NULL:
            raise InvalidProblemException()
        coco_evaluate_constraint(self.problem,
                               <double *>np.PyArray_DATA(_x),
                               <double *>np.PyArray_DATA(self.y))
        return self.y
    def recommend(self, arx):
        """Recommend a list of solutions (with len 1 in the single-objective
        case). """
        raise NotImplementedError("has never been tested, incomment this to start testing")
        cdef np.ndarray[double, ndim=1, mode="c"] _x
        assert isinstance(arx, list)
        number = len(arx)
        x = np.hstack(arx)
        x = np.array(x, copy=False, dtype=np.double, order='C')
        if np.size(x) != number * self.number_of_variables:
            raise ValueError(
                "Dimensions, `arx.shape==%s`, of input `arx` " % str(arx.shape) +
                "do not match the problem dimension `number_of_variables==%d`." 
                             % self.number_of_variables)
        _x = x  # this is the final type conversion
        if self.problem is NULL:
            raise InvalidProblemException()
        coco_recommend_solutions(self.problem, <double *>np.PyArray_DATA(_x),
                                 number)

    def add_observer(self, observer):
        """`add_observer(self, observer: Observer)`, see also `Observer`.
        
        `observer` can be `None`, in which case nothing is done. 
        """
        if observer:
            assert self.problem
            observer._update_current_observer_global()
            self.problem = coco_problem_add_observer(self.problem, _current_observer)
            self._list_of_observers.append(observer)
        return self

    def _f0(self, x):
        """"inofficial" interface to `self` with target f-value of zero. """
        return self(x) - self.final_target_fvalue1

    @property
    def list_of_observers(self):
        return self._list_of_observers
    property number_of_variables:  # this is cython syntax, not known in Python
        # this is a class definition which is instantiated automatically!?
        """Number of variables this problem instance expects as input."""
        def __get__(self):
            return self._number_of_variables

    @property
    def number_of_objectives(self):
        "number of objectives, if equal to 1, call returns a scalar"
        return self._number_of_objectives
    @property
    def number_of_constraints(self):
        "number of constraints"
        return self._number_of_constraints
    @property
    def lower_bounds(self):
        """depending on the test bed, these are not necessarily strict bounds
        """
        return self._lower_bounds
    @property
    def upper_bounds(self):
        """depending on the test bed, these are not necessarily strict bounds
        """
        return self._upper_bounds
    @property
    def evaluations(self):
        return coco_problem_get_evaluations(self.problem)
    @property
    def final_target_fvalue1(self):
        assert(self.problem)
        return coco_problem_get_final_target_fvalue1(self.problem)
    @property
    def best_observed_fvalue1(self):
        assert(self.problem)
        return coco_problem_get_best_observed_fvalue1(self.problem)

    def free(self):
        """Free the given test problem. 
        
        Not strictly necessary (unless, possibly, for the observer). `free`  
        ensures that all files associated with the problem are closed as
        soon as possible and any memory is freed. After free()ing the
        problem, all other operations are invalid and will raise an
        exception.
        """
        if self.problem != NULL:
            coco_problem_free(self.problem)
            self.problem = NULL

    def __dealloc__(self):
        # see http://docs.cython.org/src/userguide/special_methods.html
        # free let the problem_free() call(s) in coco_suite_t crash, hence
        # the possibility to set _do_free = False
        if self._do_free and self.problem != NULL:  # this is not guaranteed to work, see above link
            coco_problem_free(self.problem)
            
    # def __call__(self, np.ndarray[double, ndim=1, mode="c"] x):
    def __call__(self, x):
        """return objective function value of input `x`"""
        cdef np.ndarray[double, ndim=1, mode="c"] _x
        assert self.initialized
        x = np.array(x, copy=False, dtype=np.double, order='C')
        if np.size(x) != self.number_of_variables:
            raise ValueError(
                "Dimension, `np.size(x)==%d`, of input `x` does " % np.size(x) +
                "not match the problem dimension `number_of_variables==%d`." 
                             % self.number_of_variables)
        _x = x  # this is the final type conversion
        if self.problem is NULL:
            raise InvalidProblemException()
        coco_evaluate_function(self.problem,
                               <double *>np.PyArray_DATA(_x),
                               <double *>np.PyArray_DATA(self.y))
        return self.y[0] if self._number_of_objectives == 1 else self.y
        
    @property
    def id(self): 
        "id as string without spaces or weird characters"
        if self.problem is not NULL:
            return coco_problem_get_id(self.problem)
    
    @property    
    def name(self):
        if self.problem is not NULL:
            return coco_problem_get_name(self.problem)
            
    @property
    def index(self):
        """problem index in the benchmark `Suite` of origin"""
        return self._problem_index

    @property
    def suite(self):
        """benchmark suite this problem is from"""
        return self._suite_name
    
    @property
    def info(self):
        return str(self)

    def __str__(self):
        if self.problem is not NULL:
            objective = "%s-objective" % ('single'
                    if self.number_of_objectives == 1 
                    else str(self.number_of_objectives))
            return "%s %s problem (%s)" % (self.id, objective,  
                self.name.replace(self.name.split()[0], 
                                  self.name.split()[0] + "(%d)" 
                                  % (self.index if self.index is not None else -2)))
        else:
            return "finalized/invalid problem"
    
    def __repr__(self):
        if self.problem is not NULL:
            return "<%s(), id=%r>" % (
                    repr(self.__class__).split()[1][1:-2], 
                    # self.problem_suite, self.problem_index, 
                    self.id)
        else:
            return "<finalized/invalid problem>"
        
    def __enter__(self):
        """Allows ``with Benchmark(...).get_problem(...) as problem:``"""
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        try:
            self.free()
        except:
            pass


    # cdef class depreciated_Problem:
    #     """depreciated_Problem(problem_suite: str, problem_index: long)"""
    #     cdef coco_problem_t* problem
    #     cdef np.ndarray y  # argument for coco_evaluate
    #     # cdef public const double[:] test_bounds
    #     # cdef public np.ndarray lower_bounds
    #     # cdef public np.ndarray upper_bounds
    #     cdef public np.ndarray _lower_bounds
    #     cdef public np.ndarray _upper_bounds
    #     cdef size_t _number_of_variables
    #     cdef size_t _number_of_objectives
    #     cdef size_t _number_of_constraints
    #     cdef problem_suite  # for the record
    #     cdef problem_index  # for the record, this is not public but used in index property
    # 
    #     def __cinit__(self, problem_suite, long problem_index):
    #         # see http://docs.cython.org/src/userguide/special_methods.html
    #         cdef np.npy_intp shape[1]
    #         _problem_suite = _bstring(problem_suite)
    #         self.problem_suite = _problem_suite
    #         self.problem_index = problem_index
    #         # Implicit type conversion via passing safe, 
    #         # see http://docs.cython.org/src/userguide/language_basics.html
    #         self.problem = deprecated__coco_suite_get_problem(_problem_suite, problem_index)
    #         if self.problem is NULL:
    #             raise NoSuchProblemException(problem_suite, problem_index)
    #         self._number_of_variables = coco_problem_get_dimension(self.problem)
    #         self._number_of_objectives = coco_problem_get_number_of_objectives(self.problem)
    #         self._number_of_constraints = coco_problem_get_number_of_constraints(self.problem)
    #         self.y = np.zeros(self._number_of_objectives)
    #         ## FIXME: Inefficient because we copy the bounds instead of
    #         ## sharing the data.
    #         self._lower_bounds = -np.inf * np.ones(self._number_of_variables)
    #         self._upper_bounds = np.inf * np.ones(self._number_of_variables)
    #         # self.test_bounds = coco_problem_get_smallest_values_of_interest(self.problem)  # fails
    #         for i in range(self._number_of_variables):
    #             if coco_problem_get_smallest_values_of_interest(self.problem) is not NULL:
    #                 self._lower_bounds[i] = coco_problem_get_smallest_values_of_interest(self.problem)[i]
    #             if coco_problem_get_largest_values_of_interest(self.problem) is not NULL:
    #                 self._upper_bounds[i] = coco_problem_get_largest_values_of_interest(self.problem)[i]
    # 
    #     def add_observer(self, observer, options):
    #         """`add_observer(observer: str, options: str)`
    #         
    #         A valid observer is "observer_bbob2009" and `options`
    #         give the folder to be written into.             
    #             
    #         """
    #         _observer, _options = _bstring(observer), _bstring(options)
    #         self.problem = deprecated__coco_problem_add_observer(self.problem, _observer, _options)
    #     
    #     def constraint(self, x):
    #         """return constraint values for `x`. 
    #         
    #         By convention, constraints with values >= 0 are satisfied.
    #         """
    #         raise NotImplementedError("has never been tested, incomment this to start testing")
    #         cdef np.ndarray[double, ndim=1, mode="c"] _x
    #         x = np.array(x, copy=False, dtype=np.double, order='C')
    #         if np.size(x) != self.number_of_variables:
    #             raise ValueError(
    #                 "Dimension, `np.size(x)==%d`, of input `x` does " % np.size(x) +
    #                 "not match the problem dimension `number_of_variables==%d`." 
    #                              % self.number_of_variables)
    #         _x = x  # this is the final type conversion
    #         if self.problem is NULL:
    #             raise InvalidProblemException()
    #         coco_evaluate_constraint(self.problem,
    #                                <double *>np.PyArray_DATA(_x),
    #                                <double *>np.PyArray_DATA(self.y))
    #         return self.y
    #         
    #     def recommend(self, arx):
    #         """Recommend a list of solutions (with len 1 in the single-objective
    #         case). """
    #         raise NotImplementedError("has never been tested, incomment this to start testing")
    #         cdef np.ndarray[double, ndim=1, mode="c"] _x
    #         assert isinstance(arx, list)
    #         number = len(arx)
    #         x = np.hstack(arx)
    #         x = np.array(x, copy=False, dtype=np.double, order='C')
    #         if np.size(x) != number * self.number_of_variables:
    #             raise ValueError(
    #                 "Dimensions, `arx.shape==%s`, of input `arx` " % str(arx.shape) +
    #                 "do not match the problem dimension `number_of_variables==%d`." 
    #                              % self.number_of_variables)
    #         _x = x  # this is the final type conversion
    #         if self.problem is NULL:
    #             raise InvalidProblemException()
    #         coco_recommend_solutions(self.problem, <double *>np.PyArray_DATA(_x),
    #                                  number)
    #         
    #     property number_of_variables:
    #         """Number of variables this problem instance expects as input."""
    #         def __get__(self):
    #             return self._number_of_variables
    #     
    #     @property
    #     def number_of_objectives(self):
    #         "number of objectives, if equal to 1, call returns a scalar"
    #         return self._number_of_objectives
    #             
    #     @property
    #     def number_of_constraints(self):
    #         "number of constraints"
    #         return self._number_of_constraints
    # 
    #     @property
    #     def lower_bounds(self):
    #         """depending on the test bed, these are not necessarily strict bounds
    #         """
    #         return self._lower_bounds
    #         
    #     @property
    #     def upper_bounds(self):
    #         """depending on the test bed, these are not necessarily strict bounds
    #         """
    #         return self._upper_bounds
    #         
    #     @property
    #     def evaluations(self):
    #         return coco_problem_get_evaluations(self.problem)
    #     
    #     @property
    #     def final_target_fvalue1(self):
    #         assert(self.problem)
    #         return coco_problem_get_final_target_fvalue1(self.problem)
    #         
    #     @property
    #     def best_observed_fvalue1(self):
    #         assert(self.problem)
    #         return coco_problem_get_best_observed_fvalue1(self.problem)
    # 
    #     def free(self):
    #         """Free the given test problem. 
    #         
    #         Not strictly necessary (unless for the observer), but it will  
    #         ensure that all files associated with the problem are closed as
    #         soon as possible and any memory is freed. After free()ing the
    #         problem, all other operations are invalid and will raise an
    #         exception.
    #         """
    #         if self.problem is not NULL:
    #             coco_problem_free(self.problem)
    #             self.problem = NULL
    # 
    #     def __dealloc__(self):
    #         # see http://docs.cython.org/src/userguide/special_methods.html
    #         if self.problem is not NULL:
    #             coco_problem_free(self.problem)
    #             self.problem = NULL
    # 
    #     # def __call__(self, np.ndarray[double, ndim=1, mode="c"] x):
    #     def __call__(self, x):
    #         cdef np.ndarray[double, ndim=1, mode="c"] _x
    #         x = np.array(x, copy=False, dtype=np.double, order='C')
    #         if np.size(x) != self.number_of_variables:
    #             raise ValueError(
    #                 "Dimension, `np.size(x)==%d`, of input `x` does " % np.size(x) +
    #                 "not match the problem dimension `number_of_variables==%d`." 
    #                              % self.number_of_variables)
    #         _x = x  # this is the final type conversion
    #         if self.problem is NULL:
    #             raise InvalidProblemException()
    #         coco_evaluate_function(self.problem,
    #                                <double *>np.PyArray_DATA(_x),
    #                                <double *>np.PyArray_DATA(self.y))
    #         return self.y[0] if self._number_of_objectives == 1 else self.y
    #         
    #     @property
    #     def id(self): 
    #         "id as string without spaces or weird characters"
    #         if self.problem is not NULL:
    #             return coco_problem_get_id(self.problem)
    #     
    #     @property    
    #     def name(self):
    #         if self.problem is not NULL:
    #             return coco_problem_get_name(self.problem)
    #             
    #     @property
    #     def index(self):
    #         """problem index in the benchmark suite"""
    #         return self.problem_index
    # 
    #     @property
    #     def suite(self):
    #         """benchmark suite this problem is from"""
    #         return self.problem_suite
    #     
    #     @property
    #     def info(self):
    #         return str(self)
    # 
    #     def __str__(self):
    #         if self.problem is not NULL:
    #             objective = "%s-objective" % ('single' 
    #                     if self.number_of_objectives == 1 
    #                     else str(self.number_of_objectives))
    #             return "%s %s problem (%s)" % (self.id, objective,  
    #                 self.name.replace(self.name.split()[0], 
    #                                   self.name.split()[0] + "(%d)" 
    #                                   % self.problem_index))
    #         else:
    #             return "finalized/invalid problem"
    #     
    #     def __repr__(self):
    #         if self.problem is not NULL:
    #             return "<%s(%r, %d), id=%r>" % (
    #                     repr(self.__class__).split()[1][1:-2], self.problem_suite,
    #                     self.problem_index, self.id)
    #         else:
    #             return "<finalized/invalid problem>"
    #         
    #     def __enter__(self):
    #         """Allows ``with Benchmark(...).get_problem(...) as problem:``"""
    #         return self
    #     def __exit__(self, exception_type, exception_value, traceback):
    #         try:
    #             self.free()
    #         except:
    #             pass
    # 
    # cdef class Benchmark:
    #     """Benchmark(problem_suite: str, suite_options: str, 
    #                  observer: str, observer_options: str)
    #     
    #     The following example runs the entire bbob2009 benchmark suite
    #     on random search::
    #     
    #         >>> import numpy as np
    #         >>> from cocoex import Benchmark
    #         ... 
    #         >>> MAX_FE = 22  # max f-evaluations
    #         >>> def random_search(f, lb, ub, m):  # don't use m >> 1e5
    #         ...     candidates = lb + (ub - lb) * np.random.rand(m, len(lb))
    #         ...     return candidates[np.argmin([f(x) for x in candidates])]
    #         ...
    #         >>> solver = random_search
    #         >>> benchmark = Benchmark("suite_bbob2009", "", "observer_bbob2009", 
    #         ...                "%s_on_%s" % (solver.__name__, "bbob2009"))
    #         >>> for fun in benchmark:
    #         ...     solver(fun, fun.lower_bounds, fun.upper_bounds, MAX_FE)
    #         >>> # data should be now in "random_search_on_bbob2009" folder 
    #         >>>
    #         >>>
    #         >>> ### A more verbose loop which does exactly the same:
    #         >>>
    #         >>> problem_index = benchmark.first_problem_index
    #         >>> assert problem_index == 0  # true for bbob2009 suite
    #         >>> while problem_index >= 0:
    #         ...     fun = benchmark.get_problem(problem_index)
    #         ...     solver(fun, fun.lower_bounds, fun.upper_bounds, MAX_FE)
    #         ...     fun.free()
    #         ...     problem_index = benchmark.next_problem_index(problem_index)
    #     
    #     """
    #     cdef bytes problem_suite
    #     cdef bytes problem_suite_options
    #     cdef bytes observer
    #     cdef bytes observer_options
    #     cdef _len
    #     cdef _dimensions
    #     cdef _objectives
    # 
    #     def __cinit__(self, problem_suite, problem_suite_options, 
    #                   observer, observer_options):
    #         self.problem_suite = _bstring(problem_suite)
    #         self.problem_suite_options = _bstring(problem_suite_options)
    #         self.observer = _bstring(observer)
    #         self.observer_options = _bstring(observer_options)
    #         self._len = None
    #         self._dimensions = None
    #         self._objectives = None
    #         
    #     def get_problem(self, problem_index, *snippets):
    #         """return callable for benchmarking. 
    #         
    #         get_problem(problem_index_or_snippet: int or str, *snippets: str) -> depreciated_Problem, 
    #         where depreciated_Problem is a callable, taking an array of length 
    #         `Problem.number_of_variables` as input and return a `float` or 
    #         `np.array` (when depreciated_Problem.number_of_objectives > 1) as output. When
    #         `snippets` are given, the first problem of which the id contains all 
    #         snippets including `problem_index_or_snippet' is returned. 
    # 
    #         Example::
    #             >>> import cocoex as cc
    #             >>> b = cc.Benchmark('suite_bbob2009', "", "no_observer", "")
    #             >>> f6 = b.get_problem('f06', 'd10')
    #             >>> print(f6)
    #             bbob2009_f06_i01_d10 single-objective problem (BBOB2009(385) f06 instance 1 in 10D)
    #             >>> f6.free()
    #         """
    #         problem = self.get_problem_unobserved(problem_index, *snippets)
    #         if not problem:
    #             raise NoSuchProblemException
    #         try:
    #             problem.add_observer(self.observer, self.observer_options)
    #         except:
    #             print("adding observer %s with options %s on problem %d of suite %s \n failed" 
    #                 % (self.observer, self.observer_options, problem_index, 
    #                    self.problem_suite))
    #             problem.free()
    #             raise
    #         else:
    #             return problem
    #     
    #     def get_problem_unobserved(self, problem_index, *snippets):
    #         """`get_problem_unobserved(problem_index: int or str, *snippets: str)`
    #         return problem without observer.
    #         
    #         Useful if writing of data is not necessary. Unobserved problems
    #         do not need to be free'd explicitly. See `get_problem` for details. 
    #         """
    #         if problem_index == str(problem_index):
    #             s = problem_index
    #             problem_index = self._get_problem_index_from_id(problem_index, *snippets)
    #             if problem_index is None:
    #                 if snippets:
    #                     raise ValueError("No problem matches snippets")
    #                 else:
    #                     raise ValueError("Problem with id=%s not found" % s)
    #         try:
    #             problem = depreciated_Problem(self.problem_suite, problem_index)
    #             if not problem:
    #                 raise NoSuchProblemException
    #         except:
    #             print("problem %s of suite %s failed to initialize" 
    #                 % (str(problem_index), self.problem_suite))
    #             # any chance that problem.free() makes sense here?
    #             raise
    #         else:
    #             return problem
    #     
    #     def find_problem_ids(self, *id_snippets, verbose=False):
    #         """`find_problem_ids(*id_snippets, verbose=False)`
    #         returns all problem ids that contain each of the `id_snippets`.
    #         """
    #         res = []
    #         for p in self:
    #             if all([p.id.find(i) >= 0 for i in id_snippets]):
    #                 if verbose:
    #                     print("  id=%s, index=%d" % (p.id, p.index))
    #                 res.append(p.id)
    #         return res
    #                 
    #     def _get_problem_index_from_id(self, id, *snippets):
    #         """`_get_problem_index_from_id(id, *snippets)`
    #         returns the first problem index in the benchmark with ``problem.id==id``, 
    #         or the first problem index containing both `id` and all `snippets`, 
    #         or `None`. 
    #         
    #         See also `find_problem_ids`. 
    #         """
    #         if snippets:      
    #             try:
    #                 id = self.find_problem_ids(id, *snippets)[0]
    #             except IndexError:
    #                 return None
    #         for i in self.problem_indices:
    #             with self.get_problem_unobserved(i) as p:
    #                 found = p.id == id
    #             if found:
    #                 return i
    #         return None
    # 
    #     @property    
    #     def first_problem_index(self):
    #         "is `self.next_problem_index(-1)`"
    #         return self.next_problem_index(-1)
    #     
    #     def next_problem_index(self, problem_index):
    #         """`self.next_problem_index(-1)` is the first index. 
    #         
    #         Example::
    #         
    #             >>> from cocoex import Benchmark
    #             >>> bm = Benchmark('suite_bbob2009', '', 'observer_bbob2009', '_tmp')
    #             >>> index bm.first_problem_index
    #             >>> while index >= 0:  # -1 means no problem left
    #             ...     # do something
    #             ...     index = self.next_problem_index(index)
    #                 
    #         See also `problem_indices` for the nicer design pattern. 
    #         """
    #         return deprecated__coco_suite_get_next_problem_index(self.problem_suite, problem_index, 
    #                                        self.problem_suite_options)
    #     @property
    #     def problem_indices(self):
    #         """is an iterator over all problem indices. 
    #         
    #         Example::
    #             
    #             import cocoex
    #             bm = cocoex.Benchmark('suite_bbob2009', '', 'observer_bbob2009', '_tmp')
    #             for index in bm.problem_indices:
    #                 print("There exists a problem with index %d" % index)
    #                 # do something interesting, e.g.
    #                 with bm.get_problem(index) as fun:
    #                     from scipy.optimize import fmin
    #                     from numpy import zeros
    #                     res = fmin(fun, zeros(fun.number_of_variables))                   
    #                 # ...
    #             
    #         """
    #         index = self.first_problem_index
    #         while index >= 0:
    #             yield index
    #             index = self.next_problem_index(index)
    #         # raise StopIteration()  # not necessary
    #                 
    #     @property
    #     def dimensions(self):
    #         """return an ordered set with each problem dimensionality 
    #         (`number_of_variables`) that appears in the suite
    #         """
    #         if self._dimensions is None:
    #             s, o = set(), set()
    #             for i in self.problem_indices:
    #                 with self.get_problem_unobserved(i) as p:
    #                     s.update([p.number_of_variables])
    #                     o.update([p.number_of_objectives])
    #             self._dimensions = sorted(s)
    #             self._objectives = sorted(o)
    #         return self._dimensions
    #     
    #     @property
    #     def objectives(self):
    #         """return an ordered set with each `number_of_objectives` that appears 
    #         in the suite. Usually this set has one element only. 
    #         """
    #         if self._objectives is None:
    #             self.dimensions  # purely for the side effect, prevent code duplication
    #         return self._objectives
    # 
    #     @property
    #     def info(self):
    #         return str(self)
    # 
    #     def __call__(self, *args):
    #         """alias to get_problem_unobserved"""
    #         return self.get_problem_unobserved(*args)
    #         
    #     def __len__(self):
    #         if self._len is None:
    #             self._len = len(list(self.problem_indices))
    #         return self._len
    # 
    #     def __str__(self):
    #         if self.objectives == [1]:
    #             o = 'single-objective'
    #         elif self.objectives == [2]:
    #             o = 'bi-objective'
    #         else:
    #             o = "%s-objective" % (str(self.objectives[0]) 
    #                 if len(self.objectives) == 1 else str(self.objectives))
    #         return 'Suite with %d %s problems in dimensions %s' % (
    #             len(self), o, str(self.dimensions))
    #     
    #     def __repr__(self):
    #         return '<%s(%r, %r, %r, %r)>' % (str(type(self)).split()[1][1:-2],
    #             self.problem_suite, self.problem_suite_options, self.observer, 
    #             self.observer_options)
    #         
    #     def __iter__(self):
    #         for index in self.problem_indices:
    #             problem = self.get_problem(index)
    #             if not problem:
    #                 raise NoSuchProblemException(self.problem_suite, index)
    #             try:
    #                 yield problem
    #             except:
    #                 raise
    #             finally:  # makes this ctrl-c safe
    #                 problem.free()
    # 
    # class Benchmarks(object):
    #     """Each attribute is an unobserved Benchmark instance. Observed benchmark 
    #     instances need to have set an output folder whose name should be related
    #     to the benchmarked algorithm. """
    #     def __init__(self, benchmarks_dict):
    #         for k in benchmarks_dict:
    #             setattr(self, k, benchmarks_dict[k])
    # 
    # benchmarks = Benchmarks(dict([
    #     ['bbob2009unobserved', Benchmark("bbob2009", "", "no_observer", "")], 
    #     # ['bbob2009', Benchmark("bbob2009", "", "bbob2009_observer", "_tmp_")], 
    #      ]))
    # 
    # # has no effect
    # # del InvalidProblemException, NoSuchProblemException
