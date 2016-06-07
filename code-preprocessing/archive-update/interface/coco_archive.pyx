# -*- mode: cython -*-
#interface: c_string_type=str, c_string_encoding=ascii
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
cimport numpy as np

# Must initialize numpy or risk segfaults
np.import_array()

cdef extern from "coco.h":
    
    ctypedef struct coco_archive_t:
        pass
    
    coco_archive_t *coco_archive(char *suite_name, size_t function, 
                                 size_t dimension, size_t instance)
    int coco_archive_add_solution(coco_archive_t *archive, double f1, double f2, char *text)
    size_t coco_archive_get_number_of_solutions(coco_archive_t *archive)
    double coco_archive_get_hypervolume(coco_archive_t *archive)
    char *coco_archive_get_next_solution_text(coco_archive_t *archive)
    void coco_archive_free(coco_archive_t *archive)
    
    char* coco_set_log_level(char *level)
              
cdef bytes _bstring(s):
    if type(s) is bytes:
        return <bytes>s
    elif isinstance(s, unicode):
        return s.encode('ascii')
    else:
        raise TypeError()
            
cdef class Archive:
    """Archive of bi-objective solutions, which serves as an interface to the COCO archive implemented in C.
    """
    cdef coco_archive_t* archive # AKA _self
    cdef bytes _suite_name  
    cdef size_t _function
    cdef size_t _dimension
    cdef size_t _instance
    
    cdef size_t _number_of_solutions
    cdef double _hypervolume    
    cdef bytes _tmp_text
    
    cdef up_to_date
    
    def __cinit__(self, suite_name, function, instance, dimension):
            
        self._suite_name = _bstring(suite_name)
        self._function = function
        self._instance = instance
        self._dimension = dimension
        self.up_to_date = False
        
        self.archive = coco_archive(self._suite_name, self._function, 
                                    self._dimension, self._instance)
        
    def __dealloc__(self):
        coco_archive_free(self.archive)
                            
    def add_solution(self, f1, f2, text):        
        updated = coco_archive_add_solution(self.archive, f1, f2, _bstring(text))
        if updated:
            self.up_to_date = False            
        return updated
        
    def get_next_solution_text(self):
        self._tmp_text = coco_archive_get_next_solution_text(self.archive)
        tmp_text = self._tmp_text.decode('ascii')
        if tmp_text == "":
            return None
        return tmp_text
        
    def update(self):
        if not self.up_to_date:
            self._number_of_solutions = coco_archive_get_number_of_solutions(self.archive)            
            self._hypervolume = coco_archive_get_hypervolume(self.archive)
            self.up_to_date = True
        
    @property
    def number_of_solutions(self):
        self.update()
        return self._number_of_solutions
                
    @property
    def hypervolume(self):
        self.update()
        return self._hypervolume


def log_level(level=None):
    """Returns current log level and sets new log level if level is not None.
       :param level: Supported values: 'error' or 'warning' or 'info' or 'debug', listed with increasing verbosity,
       or '', which doesn't change anything
    """
    cdef bytes _level = _bstring(level if level is not None else "")
    return coco_set_log_level(_level)

