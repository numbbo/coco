"""Experimentation module of the COCO - COmparing Continuous Optimizers - 
framework. 

The module defines the benchmark test beds in the `Suite` class
and the output data facilities in the `Observer` class. 

Besides some documentation of the `Suite` class, a more complete
example use case can be found in the `example_experiment.py` file. 
"""
import utilities
try:
    from ._interface import Suite, Observer, known_suite_names
except Exception as _e:
    # print("numbbo/code-experiments/build/python/python/__init__.py: could not import '_interface', trying 'interface'", _e)
    from .interface import Suite, Observer, known_suite_names
  
# from .utilities import about_equal
# from .exceptions import NoSuchProblemException, InvalidProblemException
