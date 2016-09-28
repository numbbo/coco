"""Experimentation module of the COCO - COmparing Continuous Optimizers -
framework.

The module provides benchmark test beds in the `Suite` class
and output data facilities in the `Observer` class.

See the documentation of the `Suite` class:

>>> import cocoex as ex
>>> help(ex.Suite)  # doctest: +ELLIPSIS
Help on class Suite...
>>> print(ex.known_suite_names)  # doctest: +ELLIPSIS
[...

A more complete example use case can be found in the `example_experiment.py`
file.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from . import utilities
try:
    from ._interface import Suite, Observer, known_suite_names, log_level
except Exception as _e:
    # print("numbbo/code-experiments/build/python/python/__init__.py: could not import '_interface', trying 'interface'", _e)
    from .interface import Suite, Observer, known_suite_names, log_level
del absolute_import, division, print_function, unicode_literals

# from .utilities import about_equal
# from .exceptions import NoSuchProblemException, InvalidProblemException
