C Implementation
================
TODO, give a pointer?

Setup Python
============
The name of a new suite must be added to `known_suite_names` 
in `code-experiments/build/python/cython/interface.pyx`. 

Setup A Regression Test
=======================

The Python script `code-experiments/test/regression-test/create/create_suite_data.py`
can be executed to create data for the regression test (see `create_suite_data.py -h`). 
The created data files (currently with 2, 10, and 20 solutions per problem) must be
moved to `code-experiments/test/regression-test/data`. That's all there is to it. 
The suite will become part of the regression test if it is listed in `known_suite_names`
(see above).
