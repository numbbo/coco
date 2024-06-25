C Implementation of Functions and a Suite
=========================================
See [here](http://numbbo.github.io/coco-doc/C/#new-suites).

Setup Python
============
The name of a new suite must be added to `known_suite_names` 
in `code-experiments/build/python/cython/interface.pyx`. 

Setup A Regression Test
=======================

The Python script `code-experiments/test/regression-test/create/create_suite_data.py`
can be executed to create data for the regression test (see `create_suite_data.py -h`). 
The created data files (currently with 2, 10, and 20 solutions per problem) must be
moved into the numbbo/regression-tests/ repository to the folder `data/`. A default
set of instances must be also defined such that the test can call the suite with the
same instances as when the regression test data was produced. To this end, the suite
option `year: 0000` must be defined to set up the same set of instances as used for
the creation of the above regression test data.

That's all there is to it. The suite will become part of the regression test if it
is listed in `known_suite_names` (see above).

Adapt the Post-Processing
=========================

In order to show plots and tables for a new suite some changes have to be made also 
in the post-processing module. All the logic about dealing with the suites is in
`code-postprocesing/cocopp/testbedsettings.py` file. There is a dictionary called 
`suite_to_testbed` where you can set which testbed is used for a specific suite. 
An new element `'new_suite_name':'testbed_name'` should be added to the dictionary. 
If none of the existing testbed classes (they are defined in the same file) is 
appropriate then a new testbed class should be added. The class should derive from 
the base `Testbed` class or from one of the existing classes (e.g. `GECCOBBOBTestbed`). 
Only the values that are different from the base class can be specified in the new 
derived class. 
