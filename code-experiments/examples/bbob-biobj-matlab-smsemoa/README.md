Example SMS-EMOA run on the biobjective BBOB test suite
-------------------------------------------------------

This directory contains an example run of the SMS-EMOA [1,2]
implementation by Tobias Wagner and Fabian Kretzschmar [3] on the
biobjective BBOB test suite using the COCO framework.

Make sure you have MATLAB, gcc, and g++ installed. To run an
experiment, call first
```
  python do.py build-matlab-sms
```
in your shell to update the coco.c and coco.h files in this matlab
example folder and then type within matlab
```
  setup
  run_smsemoa_on_bbob_biobj
```
to compile and run the algorithm and the coco framework. 

The main "magic" of compilations is contained in the setup.m file while
the algorithm is configured and run via run_smsemoa_on_bbob_biobj.m which
should be self-explanatory.

[1] Michael Emmerich, Nicola Beume, and Boris Naujoks. An EMO algorithm
    using the hypervolume measure as selection criterion. In C. A. Coello
    Coello et al., Eds., Proc. Evolutionary Multi-Criterion Optimization,
    3rd Int'l Conf. (EMO 2005), LNCS 3410, pp. 62-76. Springer, Berlin, 2005.

[2] Boris Naujoks, Nicola Beume, and Michael Emmerich. Multi-objective
    optimisation using S-metric selection: Application to three-dimensional
    solution spaces. In B. McKay et al., Eds., Proc. of the 2005 Congress on
    Evolutionary Computation (CEC 2005), Edinburgh, Band 2, pp. 1282-1289.
    IEEE Press, Piscataway NJ, 2005.
[3] We actually use a newer version than the one from
    https://ls11-www.cs.uni-dortmund.de/rudolph/hypervolume/start
    which allows to call the objective functions through function handles
    instead of the old API which asked to hand over a string and define
    the functions in two files (like FUNCTION.m and initializeFUNCTION.m).
