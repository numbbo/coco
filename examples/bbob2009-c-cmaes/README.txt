Example CMA-ES run using "new" COCO framework
---------------------------------------------

This directory contains an example run of the (slightly modified[1]) C CMA-ES
implementation on the bbob2009 test suit using the new COCO framework. To run
an experiment, call

  % make all

to build the run_bbob executable and then start the experiment

  % ./run_bbob

After the run is complete, the newly created directory names c-cmaes-v*
contains the results ready to be post-processed.

All of the "magic" is contained in the run_bbob.c file which should be self
explanatory.

[1] The error handling was modified to log to stderr instead of a file.
