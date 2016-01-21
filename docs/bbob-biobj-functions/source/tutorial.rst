===============================================================================================
Benchmarking Multi-objective Optimizers on the ``bbob-biobj`` Suite with Coco: A Short Tutorial
===============================================================================================

The Comparing Continuous Optimizers platform (COCO) offers an almost automated way of benchmarking numerical (black-box)
optimizers. Running experiments is supported in various languages (C, Java, Matlab/Octave, python) during which the
data acquisition is automated and even hidden to the user such that she/he only needs to care about connecting
an optimization algorithm to the Coco framework and providing some cpu time for the actual experiment. The automatically
created data folder can be postprocessed with a generic python script which produces various outputs (plots, html, and
LaTeX/PDF summaries) semi-automatically.

Since its entire rewrite, the Coco platform now especially contains a multi-objective benchmark suite `bbob-biobj` with
the above described 55 functions of two objectives each. In the following, we will present briefly how to run
an experiment with Coco (in the python programming language as an example), postprocess the data, and create a
pdf with the results, preparing for a submission at the GECCO conference (i.e., in ACM style). The comparison of two
and more algorithm data sets is also provided, but not detailed here. For more information, we refer to the
documentation of the C code of Coco which every supported language is build upon: http://numbbo.github.io/coco-doc/C/.



will go here:

a short tutorial on how to use the ``bbob-biobj`` test suite for benchmarking multi-objective optimizers within the Coco
framework