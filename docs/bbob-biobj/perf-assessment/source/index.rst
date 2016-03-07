\newcommand{\IR}{\mathbf{R}}

.. bbob-biobj-experiments-doc documentation master file, created by
   sphinx-quickstart on Tue Jan 26 22:28:28 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the experimental setup description for the bbob-biobj suite!
=======================================================================

This document details the specificities of Coco's performance assessment with respect to the biobjective test suite
``bbob-biobj`` described in more detail in [bbob-biobj-functions-doc]_ .


A Set-Indicator Value Replaces the Objective Function (Tobias)
--------------------------------------------------------------
The general concepts of how the Coco platform suggests to benchmark multi-objective algorithms
is the same than in the single-objective case: for each optimization algorithm, we record the
(expected) runtimes to reach given target precisions for each problem in a given suite.
A problem thereby consists of an objective function, its dimension, and an concrete instantiation
of it (see [coco-functions-doc]_ )


* Evaluation based on the complete archive of nondominated solutions, independent of population size (Tobias)

* explain - give formula for the computation of the hypervolume (if there are no points dominating the Nadir)


Choice of Target Difficulties
-----------------------------
Choice of the targets based on best estimation of Pareto front (using all the data we have) - chosen instance wise

relative targets (in terms of the hypervolume difference to the hypervolume of the reference set)
are chosen the same for all functions, dimensions, and instances: recorded are 100 targets per order of magnitude,
equi-distantly chosen on the log-scale. Displayed are finally only 10 targets per order of magnitude, in total
51 of them between :math:`10^0` and :math:`10^{-5}`

Note that due to the approximative nature of the reference set and its hypervolume, negative hypervolume values are
possible. The Coco platform stores all



Instances and Generalization Experiment
---------------------------------------
* we record for 10 instances but display result for only 5. This will allow us to generate data for an unbiased
  generalization test on the unseen instances


Contents:

.. toctree::
   :maxdepth: 2



   
.. [bbob-biobj-functions-doc] The BBOBies. **Function Documentation of the bbob-biobj Test Suite**. http://numbbo.github.io/coco-doc/bbob-biobj/functions/

.. [coco-functions-doc] The BBOBies. **COCO: Performance Assessment**. http://numbbo.github.io/coco-doc/perf-assessment/