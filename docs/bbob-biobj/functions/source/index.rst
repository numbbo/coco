.. biobj-functions documentation master file, created by
   sphinx-quickstart on Thu Dec 24 16:35:27 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

COCO: the Black Box Optimization Benchmarking Bi-objective (bbob-biobj) Test Suite
==================================================================================

The ``bbob-biobj`` test suite contains 55 bi-objective functions in continuous domain which are derived from combining
functions of the well-known single-objective noiseless ``bbob`` test suite. It will be used as the main test suite of
the upcoming `BBOB-2016 workshop <http://numbbo.github.io/workshops/BBOB-2016/>`_ at GECCO. Besides giving the actual
function definitions and presenting their (known) properties, this documentation also aims at
summarizing the state-of-the-art in multi-objective black-box benchmarking, at giving the rational behind our approach,
and at providing a simple tutorial on how to use these functions for actual benchmarking within the Coco framework.

Please note that, for the time being, this documentation is under development and might not contain all final data.

.. figure:: _figs/examples-bbob-biobj.png
   :scale: 60

   Example plots of the Pareto front approximation, found by NSGA-II on selected ``bbob-biobj`` functions. In blue the
   non-dominated points at the end of different independent runs, in red the points that are non-dominated over all runs.



Table of Contents:
------------------

.. toctree::
   :maxdepth: 2
   :numbered: 2

   introduction
   state-of-the-art
   our-approach
   functions/index
   tutorial






Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Definitions and Terminology
---------------------------
We remind in this section different definitions.

*function, instance*
 A function within COCO is a parametrized function :math:`f_\theta:
 \mathbb{R}^n \to \mathbb{R}^m` with :math:`\theta \in \Theta` a set of
 parameters. A parameter determines a so-called instance. For example,
 :math:`\theta` encodes the location of the optimum and two different
 instances have shifted optima.
 
 The integer :math:`n` is the dimension of the search space and
 :math:`m=2` for the  ``bbob-biobj`` test suite. 
 
 
