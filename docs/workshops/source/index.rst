.. BBOB workshop series documentation master file, created by
   sphinx-quickstart on Tue Jan 12 06:46:36 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the BBOB workshop series!
================================================


The Black-box Optimization Benchmarking (BBOB) workshop series provides an easy-to-use toolchain for
benchmarking black-box optimization algorithms for continuous domains and a place to present, compare, and discuss
the performance of numerical black-box optimization algorithms. The former is realized through the Comparing Continuous
Optimizers platform (Coco).

So far, six workshops have been held (in 2009, 2010, 2012, 2013, and 2015 at GECCO and 2015 at CEC). The next workshop
is going to take place at GECCO 2016 with a :ref:`new focus on multi-objective problems <bbob2016page>` (with two
objectives as a first step).

Generally, three benchmark suites are available:

* ``bbob`` containing 24 noiseless functions
* ``bbob-noisy`` containing 30 noisy functions
* ``bbob-biobj`` containing 55 noiseless, bi-objective functions, generated from instances of the ``bbob`` suite

Note that due to the rewriting of the Coco platform, only the ``bbob`` and ``bbob-biobj`` are available in the new code, available at http://github.com/numbbo/numbbo while for the noisy test suite, the old code at http://coco.gforge.inria.fr/doku.php?id=downloads has to be used.

Contents:
.........

.. toctree::
   :maxdepth: 2

   BBOB-2016/index
   bbobbefore2016



.. `new focus on multi-objective problems`_ BBOB-2016/index.html



.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
