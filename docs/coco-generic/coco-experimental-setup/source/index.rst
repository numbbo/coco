Welcome to the generic experimental setup description for the Coco platform!
============================================================================

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Black-Box Optimization Benchmarking Procedure
=============================================

.. |ftarget| replace:: :math:`f_\mathrm{target}`
.. |nruns| replace:: :math:`\texttt{Ntrial}`
.. |DIM| replace:: :math:`D`
.. _2009: http://www.sigevo.org/gecco-2009/workshops.html#bbob
.. _2010: http://www.sigevo.org/gecco-2010/workshops.html#bbob
.. _2012: http://www.sigevo.org/gecco-2012/workshops.html#bbob
.. _BBOB-2009: http://coco.gforge.inria.fr/doku.php?id=bbob-2009-results
.. _BBOB-2010: http://coco.gforge.inria.fr/doku.php?id=bbob-2010-results
.. _BBOB-2012: http://coco.gforge.inria.fr/doku.php?id=bbob-2012
.. _GECCO: http://www.sigevo.org/gecco-2012/
.. _COCO: http://coco.gforge.inria.fr

COCO_ has been used in several workshops during the GECCO_ conference since 2009_ (BBOB-2009_).

For these workshops, a testbed of 24 noiseless functions and another of 30
noisy functions are provided. Descriptions of the functions can be found at
http://coco.gforge.inria.fr/doku.php?id=downloads

This section describes the setup of the experimental procedures and their rationales,  giving the guidelines to produce an article for a GECCO-BBOB workshop using COCO.

.. contents::
   :local:

Symbols, Constants, and Parameters
----------------------------------

For the workshops, some constants were set:

  :math:`D = 2; 3; 5; 10; 20; 40` 
    is search space dimensionalities used for all functions.

  :math:`\texttt{Ntrial} = 15` 
    is the number of trials for each single setup, 
    i.e. each function and dimensionality. The performance is evaluated over all trials.

  :math:`\Delta f = 10^{-8}`
    precision to reach, that is, a difference to the smallest
    possible function value :math:`f_\mathrm{opt}`.

  :math:`f_\mathrm{target} = f_\mathrm{opt}+\Delta f` 
    is target function value to reach for different :math:`\Delta f` values.
    The final, smallest considered target function value is
    :math:`f_\mathrm{target} = f_\mathrm{opt} + 10^{-8}`, but also larger values
    for |ftarget| are evaluated.

The |nruns| runs are conducted on different instances of the functions.

.. _sec:rationales:

Rationale for the Choice of Ntrial = 15
_______________________________________

All functions can be instantiated in different "versions" (with
different location of the global optimum and different optimal function value).
Overall |nruns| runs are conducted on different instantiations (in the
2009_ setup each instance was repeated three times). The parameter
|nruns|, the overall number of trials on each function/dimension pair,
determines the minimal measurable success rate and influences the
overall necessary CPU time.  Compared to a standard setup for testing
stochastic search procedures, we have chosen a small value, |nruns| :math:`=15`.
Consequently, within the same CPU-time budget, single trials can be
longer and conduct more function evaluations (until |ftarget| is
reached). If an algorithm terminates before |ftarget| is reached,
longer trials can simply be achieved by independent multistarts.
Because these multistarts are conducted within each trial, more
sophisticated restart strategies are feasible. **Within-trial multistarts 
never impair the used performance measures and are encouraged.** Finally, 
15 trials are sufficient to make relevant performance differences statistically
significant. [#]_

.. [#] If the number of trials is chosen *much* larger, small and 
   therefore irrelevant
   performance differences become statistically significant.

Rationale for the Choice of f\ :sub:`target`
____________________________________________

The initial search domain and the target function value are an essential part
of the benchmark function definition.  Different target function values might
lead to different characteristics of the problem to be solved, besides that
larger target values are invariably less difficult to reach. Functions might be
easy to solve up to a function value of 1 and become intricate for smaller
target values. 
We take records for a larger number of predefined target values, defined relative to the known optimal function value :math:`f_\mathrm{opt}` and in principle unbounded from above. 
The chosen value for the final (smallest) |ftarget| is somewhat arbitrary. 
Reasonable values can change by simple modifications in the function
definition. In order to safely prevent numerical precision problems, the final target is :math:`f_\mathrm{target} = f_\mathrm{opt} + 10^{-8}`.

Benchmarking Experiment
-----------------------

The real-parameter search algorithm under consideration is run on a testbed of
benchmark functions to be minimized (the implementation of the functions is
provided in C/C++, Java, MATLAB/Octave and Python). On each function and for
each dimensionality |nruns| trials are carried out (see also
:ref:`sec:rationales`). Different function *instances* can be used. 

The :file:`exampleexperiment.*` code template is provided to run this experiment. For BBOB-2012, the instances are 1 to 5 and 21 to 30. 

.. _sec:input:

Input to the Algorithm and Initialization
_________________________________________

An algorithm can use the following input:

  #. the search space dimensionality |DIM|

  #. the search domain; all functions of BBOB are defined everywhere in
     :math:`\mathbb{R}^{D}` and have their global optimum in :math:`[-5,5]^{D}`.
     Most BBOB functions have their global optimum in the range
     :math:`[-4,4]^{D}` which can be a reasonable setting for initial solutions.

  #. indication of the testbed under consideration, i.e. different
     algorithms and/or parameter settings can be used for the
     noise-free and the noisy testbed. 

  #. the final target precision delta-value :math:`\Delta f = 10^{-8}` (see above),
     in order to implement effective termination and restart mechanisms 
     (which should also prevent early termination)

  #. the target function value |ftarget|, however provided *only* for
     conclusive (final) termination of trials, in order to reduce the 
     overall CPU requirements. The target function value must not be used as
     algorithm input otherwise (not even to trigger within-trial
     restarts).

Based on these input parameters, the parameter setting and initialization of
the algorithm is entirely left to the user. As a consequence, the setting shall
be identical for all benchmark functions of one testbed (the function
identifier or any known characteristics of the function are, for natural reasons, not allowed as
input to the algorithm, see also Section :ref:`sec:tuning`).

.. _sec:stopping:

Termination Criteria and Restarts
_________________________________

Algorithms with any budget of function evaluations, small or large, are
considered in the analysis of the results. Exploiting a larger number of
function evaluations increases the chance to achieve better function values or
even to solve the function up to the final |ftarget| [#]_.
In any case, a trial can be conclusively terminated if |ftarget| is reached.
Otherwise, the choice of termination is a relevant part of the algorithm: the
termination of unsuccessful trials affects the performance. To exploit a large
number of function evaluations effectively, we suggest considering a multistart
procedure, which relies on an interim termination of the algorithm.

.. |ERT| replace:: :math:`\mathrm{ERT}`

Independent restarts do not change the main performance measure,
*expected running time* (\ |ERT|\ , see Appendix :ref:`sec:ERT`) to hit
a given target. Independent restarts mainly improve the reliability and
"visibility" of the measured value. For example, using a fast algorithm
with a small success probability, say 5% (or 1%), chances are that not a
single of 15 trials is successful. With 10 (or 90) independent restarts,
the success probability will increase to 40% and the performance will
become visible. At least four to five (here out of 15) successful trials are
desirable to accomplish a stable performance measurement. This reasoning
remains valid for any target function value (different values are
considered in the evaluation).

Restarts either from a previous solution, or with a different parameter
setup, for example with different (increasing) population sizes, might be
considered, as it has been applied quite successful [Auger:2005a]_ [Harik:1999]_.

Choosing different setups mimics what might be done in practice. All restart
mechanisms are finally considered as part of the algorithm under consideration.

.. [#] The easiest functions of BBOB can be solved
   in less than :math:`10 D` function evaluations, while on the most difficult
   functions a budget of more than :math:`1000 D^2` function
   evaluations to reach the final :math:`f_\mathrm{target} = f_\mathrm{opt} + 10^{-8}` is expected.


Time Complexity Experiment
--------------------------

In order to get a rough measurement of the time complexity of the algorithm,
the overall CPU time is measured when running the algorithm on :math:`f_8`
(Rosenbrock function) of the BBOB testbed for at least a few tens of seconds
(and at least a few iterations).  The chosen setup should reflect a "realistic
average scenario". If another termination criterion is reached, the algorithm
is restarted (like for a new trial). The *CPU-time per function evaluation* is
reported for each dimension. The time complexity experiment is conducted in the
same dimensions as the benchmarking experiment. The chosen setup, coding
language, compiler and computational architecture for conducting these
experiments are described.

The :file:`exampletiming.*` code template is provided to run this experiment. For CPU-inexpensive algorithms the timing might mainly reflect the time spent in function :meth:`fgeneric`.

.. _sec:tuning:

Parameter Setting and Tuning of Algorithms
------------------------------------------

The algorithm and the used parameter setting for the algorithm should be
described thoroughly. Any tuning of parameters to the testbed should be
described and the approximate number of tested parameter settings should be
given. 

On all functions the very same parameter setting must be used (which might well depend on the dimensionality, see Section :ref:`sec:input`). That means, *a priori* use of function-dependent parameter settings is prohibited (since 2012).  In other words, the function ID or any function characteristics (like
separability, multi-modality, ...) cannot be considered as input parameter to the algorithm. Instead, we encourage benchmarking different
parameter settings as "different algorithms" on the entire testbed. In order
to combine different parameter settings, one might use either multiple runs
with different parameters (for example restarts, see also
Section :ref:`sec:stopping`), or use (other) probing techniques for identifying
function-wise the appropriate parameters online. The underlying assumption in
this experimental setup is that also in practice we do not know in advance
whether the algorithm will face :math:`f_1` or :math:`f_2`, a unimodal or a
multimodal function... therefore we cannot adjust algorithm parameters *a
priori* [#]_.

.. [#] In contrast to most other function properties, the property of having 
   noise can usually be verified easily. Therefore, for noisy functions a
   *second* testbed has been defined. The two testbeds can be approached *a
   priori* with different parameter settings or different algorithms.



.. [Auger:2005a] A Auger and N Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769–1776. IEEE Press, 2005.
.. [Auger:2005b] A. Auger and N. Hansen. Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777–1784, 2005.
.. [Auger:2009] Anne Auger and Raymond Ros. Benchmarking the pure
   random search on the BBOB-2009 testbed. In Franz Rothlauf, editor, *GECCO
   (Companion)*, pages 2479–2484. ACM, 2009.
.. [Efron:1993] B. Efron and R. Tibshirani. *An introduction to the
   bootstrap.* Chapman & Hall/CRC, 1993.
.. [Harik:1999] G.R. Harik and F.G. Lobo. A parameter-less genetic
   algorithm. In *Proceedings of the Genetic and Evolutionary Computation
   Conference (GECCO)*, volume 1, pages 258–265. ACM, 1999.
.. [Hoos:1998] H.H. Hoos and T. Stützle. Evaluating Las Vegas
   algorithms—pitfalls and remedies. In *Proceedings of the Fourteenth 
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238–245, 1998.
.. [Price:1997] K. Price. Differential evolution vs. the functions of
   the second ICEO. In Proceedings of the IEEE International Congress on
   Evolutionary Computation, pages 153–157, 1997.


