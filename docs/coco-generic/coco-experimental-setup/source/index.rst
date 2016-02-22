.. title:: COCO: Experimental Procedure
.. image:: _images/title-image.png
.. sectnum::
  :depth: 3

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
.. _COCO: https://github.com/numbbo/coco
.. _COCOold: http://coco.gforge.inria.fr

.. |coco_problem_get_dimension| replace:: ``coco_problem_get_dimension``
.. _coco_problem_get_dimension: http://numbbo.github.io/coco-doc/C/coco_8h.html#a0dabf3e4f5630d08077530a1341f13ab

.. |coco_problem_get_largest_values_of_interest| replace:: 
  ``coco_problem_get_largest_values_of_interest``
.. _coco_problem_get_largest_values_of_interest: http://numbbo.github.io/coco-doc/C/coco_8h.html#a29c89e039494ae8b4f8e520cba1eb154

.. |coco_problem_get_smallest_values_of_interest| replace::
  ``coco_problem_get_smallest_values_of_interest``
.. _coco_problem_get_smallest_values_of_interest: http://numbbo.github.io/coco-doc/C/coco_8h.html#a4ea6c067adfa866b0179329fe9b7c458

.. |coco_problem_get_initial_solution| replace:: 
  ``coco_problem_get_initial_solution``
.. _coco_problem_get_initial_solution: http://numbbo.github.io/coco-doc/C/coco_8h.html#ac5a44845acfadd7c5cccb9900a566b32

.. |coco_problem_final_target_hit| replace:: 
  ``coco_problem_final_target_hit``
.. _coco_problem_final_target_hit: 
  http://numbbo.github.io/coco-doc/C/coco_8h.html#a1164d85fd641ca48046b943344ae9069

.. |coco_problem_get_number_of_objectives| replace:: 
  ``coco_problem_get_number_of_objectives``
.. _coco_problem_get_number_of_objectives: http://numbbo.github.io/coco-doc/C/coco_8h.html#ab0d1fcc7f592c283f1e67cde2afeb60a

.. |coco_problem_get_number_of_constraints| replace:: 
  ``coco_problem_get_number_of_constraints``
.. _coco_problem_get_number_of_constraints: http://numbbo.github.io/coco-doc/C/coco_8h.html#ad5c7b0889170a105671a14c8383fbb22

.. |coco_evaluate_function| replace:: 
  ``coco_evaluate_function``
.. _coco_evaluate_function: http://numbbo.github.io/coco-doc/C/coco_8h.html#aabbc02b57084ab069c37e1c27426b95c

.. |coco_evaluate_constraint| replace:: 
  ``coco_evaluate_constraint``
.. _coco_evaluate_constraint: 
  http://numbbo.github.io/coco-doc/C/coco_8h.html#ab5cce904e394349ec1be1bcdc35967fa

.. |coco_problem_t| replace:: 
  ``coco_problem_t``
.. _coco_problem_t: 
  http://numbbo.github.io/coco-doc/C/coco_8h.html#a408ba01b98c78bf5be3df36562d99478

.. |coco_recommend_solution| replace:: 
  ``coco_recommend_solution``
.. _coco_recommend_solution: 
  http://numbbo.github.io/coco-doc/C/coco_8h.html#afd76a19eddd49fb78c22563390437df2

.. #################################################################################
.. #################################################################################
.. #################################################################################

|
|

.. contents:: Table of Contents

|
|

Introduction
============
In this document we describe the experimental set-up for *black-box 
optimization benchmarking* with the COCO_ platform. 

.. TODO:: cite document from 2009/2010

The platform provides
several (single and bi-objective) *test suites* with black-box problems of different 
dimensions to be minimized. The platform also automatically collects the relevant 
data to display the performance results after a post-processing is applied. The number of problem evaluations to reach a certain solution quality (:math:`f`-value) is the central measure of performance. 

Terminology
-----------
function
  We talk about a **function** as a mapping
  :math:`\mathbb{R}^n\to\mathbb{R}^m` with scalable input space, that is,
  :math:`n` is not (yet) determined, and usually :math:`m\in\{1,2\}`.
  Functions are commonly parametrized such that different *instances* of the
  "same" function are available, e.g. translated or shifted versions. 
  
problem
  We talk about a **problem**, |coco_problem_t|_, as a specific *function
  instance* on which the optimization algorithm is run. Specifically, a problem
  can be described as the triple ``(dimension, function, instance)``. 
  In the context of performance
  assessment, also one or several target :math:`f`- or :math:`\Delta f`-values
  are attached to each problem. That is, a target value is added to the 
  above triple to define a single problem. 
  
runtime
  We define *runtime*, or *run-length* as the *number of evaluations* of a
  given problem, also referred to as number of *function evaluations*. 
  
suite
  A test- or benchmark-suite is a collection of problems, typically between
  twenty and a hundred, where the number of objectives :math:`m` is fixed. 

.. compare also the `COCO read me`_. .. _`COCO read me`: https://github.com/numbbo/coco/blob/master/README.md 


Conducting the Experiment
==========================

The optimization algorithm to be benchmarked is run on each problem 
of the given test suite once. There is no prescribed maximally allowed 
runtime. The longer the experiments, the more data are available to assess 
the performance accurately, however see also Section :ref:`sec:stopping`. 

.. _sec:input:

Initialization and Input to the Algorithm
------------------------------------------

An algorithm can use the following input information from each problem. For initialization: 

1. **Input and output dimensions** of the problem, specifically:

    - The search space (input) dimension via |coco_problem_get_dimension|_, 
    - The number of objectives via |coco_problem_get_number_of_objectives|_, 
      which is the "output" dimension of |coco_evaluate_function|_. 
      All functions of a single benchmark suite have the same number 
      of objectives, currently either one or two. 
    - The number of constraints via |coco_problem_get_number_of_constraints|_, 
      which is the "output" dimension of |coco_evaluate_constraint|_. *All* 
      problems of a single benchmark suite have either no constraints, or 
      one or more constraints. 

2. The **search domain of interest** via |coco_problem_get_largest_values_of_interest|_ and |coco_problem_get_smallest_values_of_interest|_. The optimum (or the pareto set) lies within the search domain of interested. If the optimizer operates on a bounded domain only, the domain of interest can be interpreted as lower and upper bounds.

3. A **feasible (initial) solution** via |coco_problem_get_initial_solution|_. 

The initial state of the optimization algorithm and its parameters shall only be based on
these input values. The initial algorithm setting is considered as part of
the algorithm and must therefore follow the same procedure for all problems of a
suite. The problem identifier or the positioning of the problem in the suite or
any (other) known characteristics of the problem are, for obvious reasons, not
allowed as input to the algorithm, see also Section
:ref:`sec:tuning`.

During an optimization run, the following (new) information is available to
the algorithm: 

#. The result, i.e. the :math:`f`-value(s), from evaluating the problem 
   at a given search point 
   via |coco_evaluate_function|_. 

#. The result from evaluating the constraints of the problem at a 
   given search point via |coco_evaluate_constraint|_. 
 
#. The result of |coco_problem_final_target_hit|_, which can be used
   to terminate a run conclusively without changing the performance assessment
   in any way. Currently, if the number of objectives :math:`m > 1`, this
   function returns always zero. 

The number of evaluations of the problem and/or constraints are the search
costs, also referred to as runtime, and used for the performance assessment of
the algorithm. 

.. _sec:recommendations:

Recommendations
----------------

The performance assessment is by default based on the :math:`f`-values of the
evaluated solutions. Each evaluation is associated with the evaluated
solution, which is considered as the currently best known approximation of the
optimum by the algorithm. The associated solution and hence the
:math:`f`-value associated *to the current (last) evaluation* can be changed
by calling |coco_recommend_solution|_. On non-noisy suites it is neither
necessary nor advantageous to recommend the same solution repeatedly.
Recommendations allow the algorithm to explore (in particular bad) solutions
without affecting the performance assessment by the choice of the solution 
of which it acquires the :math:`f`-value. 

.. On non-noisy suites the last evaluation changes the assessment only if the :math:`f`-value is better than all :math:`f`-values from previous evaluations. 



.. _sec:stopping:

Termination Criteria and Restarts
----------------------------------

Algorithms with any budget of function evaluations are considered in the
assessment of the results. Exploiting a large number of function evaluations
increases the chance to achieve better function values and improves
comparability. [#]_ In any case, a run can be conclusively terminated if
|coco_problem_final_target_hit| returns 1. Otherwise, the choice of
termination is a relevant part of the algorithm, because a timely
termination (and restart) of stagnating runs is likely to improve the performance. To
exploit a large number of function evaluations effectively, a multistart
procedure which relies on an interim termination of the algorithm is
advisable. 

Multistarts can be independent or feature a parameter sweep. Independent
restarts do not change the central performance measure. Independent restarts
mainly improve the reliability and "visibility" of the measured value. 

.. For example, using a fast algorithm
   with a small success probability, say 5% (or 1%), chances are that not a
   single of 15 trials is successful. With 10 (or 90) independent restarts,
   the success probability will increase to 40% and the performance will
   become visible. At least four to five (here out of 15) successful trials are
   desirable to accomplish a stable performance measurement. This reasoning
   remains valid for any target function value (different values are
   considered in the evaluation).

.. Restarts either from a previous solution, or with a different parameter
   setup, for example with different (increasing) population sizes, might be
   considered, as it has been applied quite successful [Auger:2005a]_ [Harik:1999]_.

.. Choosing different setups mimics what might be done in practice. All restart
   mechanisms are finally considered as part of the algorithm under consideration.

.. The easiest functions of BBOB can be solved
   in less than :math:`10 D` function evaluations, while on the most difficult
   functions a budget of more than :math:`1000 D^2` function
   evaluations to reach the final :math:`f_\mathrm{target} = f_\mathrm{opt} + 10^{-8}` is expected.

.. [#] Algorithms are only comparable up to the smallest budget given to 
  any of them. 


Time Complexity Experiment
--------------------------

In order to get a rough measurement of the time complexity of the algorithm,
the overall CPU time should be measured when running the algorithm on a single
function for at least a few tens of seconds (and at least a few iterations) in
all available dimensions. [#]_  The chosen setup should reflect a "realistic
average scenario". If another termination criterion is reached, the algorithm
is restarted (like for a new trial). The *CPU-time per function evaluation* is
reported for each dimension. The chosen setup, coding language, compiler and
computational architecture for conducting these experiments are described.

.. The :file:`exampletiming.*` code template is provided to run this experiment. For CPU-inexpensive algorithms the timing might mainly reflect the time spent in function :meth:`fgeneric`.

.. [#] On the ``bbob`` test suite the first instance of the 
  Rosenbrock function :math:`f_8` is used, that is, 
  the suite indices 105, 465, 825, 1185, 1545, 1905. 

.. _sec:tuning:

Parameter Setting and Tuning of Algorithms
------------------------------------------

.. The algorithm and the used parameter setting for the algorithm should be 
   described thoroughly. 

Any tuning of parameters to the test suite should be described and the
approximate number of tested parameter settings should be given. 

On all functions the very same parameter setting must be used (which might well depend on the dimensionality, see Section :ref:`sec:input`). That means, the *a priori* use of function-dependent parameter settings is prohibited (since 2012).  The function ID or any function characteristics (like
separability, multi-modality, ...) cannot be considered as input parameter to the algorithm. 

On the other hand, benchmarking different parameter settings as "different
algorithms" on an entire test suite is encouraged. 

.. In order to combine
   different parameter settings within a single algorithm, one can use multiple runs with
   different parameters (for example restarts, see also Section
   :ref:`sec:stopping`), or probing techniques to identify
   problem-wise the appropriate parameters online. The underlying assumption in
   this experimental setup is that also in practice we do not know in advance
   whether the algorithm will face :math:`f_1` or :math:`f_2`, a unimodal or a
   multimodal function... therefore we cannot adjust algorithm parameters *a
   priori* [#]_.

.. In contrast to most other function properties, the property of having 
   noise can usually be verified easily. Therefore, for noisy functions a
   *second* testbed has been defined. The two testbeds can be approached *a
   priori* with different parameter settings or different algorithms.



The ``bbob`` Suite
===================

This section describes the specific setup and rationales for the BBOB benchmark suite.
The descriptions of the functions can be found at http://coco.gforge.inria.fr/doku.php?id=downloads

Symbols, Constants, and Parameters
----------------------------------

For the BBOB test, the following constants are set:

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


