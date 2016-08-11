.. title:: COCO: The Experimental Procedure

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COCO: The Experimental Procedure
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

.. the next two lines are necessary in LaTeX. They will be automatically 
  replaced to put away the \chapter level as ^^^ and let the "current" level
  become \section. 

.. CHAPTERTITLE
.. CHAPTERUNDERLINE

.. |
.. |
.. .. sectnum::
  :depth: 3
.. .. contents:: Table of Contents
.. |
.. |

.. raw:: html

   See also: <I>ArXiv e-prints</I>,
   <A HREF="http://arxiv.org/abs/1603.08776">arXiv:1603.08776</A>, 2016.


.. raw:: latex

  % \tableofcontents is automatic with sphinx and moved behind abstract by swap...py
  \begin{abstract}


We present a budget-free experimental setup and procedure for benchmarking numerical
optimization algorithms in a black-box scenario. 
This procedure can be applied with the COCO_ benchmarking platform. 
We describe initialization of and input to the algorithm and touch upon the
relevance of termination and restarts. 

.. We finally reconsider parameter tuning and the concept of recommendations for benchmarking with COCO_.


.. raw:: latex

  \end{abstract}
  \newpage
  

.. _2009: http://www.sigevo.org/gecco-2009/workshops.html#bbob
.. _2010: http://www.sigevo.org/gecco-2010/workshops.html#bbob
.. _2012: http://www.sigevo.org/gecco-2012/workshops.html#bbob
.. _BBOB-2009: http://coco.gforge.inria.fr/doku.php?id=bbob-2009-results
.. _BBOB-2010: http://coco.gforge.inria.fr/doku.php?id=bbob-2010-results
.. _BBOB-2012: http://coco.gforge.inria.fr/doku.php?id=bbob-2012
.. _GECCO-2012: http://www.sigevo.org/gecco-2012/
.. _COCO: https://github.com/numbbo/coco
.. _COCOold: http://coco.gforge.inria.fr

.. |f| replace:: :math:`f`

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
  
.. |coco_problem_get_evaluations(const coco_problem_t * problem)| replace::
  ``coco_problem_get_evaluations(const coco_problem_t * problem)``
.. _coco_problem_get_evaluations(const coco_problem_t * problem): 
  http://numbbo.github.io/coco-doc/C/coco_8h.html#a6ad88cdba2ffd15847346d594974067f


.. #################################################################################
.. #################################################################################
.. #################################################################################


Introduction
============

Based on [HAN2009]_ and [HAN2010]_, we describe a comparatively simple experimental 
setup for *black-box optimization benchmarking*. We recommend to use this procedure
within the COCO_ platform [HAN2016co]_. [#]_ 

Our **central measure of performance**, to which the experimental procedure is
adapted, is the number of calls to the objective function to reach a
certain solution quality (function value or :math:`f`-value or indicator
value), also denoted as runtime. 

Terminology
-----------
*function*
  We talk about an objective *function* |f| as a parametrized mapping
  :math:`\mathbb{R}^n\to\mathbb{R}^m` with scalable input space, that is,
  :math:`n` is not (yet) determined, and usually :math:`m\in\{1,2\}`.
  Functions are parametrized such that different *instances* of the
  "same" function are available, e.g. translated or shifted versions. 
  
*problem*
  We talk about a *problem*, |coco_problem_t|_, as a specific *function
  instance* on which the optimization algorithm is run. Specifically, a problem
  can be described as the triple ``(dimension, function, instance)``. A problem
  can be evaluated and returns an :math:`f`-value or -vector. 
  In the context of performance
  assessment, a target :math:`f`- or indicator-value
  is attached to each problem. That is, a target value is added to the 
  above triple to define a single problem in this case. 
  
*runtime*
  We define *runtime*, or *run-length* [HOO1998]_
  as the *number of evaluations* 
  conducted on a given problem, also referred to as number of *function* evaluations. 
  Our central performance measure is the runtime until a given target value 
  is hit [HAN2016perf]_.

*suite*
  A test- or benchmark-suite is a collection of problems, typically between
  twenty and a hundred, where the number of objectives :math:`m` is fixed. 

.. compare also the `COCO read me`_. .. _`COCO read me`: https://github.com/numbbo/coco/blob/master/README.md 

.. [#] The COCO_ platform provides
       several (single and bi-objective) *test suites* with a collection of
       black-box optimization problems of different dimensions to be
       minimized. COCO_ automatically collects the relevant data to display
       the performance results after a post-processing is applied. 


Conducting the Experiment
=========================

The optimization algorithm to be benchmarked is run on each problem of the
given test suite once. On each problem, the very same algorithm with the same
parameter setting, the same initialzation procedure, the same budget, the same
termination and/or restart criteria etc. is used. 
There is no prescribed minimal or maximal allowed budget, the benchmarking 
setup is *budget-free*. 
The longer the experiment, the more data are available to assess the performance
accurately.
See also Section :ref:`sec:budget`. 

.. _sec:input:

Initialization and Input to the Algorithm
-----------------------------------------

An algorithm can use the following input information from each problem. 
At any time: 

*Input and output dimensions*
  as a defining interface to the problem, specifically:

    - The search space (input) dimension via |coco_problem_get_dimension|_, 
    - The number of objectives via |coco_problem_get_number_of_objectives|_, 
      which is the "output" dimension of |coco_evaluate_function|_. 
      All functions of a single benchmark suite have the same number 
      of objectives, currently either one or two. 
    - The number of constraints via |coco_problem_get_number_of_constraints|_, 
      which is the "output" dimension of |coco_evaluate_constraint|_. *All* 
      problems of a single benchmark suite have either no constraints, or 
      one or more constraints. 

*Search domain of interest*
  defined from |coco_problem_get_largest_values_of_interest|_ and |coco_problem_get_smallest_values_of_interest|_. The optimum (or each extremal solution of the Pareto set) lies within the search domain of interest. If the optimizer operates on a bounded domain only, the domain of interest can be interpreted as lower and upper bounds.

*Feasible (initial) solution* 
  provided by |coco_problem_get_initial_solution|_. 

The initial state of the optimization algorithm and its parameters shall only be based on
these input values. The initial algorithm setting is considered as part of
the algorithm and must therefore follow the same procedure for all problems of the
suite. The problem identifier or the positioning of the problem in the suite or
any (other) known characteristics of the problem are not
allowed as input to the algorithm, see also Section
:ref:`sec:tuning`.



During an optimization run, the following (new) information is available to
the algorithm: 

#. The result, i.e., the :math:`f`-value(s) from evaluating the problem 
   at a given search point 
   via |coco_evaluate_function|_. 

#. The result from evaluating the constraints of the problem at a 
   given search point via |coco_evaluate_constraint|_. 
 
#. The result of |coco_problem_final_target_hit|_, which can be used
   to terminate a run conclusively without changing the performance assessment
   in any way. Currently, if the number of objectives :math:`m > 1`, this
   function returns always zero. 

The number of evaluations of the problem and/or constraints are the search
costs, also referred to as *runtime*, and used for the performance 
assessment of the algorithm. [#]_

.. .. [#] Note, however, that the Pareto set in the bi-objective case is not always guaranteed to lie in its entirety within the region of interest.

.. [#] |coco_problem_get_evaluations(const coco_problem_t * problem)|_ is a
  convenience function that returns the number of evaluations done on ``problem``. 
  Because this information is available to the optimization algorithm anyway, 
  the convenience function might be used additionally. 
  


.. _sec:budget:

Budget, Termination Criteria, and Restarts
------------------------------------------

Algorithms and/or setups with any budget of function evaluations are 
eligible, the benchmarking setup is *budget-free*. 
We consider termination criteria to be part of the benchmarked algorithm. 
The choice of termination is a relevant part of the algorithm. 
On the one hand, allowing a larger number of function evaluations increases the chance to find solutions with better quality. On the other hand, a timely
termination of stagnating runs can improve the performance, as these evaluations
can be used more effectively.

To exploit a large(r) number of function evaluations effectively, we encourage to
use **independent restarts** [#]_, in particular for algorithms which terminate
naturally within a comparatively small budget. 
Independent restarts are a natural way to approach difficult optimization
problems and do not change the central performance measure used in COCO_ (hence it is budget-free), 
however, 
independent restarts improve the reliability, comparability [#]_, precision, and "visibility" of the measured results. 

Moreover, any **multistart procedure** (which relies on an interim termination of the algorithm) is encouraged. 
Multistarts may not be independent as they can feature a parameter sweep (e.g., increasing population size [HAR1999]_ [AUG2005]_), can be based on the outcome of the previous starts, and/or feature a systematic change of the initial conditions for the algorithm. 

After a multistart procedure has been established, a recommended procedure is
to use a budget proportional to the dimension, :math:`k\times n`, and run 
repeated experiments with increase :math:`k`, e.g. like 
:math:`3, 10, 30, 100, 300,\dots`, which is a good compromise between
availability of the latest results and computational overhead. 

An algorithm can be conclusively terminated if
|coco_problem_final_target_hit|_ returns 1. [#]_ This saves CPU cycles without
affecting the performance assessment, because there is no target left to hit. 

.. [#] The COCO_ platform provides example code implementing independent restarts. 

.. [#] Algorithms are only comparable up to the smallest budget given to 
  any of them. 

.. [#] For the ``bbob-biobj`` suite this is however currently never the case. 

.. |j| replace:: :math:`j`

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
   evaluations to reach the final :math:`f_\mathrm{target} = f_\mathrm{opt} + 10^{-8}` 
   is expected.


.. _sec:tuning:

Parameter Setting and Tuning of Algorithms
==========================================

.. The algorithm and the used parameter setting for the algorithm should be 
   described thoroughly. 

Any tuning of algorithm parameters to the test suite should be described and
*the approximate overall number of tested parameter settings or algorithm
variants and the approximate overall invested budget should be given*. 

The only recommended tuning procedure is the verification that **termination
conditions** of the algorithm are suited to the given testbed and, in case,
tuning of termination parameters. [#]_
Too early or too late termination can be identified and adjusted comparatively 
easy. 
This is also a useful prerequisite for allowing restarts to become more effective. 

On all functions the very same parameter setting must be used (which might
well depend on the dimensionality, see Section :ref:`sec:input`). That means,
the *a priori* use of function-dependent parameter settings is prohibited
(since 2012).  The function ID or any function characteristics (like
separability, multi-modality, ...) cannot be considered as input parameter to
the algorithm. 

On the other hand, benchmarking different parameter settings as "different
algorithms" on the entire test suite is encouraged. 

.. [#] For example in the single objective case, care should be 
   taken to apply termination conditions that allow to hit the final target on
   the most basic functions, like the sphere function :math:`f_1`, that is on the
   problems 0, 360, 720, 1080, 1440, and 1800 of the ``bbob`` suite.  

   In our experience, numerical optimization software frequently terminates 
   too early by default, while evolutionary computation software often 
   terminates too late by default. 

.. In order to combine
   different parameter settings within a single algorithm, one can use 
   multiple runs with
   different parameters (for example restarts, see also Section
   :ref:`sec:budget`), or probing techniques to identify
   problem-wise the appropriate parameters online. The underlying assumption in
   this experimental setup is that also in practice we do not know in advance
   whether the algorithm will face :math:`f_1` or :math:`f_2`, a unimodal or a
   multimodal function... therefore we cannot adjust algorithm parameters *a
   priori* [#]_.

.. In contrast to most other function properties, the property of having 
   noise can usually be verified easily. Therefore, for noisy functions a
   *second* testbed has been defined. The two testbeds can be approached *a
   priori* with different parameter settings or different algorithms.

.. .. # [Auger:2005a] A Auger and N Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769–1776. IEEE Press, 2005.


.. .. _sec:recommendations:

    Recommendations
    ===============

    The performance assessment is based on a set of evaluation counts
    associated with the :math:`f`-value or -vector of a solution. 
    By default, each evaluation count is associated with the respectively *evaluated*
    solution and hence its :math:`f`-value. 
    In the single-objective case, the solution associated *to the current (last)
    evaluation* can be changed by calling |coco_recommend_solution|_, thereby
    associating the :math:`f`-value of the *recommended* solution (instead of the
    *evaluated* solution) with the current evaluation count. 
    A recommendation is best viewed as the *currently best known approximation* of the
    optimum [#]_ delivered by the optimization algorithm, or as the currently most 
    desirable return value of the algorithm. 

    Recommendations allow the algorithm to explore solutions without affecting the
    performance assessment. For example, a surrogate-based algorithm can explore
    (i.e. evaluate) an arbitrarily bad solution, update the surrogate model and
    then recommend the (new) model optimizer. On non-noisy suites it is neither
    necessary nor advantageous to recommend the same solution repeatedly.

    .. On non-noisy suites the last evaluation changes the assessment only if the :math:`f`-value is better than all :math:`f`-values from previous evaluations. 

    .. [#] In the noisy scenario, a small number of the most current solutions 
      will be taken into account [HAN2016perf]_. 
      In the multi-objective scenario, the recommendation option is not available,
      because an archive of non-dominated solutions presumes that all solutions are
      evaluated. 

Time Complexity Experiment
==========================

In order to get a rough measurement of the time complexity of the algorithm, the
wall-clock or CPU time should be measured when running the algorithm on the
benchmark suite. The chosen setup should reflect a "realistic average 
scenario". [#]_ 
The *time divided by the number of function evaluations shall be presented
separately for each dimension*. The chosen setup, coding language, compiler and
computational architecture for conducting these experiments should be given.

.. The :file:`exampletiming.*` code template is provided to run this experiment. For CPU-inexpensive algorithms the timing might mainly reflect the time spent in function :math:`fgeneric`.

.. [#] 
  The example experiment code provides the timing output measured over all
  problems of a single dimension by default. It also can be used to make a record
  of the same timing experiment with "pure random search", which can serve as 
  additional base-line data. On the ``bbob`` test suite, also only the
  first instance of the Rosenbrock function :math:`f_8` had been used for this
  experiment previously, that is, the suite indices 105, 465, 825, 1185, 1545,
  1905. 
  

.. raw:: html
    
    <H2>Acknowledgments</H2>

.. raw:: latex

    \section*{Acknowledgments}

The authors would like to thank Raymond Ros, Steffen Finck, Marc Schoenauer,  
Petr Posik and Dejan Tušar for their many invaluable contributions to this work. 

This work was support by the grant ANR-12-MONU-0009 (NumBBO) 
of the French National Research Agency.


.. ############################# References ###################################
.. raw:: html
    
    <H2>References</H2>

.. [AUG2005] A. Auger and N. Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769--1776. IEEE Press, 2005.
.. .. [Auger:2005b] A. Auger and N. Hansen. Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777-1784, 2005.
.. .. [Auger:2009] A. Auger and R. Ros. Benchmarking the pure
   random search on the BBOB-2009 testbed. In F. Rothlauf, editor, *GECCO
   (Companion)*, pages 2479-2484. ACM, 2009.
.. .. [Efron:1993] B. Efron and R. Tibshirani. *An introduction to the
   bootstrap.* Chapman & Hall/CRC, 1993.

.. [HAN2016perf] N. Hansen, A. Auger, D. Brockhoff, D. Tušar, T. Tušar. 
  `COCO: Performance Assessment`__. *ArXiv e-prints*, `arXiv:1605.03560`__, 2016.
__ http://numbbo.github.io/coco-doc/perf-assessment
__ http://arxiv.org/abs/1605.03560

.. [HAN2009] N. Hansen, A. Auger, S. Finck, and R. Ros. 
   Real-Parameter Black-Box Optimization Benchmarking 2009: Experimental Setup, *Inria Research Report* RR-6828 http://hal.inria.fr/inria-00362649/en, 2009.

.. [HAN2010] N. Hansen, A. Auger, S. Finck, and R. Ros. 
   Real-Parameter Black-Box Optimization Benchmarking 2010: Experimental Setup, *Inria Research Report* RR-7215 http://hal.inria.fr/inria-00362649/en, 2010.

.. [HAN2016co] N. Hansen, A. Auger, O. Mersmann, T. Tušar, D. Brockhoff.
   `COCO: A Platform for Comparing Continuous Optimizers in a Black-Box 
   Setting`__, *ArXiv e-prints*, `arXiv:1603.08785`__, 2016. 
.. __: http://numbbo.github.io/coco-doc/
.. __: http://arxiv.org/abs/1603.08785
 
.. [HAR1999] G.R. Harik and F.G. Lobo. A parameter-less genetic
   algorithm. In *Proceedings of the Genetic and Evolutionary Computation
   Conference (GECCO)*, volume 1, pages 258-265. ACM, 1999.
.. [HOO1998] H.H. Hoos and T. Stützle. Evaluating Las Vegas
   algorithms: pitfalls and remedies. In *Proceedings of the Fourteenth 
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238-245, 1998.
.. .. [PRI1997] K. Price. Differential evolution vs. the functions of
   the second ICEO. In Proceedings of the IEEE International Congress on
   Evolutionary Computation, pages 153--157, 1997.

.. ############################## END Document #######################################
