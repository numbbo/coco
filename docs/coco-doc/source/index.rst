.. title:: COCO: Comparing Continuous Optimizers

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COCO: A platform for Comparing Continuous Optimizers in a Black-Box Setting
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

...
%%%

.. |
.. |
.. .. sectnum::
  :depth: 3
.. .. contents:: Table of Contents
.. |
.. |

.. Here we put the abstract when using LaTeX, the \abstractinrst command is defined in 
     the 'preamble' of latex_elements in source/conf.py, the text
     is defined in `abstract` of conf.py. To flip abstract and 
     table of contents, or update the table of contents, toggle 
     the \generatetoc command in the 'preamble' accordingly. 
.. raw:: latex

    \abstractinrst
    \newpage 

.. COCO is a platform for Comparing Continuous Optimizers in a black-box
  setting. It aims at automatizing the tedious and repetitive task of
  benchmarking numerical optimization algorithms to the greatest possible
  extent. We present the rationals behind the development of the platform
  and its basic structure. We furthermore detail underlying fundamental 
  concepts of COCO such as its definition of a problem, the idea of
  instances, or performance measures and give an overview of the
  available test suites.
  
  
.. _2009: http://www.sigevo.org/gecco-2009/workshops.html#bbob
.. _2010: http://www.sigevo.org/gecco-2010/workshops.html#bbob
.. _2012: http://www.sigevo.org/gecco-2012/workshops.html#bbob
.. _BBOB-2009: http://coco.gforge.inria.fr/doku.php?id=bbob-2009-results
.. _BBOB-2010: http://coco.gforge.inria.fr/doku.php?id=bbob-2010-results
.. _BBOB-2012: http://coco.gforge.inria.fr/doku.php?id=bbob-2012
.. _GECCO-2012: http://www.sigevo.org/gecco-2012/
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
  
.. |coco_problem_get_evaluations(const coco_problem_t * problem)| replace::
  ``coco_problem_get_evaluations(const coco_problem_t * problem)``
.. _coco_problem_get_evaluations(const coco_problem_t * problem): 
  http://numbbo.github.io/coco-doc/C/coco_8h.html#a6ad88cdba2ffd15847346d594974067f


.. |f| replace:: :math:`f`
.. |g| replace:: :math:`g`
.. |x| replace:: :math:`\x`


.. #################################################################################
.. #################################################################################
.. #################################################################################


Introduction
============

.. note:: search problem rather than optimization problem? 

We consider the continuous black-box optimization problem

.. math::

    f: X\subset\mathbb{R}^n \to \mathbb{R}^m \qquad n,m\ge1 

    g: X\subset\mathbb{R}^n \to \mathbb{R}^l \qquad l\ge0 \enspace.
    
.. old such that with :math:`g: X\subset\mathbb{R}^n \to \mathbb{R}^l` we have :math:`g_i(\x)\le0` for all :math:`i=1\dots l`. 

More specifically, we aim to find, as quickly as possible, one or several solutions |x| in the search space :math:`X` with *small* value(s) of :math:`f(\x)\in\mathbb{R}^m` satisfying :math:`g_i(\x)\le0` for all :math:`i=1\dots l`. 
We consider *time* to be the number of calls to the function |f|. 

A continuous optimization algorithm, also known as *solver*, addresses the above problem. 
Here, we assume that :math:`X` is known but no prior knowledge about |f| or |g| are available to the algorithm, that is, 
they are considered as a black-box which the algorithm can query with solutions 
:math:`\x\in\mathbb{R}^n`.

From these prerequisits, benchmarking optimization algorithms seems to be a
rather simple and straightforward task. We run an algorithm on a collection of
problems and display the results. However, under closer inspection,
benchmarking turns out to be surprisingly tedious, and it appears to be
difficult to get results that can be meaningfully interpreted beyond the
standard claim that one algorithm is better than another on some problems, and
vice versa. [#]_ Here, we offer a conceptual guideline for benchmarking
continuous optimization algorithms which tries to address this challenge and
has been implemented within the COCO_ framework. [#]_ 

The COCO_ framework provides the practical means for an automatized
benchmarking procedure. Benchmarking an optimization algorithm, say,
implemented in the function ``fmin`` in Python, on a benchmark suite becomes 
as simple as

.. raw:: latex

    in Figure 1. \begin{figure} %\begin{minipage}{\textwidth}
    
.. code:: python

  try: import cocoex  
  except: import bbob_pproc as cocoex
  import cocopp
  from myoptimizer import fmin
    
  suite = cocoex.Suite("bbob", "year: 2016", "")
  observer = cocoex.Observer("bbob", "result_folder: myoptimizer-on-bbob")
    
  for p in suite:
      observer.observe(p)
      fmin(p, p.initial_solution)
        
  cocopp.main('exdata/myoptimizer-on-bbob')

.. raw:: latex 

    \caption[Minimal benchmarking code in Python]{
    Python code to benchmark \texttt{fmin} on the \texttt{bbob} suite and
    display the results.

Now the file ``ppdata/ppdata.html`` can be used to browse the resulting data. 

.. raw:: latex 

    }
    \end{figure}

The COCO_ framework provides currently

- an interface to several languages in which the benchmarked optimizer
  can be written, currently C/C++, Java, Matlab/Octave, Python
- several benchmark suites or testbeds, currently all written in C
- data logging facilities via the ``Observer``
- data post-processing in Python and data display facilities in ``html``
- article LaTeX templates

The underlying philosophy of COCO_ is to provide everything which most experimenters 
needed to setup and implement if they wanted to benchmark an algorithm properly.

.. [#] It remains to be a standard procedure to present tens or even hundreds 
    of numbers in one or several tables, left to the reader to scan and compare 
    to each other [SUG2015]. 
    
    .. todo:: add reference

.. [#] See https://www.github.com/numbbo/coco or https://numbbo.github.io for implementation details. 


Why COCO_?
----------

Appart from diminishing the burden (time) and the pitfalls (and bugs
and omissions) of the repetitive coding task by many experimenters, our aim is to
provide a *conceptual guideline for better benchmarking*. Our guideline has 
the following defining features.  

.. format hint: four spaces are needed to make the continuation
     https://gist.github.com/dupuy/1855764

#. Benchmark functions are 

    #. used as black boxes for the algorithm, however they 
       are explicitly known to the scientific community. 
    #. designed to be comprehensible, to allow a meaningful 
       interpretation of performance results.
    #. difficult to "defeat", that is, they do not 
       have artificial regularities that can be (intentionally or unintentionally) 
       exploited by an algorithm. [#]_
    #. scalable with the input dimension [WHI1996]_.

#. There is no predefined budget (number of |f|-evaluations) for running an
   experiment, the experimental procedure is budget-free [BBO2016ex]_.

#. A single performance  measure is used, namely runtime measured in 
   number of |f|-evaluations [BBO2016perf]_. Runtime has the advantage to 
    
     - be easily interpretable without expert domain knowledge
     - be quantitative on the ratio scale [STE1946]_ [#]_
     - assume a wide range of values
     - aggregate over a collection of values in a very meaningful way
     
   A missing runtime value is considered as possible outcome (see below).

    
#. The display is as comprehensible, intuitive and informative as as possible. 
   Aggregation over dimension is avoided, because dimension is a known parameter 
   that can and should use for algorithm selection. 

.. [#] For example, the optimum is not in all-zeros, optima are not placed 
    on a regular grid, the function is not separable [WHI1996]_. Which 
    regularities are common place in real-world optimization problems remains 
    an open question. 

.. [#] As opposed to ranking algorithm based on their solution quality achieved
  after a given runtime.  

.. .. [#] Wikipedia__ gives a reasonable introduction to scale types.
.. .. was 261754099
.. .. __ http://en.wikipedia.org/w/index.php?title=Level_of_measurement&oldid=478392481


Terminology
-----------
.. todo:: this is a duplicate, should become shorter or go away

We specify a few terms which are used later. 

*function*
  We talk about a *function* as a parametrized mapping
  :math:`\mathbb{R}^n\to\mathbb{R}^m` with scalable input space, and usually :math:`m\in\{1,2\}`.
  Functions are parametrized such that different *instances* of the
  "same" function are available, e.g. translated or shifted versions. 
  
*problem*
  We talk about a *problem*, |coco_problem_t|_, as a specific *function
  instance* on which the optimization algorithm is run. 
  A problem
  can be evaluated and returns an |f|-value or -vector and, in case,
  a |g|-vector. 
  In the context of performance
  assessment, a target :math:`f`- or indicator-value is attached 
  to each problem. That is, a target value is added to the 
  above triple to define a single problem in this case. 
  
*runtime*
  We define *runtime*, or *run-length* [HOO1998]_
  as the *number of evaluations* 
  conducted on a given problem, also referred to as number of *function* evaluations. 
  Our central performance measure is the runtime until a given target value 
  is hit.

*suite*
  A test- or benchmark-suite is a collection of problems, typically between
  twenty and a hundred, where the number of objectives :math:`m` is fixed. 


.. |n| replace:: :math:`n`
.. |m| replace:: :math:`m`
.. |theta| replace:: :math:`\theta`
.. |i| replace:: :math:`i`
.. |j| replace:: :math:`j`
.. |t| replace:: :math:`t`
.. |fi| replace:: :math:`f_i`


Functions, Instances, Problems, and Targets 
============================================

.. Note:: The following would probably best fit into a generic document about 
   functions and test suites. 

In the COCO_ framework we consider functions, |fi|, which are for each suite
distinguished by their identifier :math:`i=1,2,\dots`. Functions are
*parametrized* by dimension, |n|, and instance number, |j|, [#]_
that is for a given |m| we have

.. math::

    \finstance_i \equiv f(n, i, j):\R^n \to \mathbb{R}^m \quad
    \x \mapsto \finstance_i (\x) = f(n, i, j)(\x)\enspace. 
    
Varying |n| or |j| leads to a variation of the problem over the same function
|i| of a given suite. 
By fixing |n| and |j| for function |fi|, we define an optimization problem
that can be presented to an optimization algorithm. That is, 
for each test suite,
the triple :math:`(n, i, j)\equiv(f_i, n, j)` uniquely defines a problem. 
Each problem receives again
an index in the suite, mapping the triple :math:`(n, i, j)` to a single
number. 

.. [#] We can think of |j| as a continuous parameter vector, as it 
  parametrizes, among others things, translations and rotations. In practice, 
  |j| is a discrete identifier for single instantiations of these parameters. 


The Instance Concept
-----------------------

As the formalization above suggests, the differentiation between function (index) 
and instance index is of purely semantic nature. 
This semantics however has important implications in how we display and
interpret the results. We interpret varying the instance parameter as 
a natural randomization [#]_ in order to 

  - generate repetitions on the functions and
  - average away irrelevant aspects of the function thereby providing

      - generality which alleviates the problem of overfitting, and
      - a fair setup which prevents intentional or unintentional exploitation of 
        irrelevant or artificial function properties. 

For example, we consider the absolute location of the optimum not a defining
function feature. Consequently, conducting several trials either with a
randomized initial solution or on instances with randomized search space
translations is equivalent, given that the optimizer behaves translation
invariant and we disregard domain bounds. 

.. [#] Changing or sweeping through a relevant feature of the problem class,
       systematically or randomized, is another possible usage of instance
       parametrization. 

Runtime and Target Values
=========================

In order to measure the runtime of an algorithm on a problem, we need to
establish a termination condition. 
For this end, we set a **target value**, |t|, which is an |f|- or
indicator-value [BBO2016biobj]_. 
For a single run, when an algorithm hits the target |t| on problem |p|, we
say it has *solved the problem* |pt|. [#]_

Now, the **runtime** is the evaluation count when the target value |t| was
hit for the first time, that is, runtime is the number of evaluations needed to 
solve the problem |pt| (but see also Recommendations_ in [BBO2016ex]_). 
Runtime can be formally written as |RT(pt)|. 

.. _Recommendations: https://www.github.com

.. old For each target value, |t|, the quadruple :math:`(f_i, n, j, t)` gives 
       raise to a runtime, |RT(pt)|, 
   When the problem :math:`(f_i, n, j)` has been solved up to the target quality |t|. 
   An algorithm solves a problem |pt| if it hits the target |t|. 
   In the context of performance evaluation, we refer to such a quadruple itself also as a *problem*. 

If the algorithm never hits the target, the runtime remains undefined --- while
it has been bound to lie in the interval  :math:`[k+1,\infty]`, where |k| is the number of
evaluations of the unsuccessful run. 
The number of determined runtime values depends on the budget the 
algorithm has been explored. Therefore, larger budgets are preferable. 

.. [#] Note the use of the term *problem* in two meanings: as the problem the
  algorithm is benchmarked on, |p|, and as the problem, |pt|, an algorithm can
  solve with a certain runtime, |RT(pt)|, or may fail to solve. Each problem
  |p| gives raise to a collection of dependent problems |pt|. Viewed as random
  variables, the events |RT(pt)| are not independent events for different values
  of |t|, but given |p|. 

.. |k| replace:: :math:`k`
.. |p| replace:: :math:`(f_i, n, j)`
.. |pt| replace:: :math:`(f_i, n, j, t)`
.. |RT(pt)| replace:: :math:`\mathrm{RT}(f_i, n, j, t)`


Restarts and Simulated Restarts
-------------------------------

An optimization algorithm is bound to terminate and return a recommended 
solution to the problem |p|. 
Independent restarts from different, randomized initial solutions are a simple
but powerful tool to increase the number of solved problems |pt| for an increasing number of |t|. [#]_

The experimenter is free to 

The runtime is monotonuous decreasing in |t|. 



Related to budget, budget-free...

 - missing value is interpreted as being above the explored budget. 
    - bla

A simulated restart adds at least the minimum runtime from a successful trial. 


.. todo::


.. [#] For a given problem |p|, the runtime values, |RT(pt)|
  are monotonuous increasing in |t|. Considered as randomvariables, these 
  runtimes are not independent. 


Aggregation
------------

.. note::

  - Missing values can be integrated over within instances [BBO2016perf]_. 

  - interpretation of aggregations, like that we draw a problem uniform at random (over all problems or over all instances), but see also [BBO2016perf]_. 


.. todo::


.. todo::



General Code Structure
===============================================

The code bases consists of two parts. 

The *Experiments* part
  defines test suites and allows to conduct experiments providing the output data. The `code base is written in C`__, and wrapped in different languages (currently Java, Python, Matlab/Octave). An amalgamation technique is used that outputs two files ``coco.h`` and ``coco.c`` which suffice to use the experiments part of the framework. 

  .. __: http://numbbo.github.io/coco-doc/C


The *post-processing* part
  processes the data and display the results. This part is entirely written in 
  Python and relies heavily on |matplotlib|_ [HUN2007]_.  

.. |matplotlib| replace:: ``matplotlib``
.. _matplotlib: http://matplotlib.org/



Test Suites
=====================
Currently, the COCO_ framework provides three different test suites. 

``bbob`` 
  contains 24 functions in five subgroups [HAN2009fun]_.

``bbob-noisy``
  contains 30 noisy problems in three subgroups [HAN2009noi]_, 
  currently only implemented in the `old code basis`_.

``bbob-biobj``
  contains 55 bi-objective (:math:`m=2`) functions in 15 subgroups [BBO2016fun]_. 
  
.. _`old code basis`: http://coco.gforge.inria.fr/doku.php?id=downloads


.. ############################# References #########################################
.. raw:: html
    
    <H2>References</H2>
    
.. author list yet to be defined

.. [BBO2016biobj] The BBOBies: Biobjective function benchmark suite. 

.. [BBO2016ex] The BBOBies: `Experimental Setup`__. 
__ https://www.github.com

.. [BBO2016perf] The BBOBies: `Performance Assessment`__. 
__ https://www.github.com

.. [BBO2016fun] The BBOBies: Biobjective Function Definitions. 

.. .. [HAN2009] Hansen, N., A. Auger, S. Finck R. and Ros (2009), Real-Parameter Black-Box Optimization Benchmarking 2009: Experimental Setup, *Inria Research Report* RR-6828 http://hal.inria.fr/inria-00362649/en

.. .. [HAN2010] Hansen, N., A. Auger, S. Finck R. and Ros (2010), Real-Parameter Black-Box Optimization Benchmarking 2010: Experimental Setup, *Inria Research Report* RR-7215 http://hal.inria.fr/inria-00362649/en

.. [HAN2009fun] N.Hansen, S. Finck, R. Ros, and A. Auger. `Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions`__. `Technical Report RR-6829`__, Inria, 2009, updated February 2010.
.. __: http://coco.gforge.inria.fr/
.. __: https://hal.inria.fr/inria-00362633

.. [HAN2009noi] N.Hansen, S. Finck, R. Ros, and A. Auger. `Real-Parameter Black-Box Optimization Benchmarking 2009: Noisy Functions Definitions`__. `Technical Report RR-6869`__, Inria, 2009, updated February 2010.
.. __: http://coco.gforge.inria.fr/
.. __: https://hal.inria.fr/inria-00369466

.. [HUN2007] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment, 
  *Computing In Science \& Engineering*, 9(3): 90-95. 


.. .. [AUG2005] A Auger and N Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769--1776. IEEE Press, 2005.
.. .. [Auger:2005b] A. Auger and N. Hansen. Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777-1784, 2005.
.. .. [Auger:2009] Anne Auger and Raymond Ros. Benchmarking the pure
   random search on the BBOB-2009 testbed. In Franz Rothlauf, editor, *GECCO
   (Companion)*, pages 2479-2484. ACM, 2009.
   
.. .. [BAR1995] R. Barr, ?. Golden, J. Kelly, M Resende, and Jr. W. Stewart. Designing and Reporting on Computational Experiments with Heuristic Methods. Journal of Heuristics, 1:9–32, 1995. 

.. .. [Efron:1993] B. Efron and R. Tibshirani. *An introduction to the
   bootstrap.* Chapman & Hall/CRC, 1993.
.. .. [HAR1999] G.R. Harik and F.G. Lobo. A parameter-less genetic
   algorithm. In *Proceedings of the Genetic and Evolutionary Computation
   Conference (GECCO)*, volume 1, pages 258-265. ACM, 1999.
.. [HOO1998] H.H. Hoos and T. Stützle. Evaluating Las Vegas
   algorithms: pitfalls and remedies. In *Proceedings of the Fourteenth 
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238-245, 1998.
.. .. [PRI1997] K. Price. Differential evolution vs. the functions of
   the second ICEO. In Proceedings of the IEEE International Congress on
   Evolutionary Computation, pages 153--157, 1997.
   
.. [STE1946] Stevens, S.S. On the theory of scales of measurement. *Science* 103(2684), pp. 677-680, 1946.

.. [WHI1996] Whitley, D., Rana, S., Dzubera, J., Mathias, K. E. Evaluating evolutionary algorithms. *Artificial intelligence*, 85(1), 245-276, 1996.


.. ############################## END Document #######################################
