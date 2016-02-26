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

.. |citeCOCOex| replace:: [COCOex]

.. |f| replace:: :math:`f`
.. |g| replace:: :math:`g`
.. |x| replace:: :math:`x`

.. role:: red
.. |todo| replace:: **todo**

.. #################################################################################
.. #################################################################################
.. #################################################################################


Introduction
============

We consider the black-box optimization problem to minimize a function :math:`f: X\subset\mathbb{R}^n \to \mathbb{R}^m, \,n,m\ge1` such that for :math:`g: X\subset\mathbb{R}^n \to \mathbb{R}^l` for all :math:`i=1\dots l` we have :math:`g_i(x)\le0`. 
More specifically, we aim to find, as quickly as possible, one or several solutions :math:`x\in X` with small value(s) of :math:`f(x)\in\mathbb{R}^m` and :math:`g_i(x)\le0`. 
We consider *time* to be the number of calls to the function |f|. 
A continuous optimization algorithm, also known as *solver*, addresses this problem. 
Here, we assume that no prior knowledge about |f| or |g| are available to the algorithm, that is, 
they are considered as a black-box the algorithm can query with solutions 
:math:`x\in\mathbb{R}^n`.

Considering this setup, benchmarking optimization algorithms seems to be a
rather simple and straightforward task. We run an algorithm on a collection of problems and display the results. Under closer inspection however it turns out to be surprisingly tedious, and it appears to be difficult to get results that can be meaningfully interpreted beyond the standard claim that one algorithm is better 
than another on some problems and vice versa. [#]_
Here, we offer a conceptual guideline for benchmarking continuous optimization algorithms which tries to address this challenge and has been implemented in the 
COCO_ framework. [#]_ 

The COCO_ framework provides the practical means for an automatized benchmarking procedure. Benchmarking an optimization algorithm, say implemented in the function ``fmin``, on a benchmark suite in Python becomes as simple as

.. code:: python

  import cocoex as ex
  import cocopp as pp
  from myoptimizer import fmin
    
  suite = ex.Suite("bbob", "2013", "")
  observer = ex.Observer("bbob", "result_folder: myoptimizer-on-bbob")
    
  for p in suite:
      observer.observe(p)
      fmin(p, p.initial_solution)
        
  pp.main('exdata/myoptimizer-on-bbob')

Now the file ``ppdata/ppdata.html`` can be used to browse the resulting data. 

The COCO_ framework provides currently

    - an interface to several languages, currently C/C++, Java, Matlab/Octave, 
      Python, in which the benchmarked optimizer might be written
    - several benchmark suites or testbeds, currently all written in C
    - data logging facilities via the ``Observer``
    - data post-processing and data display facilities in ``html``
    - article LaTeX templates

The underlying philosophy of COCO_ is to provide everything which experimenters 
need to implement, if they want to benchmark an algorithm properly.

.. Note:: talk about restarts somewhere, it's related to budget. 

Why COCO_?
----------

Appart from diminishing the burden and the pitfalls of repetitive coding by many experimenters, 
our aim is to provide a *conceptual guideline for better benchmarking*. 
Our guideline has a few defining features.  

  #. Benchmark functions are comprehensibly designed, to allow a meaningful 
     interpretation of performance results [WHI1996].

  #. Benchmark functions are difficult to "defeat", that is, they do not 
     have artificial regularities that can be (intentionally or unintentionally) 
     exploited by an algorithm. [#]_
    
  #. Benchmark functions are scalable with the input dimension [WHI1996]. 
  
  #. There is no predefined budget (number of |f|-evaluations) for running an
     experiment, the experimental 
     procedure is budget-free [COCOex]_.

  #. A single performance  measure is used, namely runtime measured in 
     number of |f|-evaluations. Runtime has the advantage to
    
     - be easily interpretable without expert domain knowledge
     - be quantitative on the ratio scale [STE1946]_. 
     - assume a wide range of values
     - aggregate over a collection of values in a very meaningful way

.. note:: later we want to talk about the interpretation of aggregations, like that we draw a problem uniform at random (over all problems or over all instances). 


Terminology
-----------
We specify a few terms which are used later. 

*function*
  We talk about a *function* as a mapping
  :math:`\mathbb{R}^n\to\mathbb{R}^m` with scalable input space, that is,
  :math:`n` is not (yet) determined, and usually :math:`m\in\{1,2\}`.
  Functions are commonly parametrized such that different *instances* of the
  "same" function are available, e.g. translated or shifted versions. 
  
*problem*
  We talk about a *problem*, |coco_problem_t|_, as a specific *function
  instance* on which the optimization algorithm is run. Specifically, a problem
  can be described as the triple ``(dimension, function, instance)``. A problem
  can be evaluated and returns an |f|-value or -vector and, in case,
  a |g|-vector. 
  In the context of performance
  assessment, additionally one or several target :math:`f`- or :math:`\Delta f`-values
  are attached to each problem. That is, a target value is added to the 
  above triple to define a single problem. 
  
*runtime*
  We define *runtime*, or *run-length* [HOO1998]_
  as the *number of evaluations* 
  conducted on a given problem, also referred to as number of *function* evaluations. 
  Our central performance measure is the runtime until a given target :math:`f`-value 
  is hit.

*suite*
  A test- or benchmark-suite is a collection of problems, typically between
  twenty and a hundred, where the number of objectives :math:`m` is fixed. 


.. [#] It remains to be a standard procedure to present tens or even hundreds 
    of numbers in one or several tables, left to the reader to scan and compare 
    to each other [SUG2015]. 

.. [#] See https://www.github.com/numbbo/coco or https://numbbo.github.io for implementation details. 

.. [#] For example, the optimum is not in all-zeros and optima are not placed 
    on a regular grid. Which regularities are common place in real-world 
    optimization problems remains an open question. 

.. .. [#] Wikipedia__ gives a reasonable introduction to scale types.
.. .. was 261754099
.. .. __ http://en.wikipedia.org/w/index.php?title=Level_of_measurement&oldid=478392481



.. Note:: (old) Reasons for having the platform - Overall appraoch in COCO ("what other do wrong and we do better")


.. |n| replace:: :math:`n`
.. |theta| replace:: :math:`\theta`
.. |i| replace:: :math:`i`


Terminology and definition of problem, function, instance, target? 
==================================================================

In the COCO_ framework we consider *parametrized* functions
:math:`\finstance:\R^n \to \mathbb{R}^m`, which are parametrized via the
parameters dimension |n| and instance |i|. By fixing |n| and |i| we
define an optimization problem that we can present to an optimization
algorithm. Varying |n| or |i| leads to a variation of the problem, while
we still talk about the same function. 

Giving each function |f| a name or an index, the triple ``(dimension |n|, 
|f|-index, instance |i|)`` defines a problem. 

Instance concept
-----------------------

Changing significant features/parameters of the problem class (systematically or randomized)
    
Generate repetitions, natural randomization
-------------------------------------------

Generality, Fairness, avoid exploitation/cheating
-------------------------------------------------


Targets
-------
To each problem, as defined above, we attach a number of target values. 



General code structure
===============================================

experiments + postprocessing

one code base: in C, wrapped in different languages (Java, Python, Matlab/Octave) for the experiments, in python for the postprocessing


Different test suites
=====================

bbob
----

bbob-biobj
----------




.. ############################# References #########################################

.. [COCOex] The BBOBies: Experimental Setup. 

.. .. [HAN2009] Hansen, N., A. Auger, S. Finck R. and Ros (2009), Real-Parameter Black-Box Optimization Benchmarking 2009: Experimental Setup, *Inria Research Report* RR-6828 http://hal.inria.fr/inria-00362649/en

.. .. [HAN2010] Hansen, N., A. Auger, S. Finck R. and Ros (2010), Real-Parameter Black-Box Optimization Benchmarking 2010: Experimental Setup, *Inria Research Report* RR-7215 http://hal.inria.fr/inria-00362649/en

.. .. [AUG2005] A Auger and N Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769--1776. IEEE Press, 2005.
.. .. [Auger:2005b] A. Auger and N. Hansen. Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777-1784, 2005.
.. .. [Auger:2009] Anne Auger and Raymond Ros. Benchmarking the pure
   random search on the BBOB-2009 testbed. In Franz Rothlauf, editor, *GECCO
   (Companion)*, pages 2479-2484. ACM, 2009.
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


.. ############################## END Document #######################################
