.. title:: COCO: Comparing Continuous Optimizers

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COCO: A platform for Comparing Continuous Optimizers in a Black-Box Setting
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

.. the next two lines are necessary in LaTeX. They will be automatically 
  replaced to put away the \chapter level as ??? and let the "current" level
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

   <SMALL><I>To cite or access this document as <tt>pdf</tt>:</I><BR>
   N. Hansen, A. Auger, O. Mersmann, T. Tušar, and D. Brockhoff (2016). 
   <A HREF="http://arxiv.org/pdf/1603.08785">
   COCO: A platform for Comparing Continuous Optimizers in a Black-Box Setting</A>. 
   <I>ArXiv e-prints</I>, 
   <A HREF="http://arxiv.org/abs/1603.08785">arXiv:1603.08785</A>.</SMALL>

.. raw:: latex

  % \tableofcontents is automatic with sphinx and moved behind abstract by swap...py
  \begin{abstract}


COCO_ is a platform for Comparing Continuous Optimizers in a black-box
setting. 
It aims at automatizing the tedious and repetitive task of
benchmarking numerical optimization algorithms to the greatest possible
extent. 
We present the rationals behind the development of the platform
as a general proposition for a guideline towards better benchmarking. 
We detail underlying fundamental concepts of 
COCO_ such as the definition of
a problem, the idea of instances, the relevance of target values, and runtime
as central performance measure. 
Finally, we  give a quick overview of the basic
code structure and the currently available test suites.
  
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

.. |example_experiment.py| replace:: ``example_experiment.py``
.. _example_experiment.py: https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment.py

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
.. |l| replace:: :math:`l`

.. role:: red
.. |todo| replace:: **todo**

.. #################################################################################
.. #################################################################################
.. #################################################################################


Introduction
============

We consider the continuous black-box optimization or search problem to minimize

.. math::

    f: X\subset\mathbb{R}^n \to \mathbb{R}^m \qquad n,m\ge1 

such that for the |l| constraints

.. math::

    g: X\subset\mathbb{R}^n \to \mathbb{R}^l \qquad l\ge0 

we have :math:`g_i(\x)\le0` for all :math:`i=1\dots l`.
More specifically, we aim to find, as quickly as possible, one or several solutions |x| in the search space :math:`X` with *small* value(s) of :math:`f(\x)\in\mathbb{R}^m` that satisfy all above constraints |g|. 
We generally consider *time* to be the number of calls to the function |f|. 

A continuous optimization algorithm, also known as *solver*, addresses the
above problem. 
Here, we assume that :math:`X` is known, but no prior knowledge about |f| or
|g| is available to the algorithm. 
That is, |f| and |g| are considered as a black-box which the algorithm can
query with solutions :math:`\x\in\mathbb{R}^n` to get the respective values
:math:`f(\x)` and :math:`g(\x)`.

From these prerequisits, benchmarking optimization algorithms seems to be a
rather simple and straightforward task. We run an algorithm on a collection of
problems and display the results. However, under closer inspection,
benchmarking turns out to be surprisingly tedious, and it appears to be
difficult to get results that can be meaningfully interpreted beyond the
standard claim that one algorithm is better than another on some problems. [#]_ 
Here, we offer a conceptual guideline for benchmarking
continuous optimization algorithms which tries to address this challenge and
has been implemented within the COCO_ framework. [#]_ 

The COCO_ framework provides the practical means for an automatized
benchmarking procedure. Installing COCO_ (in a shell) and benchmarking an
optimization algorithm, say, the function ``fmin`` from ``scipy.optimize`` 
in Python, becomes as simple [#]_ as

.. raw:: latex

    shown in the figure.

.. raw:: latex

    \begin{figure} %\begin{minipage}{\textwidth}


.. code:: bash

   $ ### get and install the code
   $ git clone https://github.com/numbbo/coco.git  # get coco using git
   $ cd coco
   $ python do.py run-python  # install Python experimental module cocoex
   $ python do.py install-postprocessing  # install post-processing :-)

.. code:: bash

   $ ### (optional) run an example from the shell
   $ cp code-experiments/build/python/example_experiment.py .
   $ python example_experiment.py     # run the current "default" experiment
   $ python -m bbob_pproc exdata/...  # run the post-processing
   $ open ppdata/index.html           # browse results

.. code:: python

   #!/usr/bin/env python
   """Python script to benchmark fmin of scipy.optimize"""
   from numpy.random import rand
   import cocoex 
   try: import cocopp  # new (future) name
   except ImportError: import bbob_pproc as cocopp  # old name
   from scipy.optimize import fmin
 
   suite = cocoex.Suite("bbob", "year: 2016", "")
   budget_multiply = 1e4  # use 1e1 or even 2 for a quick first test run
   observer = cocoex.Observer("bbob", "result_folder: myoptimizer-on-bbob")
    
   for p in suite:  # loop over all problems
       observer.observe(p)  # prepare logging of necessary data
       fmin(p, p.initial_solution)  # disp=False would silence fmin output
       while (not p.final_target_hit and  # apply restarts, if so desired
              p.evaluations < p.dimension * budget_multiplier):
           fmin(p, p.lower_bounds + (rand(p.dimension) + rand(p.dimension)) * 
                       (p.upper_bounds - p.lower_bounds) / 2)
     
   cocopp.main('exdata/myoptimizer-on-bbob')  # invoke data post-processing

.. raw:: latex 

    \caption[Minimal benchmarking code in Python]{
    Shell code for installation of \COCO\ (above), and Python code to benchmark 
    \texttt{fmin} on the \texttt{bbob} suite (below).
    
After the Python script has been executed, the file ``ppdata/index.html`` can be used 
to browse the resulting data.

.. raw:: latex 

    }
    \end{figure}

The COCO_ framework provides 

 - an interface to several languages in which the benchmarked optimizer
   can be written, currently C/C++, Java, Matlab/Octave, Python
 - several benchmark suites or testbeds, currently all written in C
 - data logging facilities via the ``Observer``
 - data post-processing in Python and data browsing through ``html``
 - article LaTeX templates.

The underlying philosophy of COCO_ is to provide everything that experimenters
need to setup and implement if they want to benchmark a given algorithm
implementation *properly*. 
A desired side effect of reusing the same framework is that data collected
over years or even decades can be effortlessly compared. [#]_
So far, the framework has been successfully used to benchmark far over a
hundred different algorithms by dozens of researchers.  

.. [#] One common major flaw is to get no
   indication of *how much* better an algorithm is. 
   That is, the results of benchmarking often provide no indication of 
   *relevance*;
   the main output is often hundreds of tabulated numbers interpretable on
   an ordinal (ranking) scale only. 
   Addressing a point of a common confusion, *statistical* significance is only
   a secondary and by no means sufficient condition for *relevance*. 
   
.. [#] Confer to `the code basis`__ on Github and the `C API documentation`__ for 
   implementation details. 

__ https://www.github.com/numbbo/coco
__ http://numbbo.github.io/coco-doc/C/
   
.. [#] See also |example_experiment.py|_ which runs
   out-of-the-box as a benchmarking Python script.  

.. [#] For example, see here__, here__ or here__ to access all data submitted 
   to the `BBOB 2009 GECCO workshop`__. 

__ http://coco.gforge.inria.fr/doku.php?id=bbob-2009-algorithms
__ http://coco.gforge.inria.fr/data-archive
__ http://coco.lri.fr/BBOB2009
__ http://coco.gforge.inria.fr/doku.php?id=bbob-2009

.. left to the reader to
   scan and compare to each other, possibly across different articles. 


Why COCO_?
----------

Appart from diminishing the time burden and the pitfalls, bugs
or omissions of the repetitive coding task for experimenters, our aim is to
provide a *conceptual guideline for better benchmarking*. Our setup and 
guideline has the following defining features.  

.. format hint: four spaces are needed to make the continuation
     https://gist.github.com/dupuy/1855764

#. Benchmark functions are

   #. used as black boxes for the algorithm, however they 
      are explicitly known to the scientific community. 
   #. designed to be comprehensible, to allow a meaningful 
      interpretation of performance results.
   #. difficult to "defeat", that is, they do not 
      have artificial regularities that can easily be (intentionally or unintentionally) 
      exploited by an algorithm. [#]_
   #. scalable with the input dimension [WHI1996]_.
  
#. There is no predefined budget (number of |f|-evaluations) for running an
   experiment, the experimental procedure is *budget-free* [HAN2016ex]_.

#. A single performance measure is used --- and thereafter aggregated and 
   displayed in several ways ---, namely **runtime**, *measured in 
   number of* |f|-*evaluations* [HAN2016perf]_. This runtime measure has the 
   advantages to 

   - be independent of the computational platform, language, compiler, coding 
     styles, and other specific experimental conditions [#]_
   - be independent, as a measurement, of the specific function on which it has
     been obtained
   - be relevant, meaningful and easily interpretable without expert domain knowledge
   - be quantitative on the ratio scale [#]_ [STE1946]_
   - assume a wide range of values 
   - aggregate over a collection of values in a meaningful way [#]_.
     
   A *missing* runtime value is considered as possible outcome (see below).
    
#. The display is as comprehensible, intuitive and informative as possible. 
   We believe that the details matter. 
   Aggregation over dimension is avoided, because dimension is a parameter 
   known in advance that can and should be used for algorithm design decisions. 
   This is possible without significant drawbacks, because all functions are 
   scalable in the dimension. 
   
We believe however that in the *process* of algorithm *design*, a benchmarking 
framework like COCO_ has its limitations. 
During the design phase, usually fewer benchmark functions should be used, the
functions and measuring tools should be tailored to the given algorithm and 
design question, and the overall procedure should usually be rather informal and
interactive with rapid iterations. 
A benchmarking framework then serves to conduct the formalized validation
experiment of the design *outcome* and can be used for regression testing. 


.. [#] For example, the optimum is not in all-zeros, optima are not placed 
    on a regular grid, most functions are not separable [WHI1996]_. The
    objective to remain comprehensible makes it more challenging to design
    non-regular functions. Which regularities are common place in real-world
    optimization problems remains an open question. 

.. [#] Runtimes measured in |f|-evaluations are widely
       comparable and designed to stay. The experimental procedure
       [HAN2016ex]_ includes however a timing experiment which records the
       internal computational effort of the algorithm in CPU or wall clock time. 

.. [#] As opposed to a ranking of algorithms based on their solution quality
       achieved after a given budget. 
       
.. [#] With the caveat that the *arithmetic average* is dominated by large values
       which can compromise its informative value.

.. .. [#] Wikipedia__ gives a reasonable introduction to scale types.
.. .. was 261754099
.. .. __ http://en.wikipedia.org/w/index.php?title=Level_of_measurement&oldid=478392481


Terminology
-----------

We specify a few terms which are used later. 

*function*
  We talk about an objective *function* as a parametrized mapping
  :math:`\mathbb{R}^n\to\mathbb{R}^m` with scalable input space, :math:`n\ge2`,
  and usually :math:`m\in\{1,2\}`.
  Functions are parametrized such that different *instances* of the
  "same" function are available, e.g. translated or shifted versions. 
  
*problem*
  We talk about a *problem*, |coco_problem_t|_, as a specific *function
  instance* on which an optimization algorithm is run. 
  A problem
  can be evaluated and returns an |f|-value or -vector and, in case,
  a |g|-vector. 
  In the context of performance assessment, a target :math:`f`- or
  indicator-value is added to define a problem. A problem is considered as
  solved when the given or the most difficult available target is obtained. 
  
*runtime*
  We define *runtime*, or *run-length* [HOO1998]_ as the *number of
  evaluations* conducted on a given problem until a prescribed target value is
  hit, also referred to as number of *function* evaluations or |f|-evaluations.
  Runtime is our central performance measure.

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


Functions, Instances, and Problems
=====================================

In the COCO_ framework we consider **functions**, |fi|, for each suite
distinguished by their identifier :math:`i=1,2,\dots` .  
Functions are further *parametrized* by the (input) dimension, |n|, and the
instance number, |j|. 
We can think of |j| as an index to a continuous parameter vector setting, as it
parametrizes, among others things, translations and rotations. In practice, |j|
is the discrete identifier for single instantiations of these parameters. 
For a given |m|, we then have

.. math::

    \finstance_i \equiv f(n, i, j):\R^n \to \mathbb{R}^m \quad
    \x \mapsto \finstance_i (\x) = f(n, i, j)(\x)\enspace. 
    
Varying |n| or |j| leads to a variation of the same function
|i| of a given suite. 
Fixing |n| and |j| of function |fi| defines an optimization **problem**
:math:`(n, i, j)\equiv(f_i, n, j)` that can be presented to the optimization algorithm. Each problem receives again
an index in the suite, mapping the triple :math:`(n, i, j)` to a single
number. 

As the formalization above suggests, the differentiation between function (index) 
and instance index is of purely semantic nature. 
This semantics however is important in how we display and
interpret the results. We interpret **varying the instance** parameter as 
a natural randomization for experiments [#]_ in order to 

 - generate repetitions on a function and
 - average away irrelevant aspects of the function definition, thereby providing
 
    - generality which alleviates the problem of overfitting, and
    - a fair setup which prevents intentional or unintentional exploitation of 
      irrelevant or artificial function properties. 

For example, we consider the absolute location of the optimum not a defining
function feature. Consequently, in a typical COCO_ benchmark suite, instances
with randomized search space translations are presented to the optimizer. [#]_


.. [#] Changing or sweeping through a relevant feature of the problem class,
       systematically or randomized, is another possible usage of instance
       parametrization. 

.. [#] Conducting either several trials on instances with randomized search space
   translations or with a randomized initial solution is equivalent, given
   that the optimizer behaves translation invariant (disregarding domain
   boundaries). 


Runtime and Target Values
=========================

In order to measure the runtime of an algorithm on a problem, we
establish a hitting time condition. 
We prescribe a **target value**, |t|, which is an |f|-value or more generally a
quality indicator-value [HAN2016perf]_ [BRO2016]_. 
For a single run, when an algorithm reaches or surpasses the target value |t|
on problem |p|, we say it has *solved the problem* |pt| --- it was successful. [#]_

Now, the **runtime** is the evaluation count when the target value |t| was
reached or surpassed for the first time. 
That is, runtime is the number of |f|-evaluations needed to solve the problem
|pt|. [#]_
*Measured runtimes are the only way how we assess the performance of an 
algorithm.* 
Observed success rates are generally translated into runtimes on a subset of
problems. 


.. Runtime can be formally written as |RT(pt)|. 

.. _Recommendations: https://www.github.com

.. old For each target value, |t|, the quadruple :math:`(f_i, n, j, t)` gives 
       raise to a runtime, |RT(pt)|, 
   When the problem :math:`(f_i, n, j)` has been solved up to the target quality |t|. 
   An algorithm solves a problem |pt| if it hits the target |t|. 
   In the context of performance evaluation, we refer to such a quadruple itself also as a *problem*. 

If an algorithm does not hit the target in a single run, this runtime remains
undefined --- while it has been bounded from below by the number of evaluations
in this unsuccessful run. 
The number of available runtime values depends on the budget the 
algorithm has explored. 
Therefore, larger budgets are preferable --- however they should not come at
the expense of abandoning reasonable termination conditions. Instead,
restarts should be done [HAN2016ex]_. 

.. [#] Reflecting the *anytime* aspect of the experimental setup, 
    we use the term *problem* in two meanings: as the problem the
    algorithm is benchmarked on, |p|, and as the problem, |pt|, an algorithm may
    solve by hitting the target |t| with the runtime, |RT(pt)|, or may fail to solve. 
    Each problem |p| gives raise to a collection of dependent problems |pt|. 
    Viewed as random variables, the events |RT(pt)| given |p| are not
    independent events for different values of |t|. 
  
.. [#] Target values are directly linked to a problem, leaving the burden to 
    define the targets with the designer of the benchmark suite. 
    The alternative, namely to present the obtained |f|- or indicator-values as results,
    leaves the (rather unsurmountable) burden to interpret the meaning of these 
    indicator values to the experimenter or the final audience. 
    Fortunately, there is an automatized generic way to generate target values
    from observed runtimes, the so-called run-length based target values
    [HAN2016perf]_. 
    

.. |k| replace:: :math:`k`
.. |p| replace:: :math:`(f_i, n, j)`
.. |pt| replace:: :math:`(f_i, n, j, t)`
.. |RT(pt)| replace:: :math:`\mathrm{RT}(f_i, n, j, t)`


.. _sec:Restarts:

Restarts and Simulated Restarts
-------------------------------

An optimization algorithm is bound to terminate and, in the single-objective case, return a recommended 
solution, |x|, for the problem, |p|. [#]_
The algorithm solves thereby all problems |pt| for which :math:`f(\x)\le t`. 
Independent restarts from different, randomized initial solutions are a simple
but powerful tool to increase the number of solved problems [HAR1999]_ --- namely by increasing the number of |t|-values, for which the problem |p|
was solved. [#]_ 
Independent restarts tend to increase the success rate, but they generally do
not *change* the performance *assessment*, because the successes materialize at
greater runtimes [HAN2016perf]_. 
Therefore, we call our approach *budget-free*. 
Restarts however "*improve the reliability, comparability, precision, and "visibility" of the measured results*" [HAN2016ex]_.

*Simulated restarts* [HAN2010]_ [HAN2016perf]_ are used to determine a runtime for unsuccessful runs. 
Semantically, *this is only valid if we can interpret different 
instances as random repetitions*. 
Resembling the bootstrapping method [EFR1994]_, when we face an unsolved problem, 
we draw uniformly at random a new |j| until we find an instance such that |pt| 
was solved. [#]_
The evaluations done on the first unsolved problem and on all subsequently
drawn unsolved problems are added to the runtime on the last problem and
are considered as runtime on the originally unsolved problem.  
This method is applied if a problem instance was not solved and is
(only) available if at least one problem instance was solved.
It allows to directly compare algorithms with different success probabilities. 

.. The minimum runtime determined by a simulated restart is the 
   minimum runtime from those solved instances which are accompanied by at least
   one unsolved instance (that is, for the same |pt| except of |j|).

.. [#] More specifically, we use the anytime scenario where we consider 
   at each evaluation the evolving quality indicator value. 

.. [#] The quality indicator is always defined such that for a given problem |p| the 
  number of acquired runtime values |RT(pt)| (hitting a target indicator value |t|)
  is monotonously increasing with the used budget. Considered as random
  variables, these runtimes are not independent. 

.. [#] More specifically, we consider the problems :math:`(f_i, n, j, t(j))` for
  all benchmarked instances |j|. The targets :math:`t(j)` depend on the instance 
  in a way to make the problems comparable. 


Aggregation
------------

A typical benchmark suite consists of about 20--100 functions with 5--15 instances for each function. For each instance, up to about 100 targets are considered for the 
performance assessment. This means we consider at least :math:`20\times5=100`, and 
up to :math:`100\times15\times100=150\,000` runtimes for the performance assessment. 
To make them amenable to the experimenter, we need to summarize these data. 

Our idea behind an aggregation is to make a statistical summary over a set or
subset of *problems of interest over which we assume a uniform distribution*. 
From a practical perspective this means to have no simple way to distinguish
between these problems and to select an optimization algorithm accordingly --- in
which case an aggregation for a single algorithm would not be helpful --- 
and that we face each problem with similar probability. 
We do not aggregate over dimension, because dimension can and 
should be used for algorithm selection. 

We have several ways to aggregate the resulting runtimes. 

 - Empirical (cumulative) distribution functions (|ECDFs|). In the domain of 
   optimization, |ECDFs| are also known as *data profiles* [MOR2009]_. We
   prefer the simple |ECDF| over the more innovative performance profiles
   [MOR2002]_ for two reasons.
   |ECDFs| (i) do not depend on other (presented) algorithms, that is, they are
   unconditionally comparable across different publications, and (ii) let us
   distinguish, for the considered algorithm, in a natural way easy problems from
   difficult problems. [#]_ 
   We usually display |ECDFs| on the log scale, which makes the area
   above the curve and the *difference area* between two curves a meaningful
   conception. 
   
   .. object/concept/element/notion/aspect/component. 
 
 - Averaging, as an estimator of the expected runtime. The average runtime 
   is often plotted against dimension to indicate scaling with dimension. 
   The *arithmetic* average is only meaningful if the underlying distribution of
   the values is similar. 
   Otherwise, the average of log-runtimes, or *geometric* average, 
   is recommended. 
   
 - Restarts and simulated restarts, see Section :ref:`sec:Restarts`, do not 
   aggregate runtimes in the literal meaning (they are literally defined only when |t| was
   hit).  They aggregate, however, time data to eventually supplement, if applicable, 
   all missing runtime values. 

.. [#] When reading a performance profile, a question immediately crossing ones 
   mind is often whether a large runtime difference is observed mainly because
   one algorithm solves the problem very quickly. 
   This question cannot be answered from the profile.
   The advantage (i) over data profiles disappears when using run-length based
   target values [HAN2016perf]_.

.. |ERT| replace:: ERT
.. |ECDF| replace:: ECDF
.. |ECDFs| replace:: ECDF

General Code Structure
===============================================

The code basis of the COCO_ code consists of two parts. 

The *experiments* part
  defines test suites, allows to conduct experiments, and provides the output
  data. The `code base is written in C`__, and wrapped in different languages
  (currently Java, Python, Matlab/Octave). An amalgamation technique is used
  that outputs two files ``coco.h`` and ``coco.c`` which suffice to run
  experiments within the COCO_ framework. 

  .. __: http://numbbo.github.io/coco-doc/C


The *post-processing* part
  processes the data and displays the resulting runtimes. This part is
  entirely written in Python and heavily depends on |matplotlib|_ [HUN2007]_.  

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
  contains 55 bi-objective (:math:`m=2`) functions in 15 subgroups [TUS2016]_. 
  
.. _`old code basis`: http://coco.gforge.inria.fr/doku.php?id=downloads


.. raw:: html
    
    <H2>Acknowledgments</H2>

.. raw:: latex

    \section*{Acknowledgments}

The authors would like to thank Raymond Ros, Steffen Finck, Marc Schoenauer,  
Petr Posik and Dejan Tušar for their many invaluable contributions to this work. 

The authors also acknowledge support by the grant ANR-12-MONU-0009 (NumBBO) 
of the French National Research Agency.


.. ############################# References ###################################
.. raw:: html
    
    <H2>References</H2>

.. this document: 
.. .. [HAN2016co] N. Hansen, A. Auger, O. Mersmann, T. Tušar, D. Brockhoff (2016).
   `COCO: A Platform for Comparing Continuous Optimizers in a Black-Box 
   Setting`__. *ArXiv e-prints*, `arXiv:1603:08785`__.
.. .. __ http://numbbo.github.io/coco-doc/
.. .. __ http://arxiv.org/abs/1603.08785

.. [BRO2016] D. Brockhoff, T. Tušar, D. Tušar, T. Wagner, N. Hansen, A. Auger, (2016). 
  `Biobjective Performance Assessment with the COCO Platform`__. *ArXiv e-prints*, `arXiv:1605.01746`__.
__ http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment
__ http://arxiv.org/abs/1605.01746

.. [HAN2016perf] N. Hansen, A. Auger, D. Brockhoff, D. Tušar, T. Tušar (2016). 
  `COCO: Performance Assessment`__. *ArXiv e-prints*, `arXiv:1605.03560`__.
__ http://numbbo.github.io/coco-doc/perf-assessment
__ http://arxiv.org/abs/1605.03560

.. .. [HAN2009] N. Hansen, A. Auger, S. Finck, and R. Ros (2009). Real-Parameter Black-Box Optimization Benchmarking 2009: Experimental Setup, *Inria Research Report* RR-6828__. __ http://hal.inria.fr/inria-00362649/en

.. .. [HAN2010ex] N. Hansen, A. Auger, S. Finck, and R. Ros (2010). 
.. Real-Parameter Black-Box Optimization Benchmarking 2010: Experimental Setup, `Research Report RR-7215`__, Inria.
.. .. __ http://hal.inria.fr/inria-00362649/en

.. [HAN2010] N. Hansen, A. Auger, R. Ros, S. Finck, and P. Posik (2010). 
  Comparing Results of 31 Algorithms from the Black-Box Optimization Benchmarking BBOB-2009. Workshop Proceedings of the GECCO Genetic and Evolutionary Computation Conference 2010, ACM, pp. 1689-1696.

.. [HAN2009fun] N. Hansen, S. Finck, R. Ros, and A. Auger (2009). 
  `Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions`__. `Research Report RR-6829`__, Inria, updated February 2010.
.. __: http://coco.gforge.inria.fr/
.. __: https://hal.inria.fr/inria-00362633

.. [HAN2009noi] N. Hansen, S. Finck, R. Ros, and A. Auger (2009). 
  `Real-Parameter Black-Box Optimization Benchmarking 2009: Noisy Functions Definitions`__. `Research Report RR-6869`__, Inria, updated February 2010.
.. __: http://coco.gforge.inria.fr/
.. __: https://hal.inria.fr/inria-00369466

.. [HAN2016ex] N. Hansen, T. Tušar, A. Auger, D. Brockhoff, O. Mersmann (2016). 
   `COCO: The Experimental Procedure`__, *ArXiv e-prints*, `arXiv:1603.08776`__.
__ http://numbbo.github.io/coco-doc/experimental-setup/
__ http://arxiv.org/abs/1603.08776

.. [HUN2007] J. D. Hunter (2007). `Matplotlib`__: A 2D graphics environment, 
  *Computing In Science \& Engineering*, 9(3): 90-95. 
.. __: http://matplotlib.org/

.. .. [AUG2005] A. Auger and N. Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769--1776. IEEE Press, 2005.
.. .. [Auger:2005b] A. Auger and N. Hansen. Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777-1784, 2005.
.. .. [Auger:2009] A. Auger and R. Ros. Benchmarking the pure
   random search on the BBOB-2009 testbed. In Franz Rothlauf, editor, *GECCO
   (Companion)*, pages 2479-2484. ACM, 2009.
   
.. .. [BAR1995] R. S. Barr, B. L. Golden, J. P. Kelly, M. G. C. Resende, and W. R. Stewart Jr. Designing and Reporting on Computational Experiments with Heuristic Methods. Journal of Heuristics, 1:9–32, 1995. 

.. [EFR1994] B. Efron and R. Tibshirani (1994). *An introduction to the
   bootstrap*. CRC Press.
.. [HAR1999] G. R. Harik and F. G. Lobo (1999). A parameter-less genetic
   algorithm. In *Proceedings of the Genetic and Evolutionary Computation
   Conference (GECCO)*, volume 1, pages 258-265. ACM.
.. [HOO1998] H. H. Hoos and T. Stützle (1998). Evaluating Las Vegas
   algorithms: pitfalls and remedies. In *Proceedings of the Fourteenth 
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238-245.
   
.. [MOR2009] J. Moré and S. Wild (2009). 
  Benchmarking Derivative-Free Optimization Algorithms. *SIAM J. Optimization*, 20(1):172-191.
   
.. [MOR2002] D. Dolan and J. J. Moré (2002). 
  Benchmarking Optimization Software with Performance Profiles. *Mathematical Programming*, 91:201-213.
   
.. .. [PRI1997] K. Price (1997). Differential evolution vs. the functions of
   the second ICEO. In *Proceedings of the IEEE International Congress on
   Evolutionary Computation*, pages 153--157.
   
.. [STE1946] S.S. Stevens (1946). 
  On the theory of scales of measurement. *Science* 103(2684), pp. 677-680.

.. [TUS2016] T. Tušar, D. Brockhoff, N. Hansen, A. Auger (2016). 
  `COCO: The Bi-objective Black Box Optimization Benchmarking (bbob-biobj) 
  Test Suite`__, *ArXiv e-prints*, `arXiv:1604.00359`__.
.. __: http://numbbo.github.io/coco-doc/bbob-biobj/functions/
.. __: http://arxiv.org/abs/1604.00359

.. [WHI1996] D. Whitley, S. Rana, J. Dzubera, K. E. Mathias (1996). 
  Evaluating evolutionary algorithms. *Artificial intelligence*, 85(1), 245-276.


.. ############################## END Document #######################################

