##############################
COCO: Performance Assessment
##############################
.. toctree::
   :maxdepth: 2



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
.. |ERT| replace:: :math:`\mathrm{ERT}`




..
   sectnum::

Introduction
=============

In this document we explain the rationale behind the performance assessment within the COCO platform. The simple but central idea is that we advocate *quantitative* performance measures as opposed to simple rankings of algorithm performances. From there on follows that run-length for function value targets are collected. We then either display expected run-length through the `Expected Running Time`_ (ERT) measure or distribution of run-length through `Empirical Cumulative Distribution Functions`_ (ECDF).

Terminology
------------
*problem*
  In the context of performance
  assessment, we talk about a *problem* as the quadruple 
  ``(dimension, function, instance, function-target-value)``. 

*instance*
  Our test functions are parametrized such that different *instances* of the same function are available. Different instances can vary by having different shifted optima, can use different random rotations that are applied to the variables, ...
  
*runtime*
  We define *runtime*, or *run-length* [HOO1998]_
  as the *number of evaluations* 
  conducted on a given problem, also referred to as number of *function* evaluations. 
  Our central performance measure is the runtime until a given target :math:`f`-value 
  is hit.
  

On Performance Measures
=======================

We advocate **performance measures** that are:

* quantitative, ideally with a ratio scale (opposed to interval or ordinal
  scale) [#]_ and with a wide variation (i.e., for example, with values ranging
  not only between 0.98 and 1.0)
* well-interpretable, in particular by having a meaning and semantics attached
  to the numbers
* relevant with respect to the "real world"
* as simple as possible

For these reasons we measure runtime to reach a target function value, denoted as fixed-target scenario in the following. 


.. _sec:verthori:

Fixed-Cost versus Fixed-Target Scenario
----------------------------------------

Two different approaches for collecting data and making measurements from
experiments are schematically depicted in Figure :ref:`fig:HorizontalvsVertical`.

.. _fig:HorizontalvsVertical:

.. figure:: HorizontalvsVertical.*
   :align: center
   :width: 60%

   Horizontal vs Vertical View

   Illustration of fixed-cost view (vertical cuts) and fixed-target view
   (horizontal cuts). Black lines depict the best function value plotted versus
   number of function evaluations.

**Fixed-cost scenario (vertical cuts)**
  Fixing a number of function evaluations (this corresponds to fixing a cost)
  and measuring the function values reached for this given number of function
  evaluations. Fixing search costs can be pictured as drawing a vertical line
  on the convergence graphs (see Figure :ref:`fig:HorizontalvsVertical` where
  the line is depicted in red).
**Fixed-target scenario (horizontal cuts)**
  Fixing a target function value and measuring the number of function
  evaluations needed to reach this target function value. Fixing a target can
  be pictured as drawing a horizontal line in the convergence graphs
  (Figure :ref:`fig:HorizontalvsVertical` where the line is depicted in blue).

It is often argued that the fixed-cost approach is close to what is needed for
real word applications where the total number of function evaluations is
limited. On the other hand, also a minimum target requirement needs to be
achieved in real world applications, for example, getting (noticeably) better
than the currently available best solution or than a competitor.

For benchmarking algorithms we prefer the fixed-target scenario over the
fixed-cost scenario since it gives *quantitative and interpretable*
data: the fixed-target scenario (horizontal cut) *measures a time*
needed to reach a target function value and allows therefore conclusions
of the type: Algorithm A is two/ten/hundred times faster than Algorithm
B in solving this problem (i.e. reaching the given target function
value). The fixed-cost scenario (vertical cut) does not give
*quantitatively interpretable* data: there is no interpretable meaning
to the fact that Algorithm A reaches a function value that is
two/ten/hundred times smaller than the one reached by Algorithm B,
mainly because there is no *a priori* evidence *how much* more difficult
it is to reach a function value that is two/ten/hundred times smaller.
This, indeed, largely depends on the specific function and on the
specific function value reached. Furthermore, for algorithms
that are invariant under certain transformations of the function value (for
example under order-preserving transformations as algorithms based on
comparisons like DE, ES, PSO), fixed-target measures can be made
invariant under these transformations by simply choosing different
target values while fixed-cost measures require the transformation
of all resulting data.


Run-length over problems
------------------------

We define a problem as a set ``(function, dimension, instance, function target)`` where the concept of instance is described in the function documentation (it is obtained by instantiating some random transformations). We collect run-length swiping over functions, dimensions, instances and targets.

.. Niko: "function target" seems misleading, as the target depends also on the instance
  (and also on the dimension). |target value| might be a possible nomenclature, we also
  have already |ftarget| defined above. 
  
.. |target value| replace:: target-:math:`f`-value
.. an alterative could be .. |target value| replace:: target value

.. Niko: I think we should discuss instances somewhere here, in particular as (a) there is 
  no generic "function document" and (b) the interpretation is related to the display
  and/or assessment. 

.. Niko: when conducting an experiment we sweep over all available 
  ``(function, dimension, instance)`` triplets. A single trial than generates the 
  quadruples. 
  
.. Niko: let me know what/where I can/should start to do/change here. 

The display of results is hence based on those collected run-length. We either used displays  based on the expected run-length |ERT| described in Section `Expected Running Time`_  or based on the distribution of run-length using empirical cumulative distribution as described in Section `Empirical Cumulative Distribution Functions`_

Interpretation of instances
***************************

Different instances of each function are used when collecting the number of function evaluations needed to reach a target for a given function. We interpret the different instances as if they are just a repetition of the same function. In this case we consider the run length collected on the different instances (for a given function, dimension and target) as independent identically distributed random variables.

.. _sec:ERT:

Expected Running Time
======================

We use the *expected running time* (|ERT|, introduced in [Price:1997]_ as
ENES and analyzed in [Auger:2005b]_ as success performance) as most
prominent performance measure. The Expected Running Time is defined as the
average number of function evaluations while the best function value was not smaller than the target

.. _eq:ERT:

.. math::
   :nowrap:

   \begin{eqnarray}
     \mathrm{ERT}(f_\mathrm{target}) &=& \frac{\#\mathrm{FEs}(f_\mathrm{best}\ge f_\mathrm{target})}{\#\mathrm{succ}}
   \end{eqnarray}


.. |nbsucc| replace:: :math:`\#\mathrm{succ}`
.. |ps| replace:: :math:`p_{\mathrm{s}}`
.. |Ts| replace:: :math:`\mathrm{RT}_\mathrm{S}`
.. |Tus| replace:: :math:`\mathrm{RT}_\mathrm{US}`

where |nbsucc| denotes the number of successful trials (successful trials are those that reached |ftarget|) and  :math:`\#\mathrm{FEs}(f_\mathrm{best}(\mathrm{FE}) \ge f_\mathrm{target})` is
the number of function evaluations
conducted in all trials, while the best function value was not smaller than
|ftarget| during the trial, i.e. the sum over all trials of:

.. _eq:SPone1:

.. math::
   \max \{\mathrm{FE} \mbox{ s.t. } f_\mathrm{best}(\mathrm{FE}) \ge f_\mathrm{target} \}.


The |ERT| coincides with the estimate of the expected running time needed to reach |ftarget| by an (hypothetical) algorithm that would conduct restarts till a successful run (i.e. reaching the target) is observed. More precisely |ERT| writes also

.. _eq:SPone2:

.. math::
   :nowrap:

   \begin{eqnarray}
     \mathrm{ERT}(f_\mathrm{target}) &=& \mathrm{RT}_\mathrm{S} + \frac{1-p_{\mathrm{s}}}{p_{\mathrm{s}}} \,\mathrm{RT}_\mathrm{US} \\
                                     &=& \frac{p_{\mathrm{s}} \mathrm{RT}_\mathrm{S} + (1-p_{\mathrm{s}}) \mathrm{RT}_\mathrm{US}}{p_{\mathrm{s}}}
   \end{eqnarray}



where the *running times* |Ts| and |Tus| denote the average number of
function evaluations for successful and unsuccessful trials, respectively (zero
for none respective trial), and |ps| denotes the fraction of successful trials.
Successful trials are those that reached |ftarget|; evaluations after
|ftarget| was reached are disregarded.

Note that|ERT| estimates the
expected running time to reach |ftarget| [Auger:2005b]_, as a function of
|ftarget|. In particular, |Ts| and |ps| depend on the |ftarget| value. Whenever
not all trials were successful, ERT also depends (strongly) on the termination
criteria of the algorithm.

.. [#] Wikipedia__ gives a reasonable introduction to scale types.
.. was 261754099
__ http://en.wikipedia.org/w/index.php?title=Level_of_measurement&oldid=478392481


Bootstrapping and Simulated Runs
================================

The |ERT| computes a single measurement from a data sample set (in our case
from |nruns| optimization runs). Bootstrapping [Efron:1993]_ can provide a
dispersion measure for this aggregated measurement: here, a "single data
sample" is derived from the original data by repeatedly drawing single trials
with replacement until a successful trial is drawn. We call also this single data sample a **simulated run**.	

The running time of the
single sample is computed as the sum of function evaluations in the drawn
trials (for the last trial up to where the target function value is reached)
[Auger:2005b]_ [Auger:2009]_. The distribution of the
bootstrapped running times is, besides its displacement, a good approximation
of the true distribution. We provide some percentiles of the bootstrapped
distribution.

Simulated runs are heavily used for displaying the distribution.

.. _sec:ECDF:

Empirical Cumulative Distribution Functions
===========================================

We exploit the "horizontal and vertical" viewpoints introduced in the last
Section :ref:`sec:verthori`. In Figure :ref:`fig:ecdf` we plot the :abbr:`ECDF
(Empirical Cumulative Distribution Function)` [#]_ of the intersection point
values (stars in Figure :ref:`fig:HorizontalvsVertical`) for 450 trials.

.. [#] The empirical (cumulative) distribution function
   :math:`F:\mathbb{R}\to[0,1]` is defined for a given set of real-valued data
   :math:`S`, such that :math:`F(x)` equals the fraction of elements in
   :math:`S` which are smaller than :math:`x`. The function :math:`F` is
   monotonous and a lossless representation of the (unordered) set :math:`S`.

.. _fig:ecdf:

.. figure:: ecdf.*
   :width: 100%
   :align: center

   ECDF

   Illustration of empirical (cumulative) distribution functions (ECDF) of
   running length (left) and precision (right) arising respectively from the
   fixed-target and the fixed-cost scenarios in Figure
   :ref:`fig:HorizontalvsVertical`. In each graph the data of 450 trials are
   shown. Left subplot: ECDF of the running time (number of function
   evaluations), divided by search space dimension |DIM|, to fall below
   :math:`f_\mathrm{opt} + \Delta f` with :math:`\Delta f = 10^{k}`, where
   :math:`k=1,-1,-4,-8` is the first value in the legend. Data for algorithms
   submitted for BBOB 2009 and :math:`\Delta f= 10^{-8}` are represented in the
   background in light brown. Right subplot: ECDF of the best achieved
   precision :math:`\Delta f` divided by 10\ :sup:`k` (thick red and upper left
   lines in continuation of the left subplot), and best achieved precision
   divided by 10\ :sup:`-8` for running times of :math:`D`, :math:`10\,D`, 
   :math:`100\,D`, :math:`1000\,D`... function evaluations (from the rightmost
   line to the left cycling through black-cyan-magenta-black).

A cutting line in Figure :ref:`fig:HorizontalvsVertical` corresponds to a
"data" line in Figure :ref:`fig:ecdf`, where 450 (30 x 15) convergence graphs
are evaluated. For example, the thick red graph in Figure :ref:`fig:ecdf` shows
on the left the distribution of the running length (number of function
evaluations) [HOO1998]_ for reaching precision
:math:`\Delta f = 10^{-8}` (horizontal cut). The graph continues on the right
as a vertical cut for the maximum number of function evaluations, showing the
distribution of the best achieved :math:`\Delta f` values, divided by 10\
:sup:`-8`. Run length distributions are shown for different target precisions
:math:`\Delta f` on the left (by moving the horizontal cutting line up- or
downwards). Precision distributions are shown for different fixed number of
function evaluations on the right. Graphs never cross each other. The
:math:`y`-value at the transition between left and right subplot corresponds to
the success probability. In the example, just under 50% for precision 10\
:sup:`-8` (thick red) and just above 70% for precision 10\ :sup:`-1` (cyan).


Simulated run-length
--------------------

Based on the interpretation of instances as pure repetitions, we build some simulated run-length from the Nruns collected data, that is from the number of function evaluations needed to reach a given target or in case the target is not reached, the number of function evaluations of the unsuccessful run. The construction of a simulated run works as follow:

We sample a run-length uniformly at random among the Nruns run-length. If this run-length correspond to a unsuccessful trial we draw uniformly again among the Nruns run-length a new run-length. We repeat this operation until we obtain a run-length corresponding to a successful trial. The simulated run-length sums up all the run-lengths till a successful trial has been sampled.

We typically generate many more simulated run-length than the number of function instances (corresponding to Nruns). 


Using simulated run-length for plotting ECDF graphs
---------------------------------------------------

The simulated run-length are used to plot the ECDF graphs: the ECDF graphs correspond to the empirical cumulative distributions of some simulated run-length generated each time the post-processing is called. As a consequence the processus of producing an ECDF graph from the collected data is stochastic and some small variations between two independent post-processing from the same data can be observed.


Understanding the different plots
==================================





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
.. [HOO1998] H.H. Hoos and T. Stützle. Evaluating Las Vegas
   algorithms—pitfalls and remedies. In *Proceedings of the Fourteenth 
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238–245, 1998.
.. [Price:1997] K. Price. Differential evolution vs. the functions of
   the second ICEO. In Proceedings of the IEEE International Congress on
   Evolutionary Computation, pages 153–157, 1997.

