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
.. |ART| replace:: :math:`\mathrm{ART}`
.. |dim| replace:: :math:`\mathrm{dim}`
.. |function| replace:: :math:`\mathrm{function}`
.. |instance| replace:: :math:`\mathrm{instance}`
.. |R| replace:: :math:`\mathbb{R}`
.. |ftheta| replace::  :math:`f_{\theta}`

..
   sectnum::

Introduction
=============

In this document we explain the rationale behind the performance assessment within the COCO platform. The simple but central idea is that we advocate *quantitative* performance measures as opposed to simple rankings of algorithm performances. From there on follows that our performance measures and displays are based on the runtime or run-length to reach a target function value.

We then either display average run-length through the `Average Running Time`_ (ART) measure or distribution of run-length through `Empirical Cumulative Distribution Functions`_ (ECDF).

When displaying the distribution of run-length, we consider the aggregation of run-length over subclasses of problems.

.. budget-free

Terminology and Definitions
----------------------------
*problem*
 A COCO problem is defined as a triple  ``(dimension,function,instance)``. In this terminology a ``function`` is actually a parametrized function and the ``instance`` is an instantiation of the parameters. More precisely let us consider a parametrized function  :math:`f_\theta: \mathbb{R}^n \to \mathbb{R}^m` for :math:`\theta \in \Theta` then a COCO problem corresponds to :math:`p=(n,f_\theta,\bar{\theta})` where :math:`n \in \mathbb{N}` is a dimension, and :math:`\bar{\theta}` is a set of parameters to instantiate the parametrized function. An algorithm optimizing the COCO problem :math:`p` will optimize :math:`\mathbf{x} \in \mathbb{R}^n \to f_{\bar{\theta}}(\mathbf{x})`. To simplify notation, in the sequel a COCO problem is denoted :math:`p=(n,f_\theta,\theta)`.
 
 In the performance assessment setting, we associate to a problem :math:`p`, a :math:`{\rm target}`, which is a function value :math:`f_{\rm target}` at which we extract the running time of the algorithm. Given that the optimal function value, that is :math:`f_{\rm opt} =  \min_{\mathbf{x}} f_{\theta}(\mathbf{x})` depends on the specific instance :math:`\theta`, the :math:`{\rm target}` function values also depends on the instance :math:`\theta`. However commonly 
 
 .. math::
 	:nowrap:

	\begin{equation} 
	\epsilon=f_{\rm target} - f_{\rm opt}
 	\end{equation}
 	
 	
 that can be thought as **precision** (or relative targets), does not depend on the instance :math:`\theta` such that we can unambiguously consider for different instances :math:`({\theta}_1, \ldots,{\theta}_K)` of a parametrized problem :math:`f_{\theta}(\mathbf{x})`, the set of targets :math:`f^{\rm target}_{{\theta}_1}, \ldots,f^{\rm target}_{{\theta}_K}` associated to a similar precision. 

*instance*
 Our test functions are parametrized such that different *instances* of the same function are available. Different instances can vary by having different shifted optima, can use different random rotations that are applied to the variables, ...  The notion of instance is introduced to generate repetition while avoiding possible exploitation of an artificial function property (like location of the optimum in zero). 
  
 We **interpret the different runs performed on different instances** of the same parametrized function in a given dimension as if they are just **independent repetitions** of the optimization algorithm on the same function. Put differently the runs performed on :math:`f_{\theta_1}, \ldots,f_{\theta_K}` with :math:`\theta_1,\ldots,\theta_K`, :math:`K` different instances of a parametrized problem :math:`f_\theta`, are assumed to be independent identically distributed.
 
 .. todo:: maybe we should insist more on this dual view of randomizing the problem class via problem isntance - choosing uniformly over set of parameters.
  
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
* independent of the programming language and computer where the algorithm was run.

This latter point excludes to use CPU time as a basis for performance measure. The shortcomings and consequences of using CPU was discussed in [Hooker:1995]_.

For these reasons we measure **runtime** to reach a target function value, that is the number of function evaluations needed to reach a target function value denoted as fixed-target scenario in the following. 


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
data: 

* the fixed-target scenario (horizontal cut) *measures a time* needed to reach a target  function value and allows therefore conclusions of the type: Algorithm A is two/ten/hundred times faster than Algorithm B in solving this problem (i.e. reaching the given target function value). 

* The fixed-cost scenario (vertical cut) does not give *quantitatively interpretable*  data: there is no interpretable meaning to the fact that Algorithm A reaches a function  value that is two/ten/hundred times smaller than the one reached by Algorithm B, mainly because there is no *a priori* evidence *how much* more difficult it is to reach a function value that is two/ten/hundred times smaller. This, indeed, largely depends on the specific function and on the specific function value reached. 
 
Furthermore, for algorithms
that are invariant under certain transformations of the function value (for
example under order-preserving transformations as algorithms based on
comparisons like DE, ES, PSO), fixed-target measures can be made
invariant under these transformations by simply choosing different
target values while fixed-cost measures require the transformation
of all resulting data.


Runtime over Problems
=========================


In order to display quantitative measurements, we have seen in the previous section that we should start from the collection of runtime for different target values. Those target values can be a :math:`f`- or indicator value (see [BBO2016biobj]_). 
In the performance assessment setting, a problem is the quadruple :math:`p=(n,f_\theta,\theta,f^{\rm target}_\theta)` where :math:`f^{\rm target}_\theta` is the target function value. This means that **we collect runtime of problems**.

Formally, the runtime of a problem is denoted as
:math:`\mathrm{RT}(n,f_\theta,\theta,f^{\rm target}_\theta)`. It is a random variable that counts the number of function evaluations needed to reach a function value lower or equal than :math:`f^{\rm target}_{\theta}`  for the first time. A run or trial that reached a target function value |ftarget| is called *successful*.

We also have to **deal with unsuccessful trials**, that is a run that did not reach a target. We then record the number of function evaluations till the algorithm is stopped. We denote the respective random variable :math:`\mathrm{RT}^{\rm us}(n,f_\theta,\theta,f^{\rm target}_\theta)`.

In order to come up with a meaningful way to compare algorithms having different probability of success (that is different probability to reach a target), we consider the conceptual **restart algorithm**: We assume that an algorithm, say called A, has a strictly positive probability |ps| to successfully solve a problem (that is to reach the associated target). The restart-A algorithm consists in restarting A till the problem is solved. The running time of the restart-A algorithm equals

.. math::
	:nowrap:

	\begin{equation*}
	\mathbf{RT}(n,f_\theta,\theta,f^{\rm target}_\theta) = \sum_{j=1}^{J-1} \mathrm{RT}^{\rm us}_j(n,f_\theta,\theta,f^{\rm target}_\theta) + \mathrm{RT}(n,f_\theta,\theta,f^{\rm target}_\theta)
	\end{equation*}

where :math:`J` is a random variable that models the number of unsuccessful runs till a success is observed, :math:`\mathrm{RT}^{\rm us}_j` are random variables corresponding to the runtime of unsuccessful trials and :math:`\mathrm{RT}` is a random variable for the runtime of successful trial.

Remark that if the probability of success is one, the restart algorithm and the original   algorithm coincide.
	
.. Note:: Considering the runtime of the restart algorithm allows to compare quantitatively the two different scenarios where

	* an algorithm converges often but relatively slowly
	* an algorithm converges less often, but once it converges, it converges fast.

The performance assessment in COCO heavily relies on this conceptual restart algorithm. However, we collect only one single sample of (successful or unsuccessful) runtime per problem while more are needed to be able to display significant data. This is where the idea of instance comes into play: We interpret different runs performed on different instances :math:`\theta_1,\ldots,\theta_K` of the same parametrized function :math:`f_\theta` as repetitions, that is as if they were performed on the same function. [#]_ 

.. [#] This assumes that instances of the same parametrized function are similar 
      to each others or that there is  not too much discrepancy in the difficulty 
      of the problem for different instances.

Runtimes collected for the different instances :math:`\theta_1,\ldots,\theta_K` of the same parametrized function :math:`f_\theta` and with respective targets associated to the same precision :math:`\epsilon` (see above) are thus assumed independent identically distributed. We denote the random variable modelling those runtimes :math:`\mathrm{RT}(n,f_\theta,\epsilon)`. We hence have a collection of runtimes (for a given parametrized function and a given precision) whose size corresponds to the number of instances of a parametrized function where the algorithm was run (typically between 10 and 15). Given that the specific instance does not matter, we write in the end the runtime of a restart algorithm of a parametrized family of function in order to reach a precision :math:`\epsilon` as

.. _eq:RTrestart:

.. math::
	:nowrap:
	:label: RTrestart 

	\begin{equation*}\label{RTrestart}
	\mathbf{RT}(n,f_\theta,\epsilon) = \sum_{j=1}^{J-1} \mathrm{RT}^{\rm us}_j(n,f_\theta,\epsilon) + \mathrm{RT}(n,f_\theta,\epsilon)
	\end{equation*}
	
	
where as above :math:`J` is a random variable modelling the number of trials needed before to observe a success, :math:`\mathrm{RT}^{\rm us}_j` are random variables modeling the number of function evaluations of unsuccessful trials and :math:`\mathrm{RT}^{\rm us}` the one for successful trials.

As we will see in Section :ref:`sec:ART` and Section :ref:`sec:ECDF` our performance display relies on the runtime of the restart algorithm, either considering the average runtime (Section :ref:`sec:ART`) or the distribution by displaying empirical cumulative distribution (Section :ref:`sec:ECDF`).


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
  

Simulated Run-length 
-----------------------

The runtime of the conceptual restart algorithm given in Equation :eq:`RTrestart` is the basis for displaying performance within COCO. We simulate some approximate samples of the runtime of the restart algorithm by constructed some so-called **simulated run-length**.

**Simulated Run-length:** Given the collection of runtimes for successful and unsuccessful trials to reach a given precision, we built a simulated run-length by repeatedly drawing with replacement among those runtimes till we draw a runtime of a successful trial. The simulated run-length is the sum of the drawn runtime.

.. Note:: The construction of a simulated run-length assumes that we have at least one runtime associated to a successful trial.

.. We call a simulated-restart, a simulated run-length concatenated to an unsuccessful 
.. runtime. That is a simulated-restart always starts from an unsuccessful trial.

.. _sec:ART:

Average Running Time
=====================

The average Running Time (|ART|) (introduced in [Price:1997]_ as
ENES and analyzed in [Auger:2005b]_ as success performance) is an estimate of the expected runtime of the restart algorithm given in Equation :eq:`RTrestart` that is used within the COCO framework. More precisely the expected runtime of the restart algorithm (on a parametrized family of function in order to reach a precision :math:`\epsilon`) writes

.. math::
    :nowrap:

	\begin{eqnarray}
	\mathbb{E}(\mathbf{RT}) & =  
	& \mathbb{E}(\mathrm{RT}^{\rm s})  + \frac{1-p_s}{p_s} 	 \mathbb{E}(\mathrm{RT}^{\rm us}) 
    \end{eqnarray}
    
    
where |ps| is the probability of success of the algorithm (to reach the underlying precision) and :math:`\mathrm{RT}^s` denotes the random variable modeling the runtime of successful runs and :math:`\mathrm{RT}^{\rm us}` the runtime of unsuccessful runs (see [Auger:2005b]_). Given a finite number of realizations of the runtime of an algorithm (run on a parametrized family of functions to reach a certain precision) that comprise at least one successful run, say :math:`\{\mathrm{RT}^{\rm us}_i, \mathrm{RT}^{\rm s}_j \}`, we can estimate the expected runtime of the restart algorithm given in the previous equation as the average runtime defined as

.. math::
    :nowrap:

	\begin{eqnarray}
	\mathrm{ART} & = & \mathrm{RT}_\mathrm{S} + \frac{1-p_{\mathrm{s}}}{p_{\mathrm{s}}} \,\mathrm{RT}_\mathrm{US} \\  & = & \frac{\sum_i \mathrm{RT}^{\rm us}_i + \sum_j \mathrm{RT}^{\rm us}_j }{\#\mathrm{succ}} \\
	& = & \frac{\#\mathrm{FEs}}{\#\mathrm{succ}} 
    \end{eqnarray}    
 
.. |nbsucc| replace:: :math:`\#\mathrm{succ}`
.. |Ts| replace:: :math:`\mathrm{RT}_\mathrm{S}`
.. |Tus| replace:: :math:`\mathrm{RT}_\mathrm{US}`
.. |ps| replace:: :math:`p_{\mathrm{s}}`


where where |Ts| and |Tus| denote the average runtime for successful and unsuccessful trials,  |nbsucc| denotes the number of successful trials and  :math:`\#\mathrm{FEs}` is
the number of function evaluations
conducted in all trials (before to reach a given precision).

Remark that while not explicitly denoted, the average runtime depends on the target and more precisely on a precision. It also depends strongly on the termination criterion of the algorithm.
    




.. _sec:ECDF:

Empirical Cumulative Distribution Functions
===========================================



We display distribution of running times through empirical cumulative distribution functions (ECDF). Formally, let us consider a set of problems :math:`\mathcal{P}` and the collection of running times to solve those problems :math:`(\mathrm{RT}_p)_{p \in \mathcal{P}}`. When the problem is not solved, the running time is infinite. The ECDF that we display is defined as


.. math::
	:nowrap:

	\begin{equation*}
	\mathrm{ECDF}(\alpha) = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} 1 \left\{ \log_{10}( \mathrm{RT}_p / n ) \leq \alpha \right\} \enspace.
	\end{equation*}

For instance, we display in Figure :ref:`fig:ecdf`, the ECDF of the running times of the pure random search algorithm on the set of problems formed by the parametrized sphere function (first function of the single-objective testsuit) with 51 relative targets uniform on a log-scale between :math:`10^2` and :math:`10^{-8}`. We can read on this plot that 20 percent of the problems were solved in about :math:`10^3` function evaluations. 


.. todo::
	* aggregation of distribution of RT (read COCO + proceed)
	* data profile.


We exploit the "horizontal and vertical" viewpoints introduced in the last
Section :ref:`sec:verthori`. In Figure :ref:`fig:ecdf` we plot the :abbr:`ECDF
(Empirical Cumulative Distribution Function)` of the intersection point
values (stars in Figure :ref:`fig:HorizontalvsVertical`) for 450 trials.

.. [#] The empirical (cumulative) distribution function
   :math:`F:\mathbb{R}\to[0,1]` is defined for a given set of real-valued data
   :math:`S`, such that :math:`F(x)` equals the fraction of elements in
   :math:`S` which are smaller than :math:`x`. The function :math:`F` is
   monotonous and a lossless representation of the (unordered) set :math:`S`.

.. _fig:ecdf:

.. figure:: pics/plots-RS-2009-bbob/pprldmany_f001_05D.*
   :width: 80%
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




.. [BBO2016biobj] The BBOBies: Biobjective function benchmark suite. 
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
.. [Hooker:1995] J. N. Hooker Testing heuristics: We have it all wrong. In Journal of
    Heuristics, pages 33-42, 1995.

   

