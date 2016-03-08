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

This document presents the main ideas and concepts of the performance assessment within the COCO platform. We start from a collection of recorded data from the algorithm benchmarked that are function values (indicator values in the case of multi-objective, function and constraint values in the case of constraint optimization) together with the number of calls to the function to reach these values. The latter represents the cost of the algorithm. This means that we exclude measuring cost in terms of CPU or wall-clock time to avoid measurement that depends on the programming language or the computer where the experiments were run [#]_. The shortcomings and unfortunate consequences of benchmarking based on CPU was discussed in [Hooker:1995]_.

From the collection of (function value, number of function evaluations) pairs, we extract runtimes (or run-length) to reach target function values. This comes as a natural consequence of our prerequisite to present *quantitative* performance measures (as opposed to simple rankings of algorithm performances).

We then either display average runtime through the `Average Runtime`_ (ART) measure or the distribution of runtimes through `Empirical Cumulative Distribution Functions`_ (ECDF). When displaying the distribution of runtimes, we consider the aggregation of runtimes over subclasses of problems excluding aggregation over dimensions given that the dimension of the problem is an information available that should be exploited (for instance for deciding on which algorithm to choose).


.. [#] We however require to provide CPU timing experiments to get a
	rough measurement of the time complexity of the algorithm.

.. budget-free

Terminology and Definitions
----------------------------

We introduce a few terms and definitions that are used in the rest of the document.

*problem, function*
 In the COCO framework, a problem is defined as a triple  ``(dimension,function,instance)``. In this terminology a ``function`` is actually a parametrized function and the ``instance`` is an instantiation of the parameters. More precisely let us consider a parametrized function  :math:`f_\theta: \mathbb{R}^n \to \mathbb{R}^m` for :math:`\theta \in \Theta` then a COCO problem corresponds to :math:`p=(n,f_\theta,\bar{\theta})` where :math:`n \in \mathbb{N}` is the dimension of the search space, and :math:`\bar{\theta}` is a set of parameters to instantiate the parametrized function. An algorithm optimizing the  problem :math:`p` will optimize :math:`\mathbf{x} \in \mathbb{R}^n \to f_{\bar{\theta}}(\mathbf{x})`. To simplify notation, in the sequel a COCO problem is denoted :math:`p=(n,f_\theta,\theta)`.
 
 In the performance assessment setting, we associate to a problem :math:`p`, a :math:`{\rm target}`, which is a function value :math:`f_{\rm target}` at which we extract the runtime of the algorithm. Given that the optimal function value, that is :math:`f_{\rm opt} =  \min_{\mathbf{x}} f_{\theta}(\mathbf{x})` depends on the specific instance :math:`\theta`, the :math:`{\rm target}` function values also depends on the instance :math:`\theta`. However the relative target or precision
 
 .. math::
 	:nowrap:

	\begin{equation} 
	\Delta f = f_{\rm target} - f_{\rm opt}
 	\end{equation}
 	
 	
 often does not depend on the instance :math:`\theta` such that we can unambiguously consider for different instances :math:`({\theta}_1, \ldots,{\theta}_K)` of a parametrized problem :math:`f_{\theta}(\mathbf{x})`, the set of targets :math:`f^{\rm target}_{{\theta}_1}, \ldots,f^{\rm target}_{{\theta}_K}` associated to a similar precision. 

*instance*
 Our test functions are parametrized such that different *instances* of the same function are available. Different instances can vary by having different shifted optima, can use different random rotations that are applied to the variables, ...  The notion of instance is introduced to generate repetition while avoiding possible exploitation of an artificial function property (like location of the optimum in zero). 
  
 We **interpret the different runs performed on different instances** of the same parametrized function in a given dimension as if they are just **independent repetitions** of the optimization algorithm on the same function. Put differently the runs performed on :math:`f_{\theta_1}, \ldots,f_{\theta_K}` with :math:`\theta_1,\ldots,\theta_K`, :math:`K` different instances of a parametrized problem :math:`f_\theta`, are assumed to be independent and identically distributed.
 
 .. Anne: maybe we should insist more on this dual view of randomizing the problem class via problem isntance - choosing uniformly over set of parameters.
  
*runtime*
  We define *runtime*, or *run-length* [HOO1998]_
  as the *number of evaluations* 
  conducted on a given problem, also referred to as number of *function* evaluations. 
  Our central performance measure is the runtime until a given target :math:`f`-value 
  is hit.
  

On Performance Measures
=======================


As it was already explained in [HAN2009]_, we advocate **performance measures** that are:

* quantitative, ideally with a ratio scale (opposed to interval or ordinal
  scale)  and with a wide variation (i.e., for example, with values ranging
  not only between 0.98 and 1.0)
* well-interpretable, in particular by having a meaning and semantics attached
  to the numbers
* relevant with respect to the "real world"
* as simple as possible.


For these reasons we measure **runtime** to reach a target function value, that is the number of function evaluations needed to reach a target function value denoted as fixed-target scenario in the following. 


.. _sec:verthori:

Fixed-Cost versus Fixed-Target Scenario
----------------------------------------

Starting from some convergence graphs we can use two different approaches for collecting data and making measurements from
experiments:

* On the one hand we can fix a cost, that is a number of function evaluations and collect the function values reached. This is referred to as **fixed-cost scenario** and corresponds to vertical cuts. Fixing search costs can be pictured as drawing a vertical line on the convergence graphs (see Figure :ref:`fig:HorizontalvsVertical` where
  the line is depicted in red).

* On the other hand we can fix a target and measure the number of function evaluations to reach this target. This is referred to as **fixed-target scenario** and corresponds to horizontal cuts. Fixing a target can
  be pictured as drawing a horizontal line in the convergence graphs
  (Figure :ref:`fig:HorizontalvsVertical` where the line is depicted in blue).


.. _fig:HorizontalvsVertical:

.. figure:: HorizontalvsVertical.*
   :align: center
   :width: 60%

   Horizontal vs Vertical View

   Illustration of fixed-cost view (vertical cuts) and fixed-target view
   (horizontal cuts). Black lines depict the best function value plotted versus
   number of function evaluations.


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

In order to come up with a meaningful way to compare algorithms having different probability of success (that is different probability to reach a target), we consider the conceptual **restart algorithm**: We assume that an algorithm, say called A, has a strictly positive probability |ps| to successfully solve a problem (that is to reach the associated target). The restart-A algorithm consists in restarting A till the problem is solved. The runtime of the restart-A algorithm equals

.. math::
	:nowrap:

	\begin{equation*}
	\mathbf{RT}(n,f_\theta,\theta,f^{\rm target}_\theta) = \sum_{j=1}^{J-1} \mathrm{RT}^{\rm us}_j(n,f_\theta,\theta,f^{\rm target}_\theta) + \mathrm{RT}^{\rm s}(n,f_\theta,\theta,f^{\rm target}_\theta)
	\end{equation*}

where :math:`J` is a random variable that models the number of unsuccessful runs till a success is observed, :math:`\mathrm{RT}^{\rm us}_j` are random variables corresponding to the runtime of unsuccessful trials and :math:`\mathrm{RT}^{\rm s}` is a random variable for the runtime of a successful trial.

Remark that if the probability of success is one, the restart algorithm and the original   algorithm coincide.
	
.. Note:: Considering the runtime of the restart algorithm allows to compare quantitatively the two different scenarios where

	* an algorithm converges often but relatively slowly
	* an algorithm converges less often, but once it converges, it converges fast.

The performance assessment in COCO heavily relies on this conceptual restart algorithm. However, we collect only one single sample of (successful or unsuccessful) runtime per problem while more are needed to be able to display significant data. This is where the idea of instances comes into play: We interpret different runs performed on different instances :math:`\theta_1,\ldots,\theta_K` of the same parametrized function :math:`f_\theta` as repetitions, that is, as if they were performed on the same function. [#]_ 

.. [#] This assumes that instances of the same parametrized function are similar 
      to each others or that there is  not too much discrepancy in the difficulty 
      of the problem for different instances.

Runtimes collected for the different instances :math:`\theta_1,\ldots,\theta_K` of the same parametrized function :math:`f_\theta` and with respective targets associated to the same relative target :math:`\Delta f` (see above) are thus assumed independent and identically distributed. We denote the random variable modeling those runtimes :math:`\mathrm{RT}(n,f_\theta,\Delta f)`. We hence have a collection of runtimes (for a given parametrized function and a given precision) whose size corresponds to the number of instances of a parametrized function where the algorithm was run (typically between 10 and 15). Given that the specific instance does not matter, we write in the end the runtime of a restart algorithm of a parametrized family of function in order to reach a relative target :math:`\Delta f` as

.. _eq:RTrestart:

.. math::
	:nowrap:
	:label: RTrestart 

	\begin{equation*}\label{RTrestart}
	\mathbf{RT}(n,f_\theta,\Delta f) = \sum_{j=1}^{J-1} \mathrm{RT}^{\rm us}_j(n,f_\theta,\Delta f) + \mathrm{RT}^{\rm s}(n,f_\theta,\Delta f)
	\end{equation*}
	
	
where as above :math:`J` is a random variable modeling the number of trials needed before to observe a success, :math:`\mathrm{RT}^{\rm us}_j` are random variables modeling the number of function evaluations of unsuccessful trials and :math:`\mathrm{RT}^{\rm s}` the one for successful trials.

As we will see in Section :ref:`sec:ART` and Section :ref:`sec:ECDF`, our performance display relies on the runtime of the restart algorithm, either considering the average runtime (Section :ref:`sec:ART`) or the distribution by displaying empirical cumulative distribution functions (Section :ref:`sec:ECDF`).


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
  

Simulated Run-lengths of Restart Algorithms
-------------------------------------------

The runtime of the conceptual restart algorithm given in Equation :eq:`RTrestart` is the basis for displaying performance within COCO. We can simulate some (approximate) samples of the runtime of the restart algorithm by constructing so-called simulated run-lengths from the available empirical data:

**Simulated Run-length:** Given a collection of runtimes for successful and unsuccessful trials to reach a given precision, we draw a simulated run-length of the restart algorithm by repeatedly drawing uniformly at random and with replacement among all given runtimes till we draw a runtime from a successful trial. The simulated run-length is then the sum of the drawn runtimes.

.. Note:: The construction of simulated run-lengths assumes that at least one runtime is associated to a successful trial.

Simulated run-lengths are in particular only interesting in the case where at least one trial is not successful. In order to remove unnecessary stochastics in the case that many (or all) trials are successful, we advocate for a derandomized version of simulated run-lengths when we are interested in drawing a batch of :math:`N` simulated run-lengths:

**Simulated Run-lengths (derandomized version):** Given a collection of runtimes for successful and unsuccessful trials to reach a given precision, we deterministically sweep through the trials and define the next simulated run-length as the run-length associated to the trial if it is successful and in the case of an unsuccessful trial as the sum of the associated run-length of the trial and the simulated run-length of the restarted algorithm as described above.

Note that the latter derandomized version to draw simulated run-lengths has the minor disadvantage that the number of samples :math:`N` is restricted to a multiple of the trials in the data set.

.. maybe we should indeed put a picture here



.. _sec:ART:

Average Runtime
=====================

The average runtime (|ART|) (introduced in [Price:1997]_ as
ENES and analyzed in [Auger:2005b]_ as success performance) is an estimate of the expected runtime of the restart algorithm given in Equation :eq:`RTrestart` that is used within the COCO framework. More precisely, the expected runtime of the restart algorithm (on a parametrized family of functions in order to reach a precision :math:`\epsilon`) writes

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


where |Ts| and |Tus| denote the average runtime for successful and unsuccessful trials,  |nbsucc| denotes the number of successful trials and  :math:`\#\mathrm{FEs}` is
the number of function evaluations
conducted in all trials (before to reach a given precision).

Remark that while not explicitly denoted, the average runtime depends on the target and more precisely on a precision. It also depends strongly on the termination criterion of the algorithm.
    




.. _sec:ECDF:

Empirical Cumulative Distribution Functions
===========================================

.. Anne: to be discussed - I talk about infinite runtime to make the definition below .. .. Anne: fine. However it's probably not precise given that runtime above :math:`10^7` are .. Anne: infinite.

We display distributions of runtimes through empirical cumulative distribution functions (ECDF). Formally, let us consider a set of problems :math:`\mathcal{P}` and a collection of runtimes to solve those problems :math:`(\mathrm{RT}_{p,k})_{p \in \mathcal{P}, 1 \leq k \leq K}` where :math:`K` is the number of runtimes per problem. When the problem is not solved, the runtimes are infinite. The ECDF that we display is defined as


.. math::
	:nowrap:

	\begin{equation*}
	\mathrm{ECDF}(\alpha) = \frac{1}{|\mathcal{P}| K} \sum_{p \in \mathcal{P},k} \mathbf{1} \left\{ \log_{10}( \mathrm{RT}_{p,k} / n ) \leq \alpha \right\} \enspace.
	\end{equation*}

It gives the *proportion of problems solved in less than a specified budget* which is read on the x-axis. For instance, we display in Figure :ref:`fig:ecdf`, the ECDF of the running times of the pure random search algorithm on the set of problems formed by the parametrized sphere function (first function of the single-objective ``bbob`` test suite) in dimension :math:`n=5` with 51 relative targets uniform on a log-scale between :math:`10^2` and :math:`10^{-8}` and :math:`K=10^3`. We can read in this plot for example that a little bit less than 20 percent of the problems were solved in less than :math:`5 \cdot 10^3 = 10^3 \cdot n` function evaluations. 

Note that we consider **runtimes of the restart algorithm**, that is, we use the idea of simulated run-lengths of the restart algorithm as described above to generate :math:`K` runtimes from typically 10 or 15 instances per function and dimension. Hence, only when no instance is solved, we consider that the runtime is infinite.


.. _fig:ecdf:

.. figure:: pics/plots-RS-2009-bbob/pprldmany_f001_05D.*
   :width: 80%
   :align: center

   ECDF

   Illustration of empirical (cumulative) distribution function (ECDF)
   of runtimes on the sphere function using 51 relative targets
   uniform on a log scale between :math:`10^2` and :math:`10^{-8}`. The
   runtimes displayed correspond to the pure random search
   algorithm in dimension 5.
   

      
**Aggregation:**

In the ECDF displayed in Figure :ref:`fig:ecdf` we have **aggregated** the runtime on several problems by displaying the runtime of the pure random search on the set of problems formed by 51 targets between :math:`10^2` and :math:`10^{-8}` on the parametrized sphere in dimension 5.

Those problems concern the same parametrized family of functions, namely a set of shifted sphere functions with different offsets in their function values. We consider also aggregation **over several parametrized functions**. We usually divide the set of parametrized functions into subgroups sharing similar properties (for instance separability, unimodality, ...) and display ECDFs which aggregate the problems induced by those functions and by all targets. See Figure :ref:`fig:ecdfgroup`.


.. _fig:ecdfgroup:

.. figure:: pics/plots-RS-2009-bbob/gr_separ_05D_05D_separ-combined.* 
   :width: 100%
   :align: center
   
   ECDF for a subgroup of functions

   **Left:** ECDF of the runtime of the pure random search algorithm for
   functions f1, f2, f3, f4 and f5 that constitute the group of
   separable functions for the ``bbob`` testsuite. **Right:** ECDF aggregated
   over all targets and functions f1, f2, f3, f4 and f5.
   

We can also naturally aggregate over all functions and hence obtain one single ECDF per algorithm per dimension. The ECDF of different algorithms can be displayed on the same graph as depicted in Figure :ref:`fig:ecdfall`.

.. _fig:ecdfall:

.. figure:: pics/plots-all2009/pprldmany_noiselessall-5and20D.* 
   :width: 100%
   :align: center
   
   ECDF over all functions and all targets

   ECDF of several algorithms benchmarked during the BBOB 2009 workshop
   in dimension 5 (left) and in dimension 20 (right) when aggregating over all functions of the ``bbob`` suite.
   
   
.. Note::  
 	The ECDF graphs are also known under the name data
 	profile (see [More:2009]_). Note, however, that the original definition of data profiles does not consider a log scale for the runtime and that data profiles are standardly used without a log scale [Rios:2012]_.
	
	We advocate not to aggregate over dimension as the dimension is 
	typically an input parameter to the algorithm that can be
	exploited to run different types of algorithms on different dimensions.



.. todo::
	* ECDF and uniform pick of a problem
	* clean up bibliography


Acknowledgements
================
This work was supported by the grant ANR-12-MONU-0009 (NumBBO) 
of the French National Research Agency.


References
==========
	

.. [HAN2009] Hansen, N., A. Auger, S. Finck R. and Ros (2009), Real-Parameter Black-Box Optimization Benchmarking 2009: Experimental Setup, *Inria Research Report* RR-6828 http://hal.inria.fr/inria-00362649/en


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
.. [More:2009] Jorge J. Moré and Stefan M. Wild. Benchmarking
	Derivative-Free Optimization Algorithms, SIAM J. Optim., 20(1), 172–191, 2009.
.. [Price:1997] K. Price. Differential evolution vs. the functions of
   the second ICEO. In Proceedings of the IEEE International Congress on
   Evolutionary Computation, pages 153–157, 1997.
.. [Hooker:1995] J. N. Hooker Testing heuristics: We have it all wrong. In Journal of
    Heuristics, pages 33-42, 1995.

   

