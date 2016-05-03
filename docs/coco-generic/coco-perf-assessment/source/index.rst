.. title:: COCO: Performance Assessment

##############################
COCO: Performance Assessment
##############################

.. .. toctree::
   :maxdepth: 2

..
   sectnum::

.. |ftarget| replace:: :math:`I^{{\rm target},\theta}`
.. |nruns| replace:: :math:`\texttt{Ntrial}`
.. |DIM| replace:: :math:`n`
.. _2009: http://www.sigevo.org/gecco-2009/workshops.html#bbob
.. _2010: http://www.sigevo.org/gecco-2010/workshops.html#bbob
.. _2012: http://www.sigevo.org/gecco-2012/workshops.html#bbob
.. _BBOB-2009: http://coco.gforge.inria.fr/doku.php?id=bbob-2009-results
.. _BBOB-2010: http://coco.gforge.inria.fr/doku.php?id=bbob-2010-results
.. _BBOB-2012: http://coco.gforge.inria.fr/doku.php?id=bbob-2012
.. _GECCO: http://www.sigevo.org/gecco-2012/
.. _COCO: https://github.com/numbbo/coco
.. .. _COCO: http://coco.gforge.inria.fr
.. |ERT| replace:: :math:`\mathrm{ERT}`
.. |aRT| replace:: :math:`\mathrm{aRT}`
.. |dim| replace:: :math:`\mathrm{dim}`
.. |function| replace:: :math:`\mathrm{function}`
.. |instance| replace:: :math:`\mathrm{instance}`
.. |R| replace:: :math:`\mathbb{R}`
.. |ftheta| replace::  :math:`f_{\theta}`


.. the next two lines are necessary in LaTeX. They will be automatically 
  replaced to put away the \chapter level as ^^^ and let the "current" level
  become \section. 

.. CHAPTERTITLE
.. CHAPTERUNDERLINE

.. raw:: latex

  % \tableofcontents is automatic with sphinx and moved behind abstract by swap...py
  \begin{abstract}

.. WHEN CHANGING THIS, CHANGE ALSO the abstract in conf.py ACCORDINGLY (though it seems the latter is not used)

We present an any-time performance assessment for benchmarking numerical
optimization algorithms in a black-box scenario,
applied within the COCO_ benchmarking platform. 
We describe... [#]_

.. [#] *ArXiv e-prints*, arXiv:1605.xxxxx__, 2016.
.. __: http://arxiv.org/abs/1605.xxxxx


.. raw:: latex

  \end{abstract}
  \newpage


Introduction
=============

.. budget-free

This document presents the main ideas and concepts of the performance assessment within the COCO platform. Opposed to simple rankings of algorithms, we aim to provide a *quantitative* and *meaningful* performance assessment, which allows for conclusions of type: Algorithm A is ten times faster than algorithm B in solving the given problem or in solving problems with a certain type of difficulties/problem features. In order to do so, we record algorithm *runtimes*, measured in number of function evaluations to reach predefined target values, during the algorithm run.

Runtimes represent the cost of the algorithm. Apart from a short, exploratory experiment [#]_, we avoid measuring the algorithm cost in CPU or wall-clock time because these depend on parameters which are difficult or impractical to control, like the programming language, coding style, the computer used to run the experiments, etc. See [Hooker:1995]_ for a discussion on shortcomings and unfortunate consequences of benchmarking based on CPU time.

 .. [#] The COCO platform provides a CPU timing experiment to get a rough estimate of the time complexity of the algorithm [HAN2016ex]_.

We can then display an average runtime (aRT, see Section `Average Runtime`_) and the empirical distribution of runtimes (ECDF, see Section `Empirical Cumulative Distribution Functions`_). When displaying the distribution of runtimes, we consider the aggregation of runtimes over subclasses of problems or over all problems. We do not aggregate over dimensions, because the dimension of the problem can be used to decide which algorithm (or algorithm variant, or parameter setting) is preferred.


Terminology and Definitions
----------------------------

.. Tea: We have this section in every documentation and every time there are some differences
   between the definitions. Would it be possible to make this more uniform? I understand that
   some documents require more detailed definitions than others, but this could be solved
   differently. For example, (I'm not sure whether the reStructuredText even supports this,
   but I hope it does), the ideal approach would be to have all definitions in a single file
   and then only "pull" the ones that should be in this document here (the same goes for the
   other documents, of course). We could then even have short and long definition variants
   for the terms that require it.
   EDIT: I see now that this section is quite different from the sections with the same
   title in the other documents (i.e., here we go into more detail and explanation why
   things are done the way they are), so maybe my proposal is less suited here than in the
   other documentations (I think we should still consider to do this at least for the other
   documentations).
   
.. It will be nice to have an online glossary at some point that will help keeping things
   consistent.

   
We introduce a few terms and definitions that are used in the rest of the document.

   
*problem, function*
 In the COCO_ framework, a problem is defined as a triple  ``(dimension,function,instance)``. In this terminology a ``function`` is actually a parametrized function (to be minimized) and the ``instance`` describes an instantiation of the parameters.
 
 More precisely, let us consider a parametrized function  :math:`f_\theta: \mathbb{R}^n \to \mathbb{R}^m` for :math:`\theta \in \Theta`, then a COCO problem corresponds to :math:`p=(n,f_\theta,\theta)` where :math:`n \in \mathbb{N}` is the dimension of the search space, and :math:`\theta` is a set of parameters to instantiate the parametrized function. Given a dimension :math:`n` and two different instances :math:`\theta_1` and :math:`\theta_2` of the same parametrized family :math:`f_{\theta}`, optimizing the associated problems means optimizing :math:`f_{\theta_1}(\mathbf{x})` and :math:`f_{\theta_2}(\mathbf{x})` for :math:`\mathbf{x} \in \mathbb{R}^n`.
 
 Typically, the performance of an optimization algorithm at time :math:`t`, which aims at optimizing a problem :math:`p=(n,f_\theta,\theta)`, is defined via a quality indicator function mapping the set of all solutions evaluated so far together with their :math:`m`-dimensional evaluation vectors, outputted by :math:`f_\theta`, to a real value. In the single-objective noiseless case, this quality indicator function simply outputs the minimal observed (feasible) function value during the first :math:`t` function evaluations. In the multi-objective case, well-known multi-objective quality indicators such as the hypervolume indicator can be used to map the entire set of already evaluated solutions ("archive") to a real value.
 
 .. Anne: I took out the theta-bar - did not look too fine to me - so I felt that I needed to add theta_1 and theta_2 as two different instances @Niko, @Tea please check and improve if possible (I am not particularly happy with the new version).
 
 
 In the performance assessment setting, we associate to a problem :math:`p` and a given quality indicator,
 one or several target values such that a problem is then a quintuple ``(dimension,function,instance,quality indicator,target)``. A target value is thereby a fixed function or quality indicator value :math:`f^{\rm target}` at which we extract the runtime of the algorithm, which we assume to be minimized as well [#]_, and which typically depends on the problem instance :math:`\theta`. Given that the optimal function or quality indicator value :math:`f^{\mathrm{opt}, \theta}` depends on the specific instance :math:`\theta`, the target function/quality indicator values also depend on the instance :math:`\theta`. However, the relative target or precision

 .. math::
 	:nowrap:

	\begin{equation}
	\Delta f = f^{\rm target,\theta} - f^{\rm opt,\theta}
 	\end{equation}

 does not depend on the instance :math:`\theta` such that we can unambiguously consider for different instances :math:`({\theta}_1, \ldots,{\theta}_K)` of a parametrized problem :math:`f_{\theta}(\mathbf{x})`, the set of targets :math:`f^{\rm target,{\theta}_1}, \ldots,f^{\rm target,{\theta}_K}` associated to the same precision. Note that in the absence of knowledge about the optimal function/quality indicator value, :math:`f^{\rm opt,\theta}` is typically replaced by the best known approximation of :math:`f^{\rm opt,\theta}`.
 
 Depending on the context, we will refer to both the original triple ``(dimension,function,instance)`` and the quintuple ``(dimension,function,instance,quality indicator,target)`` as *problem*. We say, for example, that "algorithm A is solving problem :math:`p=(n,f_\theta,\theta,I,I^{\rm target})` after :math:`t` function evaluations" if the quality indicator function value :math:`I`  during the optimization of :math:`(n,f_\theta,\theta)` reaches a value of :math:`I^{\rm target}` or lower for the first time after :math:`t` function evaluations.

 .. [#] Note that we assume without loss of generality minimization of the quality indicator here for historical reasons although the name quality indicator itself suggests maximization.
 
.. Anne: Dimo, why did you drop the theta-dependency of I^target

.. Anne: I think that we have an organization problem - this definition of
  problem,  function becomes now too long and should most likely be in a
  dedicated section where it could be expanded. 
 	
*instance*
 Our test functions are parametrized such that different *instances* of the same function are available. Different instances can vary by having different shifted optima, can use different rotations that are applied to the variables, ...  The notion of instance is introduced to generate repetition while avoiding possible exploitation of an artificial function property (like location of the optimum in zero).

 
 ..  We often **interpret different runs performed on different instances**
 .. of the same parametrized function in a given dimension as **independent
 .. repetitions** of the optimization algorithm on the same function. Put
 .. differently, the runs performed on :math:`K` different instances,
 .. :math:`f_{\theta_1}, \ldots,f_{\theta_K}`, of a parametrized problem
 .. :math:`f_\theta`, are assumed to be independent and identically
 .. distributed.

 .. Anne: maybe we should insist more on this dual view of randomizing the problem class via problem isntance - choosing uniformly over set of parameters.

 .. Tea: I'm not sure that our use of instances belongs under the definition of instances.
    I think this (important!) issue should be explained in more detail later, not here.

*runtime*
  We define *runtime*, or *run-length* [HOO1998]_
  as the *number of evaluations*, also referred to as *function* evaluations,
  conducted on a given problem until a given quality indicator target value is reached.
  Runtime is our central performance measure.


On Performance Measures
=======================

Following [HAN2009]_, we advocate **performance measures** that are

 * quantitative, ideally with a ratio scale (opposed to interval or ordinal
   scale, [STE1946]_)  and with a wide variation (i.e., for example, with typical
   values ranging not only between 0.98 and 1.0) [#]_
 * well-interpretable, in particular by having a meaning and semantics attached
   to the numbers
 * relevant and meaningful with respect to the "real world"
 * as simple as possible.

.. Tea: Can we give some more explanation here?

For these reasons we measure **runtime** to reach a target value, that is the number of function evaluations needed to reach a quality indicator target value denoted as fixed-target scenario in the following.


.. [#] The transformation :math:`x\mapsto\log(1-x)` can alleviate the problem
  in this case, given it actually zooms in mostly on relevant values.

.. _sec:verthori:

Fixed-Budget versus Fixed-Target Approach
-----------------------------------------

.. for collecting data and making measurements from experiments:

Starting from some convergence graphs, which plot the quality indicator (to be minimized) against the number of function evaluations, we have two different approaches to measure performance.

**fixed-budget approach**
    We fix a budget of function evaluations,
    and collect the function values reached. Fixing the search
    budget can be pictured as drawing a *vertical* line on the convergence
    graphs (see Figure :ref:`fig:HorizontalvsVertical` where the line is
    depicted in red).

**fixed-target approach**
    We fix a target value and measure the number of function
    evaluations, the *runtime*, to reach this target. Fixing a target can be
    pictured as drawing a *horizontal* line in the convergence graphs (Figure
    :ref:`fig:HorizontalvsVertical` where the line is depicted in blue).


.. _fig:HorizontalvsVertical:

.. figure:: HorizontalvsVertical.*
   :align: center
   :width: 60%

   **Horizontal versus Vertical View**
   
   Illustration of fixed-budget view (vertical cuts) and fixed-target view
   (horizontal cuts). Black lines depict the best quality indicator value
   plotted versus number of function evaluations.


.. It is often argued that the fixed-cost approach is close to what is needed for
   real world applications where the total number of function evaluations is
   limited. On the other hand, also a minimum target requirement needs to be
   achieved in real world applications, for example, getting (noticeably) better
   than the currently available best solution or than a competitor.

For the performance assessment of algorithms, the fixed-target approach is superior
to the fixed-budget approach since it gives *quantitative and interpretable*
data.

 * The fixed-budget approach (vertical cut) does not give *quantitatively
   interpretable*  data:
   the observation that Algorithm A reaches a function value that is two (or
   ten, or a hundred) times smaller than the one reached by Algorithm B has in
   general no interpretable meaning, mainly because there is no *a priori*
   way to determine *how much* more difficult it is to reach a function value
   that is two (or ten, or a hundred) times smaller.
   This, indeed, largely depends on the specific function and on the specific
   function value reached.

 * The fixed-target approach (horizontal cut)
   *measures the time* to
   reach a target function value. The measurement allows conclusions of the
   type: Algorithm A is two (or ten, or a hundred) times faster than Algorithm B
   in solving this problem (i.e. reaching the given target function value).

Furthermore, for algorithms that are invariant under certain transformations
of the function value (for example under order-preserving transformations, as
comparison-based algorithms like DE, ES, PSO ), fixed-target measures become
invariant under these transformations by transformation of the target values
while fixed-budget measures require the transformation of all resulting data.

.. Tea: We should add references to DE, ES and PSO.
   This last paragraph should be reformulated a bit to make it more clear.
   
.. Dimo: TODO: cite Giens paper here

Missing Values
---------------

We collect runtimes to reach targets. However not all runs successfully reach a target, see for instance Figure :ref:`fig:HorizontalvsVertical`. In this case the runtime  is undefined and we collect the maximal number of function evaluations of the corresponding run. This is a lower bound on the (non-observed) runtime to reach the target.

.. Anne: @Niko check.


A Third Approach: Runlength-based Targets
-----------------------------------------
In addition to the fixed-budget and fixed-target approaches, there is an
intermediate approach, combining the ideas of *measuring runtime* (to get
meaningful measurements) and *fixing budgets* (of our interest). The basic idea
is the following.

We first fix a reference algorithm :math:`\mathcal{A}` which we run on a
problem of interest (i.e. on a 3-tuple of parameterized function, dimension,
and instance) and for which we record runtimes to reach given target values
:math:`\mathcal{F}_{\rm target} = \{ f_{\rm target}^1, \ldots, f_{\rm target}^{|\mathcal{F}_{\rm target}|} \}`
(with :math:`f_{\rm target}^i` > :math:`f_{\rm target}^j` for all :math:`i<j`)
as in the fixed-target approach described above. The chosen reference
algorithm will serve as a baseline upon which the runlength-based targets are 
computed in the second step.

Second, we fix a set of reference budgets :math:`B = \{b_1,\ldots, b_{|B|}\}`
(in number of function evaluations) that we are interested in for the given
problem and that are increasing (:math:`b_i < b_j` for all :math:`i<j`). We
then pick, for each given budget :math:`b_i` (:math:`1\leq i\leq |B|`), the
largest target that the reference algorithm :math:`\mathcal{A}` did not reach
within the given budget and that also has not yet been chosen for smaller
budgets:

.. math::
  	:nowrap:

 	\begin{equation*}
		T_{\rm chosen}^i = \max_{1\leq j \leq | \mathcal{F}_{\rm target} |}
				f_{\rm target}^j \text{ such that }
				f_{\rm target}^{j} < f(\mathcal{A}, b_i) \text{ and }
				f_{\rm target}^j < f_{\rm chosen}^{k} \text{ for all } k<i
  	\end{equation*}

with :math:`f(\mathcal{A}, t)` being the best function (or indicator) value
found by algorithm :math:`\mathcal{A}` within the first :math:`t` function
evaluations of the performed run.

	
 .. Dimo: please check whether the notation is okay

 .. Dimo: TODO: make notation consistent wrt f_target

Note that this runlength-based targets approach is in particular used in COCO
for the scenario of (single-objective) expensive optimization in which the
artificial best algorithm of BBOB-2009 is used as reference algorithm and the
five budgets of :math:`0.5n`, :math:`1.2n`, :math:`3n`, :math:`10n`, and
:math:`50n` function evaluations are fixed (with :math:`n` being the problem
dimension).



Runtime over Problems
=========================


In order to display quantitative measurements, we have seen in the previous
section that we should start from the collection of runtimes for different
target values. These target values can be a :math:`f`- or indicator value
(see [TUS2016]_).
In the performance assessment setting, a problem is the quintuple
:math:`p=(n,f_\theta,\theta,I,I^{{\rm target},\theta})` where
:math:`I^{{\rm target},\theta}` is the target function/indicator value. This means that
**we collect runtimes of problems**.

Formally, the runtime of a problem :math:`p` is denoted as
:math:`\mathrm{RT}(p)`. It is a random
variable that counts the number of function evaluations needed to reach a
quality indicator value lower or equal than :math:`I^{{\rm target},\theta}`  for the
first time. A run or trial that reached a target function value |ftarget| is
called *successful*.

We also have to **deal with unsuccessful trials**, that is a run that did not
reach a target. We then record the number of function evaluations till the
algorithm is stopped. We denote the respective random variable
:math:`\mathrm{RT}^{\rm us}(np)`.

In order to come up with a meaningful way to compare algorithms having
different probability of success (that is different probability to reach a
target), we consider the conceptual **restart algorithm**: We assume that an
algorithm, say called A, has a strictly positive probability |ps| to
successfully solve a problem (that is to reach the associated target). The
restart-A algorithm consists in restarting A till the problem is solved. The
runtime of the restart-A algorithm to solve problem :math:`p` equals

.. math::
	:nowrap:

	\begin{equation*}
	\mathbf{RT}(p) = \sum_{j=1}^{J-1} \mathrm{RT}^{\rm us}_j(p) + \mathrm{RT}^{\rm s}(p)
	\end{equation*}

where :math:`J` is a random variable that models the number of unsuccessful
runs till a success is observed, :math:`\mathrm{RT}^{\rm us}_j` are random
variables corresponding to the runtime of unsuccessful trials and
:math:`\mathrm{RT}^{\rm s}` is a random variable for the runtime of a
successful trial.

Remark that if the probability of success is one, the restart algorithm and
the original   algorithm coincide.

.. Note:: Considering the runtime of the restart algorithm allows to compare
   quantitatively the two different scenarios where

	* an algorithm converges often but relatively slowly
	* an algorithm converges less often, but whenever it converges, it is with a fast convergence rate.

Runs on Different Instances Interpreted as Independent Repetitions
------------------------------------------------------------------
The performance assessment in COCO heavily relies on the conceptual restart algorithm. However, we collect only one single sample of (successful or unsuccessful) runtime per problem while more are needed to be able to display significant data. This is where the idea of instances comes into play: We interpret different runs performed on different instances :math:`\theta_1,\ldots,\theta_K` of the same parametrized function :math:`f_\theta` as repetitions, that is, as if they were performed on the same function. [#]_

.. [#] This assumes that instances of the same parametrized function are similar
      to each others or that there is  not too much discrepancy in the difficulty
      of the problem for different instances.

Runtimes collected for the different instances :math:`\theta_1,\ldots,\theta_K` of the same parametrized function :math:`f_\theta` and with respective targets associated to the same relative target :math:`\Delta I` (see above) are thus assumed independent and identically distributed. We denote the random variable modeling those runtimes :math:`\mathrm{RT}(n,f_\theta,\Delta I)`. We hence have a collection of runtimes (for a given parametrized function and a given relative target) whose size corresponds to the number of instances of a parametrized function where the algorithm was run (typically between 10 and 15). Given that the specific instance does not matter, we write in the end the runtime of a restart algorithm of a parametrized family of function in order to reach a relative target :math:`\Delta I` as

.. _eq:RTrestart:

.. math::
	:nowrap:
	:label: RTrestart

	\begin{equation*}\label{RTrestart}
	\mathbf{RT}(n,f_\theta,\Delta I) = \sum_{j=1}^{J-1} \mathrm{RT}^{\rm us}_j(n,f_\theta,\Delta I) + \mathrm{RT}^{\rm s}(n,f_\theta,\Delta I)
	\end{equation*}


where as above :math:`J` is a random variable modeling the number of trials needed before to observe a success, :math:`\mathrm{RT}^{\rm us}_j` are random variables modeling the number of function evaluations of unsuccessful trials and :math:`\mathrm{RT}^{\rm s}` the one for successful trials.

As we will see in Section :ref:`sec:aRT` and Section :ref:`sec:ECDF`, our performance display relies on the runtime of the restart algorithm, either considering the average runtime (Section :ref:`sec:aRT`) or the distribution by displaying empirical cumulative distribution functions (Section :ref:`sec:ECDF`).



Simulated Run-lengths of Restart Algorithms
-------------------------------------------

The runtime of the conceptual restart algorithm given in Equation :eq:`RTrestart` is the basis for displaying performance within COCO. We can simulate some (approximate) samples of the runtime of the restart algorithm by constructing so-called simulated run-lengths from the available empirical data.

**Simulated Run-length:** Given a collection of runtimes for successful and unsuccessful trials to reach a given precision, we draw a simulated run-length of the restart algorithm by repeatedly drawing uniformly at random and with replacement among all given runtimes till we draw a runtime from a successful trial. The simulated run-length is then the sum of the drawn runtimes.

.. Note:: The construction of simulated run-lengths assumes that at least one runtime is associated to a successful trial.

Simulated run-lengths are in particular only interesting in the case where at least one trial is not successful. In order to remove unnecessary stochastics in the case that many (or all) trials are successful, we advocate for a derandomized version of simulated run-lengths when we are interested in drawing a batch of :math:`N` simulated run-lengths:

**Simulated Run-lengths (derandomized version):** Given a collection of runtimes for successful and unsuccessful trials to reach a given precision, we deterministically sweep through the trials and define the next simulated run-length as the run-length associated to the trial if it is successful and in the case of an unsuccessful trial as the sum of the associated run-length of the trial and the simulated run-length of the restarted algorithm as described above.

Note that the latter derandomized version to draw simulated run-lengths has the minor disadvantage that the number of samples :math:`N` is restricted to a multiple of the trials in the data set.

.. maybe we should indeed put a picture here



.. _sec:aRT:

Average Runtime
=====================

The average runtime (|aRT|) (introduced in [Price:1997]_ as
ENES and analyzed in [Auger:2005b]_ as success performance and previously called ERT in [HAN2009]_) is an estimate of the expected runtime of the restart algorithm given in Equation :eq:`RTrestart` that is used within the COCO framework. More precisely, the expected runtime of the restart algorithm (on a parametrized family of functions in order to reach a precision :math:`\epsilon`) writes

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
	\mathrm{aRT} & = & \mathrm{RT}_\mathrm{S} + \frac{1-p_{\mathrm{s}}}{p_{\mathrm{s}}} \,\mathrm{RT}_\mathrm{US} \\  & = & \frac{\sum_i \mathrm{RT}^{\rm us}_i + \sum_j \mathrm{RT}^{\rm us}_j }{\#\mathrm{succ}} \\
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

We display distributions of runtimes through empirical cumulative distribution functions (ECDF). Formally, let us consider a set of problems :math:`\mathcal{P}` and a collection of runtimes to solve those problems :math:`(\mathrm{RT}_{p,k})_{p \in \mathcal{P}, 1 \leq k \leq K}` where :math:`K` is the number of runtimes per problem. When the problem is not solved, the undefined runtime is considered as infinite in order to make the mathematical definition consistent. The ECDF that we display is then defined as


.. math::
	:nowrap:

	\begin{equation*}
	\mathrm{ECDF}(\alpha) = \frac{1}{|\mathcal{P}| K} \sum_{p \in \mathcal{P},k} \mathbf{1} \left\{ \log_{10}( \mathrm{RT}_{p,k} / n ) \leq \alpha \right\} \enspace.
	\end{equation*}

where we use :math:`\log(\infty)=\infty`.

The ECDF gives the *proportion of problems solved in less than a specified budget* which is read on the x-axis. For instance, we display in Figure :ref:`fig:ecdf`, the ECDF of the running times of the pure random search algorithm on the set of problems formed by the parametrized sphere function (first function of the single-objective ``bbob`` test suite) in dimension :math:`n=5` with 51 relative targets uniform on a log-scale between :math:`10^2` and :math:`10^{-8}` and :math:`K=10^3`. We can read in this plot for example that a little bit less than 20 percent of the problems were solved in less than :math:`5 \cdot 10^3 = 10^3 \cdot n` function evaluations.

Note that we consider **runtimes of the restart algorithm**, that is, we use the idea of simulated run-lengths of the restart algorithm as described above to generate :math:`K` runtimes from typically 10 or 15 instances per function and dimension. Hence, only when no instance is solved, we consider that the runtime is infinite.


.. Dimo/Anne: it will be nice to have a tutorial-like explanation of how an ECDF is constructed (like what we have on the introductory BBOB slides)



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

	We advocate **not to aggregate** over dimension as the dimension is
	typically an input parameter to the algorithm that can be
	exploited to run different types of algorithms on different dimensions.

	The COCO platform does not provide ECDF aggregated over dimension.


.. Best 2009 "Algorithm"
.. ---------------------
.. Anne: Might be moved somewhere else when we will have an other section
.. Anne: for all the graphs used within COCO
.. We often display the performance of the best 2009 "algorithm". For instance in Figure .. Figure :ref:`fig:ecdfall` the leftmostleft curve displays the performance of the best .. 2009 "algorithm".




.. todo::
	* ECDF and uniform pick of a problem
	* log aRT can be read on the ECDF graphs [requires some assumptions]
	* The Different Plots Provided by the COCO Platform
		* aRT Scaling Graphs
		  The aRT scaling graphs present the average running time to
		  reach a certain 			precision (relative target)
		  divided by the dimension versus the dimension. Hence an
		  horizontal line means a linear scaling with respect to the
		  dimension.
		* aRT Loss graphs
		* Best 2009: actually now I am puzzled on this Best 2009

	  algorithm (I know what is the aRT of the best 2009, but I have
	  doubts on how we display the ECDF of the best 2009



Acknowledgements
================
This work was supported by the grant ANR-12-MONU-0009 (NumBBO)
of the French National Research Agency.


References
==========

.. [Auger:2005b] A. Auger and N. Hansen. Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777–1784, 2005.
.. [TUS2016] T. Tušar, D. Brockhoff, N. Hansen, A. Auger (2016). 
  `COCO: The Bi-objective Black Box Optimization Benchmarking (bbob-biobj) 
  Test Suite`__, *ArXiv e-prints*, `arXiv:1604.00359`__.
.. __: http://numbbo.github.io/coco-doc/bbob-biobj/functions/
.. __: http://arxiv.org/abs/1604.00359

.. [HAN2016ex] N. Hansen, T. Tušar, A. Auger, D. Brockhoff, O. Mersmann (2016). 
  `COCO: The Experimental Procedure`__, *ArXiv e-prints*, `arXiv:1603.08776`__. 
.. __: http://numbbo.github.io/coco-doc/experimental-setup/
.. __: http://arxiv.org/abs/1603.08776

.. [HAN2009] Hansen, N., A. Auger, S. Finck R. and Ros (2009), Real-Parameter
	Black-Box Optimization Benchmarking 2009: Experimental Setup, *Inria
	Research Report* RR-6828 http://hal.inria.fr/inria-00362649/en
.. [HOO1998] H.H. Hoos and T. Stützle. Evaluating Las Vegas
   algorithms—pitfalls and remedies. In *Proceedings of the Fourteenth
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238–245, 1998.
.. [More:2009] Jorge J. Moré and Stefan M. Wild. Benchmarking
	Derivative-Free Optimization Algorithms, SIAM J. Optim., 20(1), 172–191, 2009.
.. [Price:1997] K. Price. Differential evolution vs. the functions of
   the second ICEO. In Proceedings of the IEEE International Congress on
   Evolutionary Computation, pages 153–157, 1997.
.. [Rios:2012] Luis Miguel Rios and Nikolaos V Sahinidis. Derivative-free optimization:
	A review of algorithms and comparison of software implementations.
	Journal of Global Optimization, 56(3):1247– 1293, 2013.
.. [Hooker:1995] J. N. Hooker Testing heuristics: We have it all wrong. In Journal of
    Heuristics, pages 33-42, 1995.
.. [STE1946] S.S. Stevens (1946).
    On the theory of scales of measurement. *Science* 103(2684), pp. 677-680.




.. old-bib [Auger:2005a] A Auger and N Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769–1776. IEEE Press, 2005.
.. old-bib
.. old-bib [Auger:2009] Anne Auger and Raymond Ros. Benchmarking the pure
   random search on the BBOB-2009 testbed. In Franz Rothlauf, editor, *GECCO
   (Companion)*, pages 2479–2484. ACM, 2009.
.. old-bib [Efron:1993] B. Efron and R. Tibshirani. *An introduction to the
   bootstrap.* Chapman & Hall/CRC, 1993.
.. old-bib [Harik:1999] G.R. Harik and F.G. Lobo. A parameter-less genetic
   algorithm. In *Proceedings of the Genetic and Evolutionary Computation
   Conference (GECCO)*, volume 1, pages 258–265. ACM, 1999.
