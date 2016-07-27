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
.. |I| replace:: :math:`I`
.. |i| replace:: :math:`i`
.. |f| replace:: :math:`f`
.. |t| replace:: :math:`t`
.. |p| replace:: :math:`p`
.. |p3| replace:: :math:`p^3`  
.. |p5| replace:: :math:`p^5`  
.. |x| replace:: :math:`x`
.. |y| replace:: :math:`y`
.. |N| replace:: :math:`N`
.. |n| replace:: :math:`n`
.. |J| replace:: :math:`J`
.. |RTus| replace:: :math:`\mathrm{RT}^{\mathrm{us}}`
.. |RTs| replace:: :math:`\mathrm{RT}^{\mathrm{s}}`
.. |calP| replace:: :math:`\mathcal{P}`
.. |calP.| replace:: :math:`\mathcal{P}.`
.. |thetai| replace:: :math:`\theta_i`
.. |ftheta| replace::  :math:`f_{\theta}`


.. the next two lines are necessary in LaTeX. They will be automatically 
  replaced to put away the \chapter level as ??? and let the "current" level
  become \section. 

.. CHAPTERTITLE
.. CHAPTERUNDERLINE

.. raw:: html

   <I>To cite or access this document as pdf:</I><BR>
   N. Hansen, A. Auger, D. Brockhoff,  D. Tušar, and T. Tušar (2016). 
   <A HREF="http://arxiv.org/abs/1605.03560">
   COCO: Performance Assessment. <I>ArXiv e-prints</I>,
   arXiv:1605.03560</A>.

.. raw:: latex

  % \tableofcontents is automatic with sphinx and moved behind the abstract 
  % by swap...py
  
  \begin{abstract}

We present an any-time performance assessment for benchmarking numerical
optimization algorithms in a black-box scenario, applied within the COCO_ benchmarking platform. 
The performance assessment is based on *runtimes* measured in number of objective function evaluations to reach one or several quality indicator target values.
We argue that runtime is the only available measure with a generic, meaningful, and quantitative interpretation.
We discuss the choice of the target values, runlength-based targets, and the aggregation of results by using simulated restarts, averages, and empirical distribution functions. 

.. raw:: latex

  \end{abstract}
  \newpage


Introduction
=============

.. budget-free

We present ideas and concepts for performance assessment when benchmarking numerical optimization algorithms in a black-box scenario. 
Going beyond a simple ranking of algorithms, we aim
to provide a *quantitative* and *meaningful* performance assessment, which
allows for conclusions like *algorithm A is seven times faster than algorithm
B* in solving a given problem or in solving problems with certain
characteristics. 
For this end, we record algorithm *runtimes, measured in
number of function evaluations* to reach predefined target values during the
algorithm run.

Runtimes represent the cost of optimization. Apart from a short, exploratory
experiment [#]_, we do not measure the algorithm cost in CPU or wall-clock time.
See for example [HOO1995]_ for a discussion on shortcomings and
unfortunate consequences of benchmarking based on CPU time.

In the COCO_ platform [HAN2016co]_, we display average runtimes (|aRT|, see Section `Averaging Runtime`_)
and the empirical distribution function of runtimes (ECDF, see Section `Empirical Distribution Functions`_). 
When displaying runtime distributions, we consider the aggregation over 
target values and over subclasses of problems, or all problems. 


.. We do not aggregate over dimension, because the dimension of the problem can be used to decide a priori which algorithm (or algorithm variant, or parameter setting) to use.

.. [#] The COCO_ platform provides a CPU timing experiment to get a rough estimate of the time complexity of the algorithm [HAN2016ex]_.


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
   
In the COCO_ framework in general, a **problem**, or problem instance triplet, |p3|, is defined by the search space dimension |n|, the objective function |f|, to be minimized, and its instance parameters |thetai| for the instance |i|.
More concisely, we consider a set of parametrized benchmark functions
:math:`f_\theta: \mathbb{R}^n \to \mathbb{R}^m, \theta \in \Theta` and the
corresponding problems :math:`p^3 = p(n, f_\theta, \theta_i)`. 
Different instances vary by having different shifted optima, can use different rotations that are applied to the variables, have different optimal |f|-values, etc. [HAN2009fun]_.  
The instance notion is introduced to generate repetition while avoiding possible exploitation of artificial function properties (like location of the optimum in zero).
The separation of dimension and instance parameters in the notation serves as a hint to indicate that we never aggregate over dimension and always aggregate over all |thetai|-values. 

In the performance assessment setting, we associate to a problem instance
|p3| a quality indicator mapping and a target value, 
such that a problem becomes a quintuple |p5|.
Usually, the quality indicator remains the same for all problems, while we have
subsets of problems which only differ in their target value. 
 
 
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


On Performance Measures
=======================

Evaluating performance is necessarily based on performance *measures*, the
definition of which plays a crucial role for the evaluation. 
Here, we introduce a list of requirements a performance measure should satisfy in general, as well as in the context of black-box optimization specifically. 
In general, a performance measure should be

* quantitative, as opposed to a simple *ranking* of entrants (e.g., algorithms). 
  Ideally, the measure should be defined on a ratio scale (as opposed to an
  interval or ordinal=ranking scale) [STE1946]_, which allows to state that "entrant A
  is :math:`x` *times better* than entrant B". [#]_ 
* assuming a wide variation of values such that, for example, typical values do 
  not only range between 0.98 and 1.0, [#]_
* well interpretable, in particular by having meaning and semantics attached to 
  the measured numbers,
* relevant and meaningful with respect to the "real world",
* as simple and as comprehensible as possible.

.. Following [HAN2009ex]_, we advocate **performance measures** that are

.. Tea: Can we give some more explanation here?

In the context of black-box optimization, the **runtime** to reach a target value, measured in number of function evaluations, satisfies all requirements. 
Runtime is well-interpretable and meaningful with respect to the
real-world as it represents time needed to solve a problem. 
Measuring number of function evaluations avoids the shortcomings of CPU
measurements that depend on parameters like the programming language, coding
style, machine used to run the experiment, etc., that are difficult or
impractical to control.
If however algorithm internal computations dominate wall-clock time in a
practical application, comparative runtime results *in number of function
evaluations* can usually be adapted *a posteri* to reflect the practical
scenario. 
This holds also true for a speed up from parallelization.  

.. [#] A variable which lives on a ratio scale has a meaningful zero, 
   allows for division, and can be taken to the logarithm in a meaningful way. 
   See for example `Level of measurement on Wikipedia`__.

.. __: https://en.wikipedia.org/wiki/Level_of_measurement?oldid=478392481

.. [#] A transformation like :math:`x\mapsto\log(1-x)` could alleviate 
   the problem in this case, given it actually zooms in on *relevant* values.


.. _sec:verthori:

Quality Indicators
-------------------

At each evaluation count (time step) |t| of an algorithm which optimizes a problem instance |thetai| of the function |ftheta| in dimension |n|, we apply a quality indicator mapping. 
A quality indicator |I| maps the set of all solutions evaluated 
so far (or recommended [HAN2016ex]_) to a problem-dependent real value.
Then, a runtime measurement can be obtained from each of a (large) set of
problem instances :math:`p^5 = p(n, f_\theta, \theta_i, I, I^\mathrm{target,
\theta_i}_{f})`. 
The runtime on this problem instance is defined as the evaluation count 
when the quality indicator value drops below the target for the first time, otherwise runtime remains undefined. 

In the single-objective noiseless case, the quality indicator outputs
the best so far observed (i.e. minimal and feasible) function value. 

In the single-objective noisy case, the quality indicator returns the 1%-tile of
the function values of the last :math:`\lceil\ln(t + 3)^2 / 2\rceil` evaluated
(or recommended) solutions. [#]_

In the multi-objective case, the current quality indicator is based on a negative
hypervolume indicator of the set of evaluated solutions (more specifically, the
non-dominated archive) [BRO2016]_.

.. [#] This feature will only be available in the new implementation of the COCO_ framework.


Fixed-Budget versus Fixed-Target Approach
-----------------------------------------

Starting from the most basic convergence graphs which plot the evolution of a
quality indicator, to be minimized, against the number of function evaluations,
there are essentially only two ways to measure the performance.

fixed budget:
    We fix a maximal budget of function evaluations,
    and measure the reached quality indicator value. A fixed search
    budget can be pictured as drawing a *vertical* line in the figure 
    (blue line in Figure :ref:`fig:HorizontalvsVertical`).

fixed target:
    We fix a target quality value and measure the number of function
    evaluations, the *runtime*, to reach this target. A fixed target can be
    pictured as drawing a *horizontal* line in the figure (red line in Figure
    :ref:`fig:HorizontalvsVertical`).


.. _fig:HorizontalvsVertical:

.. figure:: fixed-budget-vs-fixed-target.*
   :align: center
   :width: 70%

   **Fixed-Budget versus Fixed-Target**
   
   Illustration of fixed-budget view (vertical cuts) and fixed-target view
   (horizontal cuts). Black lines depict the best quality indicator value
   plotted versus number of function evaluations. Stars depict the 
   measurements used for the performance assessment. 

.. .. TODO: the line annotations in the figure should not be vertical/horizontal but budget/target. 

.. It is often argued that the fixed-cost approach is close to what is needed for
   real world applications where the total number of function evaluations is
   limited. On the other hand, also a minimum target requirement needs to be
   achieved in real world applications, for example, getting (noticeably) better
   than the currently available best solution or than a competitor.

For the performance assessment of algorithms, the fixed-target approach is superior
to the fixed-budget approach since it gives *quantitative and interpretable*
results.

* In the fixed-budget approach (vertical cut) the observation that 
  Algorithm A reaches a quality indicator value that is, say, two
  times smaller than the one reached by Algorithm B has in general no
  interpretable meaning, mainly because there is no *a priori* way to determine
  *how much* more difficult it is to reach an indicator value that is two times
  smaller.
  This usually depends on the function, the definition of the 
  quality indicator and even the specific indicator values compared.
  The assessed measure, quality indicator, exhibits in general only an 
  ordinal (rank) scale. 

* The fixed-target approach (horizontal cut)
  *measures the runtime* to
  reach a target quality value. The measurement allows conclusions of the
  type: Algorithm A is two (or ten, or a hundred) times faster than Algorithm B
  in solving this problem. 
  The assessed measure, runtime, exhibits a ratio scale. 
  The choice of the target value is however instrumental as it determines the 
  difficulty and often the characteristic of the problem to be solved. 

Furthermore, for algorithms that are invariant under certain transformations
of the function value (for example under order-preserving transformations, as
comparison-based algorithms like DE, ES, PSO [AUG2009]_), fixed-target measures are
invariant under these transformations if the target values are transformed accordingly. That is, only the horizontal line needs to be moved. Fixed-budget measures require the transformation of all resulting measurements individually.


Missing Values
---------------
Investigating the Figure :ref:`fig:HorizontalvsVertical` more carefully, we find that not all graphs intersect with either the vertical or the horizontal line. 
On the one hand, if the fixed budget is too large, the algorithm might solve the function before the budget is exceeded. [#]_ 
The algorithm performs better than the measurement is able to reflect, which can lead to serious misinterpretations. 
The remedy is to define a *final* target value and measure instead the runtime if the final target is hit. [#]_

On the other hand, if the fixed target is too difficult, the algorithm may never hit the target under the given experimental conditions. [#]_ 
The algorithm performs worse than the experiment is able to reflect, while we still get a lower bound for this missing runtime instance. 
A possible remedy is to run the algorithm longer. 
Another possible remedy is to use the final quality indicator value as measurement. 
This measurement however should only be interpreted as ranking result, defeating the original objective. 
A third (impartial) remedy is to record the overall number of function evaluations of this run and apply simulated restarts, see below.  

.. [#] Even in continuous domain, from the view point of benchmarking, 
       or application in the real world, or numerical precision, the set of
       solutions (or of solution sets) that indisputably solve the problem has a
       volume larger than zero. 
       
.. [#] This is also advisable because declaring an algorithm better
       when it reaches, say, :math:`\mathsf{const} + 10^{-30}` instead of
       :math:`\mathsf{const} + 10^{-10}`, is more often than not unjustified.
       The former result may only indicate the lack of practical
       termination conditions. 

.. [#] However, under mildly randomized conditions, for example with a randomized initial solution, the restarted algorithm reaches any attainable target with probability one. The time needed can of course well be beyond any reasonable practical limitations. 


Target Value Setting
---------------------

.. |DI| replace:: :math:`\Delta I`

We use two different ways to defined target values. The first method is simpler but relies more heavily on properties in the function definition. The second method defines comparable targets over the *entire* benchmark suite, but relies on a reference data set. 

Fixed-Spaced Target Values
++++++++++++++++++++++++++++++++

First, we define for each problem instance :math:`p^3 = (n, f_\theta, \theta_i)` 
a *reference* quality indicator value, :math:`I^{\rm ref, \theta_i}`. 
In the single-objective case this is currently the optimal function value. 
In the multi-objective case this is currently the hypervolume indicator of an
approximation of the Pareto front [BRO2016]_. 
Based on this reference value and a set of target *precision* values, which are
independent of the instance |thetai|, we define a target value

.. math::

    I^{\rm target,\theta_i} = I^{\rm ref,\theta_i} + \Delta I \enspace

for each precision |DI|, giving rise to the product set of all problems :math:`p^3` and all precision values |DI|. The |DI|-values are usually chosen to be equally log-spaced, see also below. 


Runlength-based Target Values
++++++++++++++++++++++++++++++++
.. In addition to the fixed-budget and fixed-target approaches, there is an
  intermediate approach, combining the ideas of *measuring runtime* (to get
  meaningful measurements) and *fixing budgets* (of our interest). The 
  basic idea
  is the following.

Runlength-based target values are a novel way to define the target values based
on a reference data set. Like for *performance profiles* [DOL2002]_, the
resulting empirical distribution can be interpreted *relative to a reference
algorithm or a set of reference algorithms*. 
Unlike for performance profiles, the resulting empirical distribution *is* a
data profile [MOR2009]_ reflecting the true (opposed to relative) difficulty of the respective problems for the respective algorithm. 

We assume to have given a reference data set with recorded runtimes to reach a
prescribed, usually large set of quality indicator target values [#]_ as in the
fixed-target approach described above. 
The reference data serve as a baseline upon which the runlength-based targets are  computed. 
To simplify wordings we assume w.l.o.g. that a single reference *algorithm* has generated this data set. 

Now we choose a set of increasing reference *budgets*. To each budget, starting with the smallest, we associate the easiest (largest) target for which (i) the average runtime (taken over all respective |thetai| instances, |aRT|, see below) of the reference algorithm *exceeds* the budget and (ii, optionally) that had not been chosen for a smaller budget before. If such target does not exist, we take the final (smallest) target. 

Like this, an algorithm that reaches a target within the associated budget is better than the reference algorithm on this problem.
 
Runlength-based targets are used in COCO_ for the single-objective expensive optimization scenario. 
The artificial best algorithm of BBOB-2009 (see below) is used as reference algorithm with either the five budgets of :math:`0.5n`, :math:`1.2n`, :math:`3n`, :math:`10n`, and :math:`50n` function evaluations, where :math:`n` is the problem
dimension, or with 31 targets evenly space on the log scale between :math:`0.5n` and :math:`50n` and without the optional constraint from (ii) above. In the latter case, the empirical distribution function of the runtimes of the reference algorithm shown in a ``semilogx`` plot approximately resembles a diagonal straight line between the above two reference budgets. 

Runlength-based targets have the **advantage** to make the target value setting less
dependent on the expertise of a human designer, because only the reference
*budgets* have to be chosen a priori. Reference budgets, as runtimes, are
intuitively meaningful quantities, on which it is comparatively easy to decide
upon. 
Runlength-based targets have the **disadvantage** to depend on the choice of a reference data set, that is, they depend on a set of reference algorithms. 


.. [#] By default, the ratio between two neighboring |DI| target precision values 
   is :math:`10^{0.2}` and the largest |DI| value is (dynamically) chosen such 
   that the first evaluation of the worst algorithm hits the target. 

.. Niko: TODO: simulated runlength -> simulated runtime


Runtime Computation    
===========================

.. Niko: TODO: change |p5| to p4 and say that I is assumed? 

.. In order to display quantitative measurements, we have seen in the previous section that we should start from the collection of runtimes for different target values. 

In the performance assessment context of COCO_, a problem instance can be
defined by the quintuple :math:`p^5 = p(n, f_\theta, \theta_i, I, I^{{\rm
target}, \theta_i})`, consisting of search space dimension, function,
instantiation parameters, quality indicator mapping, and quality indicator
target value. 
From the definition of |p|, we can generate a set of problems |calP| by varying one or several of the variables. We never vary dimension |n| and always vary instances |thetai| for generating |calP.| 
For each benchmarked algorithm, a single runtime is measured on each problem instance |p5|.

From a *single run* of the algorithm on the problem instance triple
:math:`p^3 = p(n, f_\theta, \theta_i)`, we obtain a runtime measurement for *each* corresponding problem quintuple |p5| which agrees in its first three variables with |p3|.
More specifically, we measure one runtime for each target value which has been reached in this run, or equivalently, for each target precision. 
This also reflects the anytime aspect of the performance evaluation in a single run. 

Formally, the runtime :math:`\mathrm{RT}^{\rm s}(p)` is a random variable that represents the number of function evaluations needed to reach the quality indicator target value for the first time. 
A run or trial that reached the target value is called *successful*. [#]_
For *unsuccessful trials*, the runtime is not defined, but the overall number of function evaluations in the given trial is a random variable denoted by :math:`\mathrm{RT}^{\rm us}(p)`. For a single run, the value of :math:`\mathrm{RT}^{\rm us}(p)` is the same for all failed targets. 

We consider the conceptual **restart algorithm**. 
Given an algorithm has a strictly positive probability |ps| to solve a 
problem, independent restarts of the algorithm solve the problem with
probability one and exhibit the runtime

.. |RTforDI| replace:: :math:`\mathbf{RT}(n,f_\theta,\Delta I)`

.. math::
    :nowrap:
    :label: RTrestart
    
    \begin{equation*}%%remove*%%
    \label{index-RTrestart}  
      % ":eq:`RTrestart`" becomes "\eqref{index-RTrestart}" in the LaTeX
    \mathbf{RT}(n, f_\theta, \Delta I) = \sum_{j=1}^{J} \mathrm{RT}^{\rm us}_j(n,f_\theta,\Delta I) + \mathrm{RT}^{\rm s}(n,f_\theta,\Delta I)
    \enspace,
    \end{equation*}%%remove*%%

where :math:`J \sim \mathrm{BN}(1, 1 - p_{\rm s})` is a random variable with negative binomial distribution that models the number of unsuccessful runs
until one success is observed and :math:`\mathrm{RT}^{\rm us}_j` are independent
random variables corresponding to the evaluations in unsuccessful trials
[AUG2005]_. 
If the probability of success is one, :math:`J` equals zero with probability one and the restart algorithm coincides with the original algorithm.

Generally, the above equation for |RTforDI| expresses the runtime from repeated independent runs on the same problem instance (while the instance :math:`\theta_i` is not given explicitly). For the performance evaluation in the COCO_ framework, we apply the equation to runs on different instances :math:`\theta_i`, however instances from the same function, with the same dimension and the same target precision. 

.. [#] The notion of success is directly linked to a target value. A run can be successful with respect to some target values (some problems) and unsuccessful with respect to others. Success sometimes refers to the final, most difficult (smallest) target value, which implies success for all other targets in this run. 


Runs on Different Instances
-----------------------------------------------------------------------
.. The performance assessment in COCO_ heavily relies on the conceptual restart algorithm. 
.. However, we collect at most one single runtime per problem while more data points are needed to display significant data. 

Different instantiations of the parametrized functions |ftheta| are a natural way to represent randomized repetitions. 
For example, different instances implement random translations of the search space and hence a translation of the optimum [HAN2009fun]_. 
Randomized restarts on the other hand can be conducted from different initial points. 
For translation invariant algorithms both mechanisms are equivalent and can be mutually exchanged. 

Thus, we interpret runs performed on different instances :math:`\theta_1, \ldots, \theta_K` as repetitions of the same problem. 
Thereby we assume that instances of the same parametrized function |ftheta| are 
similar to each other, and more specifically that they exhibit the same runtime
distribution for each given |DI|. 

.. Runtimes collected for the different instances :math:`\theta_1, \ldots, \theta_K` of the same parametrized function :math:`f_\theta` and with respective targets associated to the same target precision :math:`\Delta I` (see above) are thus assumed independent and identically distributed. 

We hence have for each parametrized problem a set of :math:`K\approx15` independent runs, which are used to compute artificial runtimes of the conceptual restart algorithm. 

.. .. Note:: Considering the runtime of the restart algorithm allows to compare
   quantitatively the two different scenarios where

	* an algorithm converges often but relatively slowly
	* an algorithm converges less often, but whenever it converges, it is with a fast convergence rate.

.. we write in the end the runtime of a restart algorithm of a
   parametrized family of function in order to reach a relative target
   :math:`\Delta I` as

.. |K| replace:: :math:`K`

Simulated Restarts and Runtimes
-----------------------------------

.. Niko: I'd like to reserve the notion of runtime to successful (simulated) runs. 

.. simulated runtime instances of the virtually restarted algorithm

The runtime of the conceptual restart algorithm as given in :eq:`RTrestart` is the basis for displaying performance within COCO_. 
We use the |K| different runs on the same function and dimension to simulate virtual restarts with a fixed target precision. 
We assume to have at least one successful run---otherwise, the runtime remains undefined, because the virtual procedure would never stop. 
Then, we construct artificial, simulated runs from the available empirical data:
we repeatedly pick, uniformly at random with replacement, one of the |K| trials until we encounter a successful trial. 
This procedure simulates a single sample of the virtually restarted algorithm from the given data. 
As given in :eq:`RTrestart` as |RTforDI|, the measured, simulated runtime is the sum of the number of function evaluations from the unsuccessful trials added to the runtime of the last and successful trial. [#]_

.. |q| replace:: :math:`q`

.. [#] In other words, we apply :eq:`RTrestart` such that |RTs| is uniformly distributed over all measured runtimes from successful instances |thetai|, |RTus| is uniformly distributed over all evaluations seen in unsuccessful instances |thetai|, and |J| has a negative binomial distribution :math:`\mathrm{BN}(1, q)`, where |q| is the number of unsuccessful instance divided by the number of all instances.


Bootstrapping Runtimes
++++++++++++++++++++++++

In practice, we repeat the above procedure between a hundred and a few thousand times, thereby sampling :math:`N` simulated runtimes from the same underlying distribution, 
resembling the bootstrap algorithm [EFR1994]_. 
To reduce the variance in this procedure, when desired, the first trial in each sample is picked deterministically instead of randomly as the :math:`1 + (N~\mathrm{mod}~K)`-th trial from the data. [#]_
Picking the first trial data as specific instance |thetai| could also be
interpreted as applying simulated restarts to this specific instance rather than
to the entire set of problems :math:`\mathcal{P} = \{p(n, f_\theta, \theta_i, \Delta I) \;|\;
i=1,\dots,K\}`. 

.. Niko: average runtime is not based on simulated restarts, but computed directly...considering the average runtime (Section :ref:`sec:aRT`) or the distribution by displaying empirical cumulative distribution functions (Section :ref:`sec:ECDF`).

.. [#] The variance reducing effect is best exposed in the case where all runs are successful and :math:`N = K`, in which case each data is picked exactly once. 
   This example also suggests to apply a random permutation of the data before to simulate virtually restarted runs. 
   This technique is not suited when we want to estimate the deviation of the given data set from the original underlying distribution [EFR1994]_.

Rationales and Limitations
+++++++++++++++++++++++++++

Simulated restarts aggregate some of the available data and thereby extend their range of interpretation. 

* Simulated restarts allow in particular to compare algorithms with a wide range of different success probabilities by a single performance measure. [#]_ Conducting restarts is also valuable approach when addressing a difficult optimization problem in practice. 

* Simulated restarts rely on the assumption that the runtime distribution for each instance is the same. If this is not the case, they still provide a reasonable performance measure, however with less of a meaningful interpretation for the result. 

* The runtime of simulated restarts may heavily depend on **termination conditions** applied in the benchmarked algorithm, due to the evaluations spent in unsuccessful trials, compare :eq:`RTrestart`. This can be interpreted as disadvantage, when termination is considered as a trivial detail in the implementation---or as an advantage, when termination is considered a relevant component in the practical application of numerical optimization algorithms. 

* The maximal number of evaluations for which simulated runtimes are meaningful 
  and representative depends on the experimental conditions. If all runs are successful, no restarts are simulated and all runtimes are meaningful. If all runs terminated due to standard termination conditions in the used algorithm, simulated restarts reflect the original algorithm. However, if a maximal budget is imposed for the purpose of benchmarking, simulated restarts do not necessarily reflect the real performance. In this case and if the success probability drops below 1/2, the result is likely to give a too pessimistic viewpoint at or beyond the chosen maximal budget. See [HAN2016ex]_ for a more in depth discussion on how to setup restarts in the experiments. 

* If only few or no successes have been observed, we can see large effects without statistical significance. Namely, 4/15 successes are not statistically significant against 0/15 successes on a 5%-level. 

.. scipy.stats.chi2_contingency([[0, 15], [5, 10]]) -> 0.05004
   scipy.stats.fisher_exact([[0, 15], [5, 10]]) -> 0.0420
   ranksumtest(range(15), list(arange(2.5, 12)) + 5 * [100]) -> 0.94

.. [#] The range of success probabilities is bounded by the number of instances to roughly :math:`2/|K|.`

.. _sec:aRT:

Averaging Runtime
==================

The average runtime (|aRT|), introduced in [PRI1997]_ as ENES and
analyzed in [AUG2005]_ as success performance and referred to as 
ERT in [HAN2009ex]_, estimates the expected runtime of the restart
algorithm given in :eq:`RTrestart`. Generally, the set of trials
over which the average is taken is generated by varying |thetai| only. 

We compute the |aRT| from a set of trials as the sum of all evaluations in unsuccessful trials plus the sum of the runtimes in all successful trials, both divided by the number of successful trials. 


Motivation
-----------

The expected runtime of the restart algorithm writes [AUG2005]_

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbb{E}(\mathbf{RT}) & =
    & \mathbb{E}(\mathrm{RT}^{\rm s})  + \frac{1-p_\mathrm{s}}{p_\mathrm{s}}
      \mathbb{E}(\mathrm{RT}^{\rm us})
    \enspace,
    \end{eqnarray*}

where :math:`p_\mathrm{s} > 0` is the probability of success of the algorithm and notations from above are used.

.. |RTsi| replace:: :math:`\mathrm{RT}^{\rm s}_i`
.. |RTusj| replace:: :math:`\mathrm{RT}^{\rm us}_j`

Given a data set with :math:`n_\mathrm{s}\ge1` successful runs with runtimes |RTsi|, and :math:`n_\mathrm{us}` unsuccessful runs with |RTusj| evaluations, the average runtime reads

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathrm{aRT} 
    & = & 
    \frac{1}{n_\mathrm{s}} \sum_i \mathrm{RT}^{\rm s}_i + 
    \frac{1-p_{\mathrm{s}}}{p_{\mathrm{s}}}\,
    \frac{1}{n_\mathrm{us}} \sum_j \mathrm{RT}^{\rm us}_j
    \\ 
    & = & 
    \frac{\sum_i \mathrm{RT}^{\rm s}_i + \sum_j \mathrm{RT}^{\rm us}_j }{n_\mathrm{s}} 
    \\
    & = & 
    \frac{\#\mathrm{FEs}}{n_\mathrm{s}}
    \end{eqnarray*}

.. |nbsucc| replace:: :math:`n_\mathrm{s}`
.. |Ts| replace:: :math:`\mathrm{RT}_\mathrm{S}`
.. |Tus| replace:: :math:`\mathrm{RT}_\mathrm{US}`
.. |ps| replace:: :math:`p_{\mathrm{s}}`

where |ps| is the fraction of successful trials, :math:`0/0` is
understood as zero and :math:`\#\mathrm{FEs}` is the number of function
evaluations conducted in all trials before to reach the given target precision.

Rationale and Limitations
--------------------------
The average runtime, |aRT|, is taken over different instances of the same function, dimension, and target precision, as these instances are interpreted as repetitions. 
Taking the average is meaningful only if each instance obeys a similar distribution without heavy tail. 
If one instance is considerably harder than the others, the average is dominated by this instance. 
For this reason we do not average runtimes from different functions or different target precisions, which however could be done if the logarithm is taken first (geometric average). 
Plotting the |aRT| divided by dimension against dimension in a log-log plot is the recommended way to investigate the scaling behavior of an algorithm. 

.. _sec:ECDF:

Empirical Distribution Functions
===========================================

We display a set of simulated runtimes with the empirical cumulative
distribution function (ECDF), AKA empirical distribution function. 
Informally, the ECDF displays the *proportion of problems solved within a
specified budget*, where the budget is given on the |x|-axis. 
More formally, an ECDF gives for each |x|-value the fraction of runtimes which do not exceed |x|, where missing runtime values are counted in the denominator of the fraction.

Rationale, Interpretation and Limitations
------------------------------------------
Empirical cumulative distribution functions are a universal way to display *unlabeled* data in a condensed way without losing information. 
They allow unconstrained aggregation, because each data point remains separately displayed, and they remain entirely meaningful under transformation of the data (e.g. taking the logarithm). 

* The empirical distribution function from a set of problems where only the target value varies, recovers an upside-down convergence graph with the resolution steps defined by the targets [HAN2010]_.

* When runs from several instances are aggregated, the association to the single run is lost, as is the association to the function when aggregating over several functions. This is particularly problematic for data from different dimensions, because dimension can be used as decision parameter for algorithm selection. Therefore, we do not aggregate over dimension. 

* The empirical distribution function can be read in two distinct ways.

  |x|-axis as independent variable: 
    for any budget (|x|-value), we see the fraction of problems solved within
    the budget as |y|-value, where the limit value to the right is the fraction
    of solved problems with the maximal budget. 
    The resulting value satisfies above listed requirements on a 
    measurement except that it does not assume a wide range of values, because
    it is bounded from above.  
  |y|-axis as independent variable: 
    for any fraction of easiest problems
    (|y|-value), we see the maximal runtime observed on these problems on the
    |x|-axis. When plotted in ``semilogx``, a horizontal shift indicates a runtime
    difference by the respective factor, quantifiable, e.g., as "five times
    faster". The area below the |y|-value and to the left of the graph reflects
    the geometric runtime average on this subset of problems, the smaller the
    better. 

Relation to Previous Work
--------------------------
Empirical distribution functions over runtimes of optimization algorithms are also known as *data profiles* [MOR2009]_. 
They are widely used for aggregating results from different functions and different dimensions to reach a single target precision [RIO2012]_. 
In the COCO_ framework, we do not aggregation over dimension but aggregate often over a wide range of target precision values. 

.. 
    Formal Definition
    -------------------
    Formally, let us consider a set of problems :math:`\mathcal{P}` 
    and |N| simulated runtimes on each problem. 
    When the problem is not solved, the undefined runtime is considered as infinite. 
    The ECDF is defined as

    .. math::
        :nowrap:

        \begin{equation*}
        \mathrm{ECDF}(t) = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \frac{1}{N}\sum_{i=1}^N \mathbf{1} \left\{ \mathbf{RT}(p) / n  \leq t \right\} \enspace,
        \end{equation*}

    counting the number of runtimes which do not exceed the time :math:`t\times n`, divided by the number of all simulated runs. 
    The ECDF is displayed in a semi-log (lin-log, semi-logx) plot. 

Examples
----------

We display in Figure :ref:`fig:ecdf` the ECDF of the (simulated) runtimes of
the pure random search algorithm on the set of problems formed by 15 instances of the sphere function (first function of the single-objective ``bbob`` test
suite) in dimension :math:`n=5` each with 51 target precisions between :math:`10^2` and :math:`10^{-8}` uniform on a log-scale and 1000 bootstraps. 

.. Dimo/Anne: it will be nice to have a tutorial-like explanation of how an ECDF is constructed (like what we have on the introductory BBOB slides)


.. _fig:ecdf:

.. figure:: pics/plots-RS-2009-bbob/pprldmany_f001_05D.*
   :width: 70%
   :align: center

   ECDF

   Illustration of empirical (cumulative) distribution function (ECDF) of
   runtimes on the sphere function using 51 relative targets uniform on a log
   scale between :math:`10^2` and :math:`10^{-8}`. The runtimes displayed
   correspond to the pure random search algorithm in dimension 5. The 
   (big) cross is the median number of evaluations in unsuccessful runs. 


We can see in this plot, for example, that almost 20 percent of the problems 
were solved within :math:`10^3 \cdot n = 5 \cdot 10^3` function evaluations. 
Runtimes to the right of the cross at :math:`10^6` have at least one unsuccessful run. 
This can be concluded, because with pure random search each unsuccessful run exploits the maximum budget.
The small dot beyond :math:`x=10^7` depicts the overall fraction of all successfully solved functions-target pairs, i.e., the fraction of :math:`(f_\theta, \Delta I)` pairs for which at least one trial (one :math:`\theta_i` instantiation) was successful. 

We usually divide the set of all (parametrized) benchmark
functions into subgroups sharing similar properties (for instance
separability, unimodality, ...) and display ECDFs which aggregate the
problems induced by these functions and all targets. 
Figure :ref:`fig:ecdfgroup` shows the result of random search on the first 
five functions of the `bbob` testsuite, separate (left) and aggregated (right).

.. _fig:ecdfgroup:

.. figure:: pics/plots-RS-2009-bbob/gr_separ_05D_05D_separ-combined.*
   :width: 100%
   :align: center

   ECDF for a subgroup of functions

   **Left:** ECDF of the runtime of the pure random search algorithm for
   functions f1, f2, f3, f4 and f5 that constitute the group of
   separable functions for the ``bbob`` testsuite over 51 target values.
   **Right:** Aggregated ECDF of the same data, that is, all functions 
   in one graph.


Finally, we also naturally aggregate over all functions of the benchmark and
hence obtain one single ECDF per algorithm per dimension. 
In Figure :ref:`fig:ecdfall`, the ECDF of different algorithms are displayed in
a single plot. 

.. _fig:ecdfall:

.. figure:: pics/plots-all2009/pprldmany_noiselessall-5and20D.*
   :width: 100%
   :align: center

   ECDF over all functions and all targets

   ECDF of several algorithms benchmarked during the BBOB 2009 workshop
   in dimension 5 (left) and in dimension 20 (right) when aggregating over all functions of the ``bbob`` suite.

The thick maroon line with diamond markers annotated as "best 2009" corresponds to the **artificial best 2009 algorithm**: for
each set of problems with the same function, dimension and target precision, we select the algorithm with the smallest |aRT| from the `BBOB-2009 workshop`__ and use for these problems the data from the selected algorithm. 
The algorithm is artificial because we may use for the same problem and dimension but for different target values the runtime results from different algorithms. [#]_

We observe that the artificial best 2009 algorithm is about two to three time faster than the left envelope of all single algorithms and solves all problems in about :math:`10^7\, n` function evaluations.  

.. __: http://coco.gforge.inria.fr/doku.php?id=bbob-2009
 
.. [#] The best 2009 curve is not guaranteed to be an upper
       left envelope of the ECDF of all algorithms from which it is
       constructed, that is, the ECDF of an algorithm from BBOB-2009 can
       cross the best 2009 curve. This may typically happen if an algorithm
       has for the most easy problems a large runtime variation and its |aRT| is 
       not the best but the short runtimes
       show up to the left of the best 2009 graph.

..  todo
..	* ECDF and uniform pick of a problem
..	* log aRT can be read on the ECDF graphs [requires some assumptions]
..	* The Different Plots Provided by the COCO Platform
..		* aRT Scaling Graphs
..		  The aRT scaling graphs present the average running time to
..		  reach a certain 			precision (relative target)
..		  divided by the dimension versus the dimension. Hence an
..		  horizontal line means a linear scaling with respect to the
..		  dimension.
..		* aRT Loss graphs
..      * scatter plots


.. raw:: html
    
    <H2>Acknowledgments</H2>

.. raw:: latex

    \section*{Acknowledgments}

This work was supported by the grant ANR-12-MONU-0009 (NumBBO)
of the French National Research Agency.

The authors would like to thank Raymond Ros, Steffen Finck, Marc Schoenauer and  
Petr Posik for their many invaluable contributions to this work. 


.. ############################# References ##################################
.. raw:: html
    
    <H2>References</H2>


.. [AUG2005] A. Auger and N. Hansen (2005). Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777–1784.
   
.. [AUG2009] A. Auger, N. Hansen, J.M. Perez Zerpa, R. Ros and M. Schoenauer (2009). 
   Experimental comparisons of derivative free optimization algorithms (invited talk). 
   In J. Vahrenhold (Ed.), *Experimental algorithms: 8th international symposium, SEA 2009*, 
   Dortmund, LNCS 5526, pages 3-15, Springer. 

.. [BRO2016] D. Brockhoff, T. Tušar, D. Tušar, T. Wagner, N. Hansen, 
   A. Auger (2016). `Biobjective Performance Assessment with the COCO 
   Platform`__. *ArXiv e-prints*, `arXiv:1605.01746`__.
__ http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment
__ http://arxiv.org/abs/1605.01746

.. [DOL2002] E.D. Dolan, J.J. Moré (2002). Benchmarking optimization software 
   with performance profiles. *Mathematical Programming* 91.2, 201-213. 

.. [EFR1994] B. Efron and R. Tibshirani (1994). *An introduction to the
   bootstrap*. CRC Press.

.. [HAN2009ex] N. Hansen, A. Auger, S. Finck, and R. Ros (2009). Real-Parameter
   Black-Box Optimization Benchmarking 2009: Experimental Setup, 
   `Research Report RR-6828`__, Inria.
.. __: http://hal.inria.fr/inria-00362649/en

.. [HAN2016co] N. Hansen, A. Auger, O. Mersmann, T. Tušar, D. Brockhoff (2016).
   `COCO: A Platform for Comparing Continuous Optimizers in a Black-Box 
   Setting`__. *ArXiv e-prints*, `arXiv:1603:08785`__.
__ http://numbbo.github.io/coco-doc/
__ http://arxiv.org/abs/1603.08785

.. [HAN2010] N. Hansen, A. Auger, R. Ros, S. Finck, and P. Posik (2010). 
   Comparing Results of 31 Algorithms from the Black-Box Optimization 
   Benchmarking BBOB-2009. In *Workshop Proceedings of the GECCO Genetic and 
   Evolutionary Computation Conference 2010*, ACM, pp. 1689-1696

.. [HAN2009fun] N. Hansen, S. Finck, R. Ros, and A. Auger (2009). 
   Real-parameter black-box optimization benchmarking 2009: Noiseless
   functions definitions. `Research Report RR-6829`__, Inria, updated
   February 2010.
__ https://hal.inria.fr/inria-00362633

.. [HAN2016ex] N. Hansen, T. Tušar, A. Auger, D. Brockhoff, O. Mersmann (2016). 
  `COCO: The Experimental Procedure`__, *ArXiv e-prints*, `arXiv:1603.08776`__. 
__ http://numbbo.github.io/coco-doc/experimental-setup/
__ http://arxiv.org/abs/1603.08776

.. [HOO1995] J.N. Hooker (1995). Testing heuristics: We have it all wrong. 
   *Journal of Heuristics*, 1(1), pages 33-42.
   
.. [HOO1998] H.H. Hoos and T. Stützle. Evaluating Las Vegas
   algorithms—pitfalls and remedies. In *Proceedings of the Fourteenth
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238–245, 1998.

.. [MOR2009] J.J. Moré and S.M. Wild (2009). Benchmarking
   Derivative-Free Optimization Algorithms, *SIAM J. Optim.*, 20(1), 172–191.

.. [PRI1997] K. Price (1997). Differential evolution vs. the functions of
   the second ICEO. In *Proceedings of the IEEE International Congress on
   Evolutionary Computation*, pages 153–157.

.. [RIO2012] L.M. Rios and N.V. Sahinidis (2013). Derivative-free optimization:
    A review of algorithms and comparison of software implementations.
    *Journal of Global Optimization*, 56(3):1247– 1293.

.. [STE1946] S.S. Stevens (1946).
    On the theory of scales of measurement. *Science* 103(2684), pp. 677-680.
    
.. .. [TUS2016] T. Tušar, D. Brockhoff, N. Hansen, A. Auger (2016). 
  `COCO: The Bi-objective Black Box Optimization Benchmarking (bbob-biobj) 
  Test Suite`__, *ArXiv e-prints*, `arXiv:1604.00359`__.
.. .. __: http://numbbo.github.io/coco-doc/bbob-biobj/functions/
.. .. __: http://arxiv.org/abs/1604.00359


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
