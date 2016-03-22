#########################################################
Biobjective Performance Assessment with the COCO Platform
#########################################################

...
^^^

.. Here we put the abstract when using LaTeX, the \abstractinrst 
   command must be defined in the 'preamble' of latex_elements in source/conf.py,
   the text should be defined in `abstract` of conf.py. To flip abstract and table
   of contents, or update the table of contents, toggle the \generatetoc
   command in the 'preamble' accordingly. 

.. raw:: latex

    % \abstractinrst
    % \tableofcontents
    \newpage 

.. Contents:

.. .. toctree::
   :maxdepth: 2

.. |coco_problem_t| replace:: 
  ``coco_problem_t``
.. _coco_problem_t: http://numbbo.github.io/coco-doc/C/coco_8h.html#a408ba01b98c78bf5be3df36562d99478

.. _COCO: https://github.com/numbbo/coco
.. |Iref| replace:: :math:`I_\mathrm{ref}`

This document details the specificities when assessing the performance of
numerical black-box optimizers on multi-objective problems within the COCO_
platform and in particular on the biobjective test suite ``bbob-biobj``,
described in more detail in [bbob-biobj-functions-doc]_ .

The performance assessment in the COCO_ platform is invariably based on the
measurement of the *runtime* until a *quality indicator* reaches a predefined
*target value* [BBO2016perf]_. [#]_ 
On each problem, several target values are defined and for each target value
a runtime is measured (or no runtime value is available if the indicator does
not reach the target value). 
In the single-objective, noise-free case, the assessed quality indicator is, at 
each given time step, the function value of the best solution the algorithm has
obtained (evaluated or recommended, see [BBO2016ex]_) before or at this time
step. 

In the bi- and multi-objective case, the assessed quality indicator at the
given time step will be a hypervolume indicator computed from *all* solutions
obtained (evaluated or recommended) before or at this time step. 


.. todo:: REMARK: the performance assessment in itself is to the most part **not 
              relative** to the optimum or, more concisely, to an optimal indicator
              value. Conceptually, we should consider the target values as
              (i) absolute values and (ii) as variable input parameters for the 
              assessment. The choice of targets relative to the best possible
              indicator value is a useful heuristic, but no necessity. Only the *uniform* 
              choice of targets within the instances of a single problem poses a 
              significant challenge. This challenge is not necessarily 
              solved by knowing the best possible indicator value.

.. todo::   * perf assessement is relative - we face a problem: we do not have the optimum.
			* How do we deal with this problem? [ this should probably be a section]
				* estimate the optimum
				* but approximation, meant to change / be improved - therefore need to ensure compatibility
				* compatibility + easy re-estimation of the performance when the reference set is improved	
			* we do not have the optimum (except for f1)
 			* we estimate it (how: running some algorithms) and it is meant to be changed with time (improved with time)
 			* things are based on the archive of nondominated solutions
 			* we measure the hypervolume difference between the dynamic archive and this reference set.
			* negative hyp-vol diff values are expected (means the algorithm improves over the current reference set)
			* archive is improved over time, whenever we have a new point entering the archive we recompute and log the hyp-vol difference.
			
.. [#] As usual, time is considered as number of function evaluations and, 
  consequently, runtime is measured in number of function evaluations.

Definitions and Terminology
---------------------------
We remind in this section different definitions.

*function instance, problem*
 In the case of the bi-objective performance assessment within COCO_, a problem is a 4-tuple of
 
 * a *parameterized function* :math:`f_\theta: \mathbb{R}^D \to \mathbb{R}^2`,
 * its concrete parameter values :math:`\theta\in\Theta` determining its so-called
   *function instance* |i|,
 * the *problem dimension* :math:`D`, and
 * a *target value* :math:`f_{\rm target}` of an underlying quality indicator, see below.
 
 We call a problem *solved* by an optization algorithm if the algorithm
 reaches a quality indicator value at least as good as the associated target value.
 The number of function evaluations needed to surpass the target value for the first time
 is COCO_'s central performance measure. [coco-doc]_

*Pareto set* and *Pareto front*
 For a concrete function instance, i.e., a function :math:`f_\theta=(f_\alpha,f_\beta)` with
 given parameter value :math:`\theta` and dimension :math:`D`, the Pareto set is the set
 of all (Pareto-optimal) solutions for which no solutions in the search space
 :math:`\R^D` exist that
 have either an improved :math:`f_\alpha` or an improved :math:`f_\beta` value while the
 other objective function is at least as good ("solutions *non-dominated* by any other
 solution"). The image of the Pareto set in the *objective space* is called the Pareto front.
 
*ideal point*
 The ideal point (in objective space) is defined as the vector in objective space that
 contains the optimal function value for each objective *independently*, i.e. for the above
 concrete function instance, the ideal point is given by
 :math:`z_{\rm ideal}  = (\inf_{x\in \mathbb{R}^D} f_\alpha(x), \inf_{x\in \mathbb{R}^D} f_\beta(x))`.
 
*nadir point* 
 The nadir point (in objective space) consists in each objective of
 the worst value obtained by a Pareto-optimal solution. More precisely, if
 :math:`\mathcal{PO}` denotes the Pareto set, the nadir point satisfies
 :math:`z_{\rm nadir}  =  \left( \sup_{x \in \mathcal{PO}} f_\alpha(x),
 \sup_{x \in \mathcal{PO}} f_\beta(x)  \right)`.
  

Biobjective Performance Assessment in COCO: A Set-Indicator Value Replaces the Objective Function
=================================================================================================
The general concepts of how the COCO_ platform suggests to benchmark
multi-objective algorithms is the same than in the single-objective case: for
each optimization algorithm, we record the runtimes to reach given target
values for each problem in a given benchmark suite. A problem thereby
consists of a (vector-valued) objective function, its search space dimension,
and a concrete instantiation of it (see [coco-perf-assessment]_ ). 
For defining the runtime on such a problem, we consider a quality indicator
which is to be optimized (minimized). 
In the single-objective case, the quality indicator is the objective
function value. 

In the case of the ``bbob-biobj`` test suite, the quality indicator will be mostly a
negative hypervolume indicator of the *archive* :math:`A_t` of all non-dominated
solutions evaluated within the first :math:`t` function evaluations. In principal, other
quality indicators of the archive can be used as well.

To be more concrete, the indicator :math:`\IHV` used here is to be mininized and
a combination of the negative hypervolume indicator of the archive with the nadir
point as the hypervolume's reference point and the distance to the region of interest
:math:`[z_{\text{ideal}}, z_{\text{nadir}}]` after a normalization of the
objective space [#]_:

.. math::
    :nowrap:
	
	\begin{equation*}
	\IHV =  \left\{ \begin{array}{ll}     
	- \text{HV}(A_t, [z_{\text{ideal}}, z_{\text{nadir}}]) & \text{if $A_t$ dominates } z_{\text{nadir}}\\
 	dist(A_t, [z_{\text{ideal}}, z_{\text{nadir}}]) & \text{otherwise} 	
	\end{array} 	\right.\enspace .
	\end{equation*}
 
where

.. math::
    :nowrap:
	
    \begin{equation*}
    \text{HV}(A_t, z_{\text{ideal}}, z_{\text{nadir}}) = \text{VOL}\left( \bigcup_{a \in A_t} \left[\frac{f_\alpha(a)-z_{\text{ideal}, \alpha}}{z_{\text{nadir}, \alpha}-z_{\text{ideal}, \alpha}}, 1\right]\times\left[\frac{f_\beta(a)-z_{\text{ideal}, \beta}}{z_{\text{nadir}, \beta}-z_{\text{ideal}, \beta}}, 1\right]\right)
	\end{equation*}
   
is the (normalized) hypervolume of archive :math:`A_t` with respect to the nadir point :math:`(z_{\text{nadir}, \alpha}, z_{\text{nadir},\beta})` as reference point and where 

.. math::
    :nowrap:
	
    \begin{equation*}
	dist(A_t, [z_{\text{ideal}}, z_{\text{nadir}}]) = \inf_{a\in A_t, z\in [z_{\text{ideal}}, z_{\text{nadir}}]} dist\left(\frac{f(a)-z_{\text{ideal}}}{z_{\text{nadir}}-z_{\text{ideal}}}, \frac{z-z_{\text{ideal}}}{z_{\text{nadir}}-z_{\text{ideal}}}\right)
	\end{equation*}
	
is the smallest (normalized) Euclidean distance between the archive and the region of interest, see also the figures below for an illustration.

.. [#] With linear transformations of both objective functions such that the ideal point :math:`z_{\text{ideal}}= (z_{\text{ideal}, \alpha}, z_{\text{ideal}, \beta})` is mapped to :math:`[0,0]` and the nadir point :math:`z_{\text{nadir}}= (z_{\text{nadir}, \alpha}, z_{\text{nadir}, \beta})` is mapped to :math:`[1,1]`.


	

.. todo::

	* why hypervolume (can also be in principle with other indicators)?

	* why archive?




.. figure:: pics/IHDoutside.*
   :align: center
   :width: 60%

   Illustration of Coco's quality indicator (to be minimized) in the
   bi-objective case if no solution of the archive (blue filled circles)
   dominates the nadir point (black filled circle), i.e., the shortest
   distance of an archive member to the region of interest (ROI), delimited
   by the nadir point. 
   Here, it is the forth point from the left that defines
   the smallest distance.
   
.. the hypervolume of the reference set (aka the
   best known Pareto front approximation, red triangles) plus 

.. figure:: pics/IHDinside.*
   :align: center
   :width: 60%

   Illustration of Coco's quality indicator (to be minized) in the
   bi-objective case if the nadir point (black filled circle) is dominated by
   a solution in the archive (blue filled circles). The indicator is the 
   (negative) hypervolume of the archive with the nadir point as reference point. 
   The difference between the hypervolume of the reference set (aka Pareto
   front approximation, red triangles) and the hypervolume of the archive is
   given as the size of the two blue shaded areas minus the size of the green
   area.


To summarize, here the specificities of the proposed ``bbob-biobj`` performance criterion:

* algorithm performance = runtime until the quality of the archive of non-dominated 
  solutions found so far surpasses a target value

* normalization of objective space before indicator calculation such that the
  region of interest (ROI) :math:`[z_{\text{ideal}}, z_{\text{nadir}}]`, defined by
  the ideal and nadir point is mapped to :math:`[0, 1]^2`

* if nadir point is dominated by a point in the archive: quality = hypervolume of archive wrt nadir point
  as hypervolume reference point

* if nadir point is not dominated by archive: quality = negative distance of archive to the ROI

This implies:

* the quality indicator value of an archive that contains the nadir point as 
  non-dominated point is :math:`0`.

* the quality indicator value is bounded from below by :math:`-1`. 

* Because the quality of the archive is used as performance criterion, no population size has to be
  prescribed to the algorithm. In particular, steady-state and generational algorithms can be 
  compared directly as well as algorithms with varying population size and algorithms which carry
  along their external archive themselves.
  
.. * As the reference set approaches the Pareto set, the optimal quality indicator difference goes to 0`

.. * Because the reference set is always a finite approximation of the Pareto set, negative quality
  indicator differences can occur.




Choice of Target Values
=======================

For each problem instance, |i|, of the benchmark suite, a *reference
hypervolume indicator value*, |Irefi|, is computed (see below). 
This reference value is determined to represent the hypervolume value of a fairly adequate
approximation of the Pareto set. [#]_ All target indicator values are computed as 
a function of |Irefi|, namely as |Irefi| :math:`+\,t`, where the target precision 
|t| is chosen as

.. math::

  t \in \{ -10^{-4}, -10^{-4.2}, -10^{-4.4}, -10^{-4.6}, -10^{-4.8}, -10^{-5}, 0, 10^{-5}, 10^{-4.9}, 10^{-4.8}, \dots, 10^{-0.1}, 10^0 \}

That is, if not stated otherwise, the runtimes of these 58 target values are
presented (usually as empirical cumulative distribution function, ECDF). 
It is not uncommon that the quality indicator value of the algorithm never surpasses some of
these target values, which leads to missing runtime measurements. Note that the non-positive
target precisions have been included in particular to account for the fact that the
reference hypervolume indicator value is computed only for a fairly adequate
approximation of the Pareto set and thus can potentially be outperformed by an actual algorithm. 
In comparison, in the single-objective case, target precision values are typically solely
positive if the global optima are known. [coco-perf-assessment]_

.. |Irefi| replace:: :math:`I_i^\mathrm{ref}`
.. |i| replace:: :math:`i`
.. |t| replace:: :math:`t`


.. Choice of Reference Set and Target Difficulties
   ===============================================
  Choice of the targets based on best estimation of Pareto front (using all the 
  data we have) - chosen instance wise

  relative targets (in terms of the hypervolume difference to the hypervolume of the reference set)
  are chosen the same for all functions, dimensions, and instances: recorded are 100 targets 
  per order of magnitude,
  equi-distantly chosen on the log-scale.


.. Displayed are finally only 10 targets per order of magnitude, in total 51 of them between :math:`10^0` and :math:`10^{-5}`

.. Note that due to the approximative nature of the reference set and its hypervolume, negative hypervolume values are possible. The Coco platform stores all

.. Remind that performance assessment is "relative" because best
   estimation of the front is meant to change. Hence ECDF plots are meant
   to be reploted.

.. [#] As we do not know the Pareto set on any but one function, the approximation 
  could be less adequate than we are hoping for. 


Choice of the Reference Hypervolume Indicator Value
---------------------------------------------------

Opposed to the single-objective ``bbob`` test suite [HAN2009fun]_, the
biobjective ``bbob-biobj`` test suite does not provide analytical forms of
its optima. 
Except for :math:`f_1`, the Pareto set and the Pareto front are unknown. 

.. The performance assessment therefore has to be relative to the best 
  known approximations and this document details how this is implemented.


Dealing with Unknown Optima
---------------------------

.. note:: Why don't we just introduce the used indicator, as all assessment is
  based on it? It seems not necessary to introduce the 1001st time the 
  definition of dominance. The assessment is based only on an indicator value. 
  As we use hypervolume, the indicator improves iff a new non-dominated 
  solution is generated. 

The equivalent of a global optimum in the multi-objective case is the set of Pareto-optimal
or efficient solutions, also known as Pareto set. If we assume the search space to be
:math:`\mathbb{R}^n` and the minimization of two objective
functions :math:`f_1: x\in \mathbb{R}^n \mapsto f_1(x)\in\mathbb{R}` and :math:`f_1: x\in \mathbb{R}^n \mapsto f_1(x)\in\mathbb{R}`,
a solution :math:`x\in\mathbb{R}^n` is called Pareto-optimal if it is not dominated
by any other solution :math:`y\in\mathbb{R}^n` or, in other words, if

.. math::
  
  \not\exists y \text{ s.t. } (f_1(y)< f_1(x) \text{ and } f_2(y)\leq f_2(x)) \text{ or } (f_2(y)\leq f_2(x) \text{ and } f_2(y)< f_2(x)).

The image of the Pareto set under the vector-valued objective function
:math:`f(x)= (f_1(x), f_2(x))` is called Pareto front.

When combining single-objective functions to multi-objective ones as in the case of the ``bbob-biobj``
suite, one cannot expect that Pareto set and Pareto front can be described in analytical form---even
if the single-objective optima are known. Comparing algorithm performance can therefore only be
done relatively to the best known optimum. In the multi-objective
case, where with the Pareto set a set of solutions is sought, we call this approximation
**reference set**. In practice, such a reference set is typically generated by running a certain set
of algorithms on the considered problem ahead of the performance assessment.

This has two main implications:

.. todo:: "*Performance can only be judged relatively to the reference set*" seem
  just false. We can defined a target hypervolume and measure runtime entirely
  independent of the reference set. 

* Performance can only be judged relatively to the reference set. The better the algorithms
  used to create the reference set have been, the more accurate the performance assessment.

* The reference set is expected to evolve over time, in terms of becoming a better and better
  approximation of the actual Pareto set/Pareto front if more and more algorithms are
  compared.

.. The performance assessment via the Coco platform addresses both issues, see
   `Choice of Reference Set and Target Difficulties`_ and
   `Data storage and Future Recalculations of Indicator Values`_ below for details.
   Before we discuss these issues, however, let us have a look on the actual performance
   criterion used for the ``bbob-biobj`` test suite, assuming that a reference set is given.


Data storage and Future Recalculations of Indicator Values
==========================================================
Having a good approximation of the Pareto set/Pareto front is crucial in accessing
algorithm performance with the above suggested performance criterion. In order to allow
the reference set to approximate the Pareto set/Pareto front better and better over time,
the Coco platform records every non-dominated solution over the algorithm run.
Algorithm data sets, submitted through the Coco platform's web page, can therefore
be used to improve the quality of the reference set by adding all solutions to the
reference set which are non-dominated to it. 

Recording every new non-dominated solution within every algorithm run also allows to
recover the algorithm runs after the experiment and to recalculate the corresponding
hypervolume difference values if the reference set changes in the future.




Instances and Generalization Experiment
=======================================
* we record for 10 instances but display result for only 5. This will allow us to generate data for an unbiased
  generalization test on the unseen instances

  
  

Acknowledgements
================
This work was supported by the grant ANR-12-MONU-0009 (NumBBO) 
of the French National Research Agency.
  
   

.. ############################# References ##################################
.. raw:: html
    
    <H2>References</H2>

   
.. [bbob-biobj-functions-doc] The BBOBies. **Function Documentation of the bbob-biobj Test Suite**. http://numbbo.github.io/coco-doc/bbob-biobj/functions/

.. [coco-perf-assessment] The BBOBies. **COCO: Performance Assessment**. http://numbbo.github.io/coco-doc/perf-assessment/

.. [coco-doc] The BBOBies. **COCO: A platform for Comparing Continuous Optimizers in a Black-Box Setting**. http://numbbo.github.io/coco-doc/

.. [BBO2016ex] The BBOBies: `COCO: Experimental Procedure`__. 
__ http://numbbo.github.io/coco-doc/experimental-setup/

.. [BBO2016perf] The BBOBies: `Performance Assessment`__. 
__ https://www.github.com

.. [HAN2009fun] N. Hansen, S. Finck, R. Ros, and A. Auger (2009). 
  `Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions`__. `Technical Report RR-6829`__, Inria, updated February 2010.
.. __: http://coco.gforge.inria.fr/
.. __: https://hal.inria.fr/inria-00362633
