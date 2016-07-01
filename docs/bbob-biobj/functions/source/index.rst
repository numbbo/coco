.. title:: COCO: The Bi-objective Black-Box Optimization Benchmarking (bbob-biobj) Test Suite

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COCO: The Bi-objective Black-Box Optimization Benchmarking (``bbob-biobj``) Test Suite
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

.. the next two lines are necessary in LaTeX. They will be automatically 
  replaced to put away the \chapter level as ??? and let the "current" level
  become \section. 
.. CHAPTERTITLE
.. CHAPTERUNDERLINE

.. |
.. |
.. .. sectnum::
  :depth: 3
  

  :numbered:
.. .. contents:: Table of Contents
  :depth: 2
.. |
.. |

.. raw:: html

   See also: <I>ArXiv e-prints</I>,
   <A HREF="http://arxiv.org/abs/1604.00359">arXiv:1604.00359</A>, 2016.

.. raw:: latex

  % \tableofcontents TOC is automatic with sphinx and moved behind abstract by swap...py
  \begin{abstract}

The ``bbob-biobj`` test suite contains 55 bi-objective 
functions in continuous domain which are derived from combining functions of
the well-known single-objective noiseless ``bbob`` test suite. Besides giving
the actual function definitions and presenting their (known) properties, this
documentation also aims at giving the rationale behind our approach in terms
of function groups, instances, and potential objective space normalization.

.. raw:: latex

  \end{abstract}
  \newpage


.. old <p>The <code class="docutils literal"><span
  class="pre">bbob-biobj</span></code> test suite contains 55 bi-objective
  functions in continuous domain which are derived from combining functions
  of the well-known single-objective noiseless <code class="docutils
  literal"><span class="pre">bbob</span></code> test suite. It will be used
  as the main test suite of the upcoming <a
  href="http://numbbo.github.io/workshops/BBOB-2016/">BBOB-2016 workshop</a>
  at GECCO. Besides giving the actual function definitions and presenting
  their (known) properties, this documentation also aims at giving the
  rationale behind our approach in terms of function groups, instances, and
  objective space normalization. </p>

.. old The ``bbob-biobj`` test suite contains 55 bi-objective 
   functions in continuous domain which are derived from combining functions of
   the well-known single-objective noiseless ``bbob`` test suite. It will be
   used as the main test suite of the upcoming `BBOB-2016 workshop
   <http://numbbo.github.io/workshops/BBOB-2016/>`_ at GECCO. Besides giving the
   actual function definitions and presenting their (known) properties, this
   documentation also aims at summarizing the state-of-the-art in
   multi-objective black-box benchmarking, at giving the rational behind our
   approach, and at providing a simple tutorial on how to use these functions
   for actual benchmarking within the COCO_ framework.


.. _COCO: https://github.com/numbbo/coco
.. _COCOold: http://coco.gforge.inria.fr
.. |coco_problem_t| replace:: 
  ``coco_problem_t``
.. _coco_problem_t: http://numbbo.github.io/coco-doc/C/coco_8h.html#a408ba01b98c78bf5be3df36562d99478

.. summarizing the state-of-the-art in multi-objective black-box benchmarking, at 
.. and at providing a simple tutorial on how to use these functions for actual benchmarking within the COCO_ framework.

.. .. Note::
  
  For the time being, this documentation is under development and might not 
  contain all final data.

.. figure:: _figs/examples-bbob-biobj.png
   :scale: 60
   :align: center

   Example plots of the Pareto front approximation, found by NSGA-II on
   selected ``bbob-biobj`` functions. In blue the non-dominated points at the
   end of different independent runs, in red the points that are
   non-dominated over all runs.

.. Tea: f_1 and f_2 should be replaced by f_\alpha and f_\beta in all plots. Also, instead of "f16 :" do "f16: "

.. #################################################################################
.. #################################################################################
.. #################################################################################



Introduction
============

In the following, we consider bi-objective, unconstrained
**minimization** problems of the form

.. math::
  \min_{x \in \mathbb{R}^n} f(x)=(f_\alpha(x),f_\beta(x)),

where :math:`n` is the number of variables of the problem (also called
the problem dimension), :math:`f_\alpha: \mathbb{R}^n \rightarrow \mathbb{R}`
and :math:`f_\beta: \mathbb{R}^n \rightarrow \mathbb{R}` are the two
objective functions, and the :math:`\min` operator is related to the
standard *dominance relation*. A solution :math:`x\in\mathbb{R}^n`
is thereby said to *dominate* another solution :math:`y\in\mathbb{R}^n` if
:math:`f_\alpha(x) \leq f_\alpha(y)` and :math:`f_\beta(x) \leq f_\beta(y)` hold and at
least one of the inequalities is strict.

Solutions which are not dominated by any other solution in the search
space are called *Pareto-optimal* or *efficient solutions*. All
Pareto-optimal solutions constitute the *Pareto set* of which an 
approximation is sought. The Pareto set's image in the
objective space :math:`f(\mathbb{R}^n)` is called *Pareto front*.

The objective of the minimization problem is to find, with as few evaluations
of |f| as possible, a set of non-dominated solutions which is (i) as large
as possible and (ii) has |f|-values as close to the Pareto front as possible. [#]_ 

.. [#] Distance in |f|-space is defined here such that nadir and ideal point 
   have in each coordinate distance one. Neither of these points is however 
   freely accessible to the optimization algorithm. 

.. Niko: here is my take on the footnote: 1/3 of the readers know the concepts and will
   get informed by this footnote (it answers precisely the question I would ask at this 
   point reading the doc). 1/3 of the readers will understand that it is a 
   good idea to learn about the concepts of nadir and ideal point, which
   it is. It will increase their incentive to check out the next section more
   carefully. 1/3 of the readers won't get anything and move on. 

.. |f| replace:: :math:`f`

Definitions and Terminology
---------------------------
We remind in this section different definitions.

*function instance, problem*
 Each function :math:`f^\theta: \mathbb{R}^n \to \mathbb{R}^m` within COCO_ is parametrized 
 with parameter values :math:`\theta \in \Theta`. A parameter value determines a so-called *function 
 instance*. For example, :math:`\theta` encodes the location of the optimum of single-objective functions, 
 which means that different instances have shifted optima. In the ``bbob-biobj`` 
 test suite, :math:`m=2` and the function instances are determined by the instances of the underlying
 single-objective functions. 
 
 A *problem* is a function instance of a specific dimension :math:`n`.

*ideal point*
 The ideal point is defined as the vector in objective space that
 contains the optimal |f|-value for each objective *independently*. 
 More precisely let :math:`f_\alpha^{\rm opt}:= \inf_{x\in \mathbb{R}^n} f_\alpha(x)` and
 :math:`f_\beta^{\rm opt}:= \inf_{x\in \mathbb{R}^n} f_\beta(x)`, the ideal point is given by
 
 .. math::
    :nowrap:

	\begin{equation*}
	z_{\rm ideal}  =  (f_\alpha^{\rm opt},f_\beta^{\rm opt}).
    \end{equation*}
    
 
*nadir point* 
 The *nadir point* (in objective space) consists in each objective of
 the worst value obtained by a Pareto-optimal solution. More precisely,
 let :math:`\mathcal{PO}` be the set of Pareto optimal points. Then the nadir point satisfies
 
 .. math::
    :nowrap:

	\begin{equation*}
	z_{\rm nadir}  =   \left( \sup_{x \in \mathcal{PO}} f_\alpha(x),
     \sup_{x \in \mathcal{PO}} f_\beta(x)  \right).
    \end{equation*} 
    
 In the case of two objectives with a unique global minimum each (that
 is, a single point in the search space maps to the global minimum) 
    
 .. math::
    :nowrap:

	\begin{equation*}
	z_{\rm nadir}  =   \left( f_\alpha(x_{\rm opt,\beta}),
      f_\beta(x_{\rm opt,\alpha})  \right),
    \end{equation*} 
    
   
 where :math:`x_{\rm opt,\alpha}= \arg \min f_\alpha(x)` and 
 :math:`x_{\rm opt,\beta}= \arg \min f_\beta(x)`.



Overview of the Proposed ``bbob-biobj`` Test Suite
==================================================

The ``bbob-biobj`` test suite provides 55 bi-objective functions in six
dimensions (2, 3, 5, 10, 20, and 40) with a large number of possible instances. 
The 55 functions are derived from combining a subset of the 24 well-known
single-objective functions of the ``bbob`` test suite [HAN2009]_ which
has been used since 2009 in the `BBOB workshop series`__. 
While concrete details on each of
the 55 ``bbob-biobj`` functions are given in Section
:ref:`sec-test-functions`, we will detail here the main rationale behind
them together with their common properties.

__ http://numbbo.github.io/workshops

The Single-objective ``bbob`` Functions
---------------------------------------
The ``bbob-biobj`` test suite is designed to be able to assess  performance of algorithms with respect to well-identified difficulties in optimization typically  occurring in real-world problems. A multi-objective problem being a combination of single-objective problems, one can obtain multi-objective problems with representative difficulties by simply combining single objective functions with representative difficulties observed in real-world problems. For this purpose we naturally use the single-objective ``bbob`` suite [HAN2009]_.

Combining all 24 ``bbob`` functions in pairs thereby results in
:math:`24^2=576` bi-objective functions overall. We however assume that
multi-objective optimization algorithms are not sensitive to permutations of
the objective functions such that combining the 24  ``bbob`` functions and
taking out the function :math:`(g_2,g_1)` if the function :math:`(g_1,g_2)`
is present results in :math:`24 + {24 \choose 2} = 24 + (24\times23)/2 = (24\times25)/2 = 300` functions.

.. Given that most (if not all) multi-objective optimization algorithms are
.. invariant to permutations of the objective functions, a bi-objective
.. function combining for example the sphere function as the first
.. objective with the Rastrigin function as the second objective will
.. result in the same performance than if the Rastrigin function is the
.. first and the sphere function is the second objective function. 
.. Hence, we should keep only one of the resulting
.. bi-objective functions. Combining then all 24 ``bbob`` functions

.. The first objective is chosen as ``bbob`` function *i*
  and the second as ``bbob`` function *j* with *i* :math:`\leq` *j*,
  resulting in :math:`24+ {24 \choose 2} = 300` functions.

Some first tests, e.g. in [BRO2015]_, showed that having 300 functions is
impracticable in terms of the overall running time of the benchmarking
experiment.  We then decided to exploit the organization of the ``bbob``
functions into classes to choose a subset of functions. More precisely, the 24
original ``bbob`` functions are grouped into five function classes where each
class gathers functions with similar properties, namely

  1. separable functions
  2. functions with low or moderate conditioning
  3. functions with high conditioning and unimodal
  4. multi-modal functions with adequate global structure, 
  5. multi-modal functions with weak global structure.



To create the ``bbob-biobj`` suite, we choose two functions within each class. This way we do not introduce any bias towards a specific class. In addition within each class, the functions are chosen to be the most
representative without repeating similar functions. For example,
only one Ellipsoid, one Rastrigin, and one Gallagher function are
included in the ``bbob-biobj`` suite although they appear in
separate versions in the ``bbob`` suite. Finally our choice of  10 ``bbob`` functions for creating the ``bbob-biobj`` test suite is the following:

.. We chose two functions within each class
..  consider only the following 10 of the 24 ``bbob``
.. functions:


.. The above ten ``bbob`` functions have been chosen for the creation
.. of the ``bbob-biobj`` suite in a way to not introduce any bias
.. towards a specific class
.. by choosing exactly two functions per ``bbob`` function class.
.. Within each class, the functions were chosen to be the most
.. representative without repeating similar functions. For example,
.. only one Ellipsoid, one Rastrigin, and one Gallagher function are
.. included in the ``bbob-biobj`` suite although they appear in
.. separate versions in the ``bbob`` suite.



.. |f`1` in the bbob suite| replace:: :math:`f_1` in the ``bbob`` suite
.. _f`1` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=5

.. |f`2` in the bbob suite| replace:: :math:`f_2` in the ``bbob`` suite
.. _f`2` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=10

.. |f`6` in the bbob suite| replace:: :math:`f_6` in the ``bbob`` suite
.. _f`6` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=30

.. |f`8` in the bbob suite| replace:: :math:`f_8` in the ``bbob`` suite
.. _f`8` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=40

.. |f`13` in the bbob suite| replace:: :math:`f_{13}` in the ``bbob`` suite
.. _f`13` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=65

.. |f`14` in the bbob suite| replace:: :math:`f_{14}` in the ``bbob`` suite
.. _f`14` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=70

.. |f`15` in the bbob suite| replace:: :math:`f_{15}` in the ``bbob`` suite
.. _f`15` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=75

.. |f`17` in the bbob suite| replace:: :math:`f_{17}` in the ``bbob`` suite
.. _f`17` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=85

.. |f`20` in the bbob suite| replace:: :math:`f_{20}` in the ``bbob`` suite
.. _f`20` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=100

.. |f`21` in the bbob suite| replace:: :math:`f_{21}` in the ``bbob`` suite
.. _f`21` in the bbob suite: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=105

.. |bbob suite| replace:: ``bbob`` suite
.. _bbob suite: https://hal.inria.fr/inria-00362633

* Separable functions

  - Sphere (function 1 in |bbob suite|_)
  - Ellipsoid separable (function 2 in |bbob suite|_)

* Functions with low or moderate conditioning 

  - Attractive sector (function 6 in |bbob suite|_)
  - Rosenbrock original (function 8 in |bbob suite|_)

* Functions with high conditioning and unimodal 

  - Sharp ridge (function 13 in |bbob suite|_)
  - Sum of different powers (function 14 in |bbob suite|_)

* Multi-modal functions with adequate global structure 

  - Rastrigin (function 15 in |bbob suite|_)
  - Schaffer F7, condition 10 (function 17 in |bbob suite|_)

* Multi-modal functions with weak global structure 

  - Schwefel x*sin(x) (function 20 in |bbob suite|_)
  - Gallagher 101 peaks (function 21 in |bbob suite|_)

  
Using the above described pairwise combinations, this results in
having :math:`10+{10 \choose 2} = 55` bi-objective functions in
the final `bbob-biobj` suite. These functions are denoted :math:`f_1` to :math:`f_{55}` in the sequel.

.. The next section gives the
.. reasoning behind choosing exactly these 10 functions.

  

Function Groups
---------------------------------------------------------------

From combining the original ``bbob`` function classes, we obtain 15 function
classes to structure the 55 bi-objective functions of the ``bbob-biobj`` test
suite. Each function class contains three or four functions. We are listing
below the function classes and in parenthesis  the functions that belong to
the respective class:

 1. separable - separable (functions :math:`f_1`, :math:`f_2`, :math:`f_{11}`)
 2. separable - moderate (:math:`f_3`, :math:`f_4`, :math:`f_{12}`, :math:`f_{13}`)
 3. separable - ill-conditioned (:math:`f_5`, :math:`f_6`, :math:`f_{14}`, :math:`f_{15}`)
 4. separable - multi-modal (:math:`f_7`, :math:`f_8`, :math:`f_{16}`, :math:`f_{17}`)
 5. separable - weakly-structured (:math:`f_9`, :math:`f_{10}`, :math:`f_{18}`, :math:`f_{19}`)
 6. moderate - moderate (:math:`f_{20}`, :math:`f_{21}`, :math:`f_{28}`)
 7. moderate - ill-conditioned (:math:`f_{22}`, :math:`f_{23}`, :math:`f_{29}`, :math:`f_{30}`)
 8. moderate - multi-modal (:math:`f_{24}`, :math:`f_{25}`, :math:`f_{31}`, :math:`f_{32}`)
 9. moderate - weakly-structured (:math:`f_{26}`, :math:`f_{27}`, :math:`f_{33}`, :math:`f_{34}`)
 10. ill-conditioned - ill-conditioned (:math:`f_{35}`, :math:`f_{36}`, :math:`f_{41}`)
 11. ill-conditioned - multi-modal (:math:`f_{37}`, :math:`f_{38}`, :math:`f_{42}`, :math:`f_{43}`)
 12. ill-conditioned - weakly-structured (:math:`f_{39}`, :math:`f_{40}`, :math:`f_{44}`, :math:`f_{45}`)
 13. multi-modal - multi-modal (:math:`f_{46}`, :math:`f_{47}`, :math:`f_{50}`)
 14. multi-modal - weakly structured (:math:`f_{48}`, :math:`f_{49}`, :math:`f_{51}`, :math:`f_{52}`)
 15. weakly structured - weakly structured (:math:`f_{53}`, :math:`f_{54}`, :math:`f_{55}`)


.. The original ``bbob`` function classes also allow to group the
.. 55 ``bbob-biobj`` functions, dependend on the
.. classes of the individual objective functions. Depending
.. on whether two functions of the same class are combined
.. or not, these resulting 15 new function classes contain three
.. or four functions:


More details about the single functions can be found in Section :ref:`sec-test-functions`. We however first describe their common properties in the coming sections.


Normalization of Objectives
------------------------------------
None of the 55 ``bbob-biobj`` functions is explicitly normalized and the
optimization algorithms therefore have to cope with objective values in
different ranges. Typically, different orders of magnitude
between the objective values can be observed.

However, to facilitate comparison of algorithm performance over different functions, 
we normalize the objectives based on the ideal and nadir points
before calculating the hypervolume indicator [BRO2016biperf]_.
Both points can be computed, because the global 
optimum is known and is unique for the 10 ``bbob`` base functions. 
In the black-box optimization benchmarking setup, however, the values of the
ideal and nadir points are not accessible to the optimization algorithm
[HAN2016ex]_.


.. deleted: a normalization can take place as both the ideal and the nadir point are
   known internally. 

.. Note that, for example, the ``bbob-biobj`` observer of
.. the `Coco framework`_ takes this into account and normalizes the objective
.. space, see the `bbob-biobj-specific performance assessment documentation 
.. <http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/>`_ for
.. details.

.. deleted: The reasons for having knowledge about the location of both the ideal and
  the nadir point are
  * the definitions of the single-objective ``bbob`` test functions for 
  which the optimal function value and the optimal solution are known
  by design and
  * the fact that we explicitly chose only functions from the original
  ``bbob`` test suite which have a unique optimum.

.. deleted (this was a repetition from a previous section) 
   The ideal point is then always given by the objective 
   vector :math:`(f_\alpha(x_{\text{opt},\alpha}),
   f_\beta(x_{\text{opt},\beta}))` and the nadir point by the objective
   vector :math:`(f_\alpha(x_{\text{opt},\beta}),
   f_\beta(x_{\text{opt},\alpha}))` with :math:`x_{\text{opt},\alpha}` being
   the optimal solution for the first objective function :math:`f_\alpha` and
   :math:`x_{\text{opt},\beta}` being the optimal solution for the second
   objective function :math:`f_\beta`. Note that in the black-box case, we
   typically assume for the functions provided with the `Coco framework`_,
   that information about ideal and nadir points, scaling etc. is not
   provided to the algorithm.


Instances
---------
Our test functions are parametrized and instances are instantiations of the
underlying parameters (see [HAN2016co]_). The instances for the bi-objective
functions are obtained using instances of each single objective function composing the
bi-objective one. In addition, we assert two conditions:

  1. The Euclidean distance between the two single-objective optima (also called the 
  extreme optimal points) in the search space is at least :math:`10^{-4}`. 

  2. The Euclidean distance between the ideal and the nadir point in the non-normalized 
  objective space is at least :math:`10^{-1}`. 
     

.. Instances are the way in the `Coco framework`_ to perform multiple
.. algorithm runs on the same function. More concretely, the original
.. Coco documentation states

.. ::

..  All functions can be instantiated in different *versions* (with
..  different location of the global optimum and different optimal
..  function value). Overall *Ntrial* runs are conducted on different
..  instantiations.

.. Also in the bi-objective case, we provide the idea of instances by
.. relying on the instances provided within the single-objective
.. ``bbob`` suite. 
.. However, in addition, we assert that


We associate to an instance, an instance-id which is an integer. The relation between the 
instance-id, :math:`K^{f}_{\rm id}`, of a bi-objective function :math:`f = (f_\alpha, f_\beta)`
and the instance-ids, :math:`K_{\rm id}^{f_\alpha}` and :math:`K_{\rm id}^{f_\beta}`, of its 
underlying single-objective functions :math:`f_\alpha` and :math:`f_\beta` is the following:

 * :math:`K_{\rm id}^{f_\alpha} = 2 K^{f}_{\rm id} + 1` and
 * :math:`K_{\rm id}^{f_\beta} = K_{\rm id}^{f_\alpha} + 1`


If we find that above conditions are not satisfied for all dimensions and
functions in the ``bbob-biobj`` suite, we increase the instance-id of the
second objective successively until both properties are fulfilled. 
For example, the ``bbob-biobj`` instance-id
8 corresponds to the instance-id 17 for the first objective and instance-id 18 for
the second objective while for the ``bbob-biobj`` instance-id 9, the
first instance-id is 19 but for the second objective, instance-id 21 is chosen
instead of instance-id 20.

Exceptions to the above rule are, for historical reasons, the
``bbob-biobj`` instance-ids 1 and 2 in order to match the instance-ids
1 to 5 with the ones proposed in [BRO2015]_. The ``bbob-biobj``
instance-id 1 contains the single-objective instance-ids 2 and 4 and
the ``bbob-biobj`` instance-id 2 contains the two instance-ids 3 and 5.

For each bi-objective function and given dimension, the ``bbob-biobj`` suite
contains 10 instances. [#]_

.. [#] In principle, as for the instance generation for the ``bbob`` suite, 
   the number of possible instances for the ``bbob-biobj`` suite is unlimited
   [HAN2016co]_. 
   However, running some tests with too few instances will render the
   potential statistics and their interpretation problematic while even the
   tiniest observed difference can be made statistically significant with a
   high enough number of instances. A good compromise to avoid either pitfall 
   seems to lie between, say, 9 and 19 instances.
   
.. Thus, we recommend to use between 5 to 15 instances for the actual 
   benchmarking.
.. The user doesn't actually have a choice. 

.. Tea: At this point I'm missing some discussion on how in the bi-objective case instances 
   can affect more than just the "location of the optimum". 
   

.. _sec-test-functions:

The ``bbob-biobj`` Test Functions and Their Properties
======================================================

In the following, we detail all 55 ``bbob-biobj`` functions
and their properties.

.. .. todo::
   Eventually, the following shall be provided for each function:

   - plots of the best known approximations of the Pareto set and the Pareto front
   - potentially the outcomes of example algorithms
   - plots (in objective space) of randomly sampled search points
   - potentially function value distributions along cuts through the search space 

The following table gives an overview and quick access to the functions,
inner cell IDs refer to the ``bbob-biobj`` functions, outer column and row
annotations refer to the single-objective ``bbob`` functions.

+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
|       ||fb1|_ ||fb2|_ ||fb6|_ ||fb8|_ ||fb13|_||fb14|_||fb15|_||fb17|_||fb20|_||fb21|_|
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb1|_ | |f1|  | |f2|  | |f3|  | |f4|  | |f5|  | |f6|  | |f7|  | |f8|  | |f9|  | |f10| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb2|_ |       | |f11| | |f12| | |f13| | |f14| | |f15| | |f16| | |f17| | |f18| | |f19| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb6|_ |       |       | |f20| | |f21| | |f22| | |f23| | |f24| | |f25| | |f26| | |f27| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb8|_ |       |       |       | |f28| | |f29| | |f30| | |f31| | |f32| | |f33| | |f34| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb13|_|       |       |       |       | |f35| | |f36| | |f37| | |f38| | |f39| | |f40| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb14|_|       |       |       |       |       | |f41| | |f42| | |f43| | |f44| | |f45| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb15|_|       |       |       |       |       |       | |f46| | |f47| | |f48| | |f49| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb17|_|       |       |       |       |       |       |       | |f50| | |f51| | |f52| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb20|_|       |       |       |       |       |       |       |       | |f53| | |f54| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
||fb21|_|       |       |       |       |       |       |       |       |       | |f55| |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+

.. |fb1| replace:: :math:`f_1`
.. _fb1: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=5
.. |fb2| replace:: :math:`f_2`
.. _fb2: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=10
.. |fb6| replace:: :math:`f_6`
.. _fb6: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=30
.. |fb8| replace:: :math:`f_8`
.. _fb8: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=40
.. |fb13| replace:: :math:`f_{13}`
.. _fb13: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=65
.. |fb14| replace:: :math:`f_{14}`
.. _fb14: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=70
.. |fb15| replace:: :math:`f_{15}`
.. _fb15: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=75
.. |fb17| replace:: :math:`f_{17}`
.. _fb17: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=85
.. |fb20| replace:: :math:`f_{20}`
.. _fb20: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=100
.. |fb21| replace:: :math:`f_{21}`
.. _fb21: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=105

.. |f1| replace:: :ref:`f1 <f1>`
.. |f2| replace:: :ref:`f2 <f2>`
.. |f3| replace:: :ref:`f3 <f3>`
.. |f4| replace:: :ref:`f4 <f4>`
.. |f5| replace:: :ref:`f5 <f5>`
.. |f6| replace:: :ref:`f6 <f6>`
.. |f7| replace:: :ref:`f7 <f7>`
.. |f8| replace:: :ref:`f8 <f8>`
.. |f9| replace:: :ref:`f9 <f9>`
.. |f10| replace:: :ref:`f10 <f10>`
.. |f11| replace:: :ref:`f11 <f11>`
.. |f12| replace:: :ref:`f12 <f12>`
.. |f13| replace:: :ref:`f13 <f13>`
.. |f14| replace:: :ref:`f14 <f14>`
.. |f15| replace:: :ref:`f15 <f15>`
.. |f16| replace:: :ref:`f16 <f16>`
.. |f17| replace:: :ref:`f17 <f17>`
.. |f18| replace:: :ref:`f18 <f18>`
.. |f19| replace:: :ref:`f19 <f19>`
.. |f20| replace:: :ref:`f20 <f20>`
.. |f21| replace:: :ref:`f21 <f21>`
.. |f22| replace:: :ref:`f22 <f22>`
.. |f23| replace:: :ref:`f23 <f23>`
.. |f24| replace:: :ref:`f24 <f24>`
.. |f25| replace:: :ref:`f25 <f25>`
.. |f26| replace:: :ref:`f26 <f26>`
.. |f27| replace:: :ref:`f27 <f27>`
.. |f28| replace:: :ref:`f28 <f28>`
.. |f29| replace:: :ref:`f29 <f29>`
.. |f30| replace:: :ref:`f30 <f30>`
.. |f31| replace:: :ref:`f31 <f31>`
.. |f32| replace:: :ref:`f32 <f32>`
.. |f33| replace:: :ref:`f33 <f33>`
.. |f34| replace:: :ref:`f34 <f34>`
.. |f35| replace:: :ref:`f35 <f35>`
.. |f36| replace:: :ref:`f36 <f36>`
.. |f37| replace:: :ref:`f37 <f37>`
.. |f38| replace:: :ref:`f38 <f38>`
.. |f39| replace:: :ref:`f39 <f39>`
.. |f40| replace:: :ref:`f40 <f40>`
.. |f41| replace:: :ref:`f41 <f41>`
.. |f42| replace:: :ref:`f42 <f42>`
.. |f43| replace:: :ref:`f43 <f43>`
.. |f44| replace:: :ref:`f44 <f44>`
.. |f45| replace:: :ref:`f45 <f45>`
.. |f46| replace:: :ref:`f46 <f46>`
.. |f47| replace:: :ref:`f47 <f47>`
.. |f48| replace:: :ref:`f48 <f48>`
.. |f49| replace:: :ref:`f49 <f49>`
.. |f50| replace:: :ref:`f50 <f50>`
.. |f51| replace:: :ref:`f51 <f51>`
.. |f52| replace:: :ref:`f52 <f52>`
.. |f53| replace:: :ref:`f53 <f53>`
.. |f54| replace:: :ref:`f54 <f54>` 
.. |f55| replace:: :ref:`f55 <f55>` 

.. [1,2,6,8,13,14,15,17,20,21]

..  :ref:`f1 <f1>`, :ref:`f2 <f2>`, :ref:`f3 <f3>`, :ref:`f4 <f4>`,
  :ref:`f5 <f5>`, :ref:`f6 <f6>`, :ref:`f7 <f7>`, :ref:`f8 <f8>`,
  :ref:`f9 <f9>`, :ref:`f10 <f10>`, :ref:`f11 <f11>`,
  :ref:`f12 <f12>`, :ref:`f13 <f13>`, :ref:`f14 <f14>`, :ref:`f15 <f15>`,
  :ref:`f16 <f16>`, :ref:`f17 <f17>`, :ref:`f18 <f18>`, :ref:`f19 <f19>`,
  :ref:`f20 <f20>`, :ref:`f21 <f21>`, :ref:`f22 <f22>`, :ref:`f23 <f23>`,
  :ref:`f24 <f24>`, :ref:`f25 <f25>`, :ref:`f26 <f26>`, :ref:`f27 <f27>`,
  :ref:`f28 <f28>`, :ref:`f29 <f29>`, :ref:`f30 <f30>`, :ref:`f31 <f31>`,
  :ref:`f32 <f32>`, :ref:`f33 <f33>`, :ref:`f34 <f34>`, :ref:`f35 <f35>`,
  :ref:`f36 <f36>`, :ref:`f37 <f37>`, :ref:`f38 <f38>`, :ref:`f39 <f39>`,
  :ref:`f40 <f40>`, :ref:`f41 <f41>`, :ref:`f42 <f42>`, :ref:`f43 <f43>`,
  :ref:`f44 <f44>`, :ref:`f45 <f45>`, :ref:`f46 <f46>`, :ref:`f47 <f47>`,
  :ref:`f48 <f48>`, :ref:`f49 <f49>`, :ref:`f50 <f50>`, :ref:`f51 <f51>`,
  :ref:`f52 <f52>`, :ref:`f53 <f53>`, :ref:`f54 <f54>`, :ref:`f55 <f55>`.

Some Function Properties
------------------------
In the description of the 55 ``bbob-biobj`` functions below, several
general properties of objective functions will be mentioned that
are defined here in short. It depends on these properties whether the optimization problem
is easy or hard to solve.

A *separable* function does not show any dependencies between the
variables and can therefore be solved by applying :math:`n` consecutive
one-dimensional optimizations along the coordinate axes while
keeping the other variables fixed. Consequently, *non-separable*
problems must be considered. They are much more difficult to solve. The
typical well-established technique to generate non-separable
functions from separable ones is the application of a rotation matrix
:math:`\mathbf R` to :math:`x`, that is :math:`x \in \mathbb{R}^n \mapsto g(\mathbf R x)`, 
where :math:`g` is a separable function. 

A *unimodal* function has only one local minimum which is at the same
time also its global one. 
A *multimodal* function has at least two local minima which is highly common
in practical optimization problems.

*Ill-conditioning* is another typical challenge in real-parameter
optimization and, besides multimodality, probably the most common one.
In a general case, we can consider a function as ill-conditioned if for
solution points from the same level-set "the minimal displacement [...] that 
produces a given function value improvement differs by
orders of magnitude" [HAN2011]_.
Conditioning can be rigorously formalized in the
case of convex quadratic functions,
:math:`f(x) = \frac{1}{2} x^THx` where :math:`H` is a symmetric
positive definite matrix, as the condition number of the Hessian matrix
:math:`H`. Since contour lines associated to a convex quadratic function
are ellipsoids, the condition number corresponds to the square root of
the ratio between the largest axis of the ellipsoid and the shortest axis.


The proposed ``bbob-biobj`` testbed contains ill-conditioned functions
with a typical conditioning of :math:`10^6`. We believe this is a realistic
requirement, while we have seen practical problems with conditioning
as large as :math:`10^{10}`.


Domain Bounds
-------------
All bi-objective functions provided in the ``bbob-biobj`` suite are unbounded,
i.e., defined on the entire real-valued space :math:`\mathbb{R}^n`. 
The search domain of interest is defined as :math:`[-100,100]^n`, outside of 
which non-dominated solutions are quite unlikely to be found. [#]_ The majority of non-dominated solutions are likely to lie even within :math:`[-5,5]^n`. 

.. Nevertheless, they are designed such that and bound-constraint methods are likely to be competitive. 

While we believe that the domain of interest contains the Pareto set, 
due to the nature of the ``bbob-biobj`` function definitions, there is no
guarantee that this is always the case. 
However, the extremal solutions and their neighborhood ball of radius one
are guaranteed to lie within :math:`[-5,5]^n`.

.. [#] The functions |coco_problem_get_smallest_value_of_interest|_ and 
  |coco_problem_get_largest_value_of_interest|_ 
  of the COCO_ platform allow the optimizer
  to retrieve the *search domain of interest* from the |coco_problem_t|_, 
  for example to generate the initial search points. 

.. |coco_problem_get_largest_value_of_interest| replace:: ``coco_problem_get_largest_value_of_interest``
.. _coco_problem_get_largest_value_of_interest: http://numbbo.github.io/coco-doc/C/coco_8h.html#a29c89e039494ae8b4f8e520cba1eb154

.. |coco_problem_get_smallest_value_of_interest| replace:: ``coco_problem_get_smallest_value_of_interest``
.. _coco_problem_get_smallest_value_of_interest: http://numbbo.github.io/coco-doc/C/coco_8h.html#a4ea6c067adfa866b0179329fe9b7c458

The 55 ``bbob-biobj`` Functions
-------------------------------

.. _f1:

:math:`f_1`: Sphere/Sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two sphere functions (|f`1` in the bbob suite|_).

Both objectives are unimodal, highly symmetric, rotational and scale
invariant. The Pareto set is known to be a straight line and the Pareto 
front is convex. Considered as the simplest bi-objective problem in
continuous domain.

Contained in the *separable - separable* function class.


.. .. rubric:: Information gained from this function:

.. * What is the optimal convergence rate of a bi-objective algorithm?


.. _f2:

:math:`f_2`: Sphere/Ellipsoid separable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the separable ellipsoid function (|f`2` in the bbob suite|_).

Both objectives are unimodal and separable. While the first objective is
truly convex-quadratic with a condition number of 1, the second
objective is only globally quadratic with smooth local
irregularities and highly ill-conditioned with a condition number of
about :math:`10^6`.

Contained in the *separable - separable* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_1`: Is symmetry exploited?


.. _f3:

:math:`f_3`: Sphere/Attractive sector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the attractive sector function (|f`6` in the bbob suite|_).

Both objective functions are unimodal, but only the first objective is
separable and truly convex quadratic. The attractive sector
function is highly asymmetric, where only one *hypercone* (with
angular base area) with a volume of roughly :math:`(1/2)^n`
yields low function values. The optimum of it is located at the tip
of this cone. 

Contained in the *separable - moderate* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_1` and :math:`f_{20}`:  What is the
  effect of a highly asymmetric landscape in both or one
  objective?


  
.. _f4:

:math:`f_4`: Sphere/Rosenbrock original
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the original, i.e., unrotated Rosenbrock function (|f`8` in the
bbob suite|_).

The first objective is separable and truly convex, the second
objective is partially separable (tri-band structure). The first
objective is unimodal while the second objective has a local
optimum with an attraction volume of about 25\%.

Contained in the *separable - moderate* function class.


.. .. rubric:: Information gained from this function:

.. * Can the search follow a long path with :math:`n-1` changes in
  the direction when it approaches one of the extremes of the
  Pareto front/Pareto set?





.. _f5:

:math:`f_5`: Sphere/Sharp ridge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the sharp ridge function (|f`13` in the bbob suite|_).

Both objective functions are unimodal.
In addition to the simple, separable, and differentiable first
objective, a sharp, i.e., non-differentiable ridge has to be
followed for optimizing the (non-separable) second objective. The
gradient towards the ridge remains constant, when the ridge is
approached from a given point.
Approaching the ridge is initially effective, but becomes ineffective
close to the ridge when the rigde needs to be followed in direction
to its optimum.  The necessary change in *search behavior* close to
the ridge is difficult to diagnose, because the gradient
towards the ridge does not flatten out.

Contained in the *separable - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * Can the search continuously change its search direction when
  approaching one of the extremes of the Pareto front/Pareto set?
.. * What is the effect of having a non-smooth, non-differentiable
  function to optimize?


.. _f6:

:math:`f_6`: Sphere/Sum of different powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the sum of different powers function (|f`14` in the bbob suite|_).

Both objective functions are unimodal. The first objective is
separable, the second non-separable.
When approaching the second objective's optimum, the difference 
in sensitivity between different directions in search space 
increases unboundedly. 

.. In addition, the second objective function
  possesses a small solution volume.


Contained in the *separable - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:
   

.. _f7:

:math:`f_7`: Sphere/Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the Rastrigin function (|f`15` in the bbob suite|_).

In addition to the simple sphere function, the prototypical highly
multimodal Rastrigin function needs to be solved which has originally
a very regular and symmetric structure for the placement of the optima.
Here, however, transformations are performed to alleviate
the original symmetry and regularity in the second objective.

The properties of the second objective contain non-separabilty,
multimodality (roughly :math:`10^n` local optima), a conditioning of
about 10, and a large global amplitude compared to the local amplitudes.

Contained in the *separable - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * With respect to fully unimodal functions: what is the effect of
  multimodality?

  
.. _f8:

:math:`f_8`: Sphere/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the Schaffer F7 function with condition number 10 (|f`17` in
the bbob suite|_).

In addition to the simple sphere function, an asymmetric, non-separable,
and highly multimodal function needs to be solved to approach the Pareto
front/Pareto set where the frequency and amplitude of the modulation
in the second objective vary. The conditioning of the second objective
and thus the entire bi-objective function is low.

Contained in the *separable - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_7` and :math:`f_{50}`:  What is the
  effect of multimodality on a less regular function?


.. _f9:

:math:`f_9`: Sphere/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the Schwefel function (|f`20` in the bbob suite|_).

While the first objective function is separable and unimodal,
the second objective function is partially separable and highly
multimodal---having the most prominent :math:`2^n` minima located
comparatively close to the corners of the unpenalized search area. 

Contained in the *separable - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison to e.g. :math:`f_8`: What is the effect of a weak
  global structure?

  
.. _f10:

:math:`f_{10}`: Sphere/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sphere function (|f`1` in the bbob suite|_)
and the Gallagher function with 101 peaks (|f`21` in the bbob
suite|_).

While the first objective function is separable and unimodal,
the second objective function is non-separable and consists
of 101 optima with position and height being unrelated and
randomly chosen (different for each instantiation of the function).
The conditioning around the global optimum of the second
objective function is about 30.

Contained in the *separable - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Is the search effective without any global structure?


.. _f11:

:math:`f_{11}`: Ellipsoid separable/Ellipsoid separable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two separable ellipsoid functions (|f`2` in the
bbob suite|_).

Both objectives are unimodal, separable, only globally
quadratic with smooth local irregularities, and highly
ill-conditioned with a condition number of
about :math:`10^6`.

Contained in the *separable - separable* function class.

.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_1`: Is symmetry (rather: separability) exploited?


.. _f12:

:math:`f_{12}`: Ellipsoid separable/Attractive sector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the bbob suite|_) 
and the attractive sector function (|f`6` in the bbob suite|_).

Both objective functions are unimodal but only the first
one is separable. The first objective function, in addition,
is globally quadratic with smooth local irregularities, and
highly ill-conditioned with a condition number of about
:math:`10^6`. The second objective function is highly
asymmetric, where only one *hypercone* (with
angular base area) with a volume of roughly :math:`(1/2)^n`
yields low function values. The optimum of it is located at
the tip of this cone. 

Contained in the *separable - moderate* function class.

.. .. rubric:: Information gained from this function:

.. * In comparison to, for example, :math:`f_1`: Is symmetry exploited?

.. _f13:

:math:`f_{13}`: Ellipsoid separable/Rosenbrock original
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the
bbob suite|_) and the original, i.e., unrotated Rosenbrock function
(|f`8` in the bbob suite|_).

Only the first objective is separable and unimodal. The second
objective is partially separable (tri-band structure) and has a local
optimum with an attraction volume of about 25\%.
In addition, the first objective function shows smooth local
irregularities from a globally convex quadratic function and is
highly ill-conditioned with a condition number of about
:math:`10^6`. 

Contained in the *separable - moderate* function class.


.. .. rubric:: Information gained from this function:

.. * Can the search handle highly conditioned functions and follow a long
  path with :math:`n-1` changes in the direction when it approaches the
  Pareto front/Pareto set?


.. _f14:

:math:`f_{14}`: Ellipsoid separable/Sharp ridge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the
bbob suite|_) and the sharp ridge function (|f`13` in the bbob suite|_).

Both objective functions are unimodal but only the first one is
separable.

The first objective is globally quadratic but with smooth local
irregularities and highly ill-conditioned with a condition number of
about :math:`10^6`. For optimizing the second objective, a sharp,
i.e., non-differentiable ridge has to be followed.

Contained in the *separable - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * Can the search continuously change its search direction when
  approaching one of the extremes of the Pareto front/Pareto set?
.. * What is the effect of having to solve both a highly-conditioned
  and a non-smooth, non-differentiabale function to approximate
  the Pareto front/Pareto set?


.. _f15:

:math:`f_{15}`: Ellipsoid separable/Sum of different powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the
bbob suite|_) and the sum of different powers function
(|f`14` in the bbob suite|_).

Both objective functions are unimodal but only the first one is
separable.

The first objective is globally quadratic but with smooth local
irregularities and highly ill-conditioned with a condition number of
about :math:`10^6`. When approaching the second objective's optimum,
the sensitivies of the variables in the rotated search space become
more and more different.

Contained in the *separable - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * Can the Pareto front/Pareto set be approached when both a
  highly conditioned function and a function, the conditioning
  of which increases when approaching the optimum, must be solved?

.. _f16:

:math:`f_{16}`: Ellipsoid separable/Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the
bbob suite|_) and the Rastrigin function (|f`15` in the bbob suite|_).

The objective functions show rather opposite properties.
The first one is separable, the second not. The first one
is unimodal, the second highly multimodal (roughly :math:`10^n` local
optima). The first one is highly ill-conditioning (condition number of
:math:`10^6`), the second one has a conditioning of about 10. Local
non-linear transformations are performed in both objective functions
to alleviate the original symmetry and regularity of the two
baseline functions.

Contained in the *separable - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * With respect to fully unimodal functions: what is the effect of
  multimodality?
.. * With respect to low-conditioned problems: what is the effect of
  high conditioning?



.. _f17:

:math:`f_{17}`: Ellipsoid separable/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the
bbob suite|_) and the Schaffer F7 function with condition number 10
(|f`17` in the bbob suite|_).

Also here, both single objectives possess opposing properties.
The first objective is unimodal, besides small local non-linearities symmetric,
separable and highly ill-conditioned while the second objective is highly
multi-modal, asymmetric, and non-separable, with only a low conditioning.

Contained in the *separable - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the opposing difficulties posed by the
  single objectives when parts of the Pareto front (at the extremes, in the
  middle, ...) are explored?

  
.. _f18:

:math:`f_{18}`: Ellipsoid separable/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the
bbob suite|_) and the Schwefel function (|f`20` in the bbob suite|_).

The first objective is unimodal, separable and highly ill-conditioned.
The second objective is partially separable and highly multimodal---having
the most prominent :math:`2^n` minima located comparatively close to the
corners of the unpenalized search area. 


Contained in the *separable - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. .. todo::
   Give some details.


.. _f19:

:math:`f_{19}`: Ellipsoid separable/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the separable ellipsoid function (|f`2` in the
bbob suite|_) and the Gallagher function with 101 peaks (|f`21` in the bbob suite|_).

While the first objective function is separable, unimodal, and
highly ill-conditioned (condition number of about :math:`10^6`),
the second objective function is non-separable and consists
of 101 optima with position and height being unrelated and
randomly chosen (different for each instantiation of the function).
The conditioning around the global optimum of the second
objective function is about 30.

Contained in the *separable - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Is the search effective without any global structure?
.. * What is the effect of the different condition numbers
  of the two objectives, in particular when combined
  to reach the middle of the Pareto front?


.. _f20:

:math:`f_{20}`: Attractive sector/Attractive sector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two attractive sector functions (|f`6`
in the bbob suite|_).
Both functions are unimodal and highly asymmetric, where only one
*hypercone* (with angular base area) per objective with a volume of
roughly :math:`(1/2)^n` yields low function values. The objective
functions' optima are located at the tips of those two cones. 

Contained in the *moderate - moderate* function class.

.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_1` and :math:`f_{20}`:  What is the
  effect of a highly asymmetric landscape in both or one
  objective?


  
   
.. _f21:
   
:math:`f_{21}`: Attractive sector/Rosenbrock original
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the attractive sector function (|f`6`
in the bbob suite|_) and the Rosenbrock function (|f`8` in the bbob suite|_).

The first function is unimodal but highly asymmetric, where only one
*hypercone* (with angular base area) with a volume of
roughly :math:`(1/2)^n` yields low function values (with the
optimum at the tip of the cone). The second
objective is partially separable (tri-band structure) and has a local
optimum with an attraction volume of about 25\%.

Contained in the *moderate - moderate* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of relatively large search space areas
  leading to suboptimal values of the two objective
  functions?


.. _f22:
   
:math:`f_{22}`: Attractive sector/Sharp ridge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the attractive sector function (|f`6`
in the bbob suite|_) and the sharp ridge function (|f`13` in the bbob suite|_).

Both objective functions are unimodal and non-separable. The
first objective is highly asymmetric in the sense that only one
*hypercone* (with angular base area) with a volume of
roughly :math:`(1/2)^n` yields low function values (with the
optimum at the tip of the cone). For optimizing the second
objective, a sharp, i.e., non-differentiable ridge has to be followed.

Contained in the *moderate - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * What are the effects of assymmetries and non-differentiabilities
  when approaching the Pareto front/Pareto set?

  
.. _f23:
   
:math:`f_{23}`: Attractive sector/Sum of different powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the attractive sector function (|f`6`
in the bbob suite|_) and the sum of different powers function
(|f`14` in the bbob suite|_).

Both objective functions are unimodal and non-separable. The
first objective is highly asymmetric in the sense that only one
*hypercone* (with angular base area) with a volume of
roughly :math:`(1/2)^n` yields low function values (with the
optimum at the tip of the cone). When approaching the second
objective's optimum, the sensitivies of the variables in the
rotated search space become more and more different.

Contained in the *moderate - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * What are the effects of assymmetries and an increasing
  conditioning in one objective function (sum of different
  powers function) when approaching Pareto-optimal points?
  

.. _f24:
   
:math:`f_{24}`: Attractive sector/Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the attractive sector function (|f`6`
in the bbob suite|_) and the Rastrigin function
(|f`15` in the bbob suite|_).

Both objectives are non-separable, and the second one
is highly multi-modal (roughly :math:`10^n` local
optima) while the first one is unimodal. Further
properties are that the first objective is highly
assymetric and the second has a conditioning of about 10.

Contained in the *moderate - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * With respect to fully unimodal and rather symmetric functions:
  what is the effect of multimodality and assymmetry?


.. _f25:
   
:math:`f_{25}`: Attractive sector/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the attractive sector function (|f`6`
in the bbob suite|_) and the Schaffer F7 function with condition number 10
(|f`17` in the bbob suite|_).

Both objectives are non-separable and asymmetric.
While the first objective is unimodal, the second one is
a highly multi-modal function with a low conditioning where
frequency and amplitude of the modulation vary.

Contained in the *moderate - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of having to solve the relatively` simple, but
  asymmetric first objective together with the highly multi-modal
  second objective with less regularities when the Pareto front/Pareto
  Pareto set is approached?


.. _f26:
   
:math:`f_{26}`: Attractive sector/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the attractive sector function (|f`6`
in the bbob suite|_) and the Schwefel function (|f`20` in the bbob suite|_).

The first objective is non-separable, unimodal, and asymmetric.
The second objective is partially separable and highly multimodal---having
the most prominent :math:`2^n` minima located comparatively close to the
corners of the unpenalized search area. 

Contained in the *moderate - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * What are the effects of asymmetries and a weak global structure when
  different parts of the Pareto front/Pareto set are approached?

  
.. _f27:
   
:math:`f_{27}`: Attractive sector/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the attractive sector function (|f`6`
in the bbob suite|_) and the Gallagher function with 101 peaks (|f`21` in the bbob suite|_).

Both objective functions are non-separable but only the first
is unimodal. The first objective function is furthermore asymmetric.
The second objective function has 101 optima with position and height
being unrelated and randomly chosen (different for each instantiation
of the function). The conditioning around the global optimum of the second
objective function is about 30.

Contained in the *moderate - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Is the search effective without any global structure?
.. * What is the effect of the different condition numbers
  of the two objectives, in particular when combined
  to reach the middle of the Pareto front?


.. _f28:
   
:math:`f_{28}`: Rosenbrock original/Rosenbrock original
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two Rosenbrock functions (|f`8` in the bbob suite|_).

Both objectives are partially separable (tri-band structure) and have
a local optimum with an attraction volume of about 25\%.

Contained in the *moderate - moderate* function class.


.. .. rubric:: Information gained from this function:

.. * Can the search follow different long paths with $n-1$ changes in the
  direction when approaching the extremes of the Pareto front/Pareto set?
.. * What is the effect when a combination of the two paths have to 
  be solved when a point in the middle of the Pareto front/Pareto set
  is sought?

.. _f29:
   
:math:`f_{29}`: Rosenbrock original/Sharp ridge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rosenbrock function (|f`8` in the bbob suite|_) and the 
sharp ridge function (|f`13` in the bbob suite|_).

The first objective function is partially separable (tri-band structure)
and has a local optimum with an attraction volume of about 25\%.
The second objective is unimodal and non-separable and, for
optimizing it, a sharp, i.e., non-differentiable ridge has to be followed.

Contained in the *moderate - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the opposing difficulties posed by the
  single objectives when parts of the Pareto front (at the extremes, in the
  middle, ...) are explored?


.. _f30:
   
:math:`f_{30}`: Rosenbrock original/Sum of different powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rosenbrock function (|f`8` in the bbob suite|_) and the sum of different powers function
(|f`14` in the bbob suite|_).

The first objective function is partially separable (tri-band structure)
and has a local optimum with an attraction volume of about 25\%.
The second objective function is unimodal and non-separable. When
approaching the second objective's optimum, the sensitivies of the
variables in the rotated search space become more and more different.

Contained in the *moderate - ill-conditioned* function class.

.. .. rubric:: Information gained from this function:

.. * What are the effects of having to follow a long path with $n-1$ changes
  in the direction when optimizing one objective function and an increasing
  conditioning when solving the other, in particular when trying to
  approximate the Pareto front/Pareto set not close to their extremes?
  

.. _f31:
   
:math:`f_{31}`: Rosenbrock original/Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rosenbrock function (|f`8` in the bbob suite|_) and the Rastrigin function
(|f`15` in the bbob suite|_).

The first objective function is partially separable (tri-band structure)
and has a local optimum with an attraction volume of about 25\%.
The second objective function is non-separable and
highly multi-modal (roughly :math:`10^n` local
optima).

Contained in the *moderate - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * With respect to fully unimodal functions:
  what is the effect of multimodality?


.. _f32:
   
:math:`f_{32}`: Rosenbrock original/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rosenbrock function (|f`8` in the bbob suite|_) and the 
Schaffer F7 function with condition number 10
(|f`17` in the bbob suite|_).

The first objective function is partially separable (tri-band structure)
and has a local optimum with an attraction volume of about 25\%.
The second objective function is non-separable, asymmetric, and 
highly multi-modal with a low conditioning where
frequency and amplitude of the modulation vary.

Contained in the *moderate - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the different difficulties (in particular
  the high multi-modality of the second objective) when approaching
  the Pareto front/Pareto set, especially in the middle?


.. _f33:
   
:math:`f_{33}`: Rosenbrock original/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rosenbrock function (|f`8` in the bbob suite|_) and the 
Schwefel function (|f`20` in the bbob suite|_).

Both objective functions are partially separable.
While the first objective function has a local optimum with an attraction
volume of about 25\%, the second objective function is highly
multimodal---having the most prominent :math:`2^n` minima located
comparatively close to the corners of its unpenalized search area. 

Contained in the *moderate - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the different difficulties (in particular
  the high multi-modality and weak global structure of the second
  objective) when approaching the Pareto front/Pareto set,
  especially in the middle?
.. * Can the partial separability of the two objectives be detected
  and exploited?


.. _f34:
   
:math:`f_{34}`: Rosenbrock original/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rosenbrock function (|f`8` in the bbob suite|_) and 
the Gallagher function with 101 peaks (|f`21` in the bbob suite|_).

The first objective function is partially separable, the second one
non-separable. While the first objective function has a local optimum
with an attraction volume of about 25\%, the second objective function
has 101 optima with position and height being unrelated and randomly
chosen (different for each instantiation of the function). The
conditioning around the global optimum of the second objective function
is about 30.

Contained in the *moderate - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Is the search effective without any global structure?
.. * How much does the multi-modality play a role when compared to
  fully uni-modal functions?


.. _f35:
   
:math:`f_{35}`: Sharp ridge/Sharp ridge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two sharp ridge functions (|f`13` in the bbob suite|_).

Both objective functions are unimodal and non-separable and, for
optimizing them, two sharp, i.e., non-differentiable ridges have to be
followed.

Contained in the *ill-conditioned - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of having to follow non-smooth, non-differentiabale
  ridges?

  
.. _f36:
   
:math:`f_{36}`: Sharp ridge/Sum of different powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sharp ridge function (|f`13` in the bbob suite|_) and the 
sum of different powers function
(|f`14` in the bbob suite|_).

Both functions are uni-modal and non-separable.
For optimizing the first objective, a sharp, i.e., non-differentiable
ridge has to be followed.
When approaching the second objective's optimum, the sensitivies of the
variables in the rotated search space become more and more different.

Contained in the *ill-conditioned - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * What are the effects of having to follow a ridge when optimizing one
  objective function and an increasing conditioning when solving the other,
  in particular when trying to approximate the Pareto front/Pareto set not
  close to their extremes?
  

.. _f37:
   
:math:`f_{37}`: Sharp ridge/Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sharp ridge function (|f`13` in the bbob suite|_) and the Rastrigin function
(|f`15` in the bbob suite|_).

Both functions are non-separable. While the first one
is unimodal and non-differentiable at its ridge, the second objective
function is highly multi-modal (roughly :math:`10^n` local optima).

Contained in the *ill-conditioned - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What are the effects of having to follow a ridge when optimizing one
  objective function and the high multi-modality of the other,
  in particular when trying to approximate the Pareto front/Pareto set not
  close to their extremes?


.. _f38:
   
:math:`f_{38}`: Sharp ridge/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sharp ridge function (|f`13` in the bbob suite|_) and the 
Schaffer F7 function with condition number 10
(|f`17` in the bbob suite|_).

Both functions are non-separable. While the first one
is unimodal and non-differentiable at its ridge, the second objective
function is asymmetric and highly multi-modal with a low conditioning where
frequency and amplitude of the modulation vary.

Contained in the *ill-conditioned - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the different difficulties when approaching
  the Pareto front/Pareto set, especially in the middle?

  
.. _f39:
   
:math:`f_{39}`: Sharp ridge/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sharp ridge function (|f`13` in the bbob suite|_) and the 
Schwefel function (|f`20` in the bbob suite|_).

While the first objective function is unimodal, non-separable, and
non-differentiable at its ridge, the second objective function is highly
multimodal---having the most prominent :math:`2^n` minima located
comparatively close to the corners of its unpenalized search area. 

Contained in the *ill-conditioned - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the different difficulties (in particular
  the non-differentiability of the first and the high multi-modality
  and weak global structure of the second objective) when approaching
  the Pareto front/Pareto set, especially in the middle?
  
  
.. _f40:
   
:math:`f_{40}`: Sharp ridge/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sharp ridge function (|f`13` in the bbob suite|_) and the 
Gallagher function with 101 peaks (|f`21` in the bbob suite|_).

Both objective functions are non-separable.
While the first objective function is unimodal and non-differentiable at
its ridge, the second objective function
has 101 optima with position and height being unrelated and randomly
chosen (different for each instantiation of the function). The
conditioning around the global optimum of the second objective function
is about 30.

Contained in the *ill-conditioned - weakly-structured* function class.

.. .. rubric:: Information gained from this function:

.. * Is the search effective without any global structure?
.. * How much does the multi-modality of the second objective play a role
  when compared to fully uni-modal functions?


.. _f41:
   
:math:`f_{41}`: Sum of different powers/Sum of different powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two sum of different powers functions
(|f`14` in the bbob suite|_).

Both functions are uni-modal and non-separable where the sensitivies of
the variables in the rotated search space become more and more different
when approaching the objectives' optima.


Contained in the *ill-conditioned - ill-conditioned* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_{11}`:  What is the effect of rotations
  of the search space and missing self-similarity?
   
  
.. _f42:
   
:math:`f_{42}`: Sum of different powers/Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sum of different powers functions
(|f`14` in the bbob suite|_) and the Rastrigin function
(|f`15` in the bbob suite|_).

Both objective functions are non-separable. While the first one
is unimodal, the second objective
function is highly multi-modal (roughly :math:`10^n` local optima).

Contained in the *ill-conditioned - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What are the effects of having to cope with an increasing conditioning
  when optimizing one objective function and the high multi-modality of the
  other, in particular when trying to approximate the Pareto front/Pareto set
  not close to their extremes?


.. _f43:
   
:math:`f_{43}`: Sum of different powers/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sum of different powers functions
(|f`14` in the bbob suite|_) and the Schaffer F7 function with
condition number 10 (|f`17` in the bbob suite|_).

Both objective functions are non-separable. While the first one
is unimodal with an increasing conditioning once the optimum is approached,
the second objective function is asymmetric and highly multi-modal with a
low conditioning where frequency and amplitude of the modulation vary.

Contained in the *ill-conditioned - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the different difficulties when approaching
  the Pareto front/Pareto set, especially in the middle?  
  

.. _f44:
   
:math:`f_{44}`: Sum of different powers/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sum of different powers functions
(|f`14` in the bbob suite|_) and the Schwefel function (|f`20` in the bbob suite|_).

Both objectives are non-separable.
While the first objective function is unimodal,
the second objective function is highly multimodal---having the most
prominent :math:`2^n` minima located comparatively close to the corners
of its unpenalized search area. 

Contained in the *ill-conditioned - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the different difficulties (in particular
  the increasing conditioning close to the first objective's optimum
  and the high multi-modality and weak global structure of the second
  objective) when approaching the Pareto front/Pareto set, especially in
  the middle?


.. _f45:
   
:math:`f_{45}`: Sum of different powers/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the sum of different powers functions
(|f`14` in the bbob suite|_) and the Gallagher function with
101 peaks (|f`21` in the bbob suite|_).

Both objective functions are non-separable.
While the first objective function is unimodal, the second objective function
has 101 optima with position and height being unrelated and randomly
chosen (different for each instantiation of the function). The
conditioning around the global optimum of the second objective function
is about 30.

Contained in the *ill-conditioned - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Is the search effective without any global structure?
.. * How much does the multi-modality of the second objective play a role
  when compared to fully uni-modal functions?


.. _f46:
   
:math:`f_{46}`: Rastrigin/Rastrigin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two Rastrigin functions
(|f`15` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal
(roughly :math:`10^n` local optima).

Contained in the *multi-modal - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * When compared to :math:`f_{11}`: What is the effect of non-separability and
  multi-modality?


.. _f47:
   
:math:`f_{47}`: Rastrigin/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rastrigin function
(|f`15` in the bbob suite|_) and the Schaffer F7 function with
condition number 10 (|f`17` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal.

Contained in the *multi-modal - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the different distributions of local minima 
  when approaching the Pareto front/Pareto set, especially in the middle?  
  

.. _f48:
   
:math:`f_{48}`: Rastrigin/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rastrigin function
(|f`15` in the bbob suite|_) and the Schwefel function (|f`20` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal where
the first has roughly :math:`10^n` local optima and the most prominent
:math:`2^n` minima of the second objective function are located
comparatively close to the corners of its unpenalized search area. 

Contained in the *multi-modal - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * What is the effect of the large amount of local optima in both objectives 
  when approaching the Pareto front/Pareto set, especially in the middle?
  
  
.. _f49:
   
:math:`f_{49}`: Rastrigin/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Rastrigin function
(|f`15` in the bbob suite|_) and the Gallagher function with
101 peaks (|f`21` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal where
the first has roughly :math:`10^n` local optima and the second has 
101 optima with position and height being unrelated and randomly
chosen (different for each instantiation of the function).

Contained in the *multi-modal - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Is the search effective without any global structure?
.. * What is the effect of the differing distributions of local optima
  in the two objective functions? 


.. _f50:
   
:math:`f_{50}`: Schaffer F7, condition 10/Schaffer F7, condition 10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two Schaffer F7 functions with
condition number 10 (|f`17` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal.

Contained in the *multi-modal - multi-modal* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_{46}`: What is the effect of multimodality
  on a less regular function?
  

.. _f51:
   
:math:`f_{51}`: Schaffer F7, condition 10/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Schaffer F7 function with
condition number 10 (|f`17` in the bbob suite|_)
and the Schwefel function (|f`20` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal.
While frequency and amplitude of the modulation vary in an almost
regular fashion in the first objective function, the second objective
function posseses less global structure.

Contained in the *multi-modal - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * What are the effects of different global structures in the two
  objective functions?


.. _f52:
   
:math:`f_{52}`: Schaffer F7, condition 10/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Schaffer F7 function with
condition number 10 (|f`17` in the bbob suite|_)
and the Gallagher function with
101 peaks (|f`21` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal.
While frequency and amplitude of the modulation vary in an almost
regular fashion in the first objective function, the second has 
101 optima with position and height being unrelated and randomly
chosen (different for each instantiation of the function).

Contained in the *multi-modal - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Similar to :math:`f_{51}`: What are the effects of different
  global structures in the two objective functions?


.. _f53:
   
:math:`f_{53}`: Schwefel x*sin(x)/Schwefel x*sin(x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two Schwefel functions (|f`20` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal where
the most prominent :math:`2^n` minima of each objective function are
located comparatively close to the corners of its unpenalized search area.
Due to the combinatorial nature of the Schwefel function, it is likely
in low dimensions that the Pareto set goes through the origin of the
search space.

Contained in the *weakly-structured - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison with :math:`f_{50}`: What is the effect of a weak global
  structure?
.. * Can the search algorithm benefit from Pareto-optimal search points
  it can get from random samples close to the origin on some of the
  function' instances?


.. _f54:
   
:math:`f_{54}`: Schwefel x*sin(x)/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of the Schwefel function (|f`20` in the bbob suite|_) and the Gallagher function with
101 peaks (|f`21` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal.
For the first objective function, the most prominent :math:`2^n` minima
are located comparatively close to the corners of its unpenalized search
area. For the second objective, position and height of all  
101 optima are unrelated and randomly
chosen (different for each instantiation of the function).

Contained in the *weakly-structured - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * In comparison to :math:`f_{53}`: Does the total absence of a global
  structure in one objective change anything in the performance of the
  algorithm?


.. _f55:
   
:math:`f_{55}`: Gallagher 101 peaks/Gallagher 101 peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Combination of two Gallagher functions with
101 peaks (|f`21` in the bbob suite|_).

Both objective functions are non-separable and highly multi-modal.
Position and height of all 101 optima in each objective function
are unrelated and randomly chosen and thus, no global structure
is present.

Contained in the *weakly-structured - weakly-structured* function class.


.. .. rubric:: Information gained from this function:

.. * Can the Pareto front/Pareto set be found efficiently when no global
  structure can be exploited?


.. _`Coco framework`: https://github.com/numbbo/coco


.. raw:: html
    
    <H2>Acknowledgments</H2>

.. raw:: latex

    \section*{Acknowledgments}

This work was supported by the grant ANR-12-MONU-0009 (NumBBO) 
of the French National Research Agency.

 
.. ############################# References #########################################
.. raw:: html
    
    <H2>References</H2>
   

.. [BRO2016biperf] D. Brockhoff, T. Tušar, D. Tušar, T. Wagner, N. Hansen, A. Auger, (2016). 
  `Biobjective Performance Assessment with the COCO Platform`__. *ArXiv e-prints*, `arXiv:1605.01746`__.
.. __: http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment
.. __: http://arxiv.org/abs/1605.01746

.. [BRO2015] D. Brockhoff, T.-D. Tran, and N. Hansen (2015).
   Benchmarking Numerical Multiobjective Optimizers Revisited. In
   Proceedings of the 2015 GECCO Genetic and Evolutionary Computation Conference, 
   pp. 639-646, ACM. 
   
.. [HAN2016co] N. Hansen, A. Auger, O. Mersmann, T. Tušar, D. Brockhoff (2016).
   `COCO: A Platform for Comparing Continuous Optimizers in a Black-Box 
   Setting`__, *ArXiv e-prints*, `arXiv:1603.08785`__. 
.. __: http://numbbo.github.io/coco-doc/
.. __: http://arxiv.org/abs/1603.08785


.. [HAN2009] N. Hansen, S. Finck, R. Ros, and A. Auger (2009). 
   `Real-parameter black-box optimization benchmarking 2009: Noiseless
   functions definitions`__. `Research Report RR-6829`__, Inria, updated
   February 2010.
.. __: http://coco.gforge.inria.fr/
.. __: https://hal.inria.fr/inria-00362633

.. [HAN2011] N. Hansen, R. Ros, N. Mauny, M. Schoenauer, and A. Auger (2011). Impacts
	of Invariance in Search: When CMA-ES and PSO Face Ill-Conditioned and
	Non-Separable Problems. *Applied Soft Computing*. Vol. 11, pp. 5755-5769.
	Elsevier.  

.. [HAN2016ex] N. Hansen, T. Tušar, A. Auger, D. Brockhoff, O. Mersmann (2016). 
  `COCO: The Experimental Procedure`__, *ArXiv e-prints*, `arXiv:1603.08776`__. 
.. __: http://numbbo.github.io/coco-doc/experimental-setup/
.. __: http://arxiv.org/abs/1603.08776

  
