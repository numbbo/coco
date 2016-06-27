.. title:: COCO: The Linearly-Constrained Black-Box Optimization Benchmarking (bbob-constr-lin) Test Suite

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COCO: The Linearly-Constrained Black-Box Optimization Benchmarking (``bbob-constr-lin``) Test Suite
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$     

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

.. raw:: latex

  % \tableofcontents TOC is automatic with sphinx and moved behind abstract by swap...py
  \begin{abstract}

The ``bbob-constr-lin`` test suite contains 48 constrained functions in continuous domain 
which are derived from combining functions of the well-known single-objective noiseless
``bbob`` test suite with randomly-generated linear constraints. This document aims to define 
those constrained functions as well as the approach used to build the linear constraints. 
It also describes how instances, targets and runtime are considered in the constrained case.

.. raw:: latex

  \end{abstract}
  \newpage





.. _COCO: https://github.com/numbbo/coco
.. _COCOold: http://coco.gforge.inria.fr
.. |coco_problem_t| replace:: 
  ``coco_problem_t``
.. _coco_problem_t: http://numbbo.github.io/coco-doc/C/coco_8h.html#a408ba01b98c78bf5be3df36562d99478

.. |coco_evaluate_constraint| replace:: 
  ``coco_evaluate_constraint``
.. _coco_evaluate_constraint: 
  http://numbbo.github.io/coco-doc/C/coco_8h.html#ab5cce904e394349ec1be1bcdc35967fa

.. |f| replace:: :math:`f`
.. |g| replace:: :math:`g`

.. summarizing the state-of-the-art in linearly-constrained black-box benchmarking, 
.. and at providing a simple tutorial on how to use these functions for actual benchmarking within the Coco framework.

.. .. Note::
  
  For the time being, this documentation is under development and might not 
  contain all final data.


.. #################################################################################
.. #################################################################################
.. #################################################################################



Introduction
============
We consider constrained optimization problems of the form

.. math:: 
   :nowrap:
   :label: cons_prob

   \begin{eqnarray*}
   \begin{array}{rcl}
   \displaystyle\min_{x \in \mathbb{R}^D} & f(x) & \\
          \textrm{s.t.:} & g_i(x) \leq 0, & i = 1, \ldots, l,\\
   \end{array}
   \end{eqnarray*}

where :math:`D` is the number of variables of the problem (also called
the problem dimension), :math:`f: \mathbb{R}^D \rightarrow \mathbb{R}`
is the objective function, and :math:`g_i: \mathbb{R}^D \rightarrow \mathbb{R}`, :math:`i = 1, \ldots, l`, 
are the constraint functions defined by :math:`g_i(x) \equiv a_i^T x`, with :math:`a_i \in \mathbb{R}^{D}`.

A point :math:`x` is said to be a *feasible solution* if :math:`g_i(x) \leq 0` for all :math:`i`.
The set of all feasible solutions is called the *feasible set*. An *optimal
solution* is a feasible solution that has the lowest objective function value
among all the feasible solutions.

Before delving into detail about how the constrained problems are built, we familiarize the reader with some terms that are used in the `Coco framework` and, in particular, in this suite.

.. |n| replace:: :math:`n`
.. |m| replace:: :math:`m`
.. |theta| replace:: :math:`\theta`
.. |i| replace:: :math:`i`
.. |j| replace:: :math:`j`
.. |k| replace:: :math:`k`
.. |t| replace:: :math:`t`
.. |fi| replace:: :math:`f_i`

Terminology
-----------

*Objective function* or, simply, *function*
  An objective function :math:`f`, also referred to as function in `Coco`, is a parametrized mapping
  :math:`\mathbb{R}^n\to\mathbb{R}` with scalable input space, :math:`n\ge2`.
  Functions are parametrized such that different *instances* of the
  "same" function are available, e.g. translated or shifted versions. 

*Constraint*
  A constraint function :math:`g_i` is a parametrized mapping
  :math:`\mathbb{R}^n\to\mathbb{R}` with scalable input space, :math:`n\ge2`.
  Constraints are also parametrized such that different *instances* of the
  "same" constraint are available, e.g. translated or shifted versions. 
  Since translations and shiftings of linear constraints result on different
  linear constraints, different instances of a constraint in the ``bbob-constr-lin``
  test suite are equivalent to different constraints. In the next section,
  we give more details on the parameters of a constraint and how they are used
  to define the instances.

*Constrained function*
  A *constrained function* is a function :math:`f(x)` subject to constraints 
  :math:`g_i(x)\leq 0`, :math:`i=1,\ldots,l`. For example, in :eq:`cons_prob`, we are minimizing a constrained function. 
  Constrained functions are parametrized such that different *instances* of the
  "same" constrained function are available. In the next section, we describe
  how the idea of instance is employed in the constrained case.
  
*runtime*
  We define *runtime*, or *run-length* as the *number of
  evaluations* conducted on a given problem until a prescribed target value is
  hit, also referred to as the sum of the number of *objective function* evaluations 
  or |f|-evaluations and the number of *constraint* evaluations 
  or |g|-evaluations. We emphasize here that one single *constraint evaluation* in `Coco` is equivalent
  to one call to the routine that evaluates *all* the constraints at once, 
  |coco_evaluate_constraint|_. Runtime is our central performance measure.


Instances and problems
----------------------

*constrained function instance*

  Each constrained function is parametrized by the (input) dimension, |n|, its identifier |i|, 
  and the instance number, |j|.

  The parameter value |j| determines a so-called *constrained function 
  instance*. It is used in the ``bbob-constr-lin`` test suite to 

  (1) define an instance of the objective function,
  (2) define an instance of the constraints for the constrained function,
  (3) encode the location of the optimal solution of the constrained function.

  **IMPORTANT:** constrained functions with the same objective function but with different constraints are 
  distinguished from each other and thus have different identifiers.

  A constraint :math:`g_k` in a constrained function :math:`f_i` is parametrized by the (input) 
  dimension, |n|, the identifier of the constrained function, |i|, its identifier within the 
  set of constraints, :math:`k\in\{1,\ldots,l\}`, and the instance number of the constrained 
  function, |j|. The parametrized constrained function :math:`f_i` is then denoted by

  .. math::

    f(n, i, j)(\x) \quad \textrm{subject to}\quad g(n, i, k, j)(\x) \leq 0, \quad k = 1, \ldots, l.

  The rationale behind the use of parameter |k| is the following. Suppose that a constrained function
  is composed of the objective function and two linear constraints :math:`g_1(x) \equiv a_i^T x` and 
  :math:`g_2(x) \equiv a_2^T x`. The gradients :math:`a_1` and :math:`a_2` are randomly generated
  using the identifier of the constrained function, |i|, and the instance number, |j|, in the seed
  formula. However, using only these values for generating these vectors would result on
  identical gradients as the same seed would be used in the building process. By using also their identifiers, 
  1 and 2, we can generate different gradients.

  As previously mentioned, translations and shiftings of linear constraints result on different
  linear constraints. Therefore, different instances of a constraint in the ``bbob-constr-lin``
  test suite are equivalent to different constraints. 

*problem*
  We talk about a *problem*, |coco_problem_t|_, as a specific *constrained function instance* 
  on which an optimization algorithm is run. 
  A problem
  can be evaluated and returns an |f|-value or -vector and, in case,
  a |g|-vector. 
  In the context of performance assessment, a target :math:`f`- or
  indicator-value is added to define a problem. 


Raw functions and transformed functions
---------------------------------------

In the `Coco framework`, we call raw functions those functions without any linear or nonlinear transformation as opposed to a transformed function, which is a raw function where some transformation has been applied to. For example, the *raw sphere function* in the ``bbob`` test suite is given by

.. math::

  f(x) = \| x \|^2.

As it can be seen, neither linear nor nonlinear transformations have been applied to the function above. The transformed sphere function -- or simply, as it is called in `Coco`, *sphere function* -- is defined as

.. math::

  f(x) = \|x-x_{\textrm{opt}}\|^2 + f_{\textrm{opt}},

where the vector :math:`x_{\textrm{opt}}` and the scalar :math:`f_{\textrm{opt}}` are constants whose values depend on the function identifier and instance number. These constants determine the optimal solution and the optimal function value of the problem, respectively.

Linear transformations, by definition, do not change some properties of the functions to which they are applied to, such as symmetry. In order to make the functions less regular, `Coco` makes use of two nonlinear transformations, namely, :math:`T_{asy}^{\beta}` and :math:`T_{osz}` [HAN2009]_. The former is a symmetry breaking transformation while the latter introduces small, smooth but clearly visible irregularities. These nonlinear transformations can transform convex raw functions into nonconvex functions, for instance.


Overview of the proposed ``bbob-constr-lin`` test suite
=======================================================

The ``bbob-constr-lin`` test suite provides 48 constrained functions in six
dimensions (2, 3, 5, 10, 20, and 40) with a large number of possible instances. 
The 48 functions are derived from combining 8 single-objective functions 
with 6 different numbers of linear constraints: 1, 2, 10, dimension/2, dimension-1 
and dimension+1.

While concrete details on each of
the 48 ``bbob-constr-lin`` constrained functions are given in Section
:ref:`sec-test-constrained-functions`, we will detail here the main rationale behind
them together with their common properties.


Main features
-------------

We summarize below the main features of the constrained functions in the ``bbob-constr-lin`` test suite.

* Linear constraints
  
* Scalable with dimension

* Non-trivial, with a few exceptions
  
* Mostly non-separable
  
* Known optimal function values

* Use many functions already implemented in `Coco` as objective functions

* Different number of constraints: :math:`1`, :math:`2`, :math:`10`, :math:`n/2`, :math:`n-1`, :math:`n+1`

* The constraints are randomly generated

.. _subsec-how-cons-are-built:

How the constraints are built
-----------------------------

The linear constraints :math:`g_i` are defined by their gradients :math:`a_i` which are randomly generated using a normal distribution.
In order to make sure that the resulting feasible set is not empty, the following steps are considered in the generation process.
  
1) Sample :math:`l` vectors :math:`a_1`, :math:`a_2`, :math:`\ldots`, :math:`a_l`.

2) Choose a point :math:`p` that will be a feasible solution.

3) For each vector :math:`a_i` such that :math:`a_i^T p > 0`, redefine :math:`a_i = -a_i`.

4) Define the constraints :math:`g_i` using the vectors :math:`a_1`, :math:`a_2`, :math:`\ldots`, :math:`a_m`.

The algorithm above ensures a feasible half-line defined by :math:`\{\alpha p\,|\,\alpha\geq0\}`.


Defining the optimal solution
-----------------------------

The constrained functions are defined in a way such that their optimal solutions are different from the optimal solutions of their unconstrained counterparts. The reason for this choice lies in the fact that if both optimal solutions were equal, the constraints would have no major impact in the difficulty of the problem in the sense that an algorithm for unconstrained optimization could be run and obtain the optimal solution without considering any constraint.

Before describing how we define a point as an optimal solution to a constrained function, we introduce some conditions for a solution to be optimal called the *Karush-Kuhn-Tucker* conditions, or, simply, the KKT conditions. Such conditions are first-order necessary conditions for optimality given that the problem satisfies some regularity conditions. 

**The Karush-Kuhn-Tucker conditions**

Suppose that the function :math:`f` and the constraints :math:`g_i` are continuously differentiable at a point :math:`x^*` and that the problem satisfies some regularity conditions (for instance, the functions :math:`g_i` being affine, which is our case as they are linear). If :math:`x^*` is a local optimal solution, then there exist constants :math:`\mu_i`, :math:`i=1,\ldots,l` called *Lagrange multipliers*, such that

.. math::
  
  \nabla f(x^*) + \displaystyle\sum_{i=1}^m \mu_i \nabla g_i(x^*) = 0, 

  g_i(x^*) \leq 0, \quad i = 1, \ldots, l,

  \mu_i g_i(x^*) = 0, \quad i = 1, \ldots, l,

  \mu_i \geq 0, \quad i = 1, \ldots, l.

A point that satisfies the KKT conditions is called a *KKT point*. Note that a KKT point is not necessarily a local optimal solution. The KKT conditions may be sufficient for optimality if some additional conditions are satisfied; for instance, if the objective function and the constraints :math:`g_i` are convex and constinuously differentiable over :math:`\R^D`. Furthermore, when convexity holds, the KKT point is also a global optimal solution. A more general result states that if the objective function is pseudoconvex and the constraints :math:`g_i` are quasiconvex, then the KKT conditions are sufficient for optimality and the KKT point is a global optimal solution. 

**Transformations in the bbob-constr-lin test suite**

We explain here the role of transformation functions in the ``bbob-constr-lin`` test suite. In principle, a transformation function may be applied to

* the objective function only;

* the constraints only;

* the entire constrained function (objective function + constraints).

Most of the ``bbob-constr-lin`` constrained functions are built by picking up a function from the ``bbob`` test suite to be the objective function and by generating linear constraints in a way that the optimal solution is well defined and known a priori. Once the constrained function is well composed, a linear transformation is applied to it, which means that the transformation is equally applied to the objective function and to the constraints.

We note that a ``bbob`` function is a transformed function itself containing linear and possibly nonlinear transformations. However, in the definition of the ``bbob-constr-lin`` constrained functions, all the nonlinear transformations to the search space are applied first, which means that if some linear transformation (e.g. rotation, translation) is applied to the search space, then this transformation must come after all the nonlinear ones, if any. For example, consider the following transformed sphere function where both linear and nonlinear transformations have been applied to:

.. math::
  :label: trans_sphere1

  f(x) = \|z\|^2,\quad z = R\,(T_{asy}^{\beta}(T_{osz}(x))-x_{\textrm{opt}}),

where :math:`R` is a rotation matrix, :math:`T_{asy}^{\beta}` and :math:`T_{osz}` are nonlinear transformations and the constant vector :math:`x_{\textrm{opt}}` defines a translation in the search space. Note that :math:`T_{asy}^{\beta}` and :math:`T_{osz}` are first applied to the search space defined by the variables :math:`x`, and only then a translation followed by a rotation are applied. Since a ``bbob`` function that has been chosen to the define a constrained function may not follow this rule, we solve this issue by "shifting" the nonlinear transformation that are misplaced. For example, a transformed function such as

.. math::
  :label: trans_sphere2

  f(x) = \|z\|^2,\quad z = R\,T_{asy}^{\beta}T_{osz}(x-x_{\textrm{opt}}),

would become the one in :eq:`trans_sphere1`.

One particular note on nonlinear transformations is that, whenever they are present, they are equally applied to the objective function and to the constraints. That implies that, if a ``bbob`` function that has been chosen to be the objective function contains a nonlinear transformation, then this transformation will (possibly) be "shifted" and it will be applied to the constraints as well. 

Another remark is that nonlinear transformations are initially not considered in the process of definition of the optimal solutions. Therefore, in what follows, we assume that no nonlinear transformation has been applied yet.

**General construction**

7 out of the 8 objective functions - all except the Rastrigin function - composing the constrained functions in ``bbob-constr-lin`` are convex or pseudoconvex - without considering the nonlinear transformations -, which together with the fact that the linear constraints are also quasiconvex implies that a KKT point is also a global optimal solution to the constrained function. Those 7 functions are taken from the ``bbob`` test suite and have any nonlinear transformation temporarily removed. The nonlinear transformations are applied afterwards to the whole constrained function (:math:`f` and :math:`g_i`) while respecting the rule that they should come before than the linear transformations (the application of the nonlinear transformations to the constrained function will not change its optimal solution as it is proved in the next subsection). 

We start by choosing the optimal solution to be initially at the origin. Since :math:`g_i(\mathbf{0}) \equiv a_i^T \mathbf{0} = 0 \leq 0`, for all :math:`i`, it follows that the origin is a feasible solution, which implies that the second KKT condition is satisfied. In order to achieve optimality, we define the gradient of the first constraint, :math:`a_1`, as :math:`a_1=-\nabla f(\mathbf{0})`. All the other linear constraints are randomly generated using a normal distribution as described in Subsection :ref:`subsec-how-cons-are-built`. The point :math:`p` that is chosen to guarantee nonemptiness of the feasible set is defined here as :math:`p=\nabla f(\mathbf{0})`. By setting the Lagrange multipliers :math:`\mu_1 = 1` and :math:`\mu_i = 0` for :math:`i=2,\ldots,l`, we have that all the KKT conditions are satisfied and that the origin is a KKT point.


**Defining the optimal solution for the Rastrigin function**

The process described before works for all the constrained functions in the current test suite except the one involving the Rastrigin function whose definition in ``bbob-constr-lin`` differs from that in the ``bbob`` test suite. The constrained Rastrigin function here is defined by

.. math::
   :label: rastrigin
   :nowrap:

   \begin{eqnarray*}
   \begin{array}{rc}
                         & f(x) = 10\bigg(D - \displaystyle\sum_{i=1}^{D}\cos(2\pi v_i) \bigg) + \|v\|^2 + f_{\textrm{opt}} \\
          \textrm{subject to} & g_i(z) \leq 0, \quad i = 1, \ldots, l,\\
   \end{array}
   \end{eqnarray*}

where :math:`v = z-x_{\textrm{opt}}` and :math:`z = T_{asy}^{\beta}(T_{osz}(x))`. Without the nonlinear transformations, it becomes

.. math::
   :label: rastrigin2
   :nowrap:

   \begin{eqnarray*}
   \begin{array}{rcl}
                         & f(x) = 10\bigg(D - \displaystyle\sum_{i=1}^{D}\cos\Big(2\pi (x-x_{\textrm{opt}})_i)\Big) \bigg) + \|x-x_{\textrm{opt}}\|^2 + f_{\textrm{opt}} \\
          \textrm{subject to} & g_i(x) \leq 0, \quad i = 1, \ldots, l.\\
   \end{array}
   \end{eqnarray*}

Differently from the other 7 constrained functions, :eq:`rastrigin2` does not have a pseudoconvex objective function, but a multimodal one. Therefore, we define the optimal solution in this case in a different manner. We first set the constant vector :math:`x_{\textrm{opt}}=(-1,\ldots,-1)^T`. We then obtain a Rastrigin function whose unconstrained global optimal solution is at :math:`x_{\textrm{opt}}=(-1,\ldots,-1)^T`. By defintion, such a function contains many local optimal solution which are (approximately) located on the :math:`n`-dimensional integer lattice :math:`\mathbb{Z}^n` translated by :math:`-x_{\textrm{opt}}`.

After the translation, the origin vector is no more the unconstrained global optimal solution, but an unconstrained local optimal solution. In order to make it the constrained global optimal solution, we add a linear constraint function :math:`g_1(x) \equiv a_1^T x` whose gradient is given by :math:`a_1 = (-1,\ldots,-1)^T`. The Figure shows a 2-dimensional example of the resulting function. As it can be seen, all feasible solutions different from the origin have larger function value. Next, all the other linear constraints are randomly generated while guaranteeing that the point :math:`p=(1,\ldots,1)^T` remains feasible.

Applying nonlinear transformations
----------------------------------

As we mentioned in the previous subsection, we initially do not consider the nonlinear transformations in the building process of the constrained functions. Those transformations are applied after defining the optimal solutions. The application of nonlinear transformations to the constrained functions, however, do not affect the location of the optimal solutions already defined as we show next.

Without loss of generality, consider the minimization of a constrained function with one single linear constraint:

.. math:: 
   :nowrap:
   :label: cons_prob_trans1

   \begin{eqnarray*}
   \begin{array}{rc}
   \displaystyle\min_{x \in \mathbb{R}^D} & f(x) \\
          \textrm{s.t.:} & g(x) \equiv a^T x \leq 0. \\
   \end{array}
   \end{eqnarray*}

Let :math:`t:\Re^n \rightarrow \Re^n` be an injective transformation function.  By "applying" :math:`t` to :eq:`cons_prob_trans1`, we obtain

.. math:: 
   :nowrap:
   :label: cons_prob_trans2

   \begin{eqnarray*}
   \begin{array}{rc}
   \displaystyle\min_{x \in \mathbb{R}^D} & f(t(x)) \\
          \textrm{s.t.:} & g(t(x)) \equiv a^T t(x) \leq 0. \\
   \end{array}
   \end{eqnarray*}


Assume that :math:`x^*` is an optimal solution to :eq:`cons_prob_trans1`. Since :math:`t` is injective, it has an inverse :math:`t^{-1}`. It follows that :math:`t^{-1}(x^*)` is an optimal solution to :eq:`cons_prob_trans2`.

**Proof:** Suppose, by contradiction, that :math:`t^{-1}(x^*)` is not an optimal solution to :eq:`cons_prob_trans2`. Then there exists a point :math:`u` such that 

.. math:: 
   :label: nonlin_trans_proof

   g(t(u)) = a^T t(u) \leq 0\quad  \textrm{ and } \quad f(t(u)) < f(t(t^{-1}(x^*))) = f(x^*)

Let :math:`z = t(u)`. Then, :math:`t^{-1}(z) = u`. By :eq:`nonlin_trans_proof` and the property of inverse functions, we have that

.. math:: 

   a^T z \leq 0\quad \textrm{ and } \quad f(t(u)) = f(t(t^{-1}(z)) = f(z) < f(x^*).  

This contradicts the assumption of :math:`x^*` being a global minimum to problem :eq:`cons_prob_trans1`. **Q.E.D.**


Since the transformations :math:`T_{asy}^{\beta}` and :math:`T_{osz}` in ``Coco`` are strictly increasing functions, they both are injective, thus having inverse functions. Since the optimal solution :math:`x^*` to :eq:`cons_prob_trans1` is defined as the origin in the construction of the constrained functions, the proof given above implies that :math:`t^{-1}(x^*) = t^{-1}(\mathbf{0})` is an optimal solution to :eq:`cons_prob_trans2`. Besides that, by definition of :math:`T_{asy}^{\beta}` and :math:`T_{osz}`, we have that :math:`t(\mathbf{0})=\mathbf{0}` for any of these two transformations. Using this together with the fact that :math:`t(t^{-1}(\mathbf{0}))=\mathbf{0}` (by the property of inverse functions) and by the injectivity of :math:`t`, we must have :math:`t^{-1}(\mathbf{0})=\mathbf{0}`. This implies that the origin is still the optimal solution after any of these two nonlinear transformations have been applied to the constrained function.


Algorithm for the construction of the constrained functions
-----------------------------------------------------------

The steps for constructing the constrained functions described in the previous subsections can be summarized in an algorithmic way as it follows.

1. Remove the nonlinear transformation of the ``bbob`` function :math:`f` chosen to be the objective function.

2. Define the feasible direction :math:`p` as :math:`\nabla f(\mathbf{0})`.

3. Define the first constraint function :math:`g_1(x)` by setting its gradient to :math:`a_1 = -p`.

4. Generate the other constraints randomly while making sure that :math:`p` remais feasible for each one.

5. Possibly apply nonlinear transformations to the constrained function.

6. Move the optimal solution away from the origin by applying a translation to the constrained function.


.. _sec-test-constrained-functions:

The ``bbob-constr-lin`` constrained functions and their properties
==================================================================

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

  
