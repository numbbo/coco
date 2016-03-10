============
Introduction
============

In the following, we consider bi-objective, unconstrained **minimization**
problems of the form

.. math::
  \min_{x \in \mathbb{R}^n} f(x)=(f_1(x),f_2(x))

with :math:`n` being the number of variables of the problem (also called
the problem dimension), :math:`f_1: \mathbb{R}^n \rightarrow \mathbb{R}`
and :math:`f_2: \mathbb{R}^n \rightarrow \mathbb{R}` being the two unconstrained
objective functions, and the :math:`\min` operator related to the
standard *dominance relation*. A solution :math:`x\in\mathbb{R}^n`
is thereby said to *dominate* another solution :math:`y\in\mathbb{R}^n` if
:math:`f_1(x) \leq f_1(y)` and :math:`f_2(x) \leq f_2(y)` hold and at
least one of the inequalities is strict.

Solutions which are not dominated by any other solution in the search
space are called *Pareto-optimal* or *efficient solutions*. All
Pareto-optimal solutions constitute the *Pareto set* of which an 
approximation is sought. The Pareto set's image in the
objective space :math:`f(\mathbb{R}^n)` is called *Pareto front*.

.. The *ideal point* is defined as the vector (in objective space) 
.. containing the optimal function values of the (two) objective functions.


Definitions and Terminology
---------------------------
We remind in this section different definitions.

*function, instance*
 A function within COCO is a parametrized function :math:`f_\theta:
 \mathbb{R}^n \to \mathbb{R}^m` with :math:`\theta \in \Theta` a set of
 parameters. A parameter determines a so-called instance. For example,
 :math:`\theta` encodes the location of the optimum and two different
 instances have shifted optima.
 
 The integer :math:`n` is the dimension of the search space and
 :math:`m=2` for the  ``bbob-biobj`` test suite. 

*ideal point*
 The ideal point is defined as the vector (in objective space)
 containing the optimal function values of the (two) objective
 functions. More precisely let :math:`f_1^{\rm opt}:= \inf_{x\in \mathbb{R}^n} f_1(x)` and
 :math:`f_2^{\rm opt}:= \sup_{x\in \mathbb{R}^n} f_2(x)`, the ideal point is given by
 
 .. math::
    :nowrap:

	\begin{equation*}
	z_{\rm ideal}  =  (f_1^{\rm opt},f_2^{\rm opt})
    \end{equation*}
    

 
*nadir point* 
 The *nadir point* (in objective space) consists in each objective of
 the worst value obtained by a Pareto-optimal solution. More precisely
 let :math:`\mathcal{PO}` be the set of Pareto optimal points then the nadir point satisfies
 
 .. math::
    :nowrap:

	\begin{equation*}
	z_{\rm nadir}  =   \left( \sup_{x \in \mathcal{PO}} f_1(x),
     \sup_{x \in \mathcal{PO}} f_2(x)  \right)
    \end{equation*} 
    
    
    
