============
Introduction
============

In the following, we consider bi-objective, unconstrained **minimization**
problems of the form

.. math::
  \min_{x \in \mathbb{R}^n} f(x)

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

The *ideal point* is defined as the vector (in objective space) 
containing the optimal function values of the (two) objective functions.
The *nadir point* (in objective space) consists in each objective
of the worst value obtained by a Pareto-optimal solution.
