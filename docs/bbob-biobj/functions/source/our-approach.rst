==================================================
Overview of the Proposed ``bbob-biobj`` Test Suite
==================================================

The ``bbob-biobj`` test suite provides 55 bi-objective functions in six
dimensions (2, 3, 5, 10, 20, and 40) with arbitrary many instances (10 of them being typically fixed for a BBOB workshop).
The 55 functions are derived from combining a subset of the 24 well-known
single-objective functions of the ``bbob`` test suite which
has been used since 2009 in the `BBOB workshop series
<http://numbbo.github.io/workshops/>`_ . While concrete details on each of
the 55 ``bbob-biobj`` functions will be given in
:ref:`sec-test-functions`, we will detail here the main rationals behind
them together with their common properties.


The Single-objective ``bbob`` Functions Used
--------------------------------------------
The ``bbob-biobj`` test suite is designed to represent typical difficulties obverved in
real-world optimization problems. It is based on the fact that a multi-objective problem is a combination of single-objective functions and that thus one can build multi-objective problems with representative difficulties by simply combining single objective functions with representative difficulties observed in real-world problems.
We naturally use the single-objective ``bbob`` suite designed to be representative of many difficulties observed in real-world problems.


Combining all 24 ``bbob`` functions in pairs thereby results in
:math:`24^2=576` bi-objective functions overall. Given
that most (if not all) multi-objective optimization algorithms are
invariant to permutations of the objective functions, a
bi-objective function combining for example the sphere function
as the first objective with the Rastrigin function as the second
objective will result in the same performance than if the Rastrigin
function is the first and the sphere function is the second
objective function. Hence, we should keep only one of the resulting
bi-objective functions. Combining then all 24 ``bbob`` functions
in the way that the first objective is chosen as ``bbob`` function
*i* and the second as ``bbob`` function *j* with *i* :math:`\leq` *j*,
results in :math:`24+ {24 \choose 2} = 300` functions.

First tests, e.g. in [BTH2015a]_, showed that having 300 functions
in Coco's first bi-objective suite is impracticable in terms
of the overall running time of the benchmarking experiment. Hence,
we consider only the following 10 of the 24 ``bbob``
functions:

* Separable functions

  - Sphere (:math:`f_1` in ``bbob`` suite)
  - Ellipsoid separable (:math:`f_2` in ``bbob`` suite)

* Functions with low or moderate conditioning 

  - Attractive sector (:math:`f_6` in ``bbob`` suite)
  - Rosenbrock original (:math:`f_8` in ``bbob`` suite)

* Functions with high conditioning and unimodal 

  - Sharp ridge (:math:`f_{13}` in ``bbob`` suite)
  - Sum of different powers (:math:`f_{14}` in ``bbob`` suite)

* Multi-modal functions with adequate global structure 

  - Rastrigin (:math:`f_{15}` in ``bbob`` suite)
  - Schaffer F7, condition 10 (:math:`f_{17}` in ``bbob`` suite)

* Multi-modal functions with weak global structure 

  - Schwefel x*sin(x) (:math:`f_{20}` in ``bbob`` suite)
  - Gallagher 101 peaks (:math:`f_{21}` in ``bbob`` suite)

  
Using the above described pairwise combinations, this results in
having :math:`10+{10 \choose 2} = 55` bi-objective functions in
the final `bbob-biobj` suite. The next section gives the
reasoning behind choosing exactly these 10 functions.

  

Function Groups and the Rational Behind Our Choice of Functions
---------------------------------------------------------------
The 24 original ``bbob`` suite are grouped into five function
classes which represent common difficulties obtained in practice
(separable functions, functions with low or moderate conditioning,
functions with high conditioning and unimodal, multi-modal
functions with adequate global structure, and multi-modal
functions with weak global structure).
The idea behind the ``bbob`` function classes is that functions
within a class should share common properties and the performance
of algorithms will be more similar within than across groups.

The above ten ``bbob`` functions have been chosen for the creation
of the ``bbob-biobj`` suite in a way to not introduce any bias
towards a specific difficulty
by choosing exactly two functions per ``bbob`` function class.
Within each class, the functions were chosen to be the most
representative without repeating similar functions. For example,
only one Ellipsoid, one Rastrigin, and one Gallagher function is
included in the ``bbob-biobj`` suite although they appear in
separate versions in the ``bbob`` suite.

The original ``bbob`` function classes also allow to group the
55 ``bbob-biobj`` functions, dependend on the
classes of the individual objective functions. Depending
on whether two functions of the same class are combined
or not, these resulting 15 new function classes contain three
or four functions:

* separable - separable (functions f1, f2, f11)
* separable - moderate (f3, f4, f12, f13)
* separable - ill-conditioned (f5, f6, f14, f15)
* separable - multi-modal (f7, f8, f16, f17)
* separable - weakly-structured (f9, f10, f18, f19)
* moderate - moderate (f20, f21, f28)
* moderate - ill-conditioned (f22, f23, f29, f30)
* moderate - multi-modal (f24, f25, f31, f32)
* moderate - weakly-structured (f26, f27, f33, f34)
* ill-conditioned - ill-conditioned (f35, f36, f41)
* ill-conditioned - multi-modal (f37, f28, f42, f43)
* ill-conditioned - weakly-structured (f39, f40, f44, f45)
* multi-modal - multi-modal (f46, f47, f50)
* multi-modal - weakly structured (f48, f49, f51, f52)
* weakly structured - weakly structured (f53, f54, f55)

More details about the single functions can be found in the next
section whereas we first describe their common properties here.


Normalization, Ideal and Nadir Point
------------------------------------
None of the 55 ``bbob-biobj`` functions is explicitly normalized and the
optimization algorithms therefore have to cope with different scalings
in the two objective functions. Typically, different orders of magnitude
between the objective values can be observed.
However, to facilitate comparision between functions, a
normalization can take place as both the ideal and the nadir point are
known internally. Note that, for example, the ``bbob-biobj`` observer of
the `Coco framework`_ takes this into account and normalizes the objective
space, see the `bbob-biobj-specific performance assessment documentation 
<http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/>`_ for
details.

The reasons for having knowledge about the location of both the ideal and
the nadir point are

* the definitions of the single-objective ``bbob`` test functions for 
  which the optimal function value and the optimal solution are known
  by design and

* the fact that we explicitly chose only functions from the original
  ``bbob`` test suite which have a unique optimum.

The ideal point is then always given by the objective vector
:math:`(f_1(x_{\text{opt},1}), f_2(x_{\text{opt},2}))` and the nadir point by the
objective vector :math:`(f_1(x_{\text{opt},2}), f_2(x_{\text{opt},1}))`
with :math:`x_{\text{opt},1}` being the optimal solution for the first
objective function :math:`f_1` and :math:`x_{\text{opt},2}` being the
optimal solution for the second objective function :math:`f_2`.
Note that in the black-box case, we typically assume for the functions
provided with the `Coco framework`_, that information about ideal and
nadir points, scaling etc. is not provided to the algorithm.



Instances
---------
Instances are the way in the `Coco framework`_ to perform multiple
algorithm runs on the same function. More concretely, the original
Coco documentation states

::

  All functions can be instantiated in different *versions* (with
  different location of the global optimum and different optimal
  function value). Overall *Ntrial* runs are conducted on different
  instantiations.

Also in the bi-objective case, we provide the idea of instances by
relying on the instances provided within the single-objective
``bbob`` suite. In addition, we assert that
  
  * the distance (Euclidean norm) between the ideal and the nadir
    point (in objective space) is at least 1e1 and that
	
  * the two single-objective optima (in search space, also called
    the extreme optimal points) are not closer than :math:`10^{-4}`.
	 
In general, the two single-objective problem instances 

 * problem1_instance = 2 \* biobj_instance + 1 and
 * problem2_instance = problem1_instance + 1

are chosen to create the bi-objective problem instance ``biobj_instance``
while ``problem2_instance`` is increased successively until the two above
properties are fullfilled. For example, the ``bbob-biobj`` instance
8 consists of instance 17 for the first objective and instance 18 for
the second objective while for the ``bbob-biobj`` instance 9, the
first instance is 19 but for the second objective, instance 21 is chosen
instead of instance 20.

Exceptions to the above rule are, for historical reasons, the
``bbob-biobj`` instances 1 and 2 in order to match the instances
1 to 5 with the ones proposed in [BTH2015a]_. The ``bbob-biobj``
instance 1 contains the single-objective instances 2 and 4 and
the ``bbob-biobj`` instance 2 contains the two instances 3 and 5.

Note that the number of instances from the ``bbob-biobj`` suite is neither
limited from above nor from below. However, less than 3 instances will
render the potential statistics and their interpretation problematic
while even the smallest difference can be made statistically
significant with a high enough number of instances. Thus, we
recommend to use 5 to 15 instances for the actual benchmarking.


  
  
.. _`Coco framework`: https://github.com/numbbo/coco

.. [BTH2015a] Dimo Brockhoff, Thanh-Do Tran, Nikolaus Hansen:
   Benchmarking Numerical Multiobjective Optimizers Revisited.
   GECCO 2015: 639-646

