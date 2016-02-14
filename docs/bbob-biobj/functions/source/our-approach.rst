==================================================
Overview of the Proposed ``bbob-biobj`` Test Suite
==================================================

The ``bbob-biobj`` test suite provides 55 bi-objective functions in six
dimensions (2, 3, 5, 10, 20, and 40) with arbitrary many instances.
In particular,
the 55 functions are derived from combining a subset of the 24 well-known
single-objective functions of the ``bbob`` test suite which
has been used since 2009 in the `BBOB workshop series
<http://numbbo.github.io/workshops/>`_ . While concrete details on each of
the 55 ``bbob-biobj`` functions will be given in
:ref:`sec-test-functions`, we will detail here the main rationals behind
them together with their common properties.


The Single-objective ``bbob`` Functions Used and the Rational Behind Their Choice
---------------------------------------------------------------------------------
As a test suite in the `Coco framework`_, also the ``bbob-biobj`` functions
are supposed to be designed to represent typical difficulties obverved in
real-world optimization problems. Given the drawbacks of existing
multi-objective benchmark suites, mentioned in :ref:`sec:stateoftheart`,
and the fact that the single-objective ``bbob`` suite is considered as
representative of many difficulties observed in real-world problems, it is
only natural to combine the existing single-objective ``bbob`` functions
to multi-objective ones.

Combining all 24 ``bbob`` functions in pairs thereby results in
:math:`24^2=576` bi-objective functions overall. If we assume
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
we decided to choose only 10 of the 24 ``bbob`` functions and used
the above described combinations, resulting in
:math:`10+{10 \choose 2} = 55` bi-objective functions in the
final `bbob-biobj` suite.

TODO: why we chose those 10 functions we chose? --> 2 per function
group

TODO: mention already here all ten single-objective functions 
The final choice for the ``bbob-biobj`` suite includes the following
10 single-objective ``bbob`` functions, given with respect to their
``bbob`` function group:

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
relying on the instances provided within the ``bbob`` suite for
the 

TODO: describe how the instances are generated

TODO: recommendation: 10 instances


  
  
.. _`Coco framework`: https://github.com/numbbo/coco

.. [BTH2015a] Dimo Brockhoff, Thanh-Do Tran, Nikolaus Hansen:
   Benchmarking Numerical Multiobjective Optimizers Revisited.
   GECCO 2015: 639-646

