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
The ``bbob-biobj`` test suite is designed to be able to assess  performance of algorithms with respect to well-identified difficulties in optimization typically  occurring in real-world problems. A multi-objective problem being a combination of single-objective problems, one can obtain multi-objective problems with representative difficulties by simply combining single objective functions with representative difficulties observed in real-world problems. For this purpose we naturally use the single-objective ``bbob`` suite.

Combining all 24 ``bbob`` functions in pairs thereby results in
:math:`24^2=576` bi-objective functions overall. We however assume that multi-objective optimization algorithms are invariant or not very sensitive to permutations of the objective functions such that combining the 24  ``bbob`` functions and taking out the function :math:`(g_2,g_1)` if the function :math:`(g_1,g_2)` is present results in :math:`24+ {24 \choose 2} = 300` functions [#]_.

.. Given that most (if not all) multi-objective optimization algorithms are
.. invariant to permutations of the objective functions, a bi-objective
.. function combining for example the sphere function as the first
.. objective with the Rastrigin function as the second objective will
.. result in the same performance than if the Rastrigin function is the
.. first and the sphere function is the second objective function. 
.. Hence, we should keep only one of the resulting
.. bi-objective functions. Combining then all 24 ``bbob`` functions

.. [#] The first objective is chosen as ``bbob`` function *i*
  and the second as ``bbob`` function *j* with *i* :math:`\leq` *j*,
  results in :math:`24+ {24 \choose 2} = 300` functions.

Some first tests, e.g. in [BTH2015a]_, showed that having 300 functions
in Coco's first bi-objective suite is impracticable in terms
of the overall running time of the benchmarking experiment.  We then decided to exploit the organization of the ``bbob`` functions into classes to choose a subset of functions. More precisely the 24 original ``bbob`` functions are grouped into five function
classes where each class gathers functions with similar properties, namely
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


* Separable functions

  - Sphere (function 1 in ``bbob`` suite)
  - Ellipsoid separable (function 2 in ``bbob`` suite)

* Functions with low or moderate conditioning 

  - Attractive sector (function 6 in ``bbob`` suite)
  - Rosenbrock original (function 8 in ``bbob`` suite)

* Functions with high conditioning and unimodal 

  - Sharp ridge (function 13 in ``bbob`` suite)
  - Sum of different powers (function 14 in ``bbob`` suite)

* Multi-modal functions with adequate global structure 

  - Rastrigin (function 15 in ``bbob`` suite)
  - Schaffer F7, condition 10 (function 17 in ``bbob`` suite)

* Multi-modal functions with weak global structure 

  - Schwefel x*sin(x) (function 20 in ``bbob`` suite)
  - Gallagher 101 peaks (function 21 in ``bbob`` suite)

  
Using the above described pairwise combinations, this results in
having :math:`10+{10 \choose 2} = 55` bi-objective functions in
the final `bbob-biobj` suite. Those functions are denoted :math:`f_1` to :math:`f_{55}` in the sequel.

.. The next section gives the
.. reasoning behind choosing exactly these 10 functions.

  

Function Groups
---------------------------------------------------------------



From combining the original ``bbob`` function classes, we obtain 15 function classes to structure the 55 bi-objective functions of the ``bbob-biobj`` testsuit. Each function class contains three or four functions. We are listing below the function classes and in parenthesis  the functions that belong to the respective class:
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
 11. ill-conditioned - multi-modal (:math:`f_{37}`, :math:`f_{28}`, :math:`f_{42}`, :math:`f_{43}`)
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


Normalization, Ideal and Nadir Point
------------------------------------
None of the 55 ``bbob-biobj`` functions is explicitly normalized and the
optimization algorithms therefore have to cope with different scalings
in the two objective functions. Typically, different orders of magnitude
between the objective values can be observed.
However, to facilitate comparison between functions, a
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
    point (in objective space) is at least 1e-1 and that
	
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

