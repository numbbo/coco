.. Note:: see also docs/coco-generic/coco-experimental-setup/source/index.rst

.. |ftarget| replace:: :math:`f_\mathrm{target}`
.. |nruns| replace:: :math:`\texttt{Ntrial}`
.. |DIM| replace:: :math:`D`
.. _2009: http://www.sigevo.org/gecco-2009/workshops.html#bbob
.. _2010: http://www.sigevo.org/gecco-2010/workshops.html#bbob
.. _2012: http://www.sigevo.org/gecco-2012/workshops.html#bbob
.. _BBOB-2009: http://coco.gforge.inria.fr/doku.php?id=bbob-2009-results
.. _BBOB-2010: http://coco.gforge.inria.fr/doku.php?id=bbob-2010-results
.. _BBOB-2012: http://coco.gforge.inria.fr/doku.php?id=bbob-2012
.. _GECCO: http://www.sigevo.org/gecco-2012/
.. _COCO: https://github.com/numbbo/coco
.. _COCOold: http://coco.gforge.inria.fr

The ``bbob`` Suite
===================

This section describes the specific setup and rationales for the BBOB benchmark suite.
The descriptions of the functions can be found at http://coco.gforge.inria.fr/doku.php?id=downloads

Symbols, Constants, and Parameters
----------------------------------

For the BBOB test, the following constants are set:

  :math:`D = 2; 3; 5; 10; 20; 40` 
    is search space dimensionalities used for all functions.

  :math:`\texttt{Ntrial} = 15` 
    is the number of trials for each single setup, 
    i.e. each function and dimensionality. The performance is evaluated over all trials.

  :math:`\Delta f = 10^{-8}`
    precision to reach, that is, a difference to the smallest
    possible function value :math:`f_\mathrm{opt}`.

  :math:`f_\mathrm{target} = f_\mathrm{opt}+\Delta f` 
    is target function value to reach for different :math:`\Delta f` values.
    The final, smallest considered target function value is
    :math:`f_\mathrm{target} = f_\mathrm{opt} + 10^{-8}`, but also larger values
    for |ftarget| are evaluated.

The |nruns| runs are conducted on different instances of the functions.

.. _sec:rationales:

Rationale for the Choice of Ntrial = 15
_______________________________________

All functions can be instantiated in different "versions" (with
different location of the global optimum and different optimal function value).
Overall |nruns| runs are conducted on different instantiations (in the
2009_ setup each instance was repeated three times). The parameter
|nruns|, the overall number of trials on each function/dimension pair,
determines the minimal measurable success rate and influences the
overall necessary CPU time.  Compared to a standard setup for testing
stochastic search procedures, we have chosen a small value, |nruns| :math:`=15`.
Consequently, within the same CPU-time budget, single trials can be
longer and conduct more function evaluations (until |ftarget| is
reached). If an algorithm terminates before |ftarget| is reached,
longer trials can simply be achieved by independent multistarts.
Because these multistarts are conducted within each trial, more
sophisticated restart strategies are feasible. **Within-trial multistarts 
never impair the used performance measures and are encouraged.** Finally, 
15 trials are sufficient to make relevant performance differences statistically
significant. [#]_

.. [#] If the number of trials is chosen *much* larger, small and 
   therefore irrelevant
   performance differences become statistically significant.

Rationale for the Choice of f\ :sub:`target`
____________________________________________

The initial search domain and the target function value are an essential part
of the benchmark function definition.  Different target function values might
lead to different characteristics of the problem to be solved, besides that
larger target values are invariably less difficult to reach. Functions might be
easy to solve up to a function value of 1 and become intricate for smaller
target values. 
We take records for a larger number of predefined target values, defined relative to the known optimal function value :math:`f_\mathrm{opt}` and in principle unbounded from above. 
The chosen value for the final (smallest) |ftarget| is somewhat arbitrary. 
Reasonable values can change by simple modifications in the function
definition. In order to safely prevent numerical precision problems, the final target is :math:`f_\mathrm{target} = f_\mathrm{opt} + 10^{-8}`.



.. [Auger:2005a] A Auger and N Hansen. A restart CMA evolution strategy with
   increasing population size. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1769–1776. IEEE Press, 2005.
.. [Auger:2005b] A. Auger and N. Hansen. Performance evaluation of an advanced
   local search evolutionary algorithm. In *Proceedings of the IEEE Congress on
   Evolutionary Computation (CEC 2005)*, pages 1777–1784, 2005.
.. [Auger:2009] Anne Auger and Raymond Ros. Benchmarking the pure
   random search on the BBOB-2009 testbed. In Franz Rothlauf, editor, *GECCO
   (Companion)*, pages 2479–2484. ACM, 2009.
.. [Efron:1993] B. Efron and R. Tibshirani. *An introduction to the
   bootstrap.* Chapman & Hall/CRC, 1993.
.. [Harik:1999] G.R. Harik and F.G. Lobo. A parameter-less genetic
   algorithm. In *Proceedings of the Genetic and Evolutionary Computation
   Conference (GECCO)*, volume 1, pages 258–265. ACM, 1999.
.. [Hoos:1998] H.H. Hoos and T. Stützle. Evaluating Las Vegas
   algorithms—pitfalls and remedies. In *Proceedings of the Fourteenth 
   Conference on Uncertainty in Artificial Intelligence (UAI-98)*,
   pages 238–245, 1998.
.. [Price:1997] K. Price. Differential evolution vs. the functions of
   the second ICEO. In Proceedings of the IEEE International Congress on
   Evolutionary Computation, pages 153–157, 1997.


