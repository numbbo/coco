.. title:: COCO: PUT YOUR TITLE HERE (bbob-largescale) Test Suite

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COCO: PUT YOUR TITLE HERE (``bbob-largescale``) Test Suite
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
   <A HREF="http://arxiv.org/abs/XXXX.XXXXX">arXiv:XXXX.XXXXX</A>, 2016.

.. raw:: latex

  % \tableofcontents TOC is automatic with sphinx and moved behind abstract by swap...py
  \begin{abstract}

YOUR ABSTRACT GOES HERE AND INTO THE conf.py FILE.

.. raw:: latex

  \end{abstract}
  \newpage





.. _COCO: https://github.com/numbbo/coco
.. _COCOold: http://coco.gforge.inria.fr
.. |coco_problem_t| replace:: 
  ``coco_problem_t``
.. _coco_problem_t: http://numbbo.github.io/coco-doc/C/coco_8h.html#a408ba01b98c78bf5be3df36562d99478

.. |f| replace:: :math:`f`

.. summarizing the state-of-the-art in multi-objective black-box benchmarking, at 
.. and at providing a simple tutorial on how to use these functions for actual benchmarking within the Coco framework.

.. .. Note::
  
  For the time being, this documentation is under development and might not 
  contain all final data.


.. #################################################################################
.. #################################################################################
.. #################################################################################



Introduction
============
ADD YOUR TEXT HERE


Definitions and Terminology
---------------------------
UPDATE ACCORDING TO YOUR NEEDS:

We remind in this section different definitions.

*function instance, problem*
 Each function within COCO :math:`f^\theta: \mathbb{R}^D \to \mathbb{R}^m` is parametrized 
 with parameter values :math:`\theta \in \Theta`. A parameter value determines a so-called *function 
 instance*. For example, :math:`\theta` encodes the location of the optimum of single-objective functions, 
 which means that different instances have shifted optima. In the ``bbob-biobj`` 
 test suite, :math:`m=2` and the function instances are determined by the instances of the underlying
 single-objective functions. 
 
 A *problem* is a function instance of a specific dimension :math:`D`.

*ideal point*
 The ideal point is defined as the vector in objective space that
 contains the optimal |f|-value for each objective *independently*. 
 More precisely let :math:`f_\alpha^{\rm opt}:= \inf_{x\in \mathbb{R}^D} f_\alpha(x)` and
 :math:`f_\beta^{\rm opt}:= \inf_{x\in \mathbb{R}^D} f_\beta(x)`, the ideal point is given by
 
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



OTHER SECTIONS YOU MIGHT NEED
==================================================






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

  
