.. title:: COCO: The Large Scale Black-Box Optimization Benchmarking (bbob-largescale) Test Suite

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COCO: The Large Scale Black-Box Optimization Benchmarking (``bbob-largescale``) Test Suite
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
The ``bbob-largescale`` test suite containing 24 single objective
functions in continuous domain is an extension of the well-known
single-objective noiseless ``bbob`` test suite in large dimension.
The core idea is to make rotational transformations :math:`\textbf{R}, \textbf{Q}` in search space,
introduced in the ``bbob`` test suite, cheaper but remaining some desired
properties. This documentation presents our approach where the rotational transformation will
be replaced by the product of a permutation matrix times a block-diagonal matrix times a
permutation matrix in order to construct large scale testbeds.

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
In ``bbob-largescale`` test suite, we consider single-objective, unconstrained minimization problems
of the form

.. math::
    \min_{x \in \mathbb{R}^n} f(x),

where the number of variables, say :math:`n`, could be from hundreds to thousands. Here, we start by
extending the dimensions used in ``bbob`` test suite (up to 40) and consider values of :math:`n` up to at
least 640.

The objective is to find, as quickly as possible, one or several solutions :math:`x` in the search
space :math:`\mathbb{R}^n` with *small* value(s) of :math:`f(\x)\in\mathbb{R}`. We
generally consider *time* to be the number of calls to the function :math:`f`.

Definitions and Terminology
---------------------------
We remind in this section different definitions.

*function instance, problem*
Each function within COCO :math:`f^\theta: \mathbb{R}^n \to \mathbb{R}` is parametrized
with parameter values :math:`\theta \in \Theta`. A parameter value determines a so-called *function
instance*. For example, :math:`\theta` encodes the location of the optimum of single-objective functions,
which means that different instances have shifted optima.

A *problem* is a function instance of a specific dimension :math:`n`.

*block-diagonal matrix*
A *block-diagonal matrix* :math:`B` is a matrix of the form

.. math::
    :nowrap:

        \begin{equation*}
            B = \begin{pmatrix}
            B_1 & 0 & \dots & 0 \\
            0 & B_2 & \dots & 0 \\
            0 & 0 & \ddots & 0 \\
            0 & 0 & \dots & B_{n_b}
            \end{pmatrix}
    \end{equation*}

where :math:`n_b`is the number of blocks and :math:`B_i, 1 \leq i \ leq n_b`
are square matrices of sizes :math:`s_i \times s_i` satisfying :math:`s_i \geq 1`
and :math:`\sum_{i=1}^{n_b}s_i = n`.

*permutation matrix*
A *permutation matrix* :math:`P` is a square binary matrix that has exactly one entry of
1 in each row and each column and 0s elsewhere.

Overview of the Proposed ``bbob-largescale`` Test Suite
==================================================
The ``bbob-largescale`` test suite provides 24 functions in six dimensions
(20, 40, 80, 160, 320, and 640) within the COCO framework. The 24 functions
are extension of the 24 well-known single-objective functions of the
``bbob`` test suite [HAN2009]_ which has been used since 2009 in
the `BBOB workshop series`__. We will explain in this section how
this testbed is built, and how we intend to make it large-scale friendly.

__ http://numbbo.github.io/workshops

The Single-objective ``bbob`` Functions
---------------------------------------
The ``bbob`` testbed relies on the use of a number of raw functions from
which 24 different problems are generated. Firstly, the notion of raw function
designates functions in their basic form applied to a non-transformed (canonical
base) search space. Then, a series of transformations on the the raw function, such as
linear transformations, non-linear transformations and symmetry breaking transformations,
will be applied to obtain the ``bbob`` test functions. There are two reasons behind the
use of transformations:

(i) have non trivial problems that can not be solved by simply exploiting some of their
properties (separability, optimum at fixed position...) and

(ii) allow to generate different instances, ideally of similar difficulty, of a same problem.

Rotational transformation is one type of linear transformation which is used to avoid
separability and coordinate system independence. The rotational transformation consists in applying
an orthogonal matrix to the search space: :math:`x \rightarrow z = \textbf{R}x`, with :math:`\textbf{R}` is an
orthogonal matrix. While the other transformations used in the ``bbob`` test suite could easily extend to
large scale setting due to their linear complexity, the rotational transformation has quadratic time and
space complexities.

Extension to Large Scale
---------------------------------------
Our idea is to derive a computationally feasible large scale optimization test suite from the
``bbob`` testbed, while preserving the main characteristics of the original functions. To
achieve this goal, we replace the computationally expensive transformations, namely full orthogonal
matrices, with orthogonal transformations of linear computational complexity:
permuted orthogonal block-diagonal matrices.

Specifically, the matrix of rotational transformation will be represented as:

.. math::
    :nowrap:

        \begin{equation*}
        R = P_{left}BP_{right}
    \end{equation*}

Here, :math:`P_{left}, P_{right}` are two permutation matrices and :math:`B` is a
block-diagonal matrix of the form

.. math::
    :nowrap:

        \begin{equation*}
        B = \begin{pmatrix}
        B_1 & 0 & \dots & 0 \\
        0 & B_2 & \dots & 0 \\
        0 & 0 & \ddots & 0 \\
        0 & 0 & \dots & B_{n_b}
        \end{pmatrix}
    \end{equation*}

where :math:`n_b`is the number of blocks and :math:`B_i, 1 \leq i \ leq n_b`
are orthogonal square matrices of sizes :math:`s_i \times s_i` satisfying :math:`s_i \geq 1`
and :math:`\sum_{i=1}^{n_b}s_i = n`. Therefore, the matrix :math:`B` is also a orthogonal matrix.

This reprentation allows the rotational transformation :math:`R` to satisfy the three
desired properties:

1. Have (almost) linear cost (due to the block structure of :math:`B`): both the amount of memory
needed to store the matrix and the computational cost of applying the transformation matrix
to a solution must scale, ideally, linearly with :math:`n` or at most in :math:`nlog(n)`
or :math:`n^{1+\epsilon}` with :math:`\epsilon << 1`.

2. Introduce non-separability (applying two permutations): the desired scenario is to have
a parameter/set of parameters that allows to control the difficulty and level of
non-separability of the resulting problem in comparison to the original, non-transformed, problem.

3. Preserve, apart from separability (due to orthogonality of :math:`B`), the properties of the raw
function: as in the case when using a full orthogonal matrix, we want to preserve the
condition number and eigenvalues of the original function when it is convex quadratic.

Generating the orthogonal block matrix :math:`B`
---------------------------------------
We want to have the matrices :math:`B_i, i=1,2,...,n_b` uniformly distributed in the set of
orthogonal matrices of the same size (the orthogonal group :math:`O(s_i)`). We first
generate square matrices with entries i.i.d. standard normally distributed. Then we apply
the Gram-Schmidt process to orthogonalize these matrices.

Orthogonal block-diagonal matrices are the raw transformation matrices for our large scale functions.
Their parameters are

- :math:`n`, defines the size of the matrix,
- :math:`{s_1,\dots,s_{n_b}}`, the block sizes where :math:`n_b` is the number of blocks.

Generating the Random Permutations :math:`P`
---------------------------------------
When applying the permutations, especially :math:`P_{left}`, one wants to remain in control of the
difficulty of the resulting problem. Ideally, the permutation should have a parameterization that easily
allows to control the difficulty of the transformed problem.

We define our permutations as series of :math:`n_s` successive swaps. To have some control over the difficulty,
we want each variable to travel, in average, a fixed distance from its starting position. For this to
happen, we consider *truncated uniform swaps*.

In a truncated uniform swap, the second swap variable is chosen uniformly at random among the variables
that are within a fixed range :math:`r_s` of the first swap variable. Let :math:`i` be the index of the first
variable to be swapped and :math:`j` be that of the second swap variable, then

.. math::
    :nowrap:

        \begin{equation*}
        j \sim U({l_b(i), \dots, u_b(i)} \backslash {i}),
    \end{equation*}

where :math:`U(S)` is the uniform distribution over the set :math:`S` and :math:`l_b(i) = \max(1,i-r_s)`
and :math:`l_b(i) = \max(n,i+r_s)`.

When :math:`r_s \leq (d-1)/2`, the average distance between the first and the second swap
variable ranges from :math:`(\sqrt(2)-1)r_s + 1/2` to :math:`r_s/2 + 1/2`. It is maximal when the first
swap variable is at least :math:`r_s` away from both extremes or is one of them.

*Algorithm 1* describes the process of generating a permutation using a series of truncated uniform
swaps. The parameters for generating these permutations are:

  - :math:`n`, the number of variables,
  - :math:`n_s`, the number of swaps. Values proportional to :math:`n` will allow to make the next parameter the only free one,
  - :math:`r_s`, the swap range and eventually the only free parameter. The swap range can be equivalently defined in the form :math:`r_s = \ceil{r_r n}, with :math:`r_r \in [0, 1]`. Each variable moves in average about :math:`r_r × 50 \%` of the maximal distance :math:`n`.

The indexes of the variables are taken in a random order thanks to the permutation :math:`\pi`. This is
done to avoid any bias with regards to which variables are selected as first swap variables when less
than :math:`n` swaps are applied. We start with :math:`p` initially the identity permutation. We apply
the swaps defined above by taking :math:`p_{\pi}(1), p_{\pi}(2), \dots, p_{\pi}(n_s)`, successively, as
first swap variable. The resulting vector :math:`p` is returned as the
desired permutation.

*Algorithm 1: Truncated Uniform Permutations*

  Inputs: problem dimension :math:`n`, number of swaps :math:`n_s`, swap range :math:`r_s`.
  Output: a vector :math:`\textbf{p} \in \mathbb{N}^n`, defining a permutation.

    1.:math:`\textbf{p} \leftarrow (1, \dots,n)`
    2.generate a uniformly random permutation :math:`pi`
    3.\textbf{for} :math:`1 leq k leq n_s` \textbf{do}
    4.    :math:`i \leftarrow \pi(k), x_{\pi(k)} is the first swap variable
    5.    :math:`l_b \leftarrow \max(1,i−r_s)`
    6.    :math:`ub \leftarrow \min(d,i+r_s)`
    7.    :math:`S \leftarrow {l_b, l_b + 1, \dots, ub} \backslash {i}`
    8.    Sample :math:`j` uniformly in :math:`S`
    9.    swap :math:`p_i` and :math:`p_j`
    10.\textbf{end for}
    11.return :math:`\textbf{p}` 



Other modifications
---------------------------------------
Also, we do two main modifications to the raw functions in the ``bbob`` test suite (see some
functions below for some examples of such modified raw functions). First, functions
are normalized to have uniform target values that are comparable over a wide range
of dimensions. Second, the Cigar and Tablet functions are generalized such that they
have a constant proportion of distinct axes that remain consistent with
the ``bbob`` test suite.

.. math::
    :nowrap:

        \begin{equation*}
        f_{raw}^{CigarGen} = \gamma(n) \left(\sum_{i=1}^{\lceil n/40 \rceil} z_i^2 + 10^6 \sum_{i=\lceil n/40 \rceil+1}^n z_i^2 \right) \\
        f_{raw}^{DiffPow} = \gamma(n) \sum_{i=1}^n |z_i|^{\left(2 + 4 \times \frac{i-1}{n-1} \right)} \\
        f_{raw}^{Elli} = \gamma(n) \sum_{i=1}^n 10^{6\frac{i-1}{n-1}} z_i^2 \\
        f_{raw}^{TabletGen} = \gamma(n) \left(10^6\sum_{i=1}^{\lceil n/40 \rceil} z_i^2 + \sum_{i=\lceil n/40 \rceil+1}^n z_i^2 \right).
    \end{equation*}

where :math:`\gamma(n) = \min(1, 40/n)` for such that a constant target value (e.g., :math:`10^{-8})
represent the same level of difficulty arcross all dimensions :math:`n \geq 40.`


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


.. [aitelhara2016] O. Ait Elhara, A. Auger, N. Hansen (2016). Permuted Orthogonal Block-Diagonal
    Transformation Matrices for Large Scale Optimization Benchmarking. GECCO 2016, Jul 2016, Denver,
    United States
    .. __: https://hal.inria.fr/hal-01308566


.. [HAN2016ex] N. Hansen, T. Tušar, A. Auger, D. Brockhoff, O. Mersmann (2016). 
  `COCO: The Experimental Procedure`__, *ArXiv e-prints*, `arXiv:1603.08776`__. 
.. __: http://numbbo.github.io/coco-doc/experimental-setup/
.. __: http://arxiv.org/abs/1603.08776

  
