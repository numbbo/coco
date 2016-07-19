.. title:: COCO: The Large Scale Black-Box Optimization Benchmarking (bbob-largescale) Test Suite


COCO: The Large Scale Black-Box Optimization Benchmarking (``bbob-largescale``) Test Suite


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
single-objective noiseless ``bbob`` test suite [HAN2009]_, which has been used since 2009 in
the `BBOB workshop series`_, in large dimension. The core idea is to make rotational
transformations :math:`\textbf{R}, \textbf{Q}` in search space that
appear in the ``bbob`` test suite cheaper while retaining some desired
properties. This documentation presents our approach where the rotational transformation will
be replaced by the product of a permutation matrix times a block-diagonal matrix times a
permutation matrix in order to construct large scale testbeds.

.. _`BBOB workshop series`: http://numbbo.github.io/workshops

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

.. Some update:
   - Step ellipsoid: It has been updated the condition: \hat{z}_i > 0.5 (old) --> |\hat{z}_i| > 0.5
   - Schwelfel function:
        (1) \mathbf{z} = 100 (\mathbf{\Lambda}^{10} (\mathbf{\hat{z}} - \mathbf{x}^{\text{opt}}) + \mathbf{x}^{\text{opt}}) --> \mathbf{z} = 100 (\mathbf{\Lambda}^{10} (\mathbf{\hat{z}} - 2|\mathbf{x}^{\text{opt}}|) + 2|\mathbf{x}^{\text{opt}}|)
        (2) - frac{1}{D} sum(...) --> - frac{1}{100D} sum(...)
        (3) \hat{z}_1 = \hat{x}_1, \hat{z}_{i+1}=\hat{x}_{i+1} + 0.25 (\hat{x}_{i} - x_i^{\text{opt}}), \text{ for } i=1, \dots, n-1 --> \hat{z}_1 = \hat{x}_1, \hat{z}_{i+1}=\hat{x}_{i+1} + 0.25 (\hat{x}_{i} - 2|x_i^{\text{opt}}|), \text{ for } i=1, \dots, n-1
..


.. #################################################################################
.. #################################################################################
.. #################################################################################




Introduction
============
In the ``bbob-largescale`` test suite, we consider single-objective, unconstrained minimization problems
of the form

.. math::
    \min_{x \in \mathbb{R}^n} \ f(x),

where the number of variables :math:`n` are 20, 40, 80, 160, 320, 640.

The objective is to find, as quickly as possible, one or several solutions :math:`x` in the search
space :math:`\mathbb{R}^n` with *small* value(s) of :math:`f(x)\in\mathbb{R}`. We
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
  A *block-diagonal matrix* :math:`B` is a matrix of the form:

  .. math::
    B = \left(\begin{pmatrix}
    B_1 & 0 & \dots & 0 \\
    0 & B_2 & \dots & 0 \\
    0 & 0 & \ddots & 0 \\
    0 & 0 & \dots & B_{n_b}
    \end{pmatrix}
    \right),

  where :math:`n_b` is the number of blocks and :math:`B_i, 1 \leq i \leq n_b`
  are square matrices of sizes :math:`s_i \times s_i` satisfying :math:`s_i \geq 1`
  and :math:`\sum_{i=1}^{n_b}s_i = n`.

*permutation matrix*
  A *permutation matrix* :math:`P` is a square binary matrix that has exactly one entry of
  1 in each row and each column and 0s elsewhere.

Overview of the Proposed ``bbob-largescale`` Test Suite
=======================================================
The ``bbob-largescale`` test suite provides 24 functions in six dimensions (20, 40, 80, 160, 320, and 640) within
the COCO framework. We will explain in this section how this test suite is built and how we intend to make it
large-scale friendly.


The single-objective ``bbob`` functions
---------------------------------------
The ``bbob`` test suite relies on the use of a number of raw functions from
which 24 different problems are generated. Firstly, the raw function
is designed. Then, a series of transformations on the raw function, such as
linear transformations (e.g., translation, rotation, scaling), non-linear
transformations (e.g., :math:`T_{\text{osz}}, T_{\text{asy}}`)
will be applied to obtain the ``bbob`` test functions. For example, the test function
:math:`f_{13}(\mathbf{x})` (`Sharp Ridge function`_) with variable :math:`\mathbf{x}`
is derived from a raw function defined by

.. _Sharp Ridge function: http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf#page=65

.. math::
    f_{\text{raw}}^{\text{Sharp Ridge}}(\mathbf{z}) = z_1^2 + 100\sqrt{\sum_{i=2}^{n}z_i^2}.

Once this raw function is located, we apply a sequence of transformations: a
rotational transformation :math:`\mathbf{Q}`; then a scaling transformation
:math:`\mathbf{\Lambda}^{10}`; then a rotational transformation :math:`\mathbf{R}`; then
a translation by using the vector :math:`\mathbf{x}^{\text{opt}}` to get the relationship
:math:`\mathbf{z} =  \mathbf{Q}\mathbf{\Lambda}^{10} \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}})`; and finally
a translation on objective space by using :math:`\mathbf{f}_{\text{opt}}` to obtain the final
function in the testbed:

.. math::
    f_{13}(\mathbf{x}) = f_{\text{raw}}^{\text{Sharp Ridge}}(\mathbf{z}) + \mathbf{f}_{\text{opt}}.


There are two reasons behind the use of transformations:

(i) provide non trivial problems that can not be solved by simply exploiting some of their properties (separability, optimum at fixed position...) and
(ii) allow to generate different instances, ideally of similar difficulty, of a same problem.


In fact, rotational transformation is one type of linear transformation which is used to avoid
separability and coordinate system independence. The rotational transformation consists in applying
an orthogonal matrix to the search space: :math:`x \rightarrow z = \textbf{R}x`, where :math:`\textbf{R}` is an
orthogonal matrix. While the other transformations used in the ``bbob`` test suite could be naturally extended to
the large scale setting due to their linear complexity, the rotational transformation has quadratic time and
space complexities. Thus we need to reduce the complexity of this transformation in the large scale setting.

Extension to large scale
------------------------
Our objective is to construct a large scale test suite whose computational cost is acceptable while preserving the main
characteristics of the original functions in the ``bbob`` test suite. To this, we will replace the
full orthogonal matrices of the rotational transformations which are very expensive in large scale
setting, with the other orthogonal transformations having (almost) linear complexity: *permuted orthogonal
block-diagonal matrices* [aitelhara2016]_.

Specifically, the matrix of rotational transformation :math:`\textbf{R}` (similar to :math:`\textbf{Q}`)
will be represented as:

.. math::
    \begin{equation*}
        \textbf{R} = P_{left}BP_{right}.
    \end{equation*}

Here, :math:`P_{left} \text{ and } P_{right}` are two permutation matrices and :math:`B` is a
block-diagonal matrix of the above form. In this case, these square sub-matrices :math:`B_i, 1 \leq i \leq n_b`
are all orthogonal. Thus, the matrix :math:`B` is also an orthogonal matrix.

This reprentation allows the rotational transformation :math:`\textbf{R}` to satisfy three
desired properties:

1. Have (almost) linear cost (due to the block structure of :math:`B`).
2. Introduce non-separability.
3. Preserve the condition number and eigenvalues of the original function when it is convex quadratic since the matrix :math:`\textbf{R}` is orthogonal.


Generating the orthogonal block matrix :math:`B`
------------------------------------------------
The sub-matrices :math:`B_i, i=1,2,...,n_b` will be uniformly distributed in the set of
orthogonal matrices of the same size. To this, we firstly generate square matrices with
size :math:`s_i, i=i=1,2,...,n_b` whose entries are i.i.d. standard normally distributed.
Then we apply the Gram-Schmidt process to orthogonalize these matrices.

Orthogonal block-diagonal matrices are the raw transformation matrices for our large scale functions.
Their parameters are:

- :math:`n`, defines the size of the matrix,
- :math:`{s_1,\dots,s_{n_b}}`, the block sizes where :math:`n_b` is the number of blocks.

Generating the permutation :math:`P`
--------------------------------------------
For generating the permutation :math:`P`, we use the technique called *truncated uniform swaps*.
Here, the second swap variable is chosen uniformly at random among the variables
that are within a fixed range :math:`r_s` of the first swap variable. Let :math:`i` be the index of the first
variable to be swapped and :math:`j` be that of the second swap variable, then

.. math::
    \begin{equation*}
        j \sim U(\{l_b(i), l_b(i) + 1, \dots, u_b(i)\} \backslash \{i\}),
    \end{equation*}

where :math:`U(S)` is the uniform distribution over the set :math:`S` and :math:`l_b(i) = \max(1,i-r_s)`
and :math:`l_b(i) = \max(n,i+r_s)`. If :math:`r_s \leq (d-1)/2`, the average distance between
the first and the second swap variable ranges from :math:`(\sqrt(2)-1)r_s + 1/2` to
:math:`r_s/2 + 1/2`. It is maximal when the first swap variable is at least :math:`r_s`
away from both extremes or is one of them.

**Algorithm 1** ([aitelhara2016]_) below describes the process of generating a permutation using a
series of truncated uniform swaps with the following parameters:

- :math:`n`, the number of variables,
- :math:`n_s`, the number of swaps.
- :math:`r_s`, the swap range.

The indexes of the variables are taken in a random order thanks to the permutation :math:`\pi`. This is
done to avoid any bias with regards to which variables are selected as first swap variables when less
than :math:`n` swaps are applied. We start with :math:`p` initially the identity permutation. We apply
the swaps defined above by taking :math:`p_{\pi}(1), p_{\pi}(2), \dots, p_{\pi}(n_s)`, successively, as
first swap variable. The resulting vector :math:`p` is returned as the
desired permutation.

*Algorithm 1: Truncated Uniform Permutations*

- Inputs: problem dimension :math:`n`, number of swaps :math:`n_s`, swap range :math:`r_s.`

- Output: a vector :math:`\textbf{p} \in \mathbb{N}^n`, defining a permutation.

1. :math:`\textbf{p} \leftarrow (1, \dots,n)`
2. Generate a uniformly random permutation :math:`\pi`
3. :math:`\textbf{for } 1 \leq k \leq n_s \textbf{ do}`
4.   :math:`\text{ \ \ \ \ }  i \leftarrow \pi(k), x_{\pi(k)}` is the first swap variable
5.   :math:`\text{ \ \ \ \ }  l_b \leftarrow \max(1,i−r_s)`
6.   :math:`\text{ \ \ \ \ }  ub \leftarrow \min(d,i+r_s)`
7.   :math:`\text{ \ \ \ \ }  S \leftarrow \{l_b, l_b + 1, \dots, u_b\} \backslash \{i\}`
8.   :math:`\text{ \ \ \  }` Sample :math:`j` uniformly in :math:`S`
9.   :math:`\text{ \ \ \  }` Swap :math:`p_i` and :math:`p_j`
10. :math:`\textbf{end for}`
11. :math:`\textbf{return} \textbf{ p}`

In this test suite, we set :math:`n_s = n \text{ and } r_s = \lfloor n/3 \rfloor`. Some numerical
results ([aitelhara2016]_) show that with these choosen parameters, the proportion of variables that are
moved from their original position when applying Algorithm 1 is approximately 100\% for all
dimension 20, 40, 80, 160, 320, 640.

Function definition in large-scale test suite
=============================================
The Table below presents the definition of 24 functions used in the large scale test suite.
Beside the important modification on the rotational transformations, we also make two changes to the raw
functions in the ``bbob`` test suite. Firstly, functions are normalized by the
parameter :math:`\gamma(n) = \min(1, 40/n)` to have uniform target values
that are comparable over a wide range of dimensions. Secondly, the Discus, Bent Cigar, and
Sharp Ridge functions are generalized such that they have a constant proportion of distinct
axes that remain consistent with the ``bbob`` test suite.


.. list-table::
    :header-rows: 1
    :widths: 3 9 7
    :stub-columns: 0

    *  -
       -  Formulation
       -  Transformations

    *  -  **Group 1: Separable functions**
       -
       -

    *  - Sphere Function
       - :math:`f_1(\mathbf{x}) = \gamma(n) \times\sum_{i=1}^{n} z_i^2 + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = \mathbf{x} - \mathbf{x}^{\text{opt}}`

    *  - Ellipsoidal Function
       - :math:`f_2(\mathbf{x}) = \gamma(n) \times\sum_{i=1}^{n}10^{6\frac{i - 1}{n - 1}} z_i^2+ \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = T_{\text{osz}}\left(\mathbf{x} - \mathbf{x}^{\text{opt}}\right)`

    *  - Rastrigin Function
       - :math:`f_3(\mathbf{x}) = \gamma(n) \times\left(10n - 10\sum_{i=1}^{n}\cos\left(2\pi z_i \right) + ||z||^2\right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = \mathbf{\Lambda}^{10} T_{\text{asy}}^{0.2} \left( T_{\text{osz}}\left(\mathbf{x} - \mathbf{x}^{\text{opt}}\right) \right)`

    *  - :math:`\text{B\"{u}che-Rastrigin Function}`
       - :math:`f_4(\mathbf{x}) = \gamma(n) \times\left(10n - 10\sum_{i=1}^{n}\cos\left(2\pi z_i \right) + ||z||^2\right) + 100f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`z_i = s_i T_{\text{osz}}\left(x_i - x_i^{\text{opt}}\right),\\ \mathbf{s}_i = \begin{cases} 10 \times 10^{\frac{1}{2} \ \frac{i-1}{n - 1}} & \text{if } z_i >0 \text{ and } i \text{ odd}\\ 10^{\frac{1}{2} \ \frac{i - 1}{n - 1}} & \text{otherwise} \end{cases}, \text{for } i = 1,\dots, n`

    *  - Linear Slope
       - :math:`f_5(\mathbf{x}) = \gamma(n)\times \sum_{i=1}^{n}\left( 5 \vert s_i \vert - s_i z_i \right) + \mathbf{f}_{\text{opt}}`
       - :math:`z_i = \begin{cases} x_i & \text{if } x_i^{\mathrm{opt}}x_i < 5^2 \\ x_i^{\mathrm{opt}} & \text{otherwise} \end{cases} \text{ for } i=1, \dots, n, \\ s_i = \text{sign} \left(x_i^{\text{opt}}\right) 10^{\frac{i-1}{n-1}}, \text{ for } i=1, \dots, n, \\ \mathbf{x}^{\text{opt}} = \mathbf{z}^{\text{opt}} = 5\times \mathbf{1}_{-}^+`

    *  -  **Group 2: Functions with low or moderate conditioning**
       -
       -

    *  - Attractive Sector Function
       - :math:`f_6(\mathbf{x}) = \gamma(n) \times T_{\text{osz}}\left(\sum_{i=1}^{n}\left( s_i z_i\right)^2 \right)^{0.9} + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} =  \mathbf{Q} \mathbf{\Lambda}^{10}  \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}}) \\ s_i = \begin{cases} 10^2 & \text{if } z_i \times x_i^{\mathrm{opt}} > 0\\ 1 & \text{otherwise}\end{cases}, \text{for } i=1,\dots, n`

    *  - Step Ellipsoidal Function
       - :math:`f_7(\mathbf{x}) = \gamma(n) \times 0.1 \max\left(\vert \hat{z}_1\vert/10^4, \sum_{i=1}^{n}10^{2\frac{i - 1}{n - 1}}z_i^2\right) + f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{\hat{z}} = \mathbf{\Lambda}^{10}  \mathbf{R}(\mathbf{x}-\mathbf{x}^{\text{opt}}) \\ \tilde{z}_i= \begin{cases} \lfloor 0.5 + \hat{z}_i \rfloor & \text{if }  |\hat{z}_i| > 0.5 \\ \lfloor 0.5 + 10 \hat{z}_i \rfloor /10 & \text{otherwise} \end{cases}, \text{for } i=1,\dots, n \\ \mathbf{z} =  \mathbf{Q} \mathbf{\tilde{z}}`

    *  - Rosenbrock Function, original
       - :math:`f_8(\mathbf{x}) = \gamma(n) \times\sum_{i=1}^{n} \left(100 \left(z_{i}^2 - z_{i+1}\right)^2 + \left(z_{i} - 1\right)^2\right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = \max\left(1, \dfrac{\sqrt{n}}{8}\right)(\mathbf{x} - \mathbf{x}^{\text{opt}})+ \mathbf{1} ,\\ \mathbf{z}^{\text{opt}} = \mathbf{1}`

    *  - Rosenbrock Function, rotated
       - :math:`f_9(\mathbf{x}) = \gamma(n) \times\sum_{i=1}^{n} \left(100 \left(z_{i}^2 - z_{i+1}\right)^2 + \left(z_{i} - 1\right)^2\right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = \max\left(1, \dfrac{\sqrt{n}}{8}\right) \mathbf{R} \mathbf{x} + \dfrac{\mathbf{1}}{2},\\ \mathbf{z}^{\text{opt}} = \mathbf{1}`

    *  -  **Group 3: Functions with high conditioning and unimodal**
       -
       -

    *  - Ellipsoidal Function
       - :math:`f_{10}(\mathbf{x}) = \gamma(n) \times\sum_{i=1}^{n}10^{6\frac{i - 1}{n - 1}} z_i^2  + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = T_{\text{osz}} ( \mathbf{R} (\mathbf{x} - \mathbf{x}^{\text{opt}}))`

    *  - Discus Function
       - :math:`f_{11}(\mathbf{x}) = \gamma(n) \times\left(10^6\sum_{i=1}^{\lceil n/40 \rceil}z_i^2 + \sum_{i=\lceil n/40 \rceil+1}^{n}z_i^2\right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = T_{\text{osz}}( \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}}))`

    *  - Bent Cigar Function
       - :math:`f_{12}(\mathbf{x}) = \gamma(n) \times\left(\sum_{i=1}^{\lceil n/40 \rceil}z_i^2 + 10^6\sum_{i=\lceil n/40 \rceil + 1}^{n}z_i^2 \right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} =  \mathbf{R} T_{\text{asy}}^{0.5}( \mathbf{R}((\mathbf{x} - \mathbf{x}^{\text{opt}}))`

    *  - Sharp Ridge Function
       - :math:`f_{13}(\mathbf{x}) = \gamma(n) \times\left(\sum_{i=1}^{\lceil n/40 \rceil}z_i^2 + 100\sqrt{\sum_{i=\lceil n/40 \rceil + 1}^{n}z_i^2} \right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} =  \mathbf{Q}\mathbf{\Lambda}^{10} \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}})`

    *  - Different Powers Functiovn
       - :math:`f_{14}(\mathbf{x}) = \gamma(n) \times\sum_{i=1}^{n} \vert z_i\vert ^{\left(2 + 4 \times \frac{i-1}{n- 1}\right)} + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} =  \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}})`

    *  -  **Group 4: Multi-modal functions with adequate global structure**
       -
       -

    *  - Rastrigin Function
       - :math:`f_{15}(\mathbf{x}) = \gamma(n) \times\left(10n - 10\sum_{i=1}^{n}\cos\left(2\pi z_i \right) + ||\mathbf{z}||^2\right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} =  \mathbf{R} \mathbf{\Lambda}^{10}  \mathbf{Q} T_{\text{asy}}^{0.2} \left(T_{\text{osz}} \left(\mathbf{R}\left(\mathbf{x} - \mathbf{x}^{\text{opt}} \right) \right) \right)`

    *  - Weierstrass Function
       - :math:`f_{16}(\mathbf{x}) = \gamma(n) \times 10\left( \dfrac{1}{n} \sum_{i=1}^{n} \sum_{k=0}^{11} \dfrac{1}{2^k} \cos \left( 2\pi 3^k \left( z_i + 1/2\right) \right) - f_0\right)^3 + \dfrac{10}{n}f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} =  \mathbf{R}\mathbf{\Lambda}^{1/100} \mathbf{Q}T_{\text{osz}}( \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}})) \\ f_0= \sum_{k=0}^{11} \dfrac{1}{2^k} \cos(\pi 3^k)`

    *  - Schaffers F7 Function
       - :math:`f_{17}(\mathbf{x}) = \gamma(n) \times\left(\dfrac{1}{n-1} \sum_{i=1}^{n-1} \left(\sqrt{s_i} + \sqrt{s_i}\sin^2\left( 50 (s_i)^{1/5}\right)\right)\right)^2 + 10f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = \mathbf{\Lambda}^{10}  \mathbf{Q} T_{\text{asy}}^{0.5}( \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}})) \\ s_i= \sqrt{z_i^2 + z_{i+1}^2}, i=1,\dots, n-1`

    *  - Schaffers F7 Function, moderately ill-conditioned
       - :math:`f_{18}(\mathbf{x}) = \gamma(n) \times\left(\dfrac{1}{n-1} \sum_{i=1}^{n-1} \left(\sqrt{s_i} + \sqrt{s_i}\sin^2\left( 50 (s_i)^{1/5}\right)\right)\right)^2 + 10f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = \mathbf{\Lambda}^{1000}  \mathbf{Q} T_{\text{asy}}^{0.5}( \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}})) \\ s_i= \sqrt{z_i^2 + z_{i+1}^2}, i=1,\dots, n-1`

    *  - Composite Griewank-Rosenbrock Function F8F2
       - :math:`f_{19}(\mathbf{x}) = \gamma(n)\times\left(\dfrac{10}{n-1} \sum_{i=1}^{n-1} \left( \dfrac{s_i}{4000} - \cos\left(s_i \right)\right) + 10 \right) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} = \max\left(1, \dfrac{\sqrt{n}}{8}\right) \mathbf{R} \mathbf{x} + \dfrac{\mathbf{1}}{2}, \\ s_i= 100(z_i^2 - z_{i+1})^2 + (z_i - 1)^2, i=1,\dots, n-1, \\ \mathbf{z}^{\text{opt}} = \mathbf{1}`

    *  -  **Group 5: Multi-modal functions with weak global structure**
       -
       -

    *  - Schwefel Function
       - :math:`f_{20}(\mathbf{x}) = \gamma(n)\times\left(-\dfrac{1}{n} \sum_{i=1}^{n} z_i\sin\left(\sqrt{\vert z_i\vert}\right)\right) + 4.189828872724339 + 100f_{pen}(\mathbf{z}/100)+\mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{\hat{x}} = 2 \times \mathbf{1}_{-}^{+} \otimes \mathbf{x}, \\ \hat{z}_1 = \hat{x}_1, \hat{z}_{i+1}=\hat{x}_{i+1} + 0.25 \left(\hat{x}_{i} - 2\left|x_i^{\text{opt}}\right|\right), \text{ for } i=1, \dots, n-1, \\ \mathbf{z} = 100 \left(\mathbf{\Lambda}^{10} \left(\mathbf{\hat{z}} - 2\left|\mathbf{x}^{\text{opt}}\right|\right) + 2\left|\mathbf{x}^{\text{opt}}\right|\right), \\ \mathbf{x}^{\text{opt}} = 4.2096874633/2 \mathbf{1}_{-}^{+}`

    *  - Gallagher’s Gaussian 101-me Peaks Function
       - :math:`f_{21}(\mathbf{x}) = \gamma(n)\times\left(10 - \max_{i=1}^{101} w_i \exp\left(- \dfrac{1}{2n} (\mathbf{z} - \mathbf{y}_i)^T\mathbf{R}^T\mathbf{C_i}\mathbf{R} (\mathbf{z} - \mathbf{y}_i) \right) \right)^2 + f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`w_i = \begin{cases} 1.1 + 8 \times \dfrac{i-2}{99} & \text{for } 2 \leq i \leq 101 \\ 10 & \text{for } i = 1 \end{cases} \\ \\ \mathbf{C_i} = \Lambda^{\alpha_i}/\alpha_i^{1/4} \text{where } \Lambda^{\alpha_i} \text{ is defined as usual, but with randomly \\ permuted diagonal elements. For } i=1,\dots, 101, \alpha_i \text{ is drawn uniformly \\ randomly from the set } \left\{1000^{2\frac{j}{99}}, j = 0,\dots, 99 \right\} \text{without replacement, and } \\ \alpha_i = 1000 \text{ for } i = 1. \\ \\ \text{The local optima } \mathbf{y}_i \text{ are uniformly drawn from the domain } [-5,5]^n \text{ for } \\ i = 2,...,101 \text{ and } \mathbf{y}_1 \in [-4,4]^n. \text{ The global optimum is at } \mathbf{x}^{\text{opt}} = \mathbf{y}_1.`

    *  - Gallagher’s Gaussian 21-hi Peaks Function
       - :math:`f_{22}(\mathbf{x}) = \gamma(n)\times\left(10 - \max_{i=1}^{21} w_i \exp\left(- \dfrac{1}{2n} (\mathbf{z} - \mathbf{y}_i)^T \mathbf{R}^T\mathbf{C_i}\mathbf{R} (\mathbf{z} - \mathbf{y}_i) \right) \right)^2 + f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`w_i = \begin{cases} 1.1 + 8 \times \dfrac{i-2}{19} & \text{for } 2 \leq i \leq 21 \\ 10 & \text{for } i = 1 \end{cases} \\ \\ \mathbf{C_i} = \Lambda^{\alpha_i}/\alpha_i^{1/4} \text{where } \Lambda^{\alpha_i} \text{ is defined as usual, but with randomly \\ permuted diagonal elements. For } i=1,\dots, 21, \alpha_i \text{ is drawn uniformly \\ randomly from the set } \left\{1000^{2\frac{j}{19}}, j = 0,\dots, 19 \right\} \text{without replacement, and } \\ \alpha_i = 1000^2 \text{ for } i = 1. \\ \\ \text{The local optima } \mathbf{y}_i \text{ are uniformly drawn from the domain } [-4.9, 4.9]^n \text{ for } \\ i = 2,..., 21 \text{ and } \mathbf{y}_1 \in [-3.92, 3.92]^n. \text{ The global optimum is at } \mathbf{x}^{\text{opt}} = \mathbf{y}_1.`

    *  - Katsuura Function
       - :math:`f_{23}(\mathbf{x}) = \gamma(n)\times\left(\dfrac{10}{n^2} \prod_{i=1}^{n} \left( 1 + i \sum_{j=1}^{32} \dfrac{\vert 2^j z_i - [2^j z_i]\vert}{2^j}\right)^{10/n^{1.2}} - \dfrac{10}{n^2}\right) + f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{z} =  \mathbf{Q}\mathbf{\Lambda}^{100}  \mathbf{R}(\mathbf{x} - \mathbf{x}^{\text{opt}})`

    *  - Lunacek bi-Rastrigin Function
       - :math:`f_{24}(\mathbf{x}) = \gamma(n)\times\Big(\min\big( \sum_{i=1}^{n} (\hat{x}_i - \mu_0)^2, n + s\sum_{i=1}^{n}(\hat{x}_i - \mu_1)^2\big) + 10 \big(n - \sum_{i=1}^{n}\cos(2\pi z_i) \big)\Big) + \\ + 10^{4}f_{pen}(\mathbf{x}) + \mathbf{f}_{\text{opt}}`
       - :math:`\mathbf{\hat{x}} = 2 \text{sign}(\mathbf{x}^{\text{opt}}) \otimes \mathbf{x}, \mathbf{x}^{\text{opt}} = \mu_0 \mathbf{1}_{-}^{+} \\ \mathbf{z} =  \mathbf{Q}\mathbf{\Lambda}^{100} \mathbf{R}(\mathbf{\hat{x}} - \mu_0\mathbf{1}) \\ \mu_0 = 2.5, \mu_1 = -\sqrt{\dfrac{\mu_0^{2} - 1}{s}}, s = 1 - \dfrac{1}{2\sqrt{n + 20} - 8.2}`


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


.. [HAN2016ex] N. Hansen, T. Tušar, A. Auger, D. Brockhoff, O. Mersmann (2016). 
  `COCO: The Experimental Procedure`__, *ArXiv e-prints*, `arXiv:1603.08776`__. 
.. __: http://numbbo.github.io/coco-doc/experimental-setup/
.. __: http://arxiv.org/abs/1603.08776


.. [aitelhara2016] O. Ait Elhara, A. Auger, N. Hansen (2016). `Permuted Orthogonal Block-Diagonal
   Transformation Matrices for Large Scale Optimization Benchmarking`__. GECCO 2016, Jul 2016, Denver,
   United States.
.. __: https://hal.inria.fr/hal-01308566

