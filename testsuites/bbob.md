---
layout: default
permalink: /testsuites/bbob
parent: test suites
nav_order: 1
title: bbob
has_toc: false
---


# The <font face="Courier">bbob</font> Test Suite

<table>
	<tr>
		<td style="width=50%">
			The blackbox optimization benchmarking (bbob) test suite is COCO's standard and most established test suite with 24 noiseless, scalable test functions. Each function is provided in various dimensions (2, 3, 5, 10, 20, 40) and available in an arbitrary dimensions and number of instances. Links to their definition as well as to visualizations of their properties can be found in the table.
		</td>
		<td>
			<a href="ecdf-bbob.html"><img src="examplefigure_all.png" alt="ECDF of runtimes for 31 algorithms on the bbob suite" width="100%"/></a>
		</td>
	</tr>
</table>

<table align="center" style="width:50%" th, td {
  padding-top: 100px;
  padding-bottom: 0px}
<tr>
   <th colspan=2 style="text-align:left">1 Separable Functions</th>
</tr>
<tr>
	<td style="width:5%">f1</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=5">Sphere Function</a></td>
</tr><tr>
	<td>f2</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=10">Separable Ellipsoidal Function</a></td>
</tr><tr>
	<td>f3</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=15">Rastrigin Function</a></td>
</tr><tr>
	<td>f4</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=20">Büche-Rastrigin Function</a></td>
</tr><tr>
	<td>f5</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=25">Linear Slope</a></td>
</tr>
<tr>
   <th colspan=2 style="text-align:left">2 Functions with low or moderate conditioning</th>
</tr>
<tr>
	<td>f6</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=30">Attractive Sector Function</a></td>
</tr><tr>
	<td>f7</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=35">Step Ellipsoidal Function</a></td>
</tr><tr>
	<td>f8</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=40">Rosenbrock Function, original</a></td>
</tr><tr>
	<td>f9</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=45">Rosenbrock Function, rotated</a></td>
</tr>
<tr>
   <th colspan=2 style="text-align:left">3 Functions with high conditioning and unimodal</th>
</tr>
<tr>
	<td>f10</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=50">Ellipsoidal Function</a></td>
</tr><tr>
	<td>f11</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=55">Discus Function</a></td>
</tr><tr>
	<td>f12</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=60">Bent Cigar Function</a></td>
</tr><tr>
	<td>f13</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=65">Sharp Ridge Function</a></td>
</tr><tr>
	<td>f14</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=70">Different Powers Function</a></td>
</tr>
<tr>
   <th colspan=2 style="text-align:left">4 Multi-modal functions with adequate global structure</th>
</tr>
<tr>
	<td>f15</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=75">Rastrigin Function</a></td>
</tr><tr>
	<td>f16</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=80">Weierstrass Function</a></td>
</tr><tr>
	<td>f17</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=85">Schaffer's F7 Function</a></td>
</tr><tr>
	<td>f18</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=90">Schaffer's F7 Function, moderately ill-conditioned</a></td>
</tr><tr>
	<td>f19</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=95">Composite Griewank-Rosenbrock Function F8F2</a></td>
</tr>
<tr>
   <th colspan=2 style="text-align:left">5 Multi-modal functions with weak global structure</th>
</tr>
<tr>
	<td>f20</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=100">Schwefel Function</a></td>
</tr><tr>
	<td>f21</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=105">Gallagher's Gaussian 101-me Peaks Function</a></td>
</tr><tr>
	<td>f22</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=110">Gallagher's Gaussian 21-hi Peaks Function</a></td>
</tr><tr>
	<td>f23</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=115">Katsuura Function</a></td>
</tr><tr>
	<td>f24</td><td><a href="https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=120">Lunacek bi-Rastrigin Function</a></td>
</tr>
</table>


Only f1 and f5 are purely quadratic or linear respectively.

See also [N. Hansen et al (2010)](https://dl.acm.org/doi/pdf/10.1145/1830761.1830790). [Comparing Results of 31 Algorithms from the Black-Box Optimization Benchmarking BBOB-2009.](https://dl.acm.org/doi/pdf/10.1145/1830761.1830790) [Workshop Proceedings of the GECCO Genetic and Evolutionary Computation Conference 2010, ACM.](https://dl.acm.org/doi/pdf/10.1145/1830761.1830790) 

A list of all so-far benchmarked algorithms on the bbob suite together with their links to papers describing the experiment can be found in our <a href="https://numbbo.github.io/data-archive/bbob/">bbob data archive</a>. Postprocessed data can be found <a href="https://numbbo.github.io/ppdata-archive">here</a>. For detailed explanations of how to use the functions in a COCO benchmarking experiment, please go to the <a href="https://github.com/numbbo/coco">COCO code page</a> on Github.

<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
