---
layout: default
permalink: /testsuites/bbob-noisy
parent: Test Suites
nav_order: 2
title: bbob-noisy
has_toc: false
---


# The bbob-noisy Test Suite

---

The bbob-noisy test suite contains 30 noisy, scalable test functions with three different noise types and levels. Each function is provided in various dimension (2, 3, 5, 10, 20, 40) and available in an arbitrary number of instances.


! Note that, for now, experiments on this test suite must be done with the [previous code base](../oldcode/bboball15.03.tar.gz).


The bbob-noisy functions
------------------------

The 30 noisy test functions of the bbob-noisy test suite are

<table align="center" style="width:50%">
<tr>
   <th colspan=3 style="text-align:left">1 Functions with moderate noise</th>
</tr>
<tr>
	<td style="width:2%">1.1</td><td style="width:2%">f101</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=7">Sphere with moderate Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f102</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=12">Sphere with moderate uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f103</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=17">Sphere with moderate seldom Cauchy noise</a></td>
</tr><tr>
	<td>1.2</td><td>f104</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=22">Rosenbrock with moderate Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f105</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=28">Rosenbrock with moderate uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f106</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=33">Rosenbrock with moderate uniform noise</a></td>
</tr>
<tr>
   <th colspan=3 style="text-align:left">2 Functions with severe noise</th>
</tr>
<tr>
	<td>2.1</td><td>f107</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=38">Sphere with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f108</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=43">Sphere with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f109</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=48">Sphere with seldom Cauchy noise</a></td>
</tr><tr>
	<td>2.2</td><td>f110</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=53">Rosenbrock with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f111</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=58">Rosenbrock with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f112</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=63">Rosenbrock with seldom Cauchy noise</a></td>
</tr><tr>
	<td>2.3</td><td>f113</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=68">Step ellipsoid with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f114</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=73">Step ellipsoid with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f115</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=78">Step ellipsoid with seldom Cauchy noise</a></td>
</tr><tr>
	<td>2.4</td><td>f116</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=83">Ellipsoid with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f117</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=88">Ellipsoid with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f118</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=93">Ellipsoid with seldom Cauchy noise</a></td>
</tr><tr>
	<td>2.5</td><td>f119</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=98">Different Powers with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f120</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=102">Different Powers with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f121</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=106">Different Powers with seldom Cauchy noise</a></td>
</tr>
<tr>
   <th colspan=3 style="text-align:left">3 Highly multi-modal functions with severe noise
</th>
</tr>
<tr>
	<td>3.1</td><td>f122</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=110">Schaffer's F7 with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f123</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=115">Schaffer's F7 with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f124</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=120">Schaffer's F7 with seldom Cauchy noise</a></td>
</tr><tr>
	<td>3.2</td><td>f125</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=125">Composite Griewank-Rosenbrock with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f126</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=129">Composite Griewank-Rosenbrock with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f127</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=133">Composite Griewank-Rosenbrock with seldom Cauchy noise</a></td>
</tr><tr>
	<td>3.3</td><td>f128</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=137">Gallagher's Gaussian Peaks 101-me with Gaussian noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f129</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=143">Gallagher's Gaussian Peaks 101-me with uniform noise</a></td>
</tr><tr>
	<td>&nbsp;</td><td>f130</td><td><a href="../oldcode/bbobdocnoisyfunctionsdef.pdf#page=148">Gallagher's Gaussian Peaks 101-me with seldom Cauchy noise</a></td>
</tr>
</table>

<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
