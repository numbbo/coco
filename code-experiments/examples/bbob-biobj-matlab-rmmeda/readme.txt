This package contains the Matlab source code of paper Q. Zhang, A. Zhou, Y. Jin. 'Modelling the Regularity in an Estimation of Distribution Algorithm for Continuous Multiobjective Optimisation with Variable Linkages'(http://cswww.essex.ac.uk/staff/qzhang/mypublication.htm).

v 0.1, Oct. 14 2007

File list:
LPCA.m  		- Local PCA algorithm
LPCAGenerator.m 	- EDA generator based on Local PCA
MOSelector.cpp	- selector in MOO algorithm, implement in C
ParetoFilter.cpp	- Pareto filter, implement in C
readme		- readme file
RMMEDA.m		- main algorithm of RM-MEDA
TEC.m 		- example(settings)


To use it, please use the following command to compile two components:

	mex MOSelector.cpp
	mex ParetoFilter.cpp

and run TEC() to see the example.

The package is tested under Windows Vista Business (Operation System) with Visual Studio 2005 (C++ compiler) and SUSE Linux 10.2 (Operation System) with GCC (C++ compiler).

For more information, please read the help in each .m file or send mails to amzhou@gmail.com or qzhang@essex.ac.uk.


 