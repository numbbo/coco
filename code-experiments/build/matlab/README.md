NumBBO/CoCO Framework in Matlab and Octave (Experimental Part)
==============================================================

Prerequisites
-------------

The simplest way to check the prerequisits is to go directly to [_Getting Started_](#Getting-Started)
below and give it a try. Then act upon failure, as in this case probably one of
the following is lacking: 

- Matlab is installed (version >=2008a) and in the path. Alternatively, you can also use the
  open source platform Octave (version >=4.0.0).

- Python is installed in the right version (>=2.6). We recommend using the Anaconda package
  (https://www.continuum.io) for installing python.
  
- A C compiler, like `gcc`, which is invoked by `mex`. For details handling non-supported versions,
  in particular when using `gcc` with MATLAB versions before 2015b, please follow
  http://gnumex.sourceforge.net/oldDocumentation/index.html


Getting Started
---------------

See _Getting Started_ [here](../../../README.md#Getting-Started) for the first steps. Then

- Copy (and rename) the folder `code-experiments/build/matlab` to a place (and name)
  of your choice. Modify the [`exampleexperiment.m`](./code-experiments/build/matlab/exampleexperiment.m`) 
  file to include the solver of your choice (instead of random search in `my_optimizer`).
  Do not forget to also choose the right benchmarking suite and the corresponding observer.

- Execute the modified file either from a system shell like 
  ```
      matlab exampleexperiment_new_name.m
  ```
  or
  ```
      octave exampleexperiment_new_name.m
  ```
  or in the Matlab/Octave shell like
  ```
      >>> example_experiment_new_name
  ```
  
See [here](../../../README.md#Getting-Started-pp) for the remaining steps to investate the resulting data.


Details and Known Issues
------------------------
- All of the compilation takes place in the `setup.m` file.
- For using gcc with older MATLAB versions than 2015b, please follow
  http://gnumex.sourceforge.net/oldDocumentation/index.html
- In case that the current Matlab wrapper of the Coco code does not work immediatly on
  the Mac OSX operating system, in particular if some parts have been updated lately, we
  suggest to try to add
  ```
  #define char16_t UINT16_T
  ```
  right before the `#include "mex.h"` lines of the corresponding C files. This holds
  especially for the more complicated example in `../../examples/bbob-biobj-matlab-smsemoa/`.


Tested Environments
-------------------
We cannot guarantee that the Matlab wrapper works on any combination of operating system
and compiler. However, compilation has been tested successfully under
   - Mac OSX with MATLAB 2012a 
   - Windows 7 with MATLAB 2014a and Visual Studio Professional.
   - Windows 10 with MATLAB 2008a and MinGW
   - Windows XP with MATLAB 2008b and MinGW
   - Windows 7 with GNU Octave 4.0.0
  
