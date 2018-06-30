# Requirements for the `top-trumps` test suite

The `top-trumps` problems are available in the shared C++ 
library `librw-top-trumps.so` (or `rw-top-trumps.dll`). The library is compiled by running 
````
python do.py build-rw-top-trumps
````
from the root directory of the repository.

Currently, the library needs to be in the same folder as the experiment executable 
(TODO: take care of this!).

If using C or Python, there is no need to compile the library separately, the "usual" commands 
````
python do.py build-c
````
and
````
python do.py build-python
````
already take care of this.
