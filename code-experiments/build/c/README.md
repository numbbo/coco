NumBBO/CoCO Framework in C (Experimental Part)
==============================================

Prerequisites
-------------

The simplest way to check the prerequisits is to go directly to [_Getting Started_](#Getting-Started)
below and give it a try. Then act upon failure, as in this case probably one of
the following is lacking: 

- Python is installed (version >=2.6). If this is not the case, check out
  [Anaconda](https://www.continuum.io), as this provides additionally all
  Python packages necessary to run the COCO postprocessing as well as the
  ipython shell. The postprocessing needs currently Python < 3.0, i.e. 2.6 or 2.7.

- `make` is installed and works with one of the Makefiles provided in 
  this folder. You might want to type `make` within this folder to see 
  whether this works. 
  
- A C compiler, like `gcc`, which is invoked by `make`, is installed. 

### Nmake

Instead of `make` and `Makefile` you can use `nmake` and the corresponding `NMakefile`

### CMake

Instead of `make` you can use `cmake` and the corresponding `CMakeLists.txt`:

```
mkdir build
cd build
cmake ../
ninja # or make, depending on your version of CMake
```

### meson

Instead of `make` you can use `meson` and `ninja`: 

```
meson setup build
meson compile -C build
```

Getting Started
---------------

See [here](../../../README.md#Getting-Started) for the first steps. Then

- Copy the files `example_experiment.c`, `coco.c`, `coco.h` and `Makefile` to a folder
  of your choice. Modify the `example_experiment.c` file to include the solver of your
  choice (instead of  `random_search`). Do not forget to also choose the right
  benchmarking suite and the corresponding observer.

- Invoke `make` to compile and run your experiment.
