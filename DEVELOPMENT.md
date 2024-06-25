Working with the `coco` code base
=================================

## Before you start

To work with the `coco` source code, you need quite a few tools in very specific versions.
We provide a Conda environment file `env.yaml` and recommend you use it to setup your development environment.
It is regularly tested on Linux, Windows and macOS and is what is used in CI.

Following these steps should setup a fresh `coco` environment:

1. Create or update a new conda environment with all the requirements by running
 
   ```sh
   conda env create -f env.yaml
   ``` 
   or if you've already setup a `coco` environment
   ```sh
   conda env update -f env.yaml --prune
   ``` 

1. Activate environment
   ```sh
   conda activate coco
   ```

You now have all required dependencies to work on the code base.
If you notice that something is missing, please let us know by opening an [issue](https://github.com/numbbo/coco/issues/new/choose).

### Reusing a conda environment

Say you already have a conda environment named `foo` that you want to reuse because it contains some additional packages you need for development.
Then, once you've activated the environment, you can run

```sh
conda update -f env.yaml
```

and it will install all the required development dependencies into your existing environment.

## Working with the Experimental Code

Before any of the bindings can be built and after every change to the core files (in `code-experiments/src/`), you need to run `scripts/fabricate`.
The job of `fabricate` is to bundle the C files and place them in all the right places so that the language specific build tools can find them when building the respective bindings.
`fabricate` also updates the binding metadata to include version information and any other changes that need to be made from one build to the next.
It is harmless to run `fabricate` too often, but forgetting to run it means you might be using stale sources.

### Running unit tests

Change to the unit tests directory
```sh
cd code-experiments/test/unit-test
```

Rerun `fabricate` if not already done
```sh
python ../../../scripts/fabricate
```

Build tests using [`cmake`](https://cmake.org/cmake/help/latest/manual/cmake.1.html)
```sh
cmake -B build
cmake --build build
```

Run tests using [`ctest`](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
```sh
ctest --test-dir build     
```

### Regression tests

Note that I'm not sure the regression tests are really useful at the moment. But here goes:

Change to the regression tests directory
```sh
cd code-experiments/test/regression-test
```

The regression tests require the `cocoex` Python package. Install it first
```sh
python -m pip uninstall -y cocoex
python -m pip install ../../build/python/  
```

Now run the regression tests

```sh
python test_suites.py   
```
and 
```sh
python test_logger.py
```

### Integration tests

Still need to fix those up.

### `cocoex` Python package

Change to the `cocoex` Python package directory
```sh
cd code-experiments/build/python
```

Install the package from source
```sh
pip uninstall -y cocoex
pip install .
```
Under macOS with an ARM processor it may be necessary to run `arch -arm64 pip install .` instead.

Install and run `pytest`
```sh
pip install pytest
python -m pytest test
```

## Working with the Postprocessing Code

Before you begin, always run `python scripts/fabricate` to update any auto-generated files.
Then, you can install the `cocopp` package using pip:

```sh
python -m pip uninstall -y cocopp
cd code-postprocessing/
python -m pip install .
```

If you are working on `cocopp`, you can use an editable install and Python will pick up any changes you make without having to reinstall:

```sh
python -m pip uninstall -y cocopp
cd code-postprocessing
python -m pip install --editable .
```
Running `python -m cocopp.test` may not work on this installation.
