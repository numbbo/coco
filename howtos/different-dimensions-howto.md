How to experiment with different dimensions
===========================================

The problem dimensions are basically hard-coded in the suites, but this doesn't mean 
that you cannot (or should not) run experiments with different dimensions. To do so,
follow these three steps:

##### 1. Find the suite you are interested in

Look in the `code-experiments/src` folder for a file named `suite_NAME.c`, where `NAME`
is the name of the suite you are interested in. In that file, find the 
`suite_NAME_initialize(void)` function. 

##### 2. Change the suite dimensions

The array of dimensions looks something like: 
```c
  const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };
```

Use the dimensions you wish (see the disclaimer below).

##### 3. Recompile the code

You need to recompile the code, which is done by any of the `python do.py build-LANG` or
`python do.py run-LANG` commands (where `LANG` is one of the supported languages).

You should now be able to run the experiments using the new dimensions.  

##### Disclaimer

The `bbob` functions might not work for `d=1`.  
