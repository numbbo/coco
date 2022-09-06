  - `build/` contains the builds which can be used to run experiments. 
    See [here](https://github.com/numbbo/coco/blob/master/README.md) 
    and `do.py` in the root folder for how to build the code.  
  - `documentation/` doesn't contain anything really useful yet
  - `examples/` contains (more) examples, but see `build` first
  - `src/` contains the source code.
  - `test/` contains unit and integration tests. 

# New features

## Compute the inverse transform of non-linear transformations

The goal is to be able to query the inverse of the oscillating and asymmetric transformations.
They are respectively in `transform_vars_asymmetric.c` and `transform_vars_oscillating.c`.
The transformations are strictly monotonous, hence invertible but their inverse does not yield a closed-form (a tool like SymPy is not able to get a closed).
There is still hope to compute the inverse numerically using a _root finding_ algorithm like Brent's method.
This method is implemented in implemented in `brentq.c` under the `brentq` function.
It takes a `callback_type` first argument which is the mathematical function $T$ for which we want to find a root.
The `brentq` function is then wrapped into another function, `brentinv`, which features a simple heuristic to find the initial interval and also create some kind of lambda function $T(x) - y$ such that we can invert $T$ at any specified point $y$.

We described the procedure to invert a univariate mathematical function.
In the case of the bbob-constrained test suite, we want to invert a multivariate transformation.
This transformation applies coordinate-wise so it falls back to apply this procedure to each coordinate.
Some changes to the existing code were needed to do so.

## Changes

### to the transformation code

*I describe what was done for the asymmetric transformation but this applies to the oscillating as well (where it is even simpler because it does not depend on the dimension).*

First, the asymmetric transformation code was duplicated in `transform_vars_asymmetric_evaluate_function` and in `transform_vars_asymmetric_evaluate_constraint`.
So I extracted the transformation from these into the `static transform_vars_asymmetric_data_t *tasy` function. 
Then, I had to create the univariate transformation `tasy_uv` from the existing `tasy` function.
The signature of this function is static double `tasy_uv(double xi, tasy_data *d)` and `tasy_data` is a new structure containing additional parameters to the transformation.
In particular, the asymmetric transformation depends on the coordinate index `i` and total number of variables `n` so these are two elements of the structure.
This data can be passed to the brentinv method because it signature is `double brentinv(callback_type f, double y, void *func_data)`.

### to the declaration code

The inverse transform needs to be applied to the feasible direction in the same time as we apply the transformation to the problem.
A proposition is then to have a new function calling the `brentinv` method coordinate wise, and to call this function in the method to instantiate each problem in the `suite_cons_bbob_problems.c` file.

For example, in  `static coco_problem_t *f_ellipsoid_c_linear_cons_bbob_problem_allocate`, just after

```c
problem = transform_vars_oscillate(problem);
feasible_direction = transform_inv_feas_dir_asymmetric(problem, feasible_direction)  # this is new
transform_inv_feas_dir_asymmetric(problem, feasible_direction)  # or this (why problem is not changed inplace ?)
```

### to the testing code

TODO

### other TODOs:

- License
- 
### Status

06/09/22:
- code compiles but instantiation of the bbob-constrained suite in python crashes
- if `transform_inv_feas_dir_*` is not used, the code fails at runtime calling a function where a nonlin transform is applied
Understand [function points](https://www.geeksforgeeks.org/function-pointer-in-c/)

Which one ?

```c
typedef double (*callback_type)(double, void*);
```

```c
static double tosz_uv_inv(double yi, tosz_data *d) {
  double xi;
  callback_type fun_ptr = &tosz_uv;
  xi = brentinv(fun_ptr, yi, d);
  return xi;
}
```

```c
static double tosz_uv_inv(double yi, tosz_data *d) {
  double xi;
  double (*fun_ptr)(double, void*) = &tosz_uv;
  xi = brentinv(fun_ptr, yi, d);
  return xi;
}
```