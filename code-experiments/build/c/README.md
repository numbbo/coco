# C/C++ bindings for NumBBO/CoCO Experiment Framework 

## Getting Started

Building the C code should be fairly straight forward. 
We strive to make the code as portable as possible and provide two example build environments.

### CMake

Building the example experiment using CMake has minimal requirements:

- CMake version 3.19 or greater
- Working C compiler supporting C99 (any recent gcc, clang or msvc will do)

To build the included example experiment, run

```
cmake -B build .
cmake --build build
```

This will give you a debug build of the core COCO code and an example
experiment in the `build/` subdirectory.

:::{note}
On Windows the final executable is often placed in a subdirectory of the
`build/` directory depending on the build type. Check in `build/Debug/` or
`build/Release/`.
:::


### meson

Instead of `cmake`, you can also use `meson` and `ninja`: 

```
meson setup build
meson compile -C build
```
