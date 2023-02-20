# NumBBO/CoCO Framework in Java (Experimental Part)

Before you start, be aware that compiling native code for Java using JNI is clunky at best.
We will be using [`cmake`](https://cmake.org) and the included `CMakeLists.txt`.
This is not the only option, you can also manually build the included Java and C files.
That however is outside the scope of this document.

## Build using `cmake`

Configure the build using `cmake`:

```
cmake -B build .
```

Build Classes and native library

```
cmake --build build
```

You should now have a `coco.jar` and `libCocoJNI.so` or `CocoJNI.dll` in the `build/` subdirectory. 
You need *both* to run the experiment!

Calling 
```
java -classpath build/coco.jar -Djava.library.path=build/ ExampleExperiment
```
will run the experiment and write the results into `exdata/`.

## Details

### Content of the build/java folder


This folder contains necessary source files to generate the shared library
for calling coco C funtions and an example of testing a java optimizer on
the coco benchmark

Files:
- `CocoJNI.java`: class declaring native methods (methods that have to be written in C and that will call C functions of `coco.c`)
- `CocoJNI.h` & `CocoJNI.c`: files defining native methods in CocoJNI.java.
  These two files will be used to generate the shared library
- `Benchmark.java`, `Problem.java`, `Suite.java`, `Observer.java`: Java classes Benchmark, Problem, Suite, Observer
- `ExampleExperiment.java`: defines an optimizer and tests it on the coco
  benchmark
