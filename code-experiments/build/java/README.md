NumBBO/CoCO Framework in Java (Experimental Part)
=================================================

If you have setup and tested everything once with `python do.py run-java`, 
see [here](https://github.com/numbbo/coco/blob/master/README.md), 
type:

    javac *java
    
to (re-)compile the example experiment and 

    java ExampleExperiment

to run it. Then change `ExampleExperiment.java` as needed.  

Details
-------

### Content of the build/java folder


This folder contains necessary source files to generate the shared library
for calling coco C funtions and an example of testing a java optimizer on
the coco benchmark

Files:
- CocoJNI.java: class declaring native methods (methods that have to be
  written in C and that will call C functions of coco.c)
- CocoJNI.h & CocoJNI.C: files defining native methods in CocoJNI.java.
  These two files will be used to generate the shared library
- Benchmark.java, Problem.java, Suite.java, Observer.java: define java
  classes (Benchmark, Problem, Suite, Observer)
- ExampleExperiment.java: defines an optimizer and tests it on the coco
  benchmark

*****************************************************

### Compilation without `python do.py`

#### for Linux and OSX (tested on a Mac OS X version 10.9.5)
For generating the shared library, under build/java do:
```
	gcc -I/System/Library/Frameworks/JavaVM.framework/Headers -c CocoJNI.c
	gcc -dynamiclib -o libCocoJNI.jnilib CocoJNI.o
```
To run the example:
- first, compile all the .java files (by typing `javac *.java` for example)
- then type `java ExampleExperiment` to run the example experiment

#### for Windows without Cygwin and with 32bit MinGW gcc compiler
For generating the shared library, under build/java do:
```
	gcc -Wl,--kill-at -I"C:\PATH_TO_YOUR_JDK\include" -I"C:\PATH_TO_YOUR_JDK\include\win32" -shared -o CocoJNI.dll CocoJNI.c
```
You should have now a CocoJNI.dll file in this folder. Now run the example:
- first, compile all the .java files (by typing `javac *.java` for example)
- then run the example experiment by typing `java ExampleExperiment`

#### for Windows with Cygwin and the x86_64-w64-mingw32-gcc compiler
For generating the shared library, under build/java do:
```
	x86_64-w64-mingw32-gcc -D __int64="long long" -Wl,--add-stdcall-alias -I"C:\PATH_TO_YOUR_JDK\include" -I"C:\PATH_TO_YOUR_JDK\include\win32" -shared -o CocoJNI.dll CocoJNI.c
```
You should have now a CocoJNI.dll file in this folder. Now run the example:
- first, compile all the .java files (by typing `javac *.java` for example)
- then run the example experiment by typing `java ExampleExperiment`
