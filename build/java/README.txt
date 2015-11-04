*****************************************************
*          Content of build/java directory          *
*****************************************************

build/java: contains necessary source files to generate the shared library for calling coco C funtions and an example of testing a java optimizer on the coco benchmark

Files:
- JNIinterface.java: class declaring native methods (methods that have to be written in C and that will call C functions of coco.c)
- JNIinterface.h & JNIinterface.C: files defining native methods in JNIinterface.java. These two files will be used to generate the shared library
- Benchmark.java, Problem.java, and NoSuchProblemException.java: define java classes (Benchmark, Problem, and NoSuchProblemException)
- demo.java: defines an optimiser and tests it on the coco benchmark

Generating the shared library (tested on a Mac OS X version 10.9.5):
Under build/java do:
	gcc -I/System/Library/Frameworks/JavaVM.framework/Headers -c JNIinterface.c
	gcc -dynamiclib -o libJNIinterface.jnilib JNIinterface.o

To run the example:
- first, compile all the .java files (javac *.java for example)
- then run demo.o by typing java demo




Generating the shared library (for Windows without Cygwin, and with 32bit MinGW gcc compiler):
Under build/java do:
	gcc -Wl,--kill-at -I"C:\PATH_TO_YOUR_JDK\include" -I"C:\PATH_TO_YOUR_JDK\include\win32" -shared -o JNIinterface.dll JNIinterface.c

You should have now a JNIinterface.dll file in this folder. Now run the example:
- first, compile all the .java files (by typing 'javac *.java' for example)
- then run the demo by typing 'java demo'

