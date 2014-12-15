{\rtf1\ansi\ansicpg1252\cocoartf1265\cocoasubrtf210
{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;}
\paperw11900\paperh16840\margl1440\margr1440\vieww28600\viewh15380\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f0\fs28 \cf0 \CocoaLigature0 *****************************************************\
*          Content of build/java directory          *\
*****************************************************\
\
build/java: contains necessary source files to generate the shared library for calling coco C funtions and an example of testing a java optimizer on the coco benchmark\
\
Files:\
- JNIinterface.java: class declaring native methods (methods that have to be written in C and that will call C functions of coco.c)\
- JNIinterface.h & JNIinterface.C: files defining native methods in JNIinterface.java. These two files will be used to generate the shared library\
- Benchmark.java & Problem.java NoSuchProblemException.java: define java classes (Benchmark, Problem, and NoSuchProblemException)\
- demo.java: defines an optimiser and tests it on the coco benchmark\
\
Generating the shared library (tested on a Mac OS X version 10.9.5):\
Under build/java do:\
	gcc -I/System/Library/Frameworks/JavaVM.framework/Headers -I/System/Library/Frameworks/JavaVM.framework/Headers -c JNIinterface.c\
	gcc -dynamiclib -o libJNIinterface.jnilib JNIinterface.o\
\
To run the example:\
- first, compile all the .java files (javac *.java for example)\
- then run demo.o by typing java demo}