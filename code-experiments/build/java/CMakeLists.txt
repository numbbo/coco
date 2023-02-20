cmake_minimum_required(VERSION 3.20)
 
project(COCO LANGUAGES C)

find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)

add_jar(coco
  Benchmark.java 
  CocoJNI.java 
  ExampleExperiment.java
  Observer.java
  Problem.java
  Suite.java
  GENERATE_NATIVE_HEADERS coco-native)

add_library(CocoJNI SHARED CocoJNI.c)
target_link_libraries(CocoJNI PRIVATE coco-native)
target_include_directories(CocoJNI PRIVATE ${JNI_INCLUDE_DIRS})
