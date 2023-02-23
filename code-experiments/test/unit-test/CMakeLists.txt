cmake_minimum_required(VERSION 3.19)
project(COCOUnitTest
        DESCRIPTION "COCO C unit tests"
        LANGUAGES C)

include(CTest)

find_library(LIBM m)
find_library(LIBRT rt)

add_library(about_equal STATIC about_equal.c about_equal.h)
target_include_directories(about_equal PUBLIC .)
if(MATH_LIBRARY)
    target_link_libraries(about_equal PUBLIC ${MATH_LIBRARY})
endif()

macro(minunit_test name)  
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PUBLIC about_equal)
  if (LIBM)
    target_link_libraries(${name} PUBLIC ${LIBM})
  endif()
  if (LIBRT)
    target_link_libraries(${name} PUBLIC ${LIBRT})
  endif()
  add_test(NAME ${name} 
           COMMAND $<TARGET_FILE:${name}>
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endmacro()

minunit_test(test_biobj_utilities)
minunit_test(test_brentq)
minunit_test(test_coco_archive)
minunit_test(test_coco_observer)
minunit_test(test_coco_problem)
minunit_test(test_coco_string)
minunit_test(test_coco_utilities)
minunit_test(test_logger_biobj)
minunit_test(test_mo_utilities)
