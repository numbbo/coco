cmake_minimum_required(VERSION 3.19)
project(COCOUnitTest
        DESCRIPTION "COCO C unit tests"
        LANGUAGES C)

include(CTest)

find_library(LIBM m)
find_library(LIBRT rt)

if(MATH_LIBRARY)
    target_link_libraries(about_equal PUBLIC ${MATH_LIBRARY})
endif()

macro(integration_test name) 
  add_executable(${name} ${name}.c ${ARGN})  
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

integration_test(test_bbob-constrained coco.c)
integration_test(test_bbob-largescale coco.c)
integration_test(test_bbob-mixint)
integration_test(test_biobj coco.c)
integration_test(test_instance_extraction coco.c)