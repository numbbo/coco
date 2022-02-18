cmake_minimum_required(VERSION 3.19)
project(ExampleExperiment
  DESCRIPTION "Example COCO experiment"
  LANGUAGES C)

find_library(MATH_LIBRARY m)

include(CheckCompilerFlag)
check_compiler_flag(C "-pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion" CC_HAS_WALL_ETC)

add_library(coco STATIC coco.c coco.h)
target_include_directories(coco PUBLIC .)
if(MATH_LIBRARY)
    target_link_libraries(coco PUBLIC ${MATH_LIBRARY})
endif()
# Add warning flags
if (MSVC)
  target_compile_options(coco PRIVATE "/W3")
elseif (CC_HAS_WALL_ETC)
  target_compile_options(coco PRIVATE -pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion)
endif()

add_executable(example_experiment example_experiment.c)
target_link_libraries(example_experiment PUBLIC coco)
if(MATH_LIBRARY)
    target_link_libraries(example_experiment PUBLIC ${MATH_LIBRARY})
endif()
