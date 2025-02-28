cmake_minimum_required(VERSION 3.22)

project(cornerstone-octree CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(MPI)
find_package(OpenMP REQUIRED)

# if INTERFACE_LINK_OPTIONS for OpenMP_CXX is not set, we set
# INTERAFACE_LINK_OPTIONS = INTERFACE_COMPILE_OPTIONS
# i.e. we add e.g. the -fopenmp flag to the linker
get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS)
if (NOT OpenMP_CXX_INTERFACE_LINK_OPTIONS)
    # fill in using INTERFACE_COMPILE_OPTIONS
    get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OpenMP_CXX_INTERFACE_LINK_OPTIONS})
endif()

include(CTest)
include(CheckLanguage)

set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_STANDARD 17)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    enable_language(HIP)
    find_package(hip)
    set(CMAKE_HIP_STANDARD 17)
endif()

add_subdirectory(include)
if (BUILD_TESTING)
    add_subdirectory(test)
endif()

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind module
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Test if we have the right version.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --version
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_VER)
if(${NB_VER} VERSION_LESS 2.0.0)
  message(FATAL_ERROR "Require nanobind version greater than 2.0.0, but found ${NB_VER}")
endif()

nanobind_add_module(cornerstone python/extensions.cpp)
target_include_directories(cornerstone PRIVATE include test)
