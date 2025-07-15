# 1. Basic
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(ai2pot)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
#set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
#LD_PRELOAD=/data/app/gcc/11.3.0/lib64/libasan.so python3 /data/home/liuhanyu/mycode/AI2Pot/test/test1.py

set(CMAKE_GTEST_DISCOVERY_TIMEOUT 600)

set(CUDA_ARCHITECTURES 61)
set(CUDA_STANDARD 17)


# 2. include FetchContent
## 2.1. Way 1: Build gtest
#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        GIT_REPOSITORY https://github.com/google/googletest.git
#        GIT_TAG v1.14.0
#)
#FetchContent_MakeAvailable(googletest)
#include(GoogleTest)

## 2.1. Way 2: Build gtest
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0) # Compitable with torch2.x
add_subdirectory("$ENV{AI2POT_PATH}/source/ext/googletest-main")
include(GoogleTest)


## 2.2. pybind11
execute_process(COMMAND python3 -c "import pybind11; print(pybind11.__path__[0])" OUTPUT_VARIABLE PYBIND11_PATH)
string(REPLACE "\n" "/share/cmake" PYBIND11_CMAKE_PREFIX_PATH "${PYBIND11_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PREFIX_PATH})
find_package(pybind11 CONFIG REQUIRED)

## 2.3. numpy
execute_process(COMMAND python3 -c "import numpy as np;print(np.__path__[0])" OUTPUT_VARIABLE NUMPY_ROOT_PATH)
string(REPLACE "\n" "/core/include" NUMPY_INCLUDE_PATH ${NUMPY_ROOT_PATH})

## 2.4. torch
execute_process(COMMAND python3 -c "import torch;print(torch.__path__[0])" OUTPUT_VARIABLE TORCH_PATH)
string(REPLACE "\n" "/share/cmake" TORCH_CMAKE_PREFIX_PATH ${TORCH_PATH})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

## 2.5. FindOpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## 2.6. CUDA
if (USE_CUDA)
    find_package(CUDA)
    if (CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION_STRING}")
        enable_language(CXX CUDA)
        set(CMAKE_CUDA_ARCHITECTURES 61)
    else()
        message(STATUS "CUDA not found, compile only CPU version.")
        enable_language(CXX)
    endif()
endif()

# 3. dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. add_subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nblist)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/correction)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/deepmd)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/mtp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/mtpr)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/nnmtp)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/ace)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fvt)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gst)
