# 1. Basic
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(matersdk)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 2. include FetchContent
## 2.1. Way 1: Build gtest
#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        GIT_REPOSITORY https://github.com/google/googletest.git
#        GIT_TAG v1.12.0
#)
#FetchContent_MakeAvailable(googletest)
#include(GoogleTest)

## 2.2. Way 2: Build gtest
add_subdirectory("$ENV{AI2POT_PATH}/source/ext/googletest-main")
include(GoogleTest)


#list(APPEND CMAKE_PREFIX_PATH "/data/home/liuhanyu/anaconda3/envs/mlff38/lib/python3.8/site-packages/pybind11/share/cmake/")
#find_package(pybind11 CONFIG REQUIRED)
execute_process(COMMAND python3 -c "import pybind11; print(pybind11.__path__[0])" OUTPUT_VARIABLE PYBIND11_PATH)
string(REPLACE "\n" "/share/cmake" PYBIND11_CMAKE_PREFIX_PATH "${PYBIND11_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PREFIX_PATH})
find_package(pybind11 CONFIG REQUIRED)

# FindOpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/deepmd)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/mtp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/mtpr)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/descriptor/ace)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fvt)
