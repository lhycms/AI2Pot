# 2. Include package
execute_process(COMMAND python3 -c "import pybind11; print(pybind11.__path__[0])" OUTPUT_VARIABLE PYBIND11_PATH)
string(REPLACE "\n" "/share/cmake" PYBIND11_CMAKE_PREFIX_PATH ${PYBIND11_PATH})
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PREFIX_PATH})
find_package(pybind11 CONFIG REQUIRED)

# 3. 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/nblist)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/nblist)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/nblist)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. 
#add_library(CpuNeighborList SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/CpuNeighborList.cc)
#add_library(structure SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/structure.cc)
###pybind11_add_module(structure ${CMAKE_CURRENT_SOURCE_DIR}/bind/structure_bind.cc)


# 6.
if (BUILD_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

    # add_executable(test_CpuNeighborList ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_CpuNeighborList.cc)
    # target_link_libraries(test_CpuNeighborList PRIVATE gtest_main gtest CpuNeighborList)
    
    add_executable(test_basicStructureInfo ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basicStructureInfo.cc)
    target_link_libraries(test_basicStructureInfo PRIVATE gtest_main gtest)

    add_executable(test_structure ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_structure.cc)
    target_link_libraries(test_structure PRIVATE gtest_main gtest)

    add_executable(test_binLinkedList ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_binLinkedList.cc)
    target_link_libraries(test_binLinkedList PRIVATE gtest_main gtest)

    add_executable(test_neighborList ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_neighborList.cc)
    target_link_libraries(test_neighborList PRIVATE gtest gtest_main)

    # gtest_discover_tests(test_CpuNeighborList)    
    gtest_discover_tests(test_basicStructureInfo)
    gtest_discover_tests(test_structure)
    gtest_discover_tests(test_binLinkedList)
    gtest_discover_tests(test_neighborList)
endif()