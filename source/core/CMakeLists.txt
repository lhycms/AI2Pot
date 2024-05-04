#
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/core)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/core)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/core)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# 5. 
if(BUILD_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    
    add_executable(test_vecx ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vecx.cc)
    target_link_libraries(test_vecx PRIVATE gtest_main gtest)

    add_executable(test_AlignedArray ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_AlignedArray.cc)
    target_link_libraries(test_AlignedArray PRIVATE gtest_main gtest)

    add_executable(test_hardware ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hardware.cc)
    target_link_libraries(test_hardware PRIVATE gtest_main gtest)

    add_executable(test_vec3Operation ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vec3Operation.cc)
    target_link_libraries(test_vec3Operation PRIVATE gtest gtest_main)

    add_executable(test_arrayUtils ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_arrayUtils.cc)
    target_link_libraries(test_arrayUtils PRIVATE gtest gtest_main)
    
    gtest_discover_tests(test_vecx)
    gtest_discover_tests(test_AlignedArray)
    gtest_discover_tests(test_hardware)
    gtest_discover_tests(test_vec3Operation)
    gtest_discover_tests(test_arrayUtils)
endif()