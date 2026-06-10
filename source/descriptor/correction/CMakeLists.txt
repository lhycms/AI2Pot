# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/correction)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/correction)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/correction)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. add subdirectory
if (USE_CUDA AND CUDA_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()

# 5. compile
if (BUILD_TEST)
    add_executable(testPairZbl ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pair_zbl.cc)
    target_link_libraries(testPairZbl PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(testPairZbl PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(testPairZbl DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})

    add_executable(testGroupZbl ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_group_zbl.cc)
    target_link_libraries(testGroupZbl PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(testGroupZbl PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(testGroupZbl DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
endif()
