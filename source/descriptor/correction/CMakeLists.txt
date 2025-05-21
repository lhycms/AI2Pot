# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/correction)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/correction)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/correction)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. add subdirectory


# 5. compile
if (BUILD_TEST)
    add_executable(test_pair_zbl ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pair_zbl.cc)
    target_link_libraries(test_pair_zbl PRIVATE gtest gtest_main)
    gtest_discover_tests(test_pair_zbl DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})

    add_executable(test_group_zbl ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_group_zbl.cc)
    target_link_libraries(test_group_zbl PRIVATE gtest gtest_main)
    gtest_discover_tests(test_group_zbl DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
endif()
