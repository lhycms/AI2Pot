# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. 


# 5. compile


# 6. compile test
if (BUILD_TEST)
    add_executable(test_nep_utilities ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_utilities.cc)
    target_link_libraries(test_nep_utilities PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_utilities)
endif()
