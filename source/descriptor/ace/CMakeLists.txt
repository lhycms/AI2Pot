# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/ace)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/ace)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/ace)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# 5. compile 

# 6. compile test
if (BUILD_TEST)
    add_executable(test_radial_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_radialBasis.cc)
    target_link_libraries(test_radial_basis PRIVATE gtest gtest_main)
    gtest_discover_tests(test_radial_basis)

    add_executable(test_angular_utils ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_angularUtils.cc)
    target_link_libraries(test_angular_utils PRIVATE gtest gtest_main)
    gtest_discover_tests(test_angular_utils)

    add_executable(test_descriptor ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_descriptor.cc)
    target_link_libraries(test_descriptor PRIVATE gtest gtest_main)
    gtest_discover_tests(test_descriptor)
endif()