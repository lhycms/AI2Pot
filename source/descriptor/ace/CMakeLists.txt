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

    add_executable(test_sinlm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sinlm.cc)
    target_link_libraries(test_sinlm PRIVATE gtest gtest_main)
    gtest_discover_tests(test_sinlm)

    add_executable(test_nep_descriptor ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_descriptor.cc)
    target_link_libraries(test_nep_descriptor PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_descriptor)
endif()