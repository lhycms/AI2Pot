# 3.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/nep/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/nep/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/nep/gpu)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. compile


# 6. compile test
if (BUILD_TEST)
    add_executable(test_nep_basis_gpu ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_basis.cu)
    target_link_libraries(test_nep_basis_gpu PRIVATE gtest gtest_main)

    add_executable(test_nep_utilities_gpu ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_utilities.cu)
    target_link_libraries(test_nep_utilities_gpu PRIVATE gtest gtest_main)

    add_executable(test_blm_gpu ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_Blm.cu)
    target_link_libraries(test_blm_gpu PRIVATE gtest gtest_main)

    add_executable(test_nep_gpu ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep.cu)
    target_link_libraries(test_nep_gpu PRIVATE gtest gtest_main)

    gtest_discover_tests(test_nep_basis_gpu DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_nep_utilities_gpu DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_blm_gpu DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_nep_gpu DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
endif()
