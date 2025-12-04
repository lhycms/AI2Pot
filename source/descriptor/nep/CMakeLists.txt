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
    add_executable(test_nep_index ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_index.cc)
    target_link_libraries(test_nep_index PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_index)

    add_executable(test_Blm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_Blm.cc)
    target_link_libraries(test_Blm PRIVATE gtest gtest_main)
    gtest_discover_tests(test_Blm)

    add_executable(test_nep_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basis.cc)
    target_link_libraries(test_nep_basis PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_basis)


    add_executable(test_nep ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep.cc)
    target_link_libraries(test_nep PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep)
endif()
