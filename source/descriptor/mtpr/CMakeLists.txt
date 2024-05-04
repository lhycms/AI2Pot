# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. compile
add_library(mtpParam SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParam.cc)


# 6. compile test
add_executable(test_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basis.cc)
target_link_libraries(test_basis PRIVATE gtest gtest_main)

add_executable(test_mtpParam ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpParam.cc)
target_link_libraries(test_mtpParam PRIVATE mtpParam gtest gtest_main)

add_executable(test_mtpBasis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasis.cc)
target_link_libraries(test_mtpBasis PRIVATE mtpParam gtest gtest_main)

gtest_discover_tests(test_basis)
gtest_discover_tests(test_mtpParam)
gtest_discover_tests(test_mtpBasis)