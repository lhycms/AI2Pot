# 3. 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/nnmtp/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/nnmtp/gpu)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/nnmtp/gpu)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. compile


# 6. compule test
if (BUILD_TEST)
    add_executable(test_nnmtp_utils ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpUtils.cu)
    target_compile_definitions(test_nnmtp_utils PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_nnmtp_utils PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nnmtp_utils DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})

    add_executable(test_nnmtp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtp.cu)
    target_compile_definitions(test_nnmtp PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_nnmtp PRIVATE mtpParam_nnMtp gtest gtest_main)
    gtest_discover_tests(test_nnmtp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
endif()
