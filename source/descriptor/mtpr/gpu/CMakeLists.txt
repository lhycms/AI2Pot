# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


# 5. compile
if (NOT DEFINED MAX_MTP_LEVEL)
    message(STATUS "MAX_MTP_LEVEL is set to 16")
    set(MAX_MTP_LEVEL 16)
else()
    message(STATUS "MAX_MTP_LEVEL is set to ${MAX_MTP_LEVEL}")
    #set(MAX_MTP_LEVEL ${MAX_MTP_LEVEL})
endif()


# 6. compile test
if (BUILD_TEST)
    add_executable(test_mtp_utilities ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtp_utilities.cu)
    target_compile_definitions(test_mtp_utilities PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_mtp_utilities PRIVATE gtest gtest_main)
    
    add_executable(test_mtp_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasis.cu)
    target_compile_definitions(test_mtp_basis PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_mtp_basis PRIVATE mtpParam gtest gtest_main)

    add_executable(test_linear_mtp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtp.cu)
    target_compile_definitions(test_linear_mtp PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_linear_mtp PRIVATE mtpParam gtest gtest_main)

    add_executable(test_linear_mtp_loss ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtpLoss.cu)
    target_compile_definitions(test_linear_mtp_loss PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_linear_mtp_loss PRIVATE gtest gtest_main)
    
    gtest_discover_tests(test_mtp_utilities DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_mtp_basis DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_linear_mtp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_linear_mtp_loss DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
endif()
