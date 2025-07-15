# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/nnmtp)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/nnmtp)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/nnmtp)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. Add subdirectory
if (NOT DEFINED MAX_MTP_LEVEL)
    message(STATUS "MAX_MTP_LEVEL is set to 18")
    set(MAX_MTP_LEVEL 18)
else()
    message(STATUS "MAX_MTP_LEVEL is set to ${MAX_MTP_LEVEL}")
endif()


# 5. compile
add_library(mtp_param SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParam.cc)
target_include_directories(mtp_param PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)


# 6. compile test
if (BUILD_TEST)
    add_executable(test_nn_mtp_utils ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtpUtils.cc)
    target_link_libraries(test_nn_mtp_utils PRIVATE gtest gtest_main)

    add_executable(test_nn_mtp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtp.cc)
    target_link_libraries(test_nn_mtp PRIVATE mtp_param gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nn_mtp PRIVATE USE_OPENMP)
    endif()

    gtest_discover_tests(test_nn_mtp_utils)
    gtest_discover_tests(test_nn_mtp)
endif()