# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


# 5. compile


# 6. compile test
if (BUILD_TEST)
    add_executable(test_mtp_utilities ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtp_utilities)
    set_target_properties(test_mtp_utilities PROPERTIES
        CUDA_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD}
        CUDA_SEPERATE_COMPILATION on
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    target_link_libraries(test_mtp_utilities gtest gtest_main)
    
    
    gtest_discover_tests(test_mtp_utilities)
endif()
