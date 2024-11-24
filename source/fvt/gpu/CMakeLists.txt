# 3. 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/fvt/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/fvt/gpu)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/fvt/gpu)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5.


# 6. 
if (BUILD_TEST)
    add_executable(test_force_sr_gpu ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_force_sr.cu)
    set_target_properties(test_force_sr_gpu PROPERTIES
        CUDA_STANDARD ${CMAKE_CXX_STANDARD}
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
    target_link_libraries(test_force_sr_gpu PRIVATE gtest gtest_main)

    gtest_discover_tests(test_force_sr_gpu)
endif()
