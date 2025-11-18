# 3. Set Dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/fitutils/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/fitutils/gpu)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/fitutils/gpu)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. Compile
add_library(descriptor_statistics_torch_launcher SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptorsStatistics_torch_launcher.cu)
set_target_properties(descriptor_statistics_torch_launcher PROPERTIES
    CUDA_SEPARABLE_COMPILATION on
    CUDA_STANDARD ${CUDA_STANDARD}
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})


# 6. Compile test
if (BUILD_TEST)
    add_executable(test_AllTypeDescriptorsStatistics ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_AllTypeDescriptorsStatistics.cu)
    target_link_libraries(test_AllTypeDescriptorsStatistics PRIVATE gtest gtest_main)
endif()
