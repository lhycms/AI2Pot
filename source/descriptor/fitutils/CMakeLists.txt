# 3. Set Dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/fitutils)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/fitutils)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/fitutils)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. Add subdirectory
if (CUDA_FOUND AND USE_CUDA)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()


# 5. Compile
if (TORCH_OP)
    # 5.1.
    add_library(target_statistics_op_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/targetStatisticsOp_cpu.cc)
    target_link_libraries(target_statistics_op_cpu PRIVATE ${TORCH_LIBRARIES})

    if (USE_CUDA AND CUDA_FOUND)
        add_library(target_statistics_op_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/targetStatisticsOp_cuda.cc)
        target_link_libraries(target_statistics_op_cuda
                              PRIVATE
                              #
                              ${TORCH_LIBRARIES})
    endif()

    # 5.2. 
    add_library(descriptor_statistics_op_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptorsStatisticsOp_cpu.cc)
    target_link_libraries(descriptor_statistics_op_cpu PRIVATE ${TORCH_LIBRARIES})

    if (USE_CUDA AND CUDA_FOUND)
        add_library(descriptor_statistics_op_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptorsStatisticsOp_cuda.cc)
        target_link_libraries(descriptor_statistics_op_cuda 
                              PRIVATE 
                              ${TORCH_LIBRARIES} descriptor_statistics_torch_launcher)
    endif()

    # 5.3. 

    # 5.4.
    add_library(fitutils_bind_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/fitutils_bind_cpu.cc)
    target_include_directories(fitutils_bind_cpu PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(fitutils_bind_cpu
                          PRIVATE
                          target_statistics_op_cpu
                          descriptor_statistics_op_cpu
                          ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

    if (USE_CUDA AND CUDA_FOUND)
        add_library(fitutils_bind_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/fitutils_bind_cuda.cc)
        target_include_directories(fitutils_bind_cuda PRIVATE ${PYTHON_INCLUDE_DIRS})
        target_link_libraries(fitutils_bind_cuda
                              PRIVATE
                              target_statistics_op_cuda
                              descriptor_statistics_op_cuda 
                              ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
    endif()
endif()


# 6. Compile test
if (BUILD_TEST)
    add_executable(test_target_statistics ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_targetStatistics.cc)
    target_link_libraries(test_target_statistics PRIVATE gtest gtest_main)

    add_executable(test_all_type_descriptors_statistics ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_AllTypeDescriptorsStatistics.cc)
    target_link_libraries(test_all_type_descriptors_statistics PRIVATE gtest gtest_main)

    add_executable(test_each_type_descriptors_statistics ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_EachTypeDescriptorsStatistics.cc)
    target_link_libraries(test_each_type_descriptors_statistics PRIVATE gtest gtest_main)

    gtest_discover_tests(test_target_statistics)
    gtest_discover_tests(test_all_type_descriptors_statistics)
endif()
