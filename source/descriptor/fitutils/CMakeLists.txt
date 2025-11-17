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
    add_library(target_statistics_op SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/targetStatisticsOp.cc)
    target_link_libraries(target_statistics_op PRIVATE ${TORCH_LIBRARIES})

    add_library(descriptor_statistics_op SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptorsStatisticsOp.cc)
    target_link_libraries(descriptor_statistics_op PRIVATE ${TORCH_LIBRARIES})

    add_library(fitutils_bind SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/fitutils_bind.cc)
    target_include_directories(fitutils_bind PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(fitutils_bind PRIVATE target_statistics_op descriptor_statistics_op ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
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

