# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. Add
if (CUDA_FOUND AND USE_CUDA)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()


# 5. compile
if (TORCH_OP)
    ## 5.1. nepOp (CPU and CUDA)
    add_library(nepOp_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/nepOp_cpu.cc)
    target_include_directories(nepOp_cpu PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(nepOp_cpu PRIVATE ${TORCH_LIBRARIES_CPU})
    if (USE_OPENMP)
        target_compile_definitions(nepOp_cpu PRIVATE USE_OPENMP)
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(nepOp_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/nepOp_cuda.cc)
        target_link_libraries(nepOp_cuda
                              PRIVATE
                              ${TORCH_LIBRARIES}
                              nep_torch_launcher
                              nep_loss_torch_launcher
                              zbl_torch_launcher)
        target_compile_definitions(nepOp_cuda PRIVATE USE_CUDA)
    endif()

    ## 5.2. nep_bind (CPU and CUDA)
    add_library(nep_bind_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/nep_bind_cpu.cc)
    target_include_directories(nep_bind_cpu PRIVATE ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(nep_bind_cpu PRIVATE ${TORCH_LIBRARIES_CPU} ${PYTHON_LIBRARIES} nepOp_cpu)
    if (USE_OPENMP)
        target_compile_definitions(nep_bind_cpu PRIVATE USE_OPENMP)
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(nep_bind_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/nep_bind_cuda.cc)
        target_include_directories(nep_bind_cuda PRIVATE ${PYTHON_INCLUDE_DIRS})
        target_link_libraries(nep_bind_cuda
                              PRIVATE
                              ${PYTHON_LIBRARIES}
                              ${TORCH_LIBRARIES}
                              nep_torch_launcher
                              nep_loss_torch_launcher
                              zbl_torch_launcher
                              nepOp_cuda)
        target_compile_definitions(nep_bind_cuda PRIVATE USE_CUDA)
    endif()
endif()


# 6. compile test
if (BUILD_TEST)
    add_executable(test_nep_index ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_index.cc)
    target_link_libraries(test_nep_index PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_index)

    add_executable(test_Blm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_Blm.cc)
    target_link_libraries(test_Blm PRIVATE gtest gtest_main)
    gtest_discover_tests(test_Blm)

    add_executable(test_nep_basis_rq ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basis.cc)
    target_link_libraries(test_nep_basis_rq PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_basis_rq)

    add_executable(test_nep_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nepBasis.cc)
    target_link_libraries(test_nep_basis PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_basis)

    add_executable(test_nep ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep.cc)
    target_link_libraries(test_nep PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nep PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep)

    add_executable(test_nep_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_cpu_launcher.cc)
    target_link_libraries(test_nep_cpu_launcher PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nep_cpu_launcher PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep_cpu_launcher)

    add_executable(test_nep_loss ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nepLoss.cc)
    target_link_libraries(test_nep_loss PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nep_loss PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep_loss)

    add_executable(test_nep_loss_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nepLoss_cpu_launcher.cc)
    target_link_libraries(test_nep_loss_cpu_launcher PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nep_loss_cpu_launcher PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep_loss_cpu_launcher)
endif()
