# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/nnmtp)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/nnmtp)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/nnmtp)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. Add subdirectory
if (NOT DEFINED MAX_MTP_LEVEL)
    message(STATUS "MAX_MTP_LEVEL is set to 16")
    set(MAX_MTP_LEVEL 16)
else()
    message(STATUS "MAX_MTP_LEVEL is set to ${MAX_MTP_LEVEL}")
endif()

if (USE_CUDA AND CUDA_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()


# 5. compile
add_library(mtpParam_nnMtp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParam.cc)
target_include_directories(mtpParam_nnMtp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (TORCH_OP)
    ## 5.0. 
    add_library(mtpParamOp_nnMtp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParamOp.cc)
    target_include_directories(mtpParamOp_nnMtp PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(mtpParamOp_nnMtp mtpParam_nnMtp ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})

    ## 5.1. nnMtpOp (CPU and CUDA)
    add_library(nnMtpOp_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/nnMtpOp_cpu.cc)
    target_include_directories(nnMtpOp_cpu PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(nnMtpOp_cpu PRIVATE ${TORCH_LIBRARIES_CPU})
    if (USE_OPENMP)
        target_compile_definitions(nnMtpOp_cpu PRIVATE USE_OPENMP)
    endif()
    if (MAX_MTP_LEVEL)
        target_compile_definitions(nnMtpOp_cpu PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(nnMtpOp_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/nnMtpOp_cuda.cc)
        target_link_libraries(nnMtpOp_cuda 
                              PRIVATE 
                              ${TORCH_LIBRARIES}
                              efv_rmse_torch_launcher
                              nnmtp_torch_launcher
                              nnmtp_loss_torch_launcher
                              zbl_torch_launcher)
        target_compile_definitions(nnMtpOp_cuda PRIVATE USE_CUDA)
        if (MAX_MTP_LEVEL)
            target_compile_definitions(nnMtpOp_cuda PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
        endif()
    endif()

    ## 5.2. nnMtp_bind (CPU and CUDA)
    add_library(nnmtp_bind_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/nnmtp_bind_cpu.cc)
    target_include_directories(nnmtp_bind_cpu PRIVATE ${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
    target_link_libraries(nnmtp_bind_cpu
                          PRIVATE
                          mtpParam_nnMtp
                          mtpParamOp_nnMtp
                          nnMtpOp_cpu
                          ${TORCH_LIBRARIES_CPU}
                          ${PYTHON_LIBRARIES})
    if (USE_OPENMP)
        target_compile_definitions(nnmtp_bind_cpu PRIVATE USE_OPENMP)
    endif()
    if (MAX_MTP_LEVEL)
        target_compile_definitions(nnmtp_bind_cpu PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(nnmtp_bind_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/nnmtp_bind_cuda.cc)
        target_include_directories(nnmtp_bind_cuda PRIVATE ${PYTHON_INCLUDE_DIRS})
        target_link_libraries(nnmtp_bind_cuda 
                              PRIVATE
                              ${PYTHON_LIBRARIES}
                              ${TORCH_LIBRARIES}
                              nnMtpOp_cuda)
        target_compile_definitions(nnmtp_bind_cuda PRIVATE USE_CUDA)
        if (MAX_MTP_LEVEL)
            target_compile_definitions(nnmtp_bind_cuda PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
        endif()
    endif()
endif()



# 6. compile test
if (BUILD_TEST)
    add_executable(test_nnMtpUtils ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtpUtils.cc)
    target_link_libraries(test_nnMtpUtils PRIVATE gtest gtest_main)

    add_executable(test_nnMtp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtp.cc)
    target_link_libraries(test_nnMtp PRIVATE mtpParam_nnMtp gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nnMtp PRIVATE USE_OPENMP)
    endif()

    add_executable(test_nnMtpLoss ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtpLoss.cc)
    target_link_libraries(test_nnMtpLoss PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nnMtp PRIVATE USE_OPENMP)
    endif()

    add_executable(test_nnMtp_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtp_cpu_launcher.cc)
    target_link_libraries(test_nnMtp_cpu_launcher 
                          PRIVATE 
                          mtpParam_nnMtp
                          gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_nnMtp_cpu_launcher PRIVATE USE_OPENMP)
    endif()
    
    add_executable(test_nnMtpLoss_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nnMtpLoss_cpu_launcher.cc)
    target_link_libraries(test_nnMtpLoss_cpu_launcher PRIVATE gtest gtest_main)

    gtest_discover_tests(test_nnMtpUtils)
    gtest_discover_tests(test_nnMtp)
    gtest_discover_tests(test_nnMtp_cpu_launcher)
    gtest_discover_tests(test_nnMtpLoss_cpu_launcher)
endif()