# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/api)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/api)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/api)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. Compile
if (TORCH_OP)
    add_library(node_calculator SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/node_calculator.cc)
    
    if (USE_CUDA)
        target_include_directories(node_calculator PRIVATE ${Python_INCLUDE_DIRS})
        target_link_libraries(
            node_calculator
            PRIVATE
            ${TORCH_LIBRARIES}
            # mtpr
            mtpParamOp
            mtpr_bind_cpu
            linearMtpOp_cpu
            mtpr_bind_cuda
            linearMtpOp_cuda
            # nep
            nep_bind_cpu
            nepOp_cpu
            nep_bind_cuda
            nepOp_cuda
            # nnmtp
            mtpParamOp_nnMtp
            nnmtp_bind_cpu
            nnMtpOp_cpu
            nnmtp_bind_cuda
            nnMtpOp_cuda
            Python::Python)
    else()
        target_include_directories(node_calculator PRIVATE ${TORCH_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
        target_link_libraries(
            node_calculator
            PRIVATE
            ${TORCH_LIBRARIES_CPU}
            # mtpr
            mtpParamOp
            mtpr_bind_cpu
            linearMtpOp_cpu
            # nep
            nep_bind_cpu
            nepOp_cpu
            # nnmtp
            mtpParamOp_nnMtp
            nnmtp_bind_cpu
            nnMtpOp_cpu
            Python::Python)
    endif()
endif()

# 6. Compile test
if (TORCH_OP)
    add_executable(node_calculator_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_node_calculator.cc)
    
    if (USE_CUDA)
        target_include_directories(node_calculator_test PRIVATE ${Python_INCLUDE_DIRS})
        target_link_libraries(
            node_calculator_test
            PRIVATE
            gtest gtest_main
            ${TORCH_LIBRARIES}
            node_calculator
            # mtpr
            mtpParamOp
            mtpr_bind_cpu
            linearMtpOp_cpu
            mtpr_bind_cuda
            linearMtpOp_cuda
            # nep
            nep_bind_cpu
            nepOp_cpu
            nep_bind_cuda
            nepOp_cuda
            # nnmtp
            mtpParamOp_nnMtp
            nnmtp_bind_cpu
            nnMtpOp_cpu
            nnmtp_bind_cuda
            nnMtpOp_cuda
            Python::Python)
    else()
        target_include_directories(node_calculator_test PRIVATE ${TORCH_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
        target_link_libraries(
            node_calculator_test
            PRIVATE
            gtest gtest_main
            ${TORCH_LIBRARIES_CPU}
            node_calculator
            # mtpr
            mtpParamOp
            mtpr_bind_cpu
            linearMtpOp_cpu
            # nep
            nep_bind_cpu
            nepOp_cpu
            # nnmtp
            mtpParamOp_nnMtp
            nnmtp_bind_cpu
            nnMtpOp_cpu
            Python::Python)
    endif()
endif()
