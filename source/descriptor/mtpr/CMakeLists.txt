# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. Add subdirectory
if (NOT DEFINED MAX_MTP_LEVEL)
    message(STATUS "MAX_MTP_LEVEL is set to 18")
    set(MAX_MTP_LEVEL 18)
else()
    message(STATUS "MAX_MTP_LEVEL is set to ${MAX_MTP_LEVEL}")
endif()

if (CUDA_FOUND AND USE_CUDA)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()


# 5. compile
add_library(mtpParam SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParam.cc)

if (TORCH_OP)
    add_library(mtpParamOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParamOp.cc)
    target_include_directories(mtpParamOp PRIVATE ${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
    target_link_libraries(mtpParamOp PRIVATE mtpParam ${TORCH_LIBRARIES_CPU})

    ## 5.1. linear_mtp_gram_and_cross_op
    add_library(linear_mtp_gram_and_cross_op_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/linear_mtp_gram_and_cross_op_cpu.cc)
    target_include_directories(linear_mtp_gram_and_cross_op_cpu PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(linear_mtp_gram_and_cross_op_cpu PRIVATE ${TORCH_LIBRARIES_CPU})
    if (USE_OPENMP)
        target_compile_definitions(linear_mtp_gram_and_cross_op_cpu PRIVATE USE_OPENMP)
    endif()
    if (MAX_MTP_LEVEL)
        target_compile_definitions(linear_mtp_gram_and_cross_op_cpu PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(linear_mtp_gram_and_cross_op_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/linear_mtp_gram_and_cross_op_cuda.cc)
        target_link_libraries(linear_mtp_gram_and_cross_op_cuda
                              PRIVATE
                              ${TORCH_LIBRARIES}
                              linear_mtp_gram_and_cross_torch_launcher)
        target_compile_definitions(linear_mtp_gram_and_cross_op_cuda PRIVATE USE_CUDA)
        if (MAX_MTP_LEVEL)
            target_compile_definitions(linear_mtp_gram_and_cross_op_cuda PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
        endif()
    endif()

    ## 5.2. schmidt_orth_op
    add_library(schmidt_orth_op_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/schmidt_orth_op_cpu.cc)
    target_include_directories(schmidt_orth_op_cpu PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(schmidt_orth_op_cpu PRIVATE ${TORCH_LIBRARIES_CPU})
    if (USE_OPENMP)
        target_compile_definitions(schmidt_orth_op_cpu PRIVATE USE_OPENMP)
    endif()
    if (MAX_MTP_LEVEL)
        target_compile_definitions(schmidt_orth_op_cpu PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(schmidt_orth_op_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/schmidt_orth_op_cuda.cc)
        target_link_libraries(schmidt_orth_op_cuda 
                              PRIVATE
                              ${TORCH_LIBRARIES}
                              schmidt_orth_torch_launcher)
        target_compile_definitions(schmidt_orth_op_cuda PRIVATE USE_CUDA)
        if (MAX_MTP_LEVEL)
            target_compile_definitions(schmidt_orth_op_cuda PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
        endif()
    endif()

    ## 5.3. linearMtpOp
    add_library(linearMtpOp_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/linearMtpOp_cpu.cc)
    target_include_directories(linearMtpOp_cpu PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(linearMtpOp_cpu PRIVATE ${TORCH_LIBRARIES_CPU})
    if (USE_OPENMP)
        target_compile_definitions(linearMtpOp_cpu PRIVATE USE_OPENMP)
    endif()
    if (MAX_MTP_LEVEL)
        target_compile_definitions(linearMtpOp_cpu PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(linearMtpOp_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/linearMtpOp_cuda.cc)
        target_link_libraries(linearMtpOp_cuda
                              PRIVATE
                              ${TORCH_LIBRARIES}
                              linear_mtp_torch_launcher 
                              linear_mtp_loss_torch_launcher
                              zbl_torch_launcher)
        target_compile_definitions(linearMtpOp_cuda PRIVATE USE_CUDA)
        if (MAX_MTP_LEVEL)
            target_compile_definitions(linearMtpOp_cuda PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
        endif()
    endif()

    ## 5.4. mtp_bind
    add_library(mtpr_bind_cpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/mtpr_bind_cpu.cc)
    target_include_directories(mtpr_bind_cpu PRIVATE ${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
    target_link_libraries(mtpr_bind_cpu
                          PRIVATE
                          schmidt_orth_op_cpu 
                          linear_mtp_gram_and_cross_op_cpu
                          mtpParamOp
                          linearMtpOp_cpu
                          ${TORCH_LIBRARIES_CPU}
                          ${PYTHON_LIBRARIES})
    if (USE_OPENMP)
        target_compile_definitions(mtpr_bind_cpu PRIVATE USE_OPENMP)
    endif()
    if (MAX_MTP_LEVEL)
        target_compile_definitions(mtpr_bind_cpu PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    endif()

    if (USE_CUDA AND CUDA_FOUND)
        add_library(mtpr_bind_cuda SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/mtpr_bind_cuda.cc)
        target_include_directories(mtpr_bind_cuda PRIVATE ${PYTHON_INCLUDE_DIRS})
        target_link_libraries(mtpr_bind_cuda
                              PRIVATE
                              schmidt_orth_op_cuda
                              linear_mtp_gram_and_cross_op_cuda
                              linearMtpOp_cuda
                              ${TORCH_LIBRARIES}
                              ${PYTHON_LIBRARIES})
        target_compile_definitions(mtpr_bind_cuda PRIVATE USE_CUDA)
        if (MAX_MTP_LEVEL)
            target_compile_definitions(mtpr_bind_cuda PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
        endif()
    endif()
endif()


# 6. compile test
if (BUILD_TEST)
    add_executable(test_schmidt_orth ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_schmidt_orth.cc)
    target_link_libraries(test_schmidt_orth PRIVATE gtest gtest_main)


    add_executable(test_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basis.cc)
    target_link_libraries(test_basis PRIVATE gtest gtest_main)

    add_executable(test_mtpParam ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpParam.cc)
    target_link_libraries(test_mtpParam PRIVATE mtpParam gtest gtest_main)

    add_executable(test_mtpBasis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasis.cc)
    target_link_libraries(test_mtpBasis PRIVATE mtpParam gtest gtest_main)

    add_executable(test_mtpBasisGrad ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasisGrad.cc)
    target_link_libraries(test_mtpBasisGrad PRIVATE mtpParam gtest gtest_main)

    add_executable(test_linear_mtp_gram_and_cross ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linear_mtp_gram_and_cross.cc)
    target_link_libraries(test_linear_mtp_gram_and_cross PRIVATE mtpParam gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_linear_mtp_gram_and_cross PRIVATE USE_OPENMP)
    endif()

    add_executable(test_linearMtp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtp.cc)
    target_link_libraries(test_linearMtp PRIVATE mtpParam gtest gtest_main)
    #target_compile_options(test_linearMtp PRIVATE -g)
    if (USE_OPENMP)
        target_compile_definitions(test_linearMtp PRIVATE USE_OPENMP)
    endif()

    add_executable(test_linearMtp_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtp_cpu_launcher.cc)
    target_link_libraries(test_linearMtp_cpu_launcher PRIVATE mtpParam gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(test_linearMtp_cpu_launcher PRIVATE USE_OPENMP)
    endif()

    add_executable(test_linearMtpLoss ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtpLoss.cc)
    target_link_libraries(test_linearMtpLoss PRIVATE gtest gtest_main)

    add_executable(test_linearMtpLoss_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtpLoss_cpu_launcher.cc)
    target_link_libraries(test_linearMtpLoss_cpu_launcher PRIVATE gtest gtest_main)

    if (TORCH_OP)
        add_executable(test_mtpParamOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpParamOp.cc)
        target_link_libraries(test_mtpParamOp PRIVATE ${TORCH_LIBRARIES} mtpParamOp mtpParam gtest gtest_main)
        gtest_discover_tests(test_mtpParamOp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})

        #add_executable(test_mtpBasisGradOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasisGradOp.cc)
        #target_link_libraries(test_mtpBasisGradOp PRIVATE ${TORCH_LIBRARIES} mtpBasisGradOp mtpParamOp gtest gtest_main)
        #set_target_properties(test_mtpBasisGradOp PROPERTIES COMPILE_FLAGS "-g")
        #gtest_discover_tests(test_mtpBasisOp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    endif()

    gtest_discover_tests(test_schmidt_orth)
    gtest_discover_tests(test_basis)
    gtest_discover_tests(test_mtpParam)
    gtest_discover_tests(test_linear_mtp_gram_and_cross)
    gtest_discover_tests(test_mtpBasis)
    gtest_discover_tests(test_mtpBasisGrad)
endif()
