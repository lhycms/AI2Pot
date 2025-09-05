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

if (USE_CUDA AND CUDA_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()


# 5. compile
add_library(mtpParam_nnMtp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParam.cc)
target_include_directories(mtpParam_nnMtp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (TORCH_OP)
    add_library(mtpParamOp_nnMtp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParamOp.cc)
    target_include_directories(mtpParamOp_nnMtp PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(mtpParamOp_nnMtp mtpParam_nnMtp ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})


    # ==========
    #add_library(nnMtpOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/nnMtpOp.cc)
    #target_include_directories(nnMtpOp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    #target_link_libraries(nnMtpOp PRIVATE ${TORCH_LIBRARIES})
    #if (USE_OPENMP)
    #    target_compile_definitions(nnMtpOp PRIVATE USE_OPENMP)
    #endif()
    #if (USE_CUDA AND CUDA_FOUND)
    #    target_compile_definitions(nnMtpOp PRIVATE USE_CUDA)
    #    target_link_libraries(nnMtpOp PRIVATE nnmtp_torch_launcher 
    #                                          nnmtp_loss_torch_launcher
    #                                          zbl_torch_launcher)
    #endif()
    #if (MAX_MTP_LEVEL)
    #    target_compile_definitions(nnMtpOp PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    #endif()
    # ==========

    # ==========
    #add_library(nnmtp_bind SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/nnmtp_bind.cc)
    #target_include_directories(nnmtp_bind PRIVATE ${PYTHON_INCLUDE_DIRS})
    #target_link_libraries(nnmtp_bind PRIVATE mtpParamOp_nnMtp nnMtpOp ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
    #if (USE_OPENMP)
    #    target_compile_definitions(nnmtp_bind PRIVATE USE_OPENMP)
    #endif()
    #if (USE_CUDA AND CUDA_FOUND)
    #    target_compile_definitions(nnmtp_bind PRIVATE USE_CUDA)
    #endif()
    #if (MAX_MTP_LEVEL)
    #    target_compile_definitions(nnmtp_bind PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    #endif()
    # ==========
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

    gtest_discover_tests(test_nnMtpUtils)
    gtest_discover_tests(test_nnMtp)
endif()