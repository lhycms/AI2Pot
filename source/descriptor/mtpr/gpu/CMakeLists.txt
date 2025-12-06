# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtpr/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


# 5. compile
add_library(linear_mtp_torch_launcher SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/linearMtp_torch_launcher.cu)
set_target_properties(linear_mtp_torch_launcher PROPERTIES
    CUDA_SEPERATE_COMPILATION on
    CUDA_STANDARD ${CUDA_STANDARD}
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
if (USE_CUDA AND CUDA_FOUND)
    target_compile_definitions(linear_mtp_torch_launcher PRIVATE USE_CUDA MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
endif()

add_library(schmidt_orth_torch_launcher SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/schmidt_orth_torch_launcher.cu)
set_target_properties(schmidt_orth_torch_launcher PROPERTIES
    CUDA_SEPERATE_COMPILATION on
    CUDA_STANDARD ${CUDA_STANDARD}
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
if (USE_CUDA AND CUDA_FOUND)
    target_compile_definitions(linear_mtp_torch_launcher PRIVATE USE_CUDA)
endif()

add_library(linear_mtp_loss_torch_launcher SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/linearMtpLoss_torch_launcher.cu)
set_target_properties(linear_mtp_loss_torch_launcher PROPERTIES
    CUDA_SEPERATE_COMPILATION on
    CUDA_STANDARD ${CUDA_STANDARD}
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
if (USE_CUDA AND CUDA_FOUND)
    target_compile_definitions(linear_mtp_loss_torch_launcher PRIVATE USE_CUDA MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
endif()


# 6. compile test
if (BUILD_TEST)
    add_executable(test_schmidt_orth_gpu ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_schmidt_orth.cu)
    target_link_libraries(test_schmidt_orth_gpu PRIVATE gtest gtest_main)

    add_executable(test_mtp_utilities ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtp_utilities.cu)
    target_compile_definitions(test_mtp_utilities PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_mtp_utilities PRIVATE gtest gtest_main)
    
    add_executable(test_mtp_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasis.cu)
    target_compile_definitions(test_mtp_basis PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_mtp_basis PRIVATE mtpParam gtest gtest_main)

    add_executable(test_linear_mtp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtp.cu)
    target_compile_definitions(test_linear_mtp PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_linear_mtp PRIVATE mtpParam gtest gtest_main)

    add_executable(test_linear_mtp_loss ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtpLoss.cu)
    target_compile_definitions(test_linear_mtp_loss PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    target_link_libraries(test_linear_mtp_loss PRIVATE gtest gtest_main)
    
    gtest_discover_tests(test_schmidt_orth_gpu DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_mtp_utilities DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    #gtest_discover_tests(test_mtp_basis DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_linear_mtp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    gtest_discover_tests(test_linear_mtp_loss DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
endif()
