# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. Add subdirectory
if (CUDA_FOUND AND USE_CUDA)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()


# 5. compile
add_library(mtpParam SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParam.cc)

if (TORCH_OP)
    add_library(mtpParamOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParamOp.cc)
    target_link_libraries(mtpParamOp PRIVATE mtpParam ${TORCH_LIBRARIES})

    add_library(mtpBasisOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpBasisOp.cc)
    target_link_libraries(mtpBasisOp PRIVATE ${TORCH_LIBRARIES})
    if (MAX_MTP_LEVEL)
        target_compile_definitions(mtpBasisOp PRIVATE MAX_MTP_LEVEL=${MAX_MTP_LEVEL})
    endif()

    add_library(mtpBasisGradOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpBasisGradOp.cc)
    target_link_libraries(mtpBasisGradOp PRIVATE ${TORCH_LIBRARIES})

    add_library(linearMtpOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/linearMtpOp.cc)
    target_link_libraries(linearMtpOp PRIVATE ${TORCH_LIBRARIES})
    if (USE_OPENMP)
        target_compile_definitions(linearMtpOp PRIVATE USE_OPENMP)
    endif()

    add_library(mtpr_bind SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/mtpr_bind.cc)
    target_include_directories(mtpr_bind PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(mtpr_bind PRIVATE mtpParamOp mtpBasisOp linearMtpOp ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
    if (USE_OPENMP)
        target_compile_definitions(linearMtpOp PRIVATE USE_OPENMP)
    endif()
endif()


# 6. compile test
if (BUILD_TEST)
    add_executable(test_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basis.cc)
    target_link_libraries(test_basis PRIVATE gtest gtest_main)

    add_executable(test_mtpParam ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpParam.cc)
    target_link_libraries(test_mtpParam PRIVATE mtpParam gtest gtest_main)

    add_executable(test_mtpBasis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasis.cc)
    target_link_libraries(test_mtpBasis PRIVATE mtpParam gtest gtest_main)

    add_executable(test_mtpBasisGrad ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasisGrad.cc)
    target_link_libraries(test_mtpBasisGrad PRIVATE mtpParam gtest gtest_main)

    add_executable(test_linearMtp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtp.cc)
    target_link_libraries(test_linearMtp PRIVATE mtpParam gtest gtest_main)
    #target_compile_options(test_linearMtp PRIVATE -g)
    if (USE_OPENMP)
        target_compile_definitions(test_linearMtp PRIVATE USE_OPENMP)
    endif()

    add_executable(test_linearMtpLoss ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtpLoss.cc)
    target_link_libraries(test_linearMtpLoss PRIVATE gtest gtest_main)

    if (TORCH_OP)
        add_executable(test_mtpParamOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpParamOp.cc)
        target_link_libraries(test_mtpParamOp PRIVATE ${TORCH_LIBRARIES} mtpParamOp mtpParam gtest gtest_main)
        gtest_discover_tests(test_mtpParamOp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})

        add_executable(test_mtpBasisOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasisOp.cc)
        target_link_libraries(test_mtpBasisOp PRIVATE ${TORCH_LIBRARIES} mtpBasisOp mtpParamOp gtest gtest_main)
        #set_target_properties(test_mtpBasis PROPERTIES COMPILE_FLAGS "-g")
        gtest_discover_tests(test_mtpBasisOp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})

        add_executable(test_linearMtpOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_linearMtpOp.cc)
        target_link_libraries(test_linearMtpOp PRIVATE linearMtpOp ${TORCH_LIBRARIES} mtpParamOp mtpParam gtest gtest_main)
        gtest_discover_tests(test_linearMtpOp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
        if (USE_OPENMP)
            target_compile_definitions(test_linearMtp PRIVATE USE_OPENMP)
        endif()

        #add_executable(test_mtpBasisGradOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasisGradOp.cc)
        #target_link_libraries(test_mtpBasisGradOp PRIVATE ${TORCH_LIBRARIES} mtpBasisGradOp mtpParamOp gtest gtest_main)
        #set_target_properties(test_mtpBasisGradOp PROPERTIES COMPILE_FLAGS "-g")
        #gtest_discover_tests(test_mtpBasisOp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    endif()

    gtest_discover_tests(test_basis)
    gtest_discover_tests(test_mtpParam)
    gtest_discover_tests(test_mtpBasis)
    gtest_discover_tests(test_mtpBasisGrad)
endif()
