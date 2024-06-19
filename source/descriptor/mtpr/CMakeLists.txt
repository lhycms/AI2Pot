# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/mtpr)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. compile
add_library(mtpParam SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParam.cc)

if (TORCH_OP)
    execute_process(COMMAND python3 -c "import torch;print(torch.__path__[0])" OUTPUT_VARIABLE TORCH_PATH)
    string(REPLACE "\n" "/share/cmake" TORCH_CMAKE_PREFIX_PATH ${TORCH_PATH})
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
    find_package(Torch CONFIG REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    add_library(mtpParamOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpParamOp.cc)
    target_link_libraries(mtpParamOp PRIVATE mtpParam ${TORCH_LIBRARIES})

    add_library(mtpBasisOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/mtpBasisOp.cc)
    target_link_libraries(mtpBasisOp PRIVATE ${TORCH_LIBRARIES})

    add_library(mtpr_bind SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/mtpr_bind.cc)
    target_include_directories(mtpr_bind PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(mtpr_bind PRIVATE mtpParamOp mtpBasisOp ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
endif()


# 6. compile test
if (BUILD_TEST)
    add_executable(test_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basis.cc)
    target_link_libraries(test_basis PRIVATE gtest gtest_main)

    add_executable(test_mtpParam ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpParam.cc)
    target_link_libraries(test_mtpParam PRIVATE mtpParam gtest gtest_main)

    add_executable(test_mtpBasis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasis.cc)
    target_link_libraries(test_mtpBasis PRIVATE mtpParam gtest gtest_main)

    if (TORCH_OP)
        add_executable(test_mtpParamOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpParamOp.cc)
        target_link_libraries(test_mtpParamOp PRIVATE ${TORCH_LIBRARIES} mtpParamOp mtpParam gtest gtest_main)
        gtest_discover_tests(test_mtpParamOp)

        add_executable(test_mtpBasisOp ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mtpBasisOp.cc)
        target_link_libraries(test_mtpBasisOp PRIVATE ${TORCH_LIBRARIES} mtpBasisOp mtpParamOp gtest gtest_main)
        set_target_properties(test_mtpBasis PROPERTIES COMPILE_FLAGS "-g")
        gtest_discover_tests(test_mtpBasisOp)
    endif()

    gtest_discover_tests(test_basis)
    gtest_discover_tests(test_mtpParam)
    gtest_discover_tests(test_mtpBasis)
endif()