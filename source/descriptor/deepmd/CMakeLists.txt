# 3. 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/deepmd)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/deepmd)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/deepmd)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. add_library
if (TORCH_OP)
    add_library(se4pw_op SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/se4pw_op.cc)
    target_include_directories(se4pw_op PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_link_libraries(se4pw_op PRIVATE ${TORCH_LIBRARIES})

    add_library(envMatrixOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/envMatrixOp.cc)
    target_include_directories(envMatrixOp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(envMatrixOp PRIVATE ${TORCH_LIBRARIES})

    add_library(envMatrixOp_bind SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/envMatrixOp_bind.cc)
    target_include_directories(envMatrixOp_bind PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(envMatrixOp_bind PRIVATE ${TORCH_LIBRARIES} envMatrixOp ${PYTHON_LIBRARIES})
endif()


# 6. test
if (BUILD_TEST)
    add_executable(test_se ${CMAKE_CURRENT_SOURCE_DIR}/test/test_se.cc)
    target_link_libraries(test_se gtest gtest_main)

    add_executable(test_se_lmp ${CMAKE_CURRENT_SOURCE_DIR}/test/test_se_lmp.cc)
    target_link_libraries(test_se_lmp gtest gtest_main)

    add_executable(test_se4pw ${CMAKE_CURRENT_SOURCE_DIR}/test/test_se4pw.cc)
    target_link_libraries(test_se4pw gtest gtest_main)

    add_executable(demo ${CMAKE_CURRENT_SOURCE_DIR}/test/demo.cc)

    add_executable(test_envMatrix ${CMAKE_CURRENT_SOURCE_DIR}/test/test_envMatrix.cc)
    target_link_libraries(test_envMatrix PRIVATE gtest gtest_main)

    gtest_discover_tests(test_se)
    gtest_discover_tests(test_se_lmp)
    gtest_discover_tests(test_se4pw)
    gtest_discover_tests(test_envMatrix)
    
    if (TORCH_OP)
        # 6.1. test
        add_executable(test_se4pw_op ${CMAKE_CURRENT_SOURCE_DIR}/test/test_se4pw_op.cc)
        target_link_directories(test_se4pw_op PRIVATE ${CMAKE_BINARY_DIR}/libdescriptor/deepmd/)
        target_link_libraries(test_se4pw_op PRIVATE gtest gtest_main ${TORCH_LIBRARIES} se4pw_op)
    
        add_executable(test_demo_se4pw ${CMAKE_CURRENT_SOURCE_DIR}/test/demo_se4pw.cc)
        target_link_directories(test_demo_se4pw PRIVATE ${CMAKE_BINARY_DIR}/lib/descriptor/deepmd/)
        target_link_libraries(test_demo_se4pw PRIVATE gtest gtest_main ${TORCH_LIBRARIES} se4pw_op)

        add_executable(test_envMatrixOp ${CMAKE_CURRENT_SOURCE_DIR}/test/test_envMatrixOp.cc)
        target_link_directories(test_envMatrixOp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(test_envMatrixOp PRIVATE gtest gtest_main ${TORCH_LIBRARIES} envMatrixOp)

        gtest_discover_tests(test_se4pw_op DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
        gtest_discover_tests(test_demo_se4pw)
        gtest_discover_tests(test_envMatrixOp DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
    endif()
endif()
