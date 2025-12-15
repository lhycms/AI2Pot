# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/descriptor/nep)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 4. 


# 5. compile
if (TORCH_OP)
    add_library(nepOp SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/nepOp.cc)
    target_include_directories(nepOp PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(nepOp PRIVATE ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})
    if (USE_OPENMP)
        target_compile_definitions(nepOp PRIVATE USE_OPENMP)
    endif()
    if (USE_CUDA AND CUDA_FOUND)
        target_compile_definitions(nepOp PRIVATE USE_CUDA)
        # target_link ...
    endif()

    add_library(nep_bind SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/nep_bind.cc)
    target_include_directories(nep_bind PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(nep_bind PRIVATE nepOp ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
    if (USE_OPENMP)
        target_compile_definitions(nep_bind PRIVATE USE_OPENMP)
    endif()
    if (USE_CUDA AND CUDA_FOUND)
        target_compile_definitions(nep_bind PRIVATE USE_CUDA)
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

    add_executable(test_nep_basis ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basis.cc)
    target_link_libraries(test_nep_basis PRIVATE gtest gtest_main)
    gtest_discover_tests(test_nep_basis)

    add_executable(test_nep ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep.cc)
    target_link_libraries(test_nep PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(nepOp PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep)

    add_executable(test_nep_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nep_cpu_launcher.cc)
    target_link_libraries(test_nep_cpu_launcher PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(nepOp PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep_cpu_launcher)

    add_executable(test_nep_loss ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nepLoss.cc)
    target_link_libraries(test_nep_loss PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(nepOp PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep_loss)

    add_executable(test_nep_loss_cpu_launcher ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nepLoss_cpu_launcher.cc)
    target_link_libraries(test_nep_loss_cpu_launcher PRIVATE gtest gtest_main)
    if (USE_OPENMP)
        target_compile_definitions(nepOp PRIVATE USE_OPENMP)
    endif()
    gtest_discover_tests(test_nep_loss_cpu_launcher)
endif()
