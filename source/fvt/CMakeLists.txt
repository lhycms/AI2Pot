# 3. 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/fvt/)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/fvt)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/fvt)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# 5. 
if (TORCH_OP)
    add_library(force_sr_op SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/force_sr_op.cc)
    target_link_libraries(force_sr_op PRIVATE ${TORCH_LIBRARIES})

    add_library(virial_sr_op SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/virial_sr_op.cc)
    target_link_libraries(virial_sr_op PRIVATE ${TORCH_LIBRARIES})

    add_library(fvt_sr_op_bind SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/fvt_sr_op_bind.cc)
    target_include_directories(fvt_sr_op_bind PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(fvt_sr_op_bind PRIVATE force_sr_op virial_sr_op ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
endif()

# 6. 
add_executable(test_force_sr ${CMAKE_CURRENT_SOURCE_DIR}/test/test_force_sr.cc)
target_link_libraries(test_force_sr PRIVATE gtest gtest_main)
gtest_discover_tests(test_force_sr)

add_executable(test_virial_sr ${CMAKE_CURRENT_SOURCE_DIR}/test/test_virial_sr.cc)
target_link_libraries(test_virial_sr PRIVATE gtest gtest_main)
gtest_discover_tests(test_virial_sr)

if (TORCH_OP)
    add_executable(test_force_sr_op ${CMAKE_CURRENT_SOURCE_DIR}/test/test_force_sr_op.cc)
    target_link_libraries(test_force_sr_op PRIVATE gtest gtest_main force_sr_op ${TORCH_LIBRARIES})
    gtest_discover_tests(test_force_sr_op DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})

    add_executable(test_virial_sr_op ${CMAKE_CURRENT_SOURCE_DIR}/test/test_virial_sr_op.cc)
    target_link_libraries(test_virial_sr_op PRIVATE gtest gtest_main virial_sr_op ${TORCH_LIBRARIES})
    gtest_discover_tests(test_virial_sr_op DISCOVERY_TIMEOUT ${CMAKE_GTEST_DISCOVERY_TIMEOUT})
endif()
