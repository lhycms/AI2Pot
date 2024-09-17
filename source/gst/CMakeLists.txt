# 3. Set dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/gst)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/gst)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj/gst)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# 4. Add subdirectory

# 5. Compile program
add_library(gst SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bind/gst_bind.cc)
set_target_properties(gst PROPERTIES
    PREFIX ""
    SUFFIX ${PYTHON_MODULE_EXTENSION})
target_include_directories(gst PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(gst PRIVATE ${NUMPY_INCLUDE_PATH})
target_link_libraries(gst PRIVATE ${PYTHON_LIBRARIES})

# 6. Compile test
add_executable(test_altbc ${CMAKE_CURRENT_SOURCE_DIR}/test/test_altbc.cc)
target_link_libraries(test_altbc PRIVATE gtest gtest_main)
