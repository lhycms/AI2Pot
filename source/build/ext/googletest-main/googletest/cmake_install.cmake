# Install script for directory: /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/ext/googletest-main/googletest

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/rh/devtoolset-7/root/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "gtest" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/GTest/GTestTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/GTest/GTestTargets.cmake"
         "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/ext/googletest-main/googletest/CMakeFiles/Export/5d445a69522011472a13f6464cd74fcd/GTestTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/GTest/GTestTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/GTest/GTestTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/GTest" TYPE FILE FILES "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/ext/googletest-main/googletest/CMakeFiles/Export/5d445a69522011472a13f6464cd74fcd/GTestTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/GTest" TYPE FILE FILES "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/ext/googletest-main/googletest/CMakeFiles/Export/5d445a69522011472a13f6464cd74fcd/GTestTargets-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "gtest" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/GTest" TYPE FILE FILES
    "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/ext/googletest-main/googletest/generated/GTestConfigVersion.cmake"
    "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/ext/googletest-main/googletest/generated/GTestConfig.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "gtest" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/ext/googletest-main/googletest/include/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "gtest" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64" TYPE STATIC_LIBRARY FILES "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/lib/libgtest.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "gtest" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64" TYPE STATIC_LIBRARY FILES "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/lib/libgtest_main.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "gtest" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/pkgconfig" TYPE FILE FILES "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/ext/googletest-main/googletest/generated/gtest.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "gtest" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/pkgconfig" TYPE FILE FILES "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/ext/googletest-main/googletest/generated/gtest_main.pc")
endif()

