cmake_minimum_required(VERSION 3.9)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()


# configurations
if(CMAKE_CONFIGURATION_TYPES)
   set(CMAKE_CONFIGURATION_TYPES debug release profile)
   #set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
   #  "Reset the configurations to what we need"
   # FORCE)
endif()
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE release CACHE STRING
      "Choose the type of build, options are: debug release profile."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

if (UNIX)
  #message("cross compile ${CMAKE_CROSSCOMPILING}")
  #message("tool chain ${CMAKE_TOOLCHAIN_FILE}")
  if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if (NOT DEFINED ENV{CC})
      find_program(CMAKE_C_COMPILER NAMES gcc)
    endif()
    if (NOT DEFINED ENV{CXX})
      find_program(CMAKE_CXX_COMPILER NAMES g++)
    endif()
  endif()
endif(UNIX)

project (OCL)
set (OCL_VERSION_MAJ 0)
set (OCL_VERSION_MIN 1)
set (OCL_VERSION_PLV 0)
set (OCL_VERSION_STR
    "${OCL_VERSION_MAJ}.${OCL_VERSION_MIN}.${OCL_VERSION_PLV}")

# compile flags for usage with cmake -DCMAKE_BUILD_TYPE=debug/release/profile
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "CLANG")
    # warnings for all
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-attributes")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections")
    # release
    set (CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")
    set (CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -fexpensive-optimizations")
    # strip release builds
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE
        "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -s")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE
        "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -s")
    # profile
    set (CMAKE_CXX_FLAGS_PROFILE
        "${CMAKE_CXX_FLAGS_PROFILE} -pg")
    # debug flags are set by cmake already
endif()

#message(${CMAKE_CXX_COMPILER_ID})
#message(${CMAKE_LIBRARY_ARCHITECTURE})
set(OCL_GCC_ARCH "native" CACHE STRING "argument for gcc -march=")
if ("${CMAKE_LIBRARY_ARCHITECTURE}" MATCHES ^x86_64* )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} -fsched-pressure")
        set (CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} -fschedule-insns")
        set (CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} -fschedule-insns2")
        set (CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #    "${CMAKE_CXX_FLAGS_RELEASE} -fselective-scheduling")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #    "${CMAKE_CXX_FLAGS_RELEASE} -fselective-scheduling2")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #    "${CMAKE_CXX_FLAGS_RELEASE} -fsel-sched-pipelining")
    endif()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
        CMAKE_CXX_COMPILER_ID STREQUAL "CLANG")
        set (CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=${OCL_GCC_ARCH}")
        set (CMAKE_CXX_FLAGS_DEBUG
            "${CMAKE_CXX_FLAGS_DEBUG} -Og -march=${OCL_GCC_ARCH}")
        set (CMAKE_CXX_FLAGS_PROFILE
            "${CMAKE_CXX_FLAGS_PROFILE} -O3 -march=${OCL_GCC_ARCH}")
    endif()
endif()

# C++20 ?
option(OCL_USE_CXX2A "use C++2A" ON)
# C++17 ?
option(OCL_USE_CXX17 "use C++17" ON)
if (OCL_USE_CXX2A)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++2a")
else()
    if(OCL_USE_CXX17)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
    endif()
endif()

option(OCL_USE_LTO "use link time optimizations" OFF)
if (OCL_USE_LTO)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -flto")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE
      "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto -fwhole-program")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE
      "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -flto")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE
      "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} -flto")
  endif()
endif()

# create static ocl library
option(OCL_STATIC_LIB "create static libocl.a" OFF)
# Full test ?
option(OCL_FULL_TEST "perform a full test with long compile times" OFF)

include_directories(include)
include_directories(test/include)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/ocl)
include_directories(${PROJECT_BINARY_DIR}/include)

configure_file("include/ocl/config.h.in"
               "${PROJECT_BINARY_DIR}/include/ocl/config.h")

include_directories(../cftal/gcc11/include)
include_directories(../cftal/include)
include_directories(../cftal/test/include)
include_directories(../dev-cftal/include)
include_directories(../dev-cftal/test/include)
#include_directories(${PROJECT_BINARY_DIR}/cftal/include)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)
if(${Boost_FOUND})
    #message ("${Boost_FOUND}")
    #message ("${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
endif()
find_library(LIBOPENCL OpenCL)
message("open cl library: ${LIBOPENCL}")
find_library(LIBM m)
find_library(LIB_BOOST_SYSTEM boost_system)

#add_subdirectory(../cftal cftal)
add_subdirectory(src)
add_subdirectory(test)


enable_testing()

set (CPACK_PACKAGE_NAME "libocl")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "ocl library")
set (CPACK_PACKAGE_VENDOR "az")
set (CPACK_DEBIAN_PACKAGE_MAINTAINER "az")
#set (CPACK_GENERATOR "DEB;RPM;TBZ2")
set (CPACK_GENERATOR "DEB")
set (CPACK_PACKAGE_VERSION_MAJOR "${OCL_VERSION_MAJ}")
set (CPACK_PACKAGE_VERSION_MINOR "${OCL_VERSION_MIN}")
set (CPACK_PACKAGE_VERSION_PATCH "${OCL_VERSION_PLV}")
set (CPACK_PACKAGE_VERSION
  "${OCL_VERSION_MAJ}.${OCL_VERSION_MIN}:${OCL_VERSION_PLV}")
set (CPACK_DEBIAN_PACKAGE_PRIORITY "extra")

#set (CPACK_COMPONENTS_GROUPING IGNORE)
set(CPACK_COMPONENTS_ALL ocl devel tools)
set (CPACK_COMPONENT_OCL_DISPLAY_NAME "ocl-${OCL_VERSION_STR}")
set (CPACK_OCL_COMPONENT_INSTALL ON)
set (CPACK_COMPONENT_DISPLAY_DEVEL_NAME "ocl-dev")
set (CPACK_DEVEL_COMPONENT_INSTALL ON)
set (CPACK_COMPONENT_TOOLS_DISPLAY_NAME "ocl-tools")
set (CPACK_TOOLS_COMPONENT_INSTALL ON)

set(CPACK_COMPONENTS_IGNORE_GROUPS 1)

# enable component packages
set (CPACK_DEB_COMPONENT_INSTALL on)
#set (CPACK_RPM_COMPONENT_INSTALL on)

#message(${CPACK_COMPONENTS_ALL})

include (CPack)


