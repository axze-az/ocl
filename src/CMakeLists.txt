cmake_minimum_required(VERSION 3.9)
project(ocl-src)

set(has_ipo, FALSE)
#include(CheckIPOSupport)
#check_ipo_support(RESULT has_ipo)

set (CSRCS
  cl_error.cc
  cl_mem_object.cc
  cl_buffer.cc
  cl_queue.cc
  cl_context.cc
  cl_device.cc
  cl_event.cc
  cl_wait_list.cc
  cl_kernel.cc
  cl_platform.cc
  cl_program.cc
  cl_system.cc
#  be_user_event.cc
  be_devices.cc
  be_kernel_key.cc
  be_kernel_cache.cc
  be_kernel_functions.cc
  be_dev_ctx_queue.cc
  be_type_2_name.cc
  be_data.cc
  dvec_base.cc
  dvec_t.cc
  expr.cc
  expr_custom.cc
  dvec_op.cc
  dvec_func.cc
  expr_kernel.cc
  random.cc
)

# only for ides
SET (HDRS
    ../include/ocl/config.h.in
)

if (OCL_STATIC_LIB)
    add_library (ocl STATIC ${CSRCS} ${HDRS})
    install(TARGETS ocl
        DESTINATION lib LIBRARY
        COMPONENT devel)
endif()

#set_target_properties(cftal PROPERTIES OUTPUT_NAME "ocl")
add_library (ocl-so SHARED ${CSRCS} ${HDRS})
set_property(TARGET ocl-so
  PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ocl-so
  PROPERTY OUTPUT_NAME ocl)
if (has_ipo)
    set_property(TARGET ocl-so
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
set_target_properties(ocl-so PROPERTIES SOVERSION "${OCL_VERSION_MAJ}"
  VERSION "${OCL_VERSION_MAJ}.${OCL_VERSION_MIN}")

if (LIBOPENCL)
  target_link_libraries(ocl-so ${LIBOPENCL})
endif()
find_package(Threads REQUIRED)
target_link_libraries(ocl-so Threads::Threads)
find_library(LIBM m)
if (LIBM)
  target_link_libraries(ocl-so ${LIBM})
endif(LIBM)

if ("${CMAKE_LIBRARY_ARCHITECTURE}" MATCHES ^x86_64* )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #     "${CMAKE_CXX_FLAGS_RELEASE} -Winline")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #     "${CMAKE_CXX_FLAGS_RELEASE} --param inline-unit-growth=4000")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #     "${CMAKE_CXX_FLAGS_RELEASE} --param large-function-growth=4000")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #     "${CMAKE_CXX_FLAGS_RELEASE} --param max-inline-insns-single=8192")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #     "${CMAKE_CXX_FLAGS_RELEASE} --param large-stack-frame=16384")
        #set (CMAKE_CXX_FLAGS_RELEASE
        #     "${CMAKE_CXX_FLAGS_RELEASE} --param large-stack-frame-growth=4000")
    endif()
endif()

install(TARGETS ocl-so
  DESTINATION lib LIBRARY
  COMPONENT ocl
  NAMELINK_SKIP)
install(TARGETS ocl-so
  DESTINATION lib LIBRARY
  COMPONENT devel
  NAMELINK_ONLY)
