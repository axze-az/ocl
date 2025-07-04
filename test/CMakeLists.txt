cmake_minimum_required(VERSION 3.10)
project(ocl-tests)

set (TEST_SRCS
    program.cc
    ops_base.cc
)

include_directories(../..)
include_directories(../include)
include_directories(./include)

set (TEST_HDRS
)

function(target_add_test_libs arg1)
    target_link_libraries(${arg1} ${LIB_BOOST_SYSTEM})
    target_link_libraries(${arg1} ocl-so)
    target_link_libraries(${arg1} Threads::Threads)
endfunction()

add_library(ocl-test SHARED ${TEST_SRCS})
set_property(TARGET ocl-test
  PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
set_target_properties(ocl-test PROPERTIES
  SOVERSION "${OCL_VERSION_MAJ}"
  VERSION "${OCL_VERSION_MAJ}.${OCL_VERSION_MIN}")
target_link_libraries(ocl-test ocl-so)
target_link_libraries(ocl-test ${LIB_BOOST_SYSTEM})
target_link_libraries(ocl-test stdc++fs)

target_add_test_libs(ocl-test)
target_link_libraries(ocl-test stdc++fs)

function(test_prog arg1)
    add_executable(${arg1} ${arg1}.cc ${TEST_HDRS})
    target_add_test_libs(${arg1})
    target_link_libraries(${arg1} ocl-test)
    add_test(NAME ${arg1} COMMAND ${arg1})
endfunction()

# test_be_data
test_prog(test_be_data)
test_prog(test_sizes)
test_prog(test_dvec)
test_prog(test_rnd)
test_prog(test_math)
test_prog(test_horner)
test_prog(test_reduce)
test_prog(test_subnormal)
test_prog(test_mem_alloc)
test_prog(test_flops)
test_prog(dev_custom_kernel)
test_prog(dev_custom_func)
test_prog(dev_select)
test_prog(dev_xxx_of)
test_prog(dev_lookup)
test_prog(dev_misc)
test_prog(dev_bf16)

add_executable(stringify stringify.cc)
