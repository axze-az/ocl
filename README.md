# ocl

A c++-20 template based variable lenght vector library with an OpenCL
backend.

## Description

ocl is a c++-20 template based variable length vector library with an OpenCL 
backend. It defines a template class ocl::dvec<_T> with common vector 
operations like addition, subtraction, multiplication using expression 
templates.
Most of these vector operations return expressions instead real result. 
These expressions are executed during the assignment to a vector to avoid 
temporary results requiring large amounts of memory if the vector lengths
are large.

## Getting Started

### Dependencies

- cmake ist the used build system
- only linux as host and target system was tested and
- gcc or clang are the only compilers used to date

### Configuration

create a build directory in the root directory of the project,
change into the directory and then configure:

1. `mkdir build`
2. `cd build`
3. `CC=clang-18 CXX=clang++-18 cmake -DCMAKE_BUILD_TYPE=release ..`

You may also use gcc instead of clang:

3. `CC=gcc-14 CXX=g++-14 cmake -DCMAKE_BUILD_TYPE=release ..`

The compilation to a specific ABI is requested by configuring the library with

`cmake -DCMAKE_BUILD_TYPE=release -DOCL_GCC_ARCH=x86-64-v3 .. -DCFTAL_GCC_ARCH=x86-64-v3 ..`

where `x64-64-v3` is one of the possible arguments to `gcc -march=`.
The variables `OCL_GCC_ARCH` and `CFTAL_GCC_ARCH` default to `native`.


### Build and test

During the build a number of test programs are built in ./test.

## License

This project is licensed under the LGPL v2.1License.
