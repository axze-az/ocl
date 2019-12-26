#include "ocl/expr_kernel.h"

void
ocl::impl::insert_headers(std::ostream& s)
{
    // fp64 extension
    s << "#if defined (cl_khr_fp64)\n"
         "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
         "#elif defined (cl_amd_fp64)\n"
         "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
         "#endif\n";
    // fp16
    s << "#if defined (cl_khr_fp16)\n"
         "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "#endif\n\n";
}

void
ocl::impl::missing_backend_data()
{
    throw std::runtime_error("missing backend data");
}
