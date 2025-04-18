//
// Copyright (C) 2010-2025 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#include "ocl/be/types.h"

ocl::cl::kernel::
kernel(const kernel &r)
    : _id(r._id)
{
    if (_id){
        auto cr=clRetainKernel(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::kernel::
kernel(kernel &&r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::kernel&
ocl::cl::kernel::
operator=(const kernel &r)
{
    if(this != &r){
        if (_id){
            auto cr=clReleaseKernel(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
        _id = r._id;
        if (_id){
            auto cr=clRetainKernel(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
    }
    return *this;
}

ocl::cl::kernel&
ocl::cl::kernel::
operator=(kernel&& r)
{
    if(_id){
        auto cr=clReleaseKernel(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
    _id = r._id;
    r._id = 0;
    return *this;
}

ocl::cl::kernel::
~kernel()
{
    if (_id){
        auto cr=clReleaseKernel(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::kernel::
kernel(cl_kernel k, bool retain)
    : _id(k)
{
    if (_id && retain) {
        auto cr=clRetainKernel(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::kernel::
kernel(const program& pgm, const std::string& kname)
{
    cl_int cr = 0;
    _id = clCreateKernel(pgm(), kname.c_str(), &cr);
    if (_id == 0) {
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

void
ocl::cl::kernel::
info(cl_kernel_info i, size_t s, void* res, size_t* rs)
    const
{
    cl_int cr=clGetKernelInfo(_id, i, s, res, rs);
    error::throw_on(cr, __FILE__, __LINE__);
}

std::string
ocl::cl::kernel::
name()
    const
{
    size_t cnt=0;
    info(CL_KERNEL_FUNCTION_NAME, 0, nullptr, &cnt);
    // std::vector<char> vc(cnt, 0);
    char* vc=static_cast<char*>(alloca(cnt));
    info(CL_KERNEL_FUNCTION_NAME, cnt, &vc[0], nullptr);
    std::string r(&vc[0], cnt-1);
    return r;
}
 
void
ocl::cl::kernel::
work_group_info(const device& d,
                cl_kernel_work_group_info i,
                size_t s, void* res, size_t* rs)
    const
{
    cl_int cr=clGetKernelWorkGroupInfo(_id, d(), i, s, res, rs);
    error::throw_on(cr, __FILE__, __LINE__);
}

void
ocl::cl::kernel::
set_arg(size_t index, size_t size, const void* value)
{
    cl_int cr=clSetKernelArg(_id,
                             static_cast<cl_uint>(index),
                             size,
                             value);
    error::throw_on(cr, __FILE__, __LINE__);
}
