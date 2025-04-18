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

ocl::cl::program::
program(const program &r)
    : _id(r._id)
{
    if (_id){
        auto cr=clRetainProgram(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::program::
program(program &&r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::program&
ocl::cl::program::
operator=(const program &r)
{
    if(this != &r){
        if (_id){
            auto cr=clReleaseProgram(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
        _id = r._id;
        if (_id){
            auto cr=clRetainProgram(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
    }
    return *this;
}

ocl::cl::program&
ocl::cl::program::
operator=(program&& r)
{
    if(_id){
        auto cr=clReleaseProgram(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
    _id = r._id;
    r._id = 0;
    return *this;
}

ocl::cl::program::
~program()
{
    if (_id){
        auto cr=clReleaseProgram(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::program::
program(cl_program k, bool retain)
    : _id(k)
{
    if (_id && retain) {
        auto cr=clRetainProgram(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

std::vector<ocl::cl::device>
ocl::cl::program::
get_devices()
    const
{
    size_t nn=0;
    info(CL_PROGRAM_DEVICES, 0, nullptr, &nn);
    cl_device_id* d=static_cast<cl_device_id*>(alloca(nn));
    size_t cnt=nn/sizeof(cl_device_id);
    info(CL_PROGRAM_DEVICES, nn, &d[0], nullptr);
    std::vector<device> vd;
    for (size_t i=0; i<cnt; ++i) {
        vd.emplace_back(d[i]);
    }
    return vd;
}

void
ocl::cl::program::
info(cl_program_info i, size_t ps, void* p, size_t* rps)
    const
{
    cl_int cr=clGetProgramInfo(_id, i, ps, p, rps);
    error::throw_on(cr, __FILE__, __LINE__);
}

void
ocl::cl::program::
build_info(const device& d, cl_program_build_info i,
           size_t ps, void* p, size_t* rps)
    const
{
    cl_int cr=clGetProgramBuildInfo(_id, d(), i, ps, p, rps);
    error::throw_on(cr, __FILE__, __LINE__);
}

std::string
ocl::cl::program::
build_log()
    const
{
    device d=get_devices().front();
    size_t cnt=0;
    build_info(d, CL_PROGRAM_BUILD_LOG, 0, nullptr, &cnt);
    char* vc=static_cast<char*>(alloca(cnt));
    build_info(d, CL_PROGRAM_BUILD_LOG, cnt, &vc[0], nullptr);
    std::string r(&vc[0], cnt-1);
    return r;
}

void
ocl::cl::program::
build(const std::string& options)
{
    const char *options_string = 0;
    if(!options.empty()){
        options_string = options.c_str();
    }
    cl_int cr=clBuildProgram(_id, 0, 0, options_string, 0, 0);
    error::throw_on(cr, __FILE__, __LINE__);
}

ocl::cl::program
ocl::cl::program::
create_with_source(const std::string& source,
                   const context& context)
{
    const char *source_string = source.c_str();
    cl_int cr=0;
    cl_program p = clCreateProgramWithSource(context(),
                                             1,
                                             &source_string,
                                             0,
                                             &cr);
    if(!p){
        error::throw_on(cr, __FILE__, __LINE__);
    }
    return program(p, false);
}
