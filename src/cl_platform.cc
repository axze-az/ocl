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

std::string
ocl::cl::platform::
info(cl_platform_info i)
    const
{
    size_t s=0;
    info(i, 0, nullptr, &s);
    char* v=static_cast<char*>(alloca(s));
    info(i, s, &v[0], nullptr);
    std::string r(&v[0], s-1);
    return r;
}

void
ocl::cl::platform::
info(cl_platform_info i, size_t s, void* p, size_t* rps)
    const
{
    cl_int cr=clGetPlatformInfo(_id, i, s, p, rps);
    error::throw_on(cr, __FILE__, __LINE__);
}

ocl::size_t
ocl::cl::platform::
device_count(cl_device_type type)
    const
{
    cl_uint count = 0;
    cl_int ret = clGetDeviceIDs(_id, type, 0, 0, &count);
    if (ret == CL_DEVICE_NOT_FOUND)
        return 0;
    error::throw_on(ret);
    return count;
}

std::vector<ocl::cl::device>
ocl::cl::platform::
devices(cl_device_type type)
    const
{
    std::vector<device> vd;
    size_t count=device_count(type);
    if(count != 0) {
        cl_device_id* vids=
            static_cast<cl_device_id*>(alloca(count*sizeof(cl_device_id)));
        cl_int err = clGetDeviceIDs(_id, type, count, &vids[0], 0);
        error::throw_on(err);
        for (size_t i=0; i<count; ++i) {
            vd.emplace_back(vids[i]);
        }
    }
    return vd;
}
