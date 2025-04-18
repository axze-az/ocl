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

std::vector<ocl::cl::platform>
ocl::cl::system::
platforms()
{
    cl_uint count = 0;
    cl_int err=clGetPlatformIDs(0, 0, &count);
    error::throw_on(err, __FILE__, __LINE__);
    std::vector<platform> platforms;
    if(count > 0) {
        cl_platform_id* platform_ids=
            static_cast<cl_platform_id*>(alloca(count*sizeof(cl_platform_id)));
        err=clGetPlatformIDs(count, &platform_ids[0], 0);
        error::throw_on(err, __FILE__, __LINE__);
        for(cl_uint i=0; i<count; ++i){
            platforms.push_back(platform(platform_ids[i]));
        }
    }
    return platforms;
}

std::vector<ocl::cl::device>
ocl::cl::system::
devices()
{
    std::vector<device> vd;
    auto vp = platforms();
    for (const auto& p : vp) {
        const auto vpd = p.devices();
        vd.insert(vd.end(), vpd.begin(), vpd.end());
    }
    return vd;
}

