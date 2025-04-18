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

ocl::cl::buffer::
buffer(const context& context, size_t size, cl_mem_flags flags,
       void *host_ptr)
{
    cl_int err = 0;
    cl_mem& _id=(*this)();
    _id = clCreateBuffer(context(),
                         flags,
                         (std::max)(size, size_t(1)),
                         host_ptr,
                         &err);
    if(!_id){
        error::throw_on(err, __FILE__, __LINE__);
    }
}

ocl::size_t
ocl::cl::buffer::
size()
    const
{
    size_t s;
    info(CL_MEM_SIZE, sizeof(s), &s, nullptr);
    return s;
}

