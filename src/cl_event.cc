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

ocl::cl::event::
event(const event &r)
    : _id(r._id)
{
    if (_id){
        auto cr=clRetainEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::event::
event(event &&r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::event&
ocl::cl::event::
operator=(const event &r)
{
    if(this != &r){
        if (_id){
            auto cr=clReleaseEvent(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
        _id = r._id;
        if (_id){
            auto cr=clRetainEvent(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
    }
    return *this;
}

ocl::cl::event&
ocl::cl::event::
operator=(event&& r)
{
    if(_id){
        auto cr=clReleaseEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
    _id = r._id;
    r._id = 0;
    return *this;
}

ocl::cl::event::
~event()
{
    if (_id){
        auto cr=clReleaseEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::event::
event(cl_event c, bool retain)
    : _id(c)
{
    if (_id && retain) {
        auto cr=clRetainEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

void
ocl::cl::event::
wait()
{
    cl_int ret = clWaitForEvents(1, &_id);
    error::throw_on(ret, __FILE__, __LINE__);
}

