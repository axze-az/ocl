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

cl_event
ocl::cl::user_event::create(const context& ctx)
{
    cl_int err;
    cl_event ev=clCreateUserEvent(ctx(), &err);
    error::throw_on(err, __FILE__, __LINE__);
    return ev;
}

ocl::cl::user_event::user_event(const context& ctx)
    : event(create(ctx))
{
}

void
ocl::cl::user_event::set_status(cl_int exec_status)
{
    cl_int err=clSetUserEventStatus((*this)(), exec_status);
    error::throw_on(err, __FILE__, __LINE__);
}
