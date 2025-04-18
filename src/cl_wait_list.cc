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

ocl::cl::wait_list::
wait_list()
    : _ve()
{
}

ocl::cl::wait_list::
wait_list(const event& event)
    : _ve(1, event)
{
}
ocl::cl::wait_list::
wait_list(const wait_list& r)
    : _ve(r._ve)
{
}

ocl::cl::wait_list::
wait_list(std::initializer_list<event> events)
    : _ve(events)
{
}

ocl::cl::wait_list&
ocl::cl::wait_list::operator=(const wait_list& r)
{
    if (&r != this)
        _ve= r._ve;
    return *this;
}

ocl::cl::wait_list::
wait_list(wait_list&& r)
    : _ve(std::move(r._ve))
{
}

ocl::cl::wait_list&
ocl::cl::wait_list::
operator=(wait_list&& r)
{
    _ve = std::move(r._ve);
    return *this;
}

ocl::cl::wait_list::
~wait_list()
{
}

bool
ocl::cl::wait_list::
empty() const
{
    return _ve.empty();
}

ocl::size_t
ocl::cl::wait_list::
size() const
{
    return _ve.size();
}

void ocl::cl::wait_list::
clear()
{
    return _ve.clear();
}

const cl_event* ocl::cl::wait_list::
get_event_ptr() const
{
    const cl_event* pe=nullptr;
    if (!_ve.empty()) {
        pe = reinterpret_cast<const cl_event*>(&_ve[0]);
    }
    return pe;
}

void
ocl::cl::wait_list::
reserve(size_t new_capacity)
{
    _ve.reserve(new_capacity);
}

void
ocl::cl::wait_list::
insert(const event& event)
{
    _ve.push_back(event);
}

void
ocl::cl::wait_list::
wait() const
{
    cl_int err=clWaitForEvents(size(), get_event_ptr());
    error::throw_on(err, __FILE__, __LINE__);
}

const ocl::cl::event&
ocl::cl::wait_list::operator[](size_t pos)
    const
{
    return _ve[pos];
}

ocl::cl::event&
ocl::cl::wait_list::operator[](size_t pos)
{
    return _ve[pos];
}

ocl::cl::wait_list::iterator
ocl::cl::wait_list::
begin()
{
    return _ve.begin();
}

ocl::cl::wait_list::
const_iterator
ocl::cl::wait_list::
begin() const   
{
    return _ve.begin();
}

ocl::cl::wait_list::const_iterator
ocl::cl::wait_list::
cbegin() const
{
    return _ve.begin();
}

ocl::cl::wait_list::
iterator
ocl::cl::wait_list::
end()
{
    return _ve.end();
}

ocl::cl::wait_list::
const_iterator
ocl::cl::wait_list::
end() const
{
    return _ve.end();
}

ocl::cl::wait_list::
const_iterator ocl::cl::wait_list::
cend() const
{
    return _ve.end();
}
