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
#include "ocl/be/kernel_cache.h"

ocl::be::scoped_lock::scoped_lock(mutex& m)
    : _mtx(m)
{
    _mtx.lock();
}

ocl::be::scoped_lock::~scoped_lock()
{
    _mtx.unlock();
}

ocl::be::kernel_cache::kernel_cache()
    : _kmap(), _mtx()
{
}

ocl::be::kernel_cache::iterator
ocl::be::kernel_cache::
find(const kernel_key& cookie)
{
    return _kmap.find(cookie);
}

void
ocl::be::kernel_cache::
erase(iterator f)
{
    _kmap.erase(f);
}

std::pair<ocl::be::kernel_cache::iterator, bool>
ocl::be::kernel_cache::
insert(const kernel_key& cookie, const kernel_handle& v)
{
    return _kmap.insert(std::make_pair(cookie, v));
}

void
ocl::be::kernel_cache::clear()
{
    _kmap.clear();
}

std::size_t
ocl::be::kernel_cache::size() const
{
    return _kmap.size();
}

