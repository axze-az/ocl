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
#include "ocl/be/kernel_functions.h"

ocl::be::kernel_functions::kernel_functions()
    : _s()
{
}

ocl::be::kernel_functions::kernel_functions(const kernel_functions& r)
    : _s(r._s)
{
}

ocl::be::kernel_functions::kernel_functions(kernel_functions&& r)
    : _s(std::move(r._s))
{
}

ocl::be::kernel_functions&
ocl::be::kernel_functions::operator=(const kernel_functions& r)
{
    if (&r != this) {
        _s = r._s;
    }
    return *this;
}

ocl::be::kernel_functions&
ocl::be::kernel_functions::operator=(kernel_functions&& r)
{
    _s = std::move(r._s);
    return *this;
}

ocl::be::kernel_functions::~kernel_functions()
{
}

bool
ocl::be::kernel_functions::insert(const std::string& fn)
{
    return _s.insert(fn).second;
}

bool
ocl::be::kernel_functions::insert(const std::string_view& fn)
{
    return insert(std::string(fn));
}
