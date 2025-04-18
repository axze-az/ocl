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
#include "ocl/be/type_2_name.h"
#include <cxxabi.h>
#include <memory>
#include <sstream>

std::string
ocl::be::demangle( const char* mangled_name )
{
    std::size_t len = 0 ;
    int status = 0 ;
    std::unique_ptr< char, decltype(&std::free) > ptr(
        __cxxabiv1::__cxa_demangle( mangled_name, nullptr, &len, &status ),
        &std::free ) ;
    return ptr.get() ;
}

std::string
ocl::be::
type_2_name_vec_t(const char* tname, size_t n)
{
    return type_2_name_vec_t(std::string_view(tname), n);
}

std::string
ocl::be::
type_2_name_vec_t(const std::string_view& tname, size_t n)
{
    std::ostringstream s;
    s << tname << n;
    return s.str();
}
