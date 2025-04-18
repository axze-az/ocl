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
#include "ocl/be/kernel_key.h"
#include <iomanip>

ocl::be::kernel_key::kernel_key(const void* p, const std::string& s)
    : _p(p), _s(s)
{
}

ocl::be::kernel_key::kernel_key(const kernel_key& r)
    : _p(r._p), _s(r._s)
{
}

ocl::be::kernel_key::kernel_key(kernel_key&& r)
    : _p(std::move(r._p)), _s(std::move(r._s))
{
}

ocl::be::kernel_key&
ocl::be::kernel_key::operator=(const kernel_key& r)
{
    if (&r != this) {
        _p = r._p;
        _s = r._s;
    }
    return *this;
}

ocl::be::kernel_key&
ocl::be::kernel_key::operator=(kernel_key&& r)
{
    _p = std::move(r._p);
    _s = std::move(r._s);
    return *this;
}

ocl::be::kernel_key::~kernel_key()
{
}

bool
ocl::be::operator<(const kernel_key& a, const kernel_key& b)
{
    return ((a.h() < b.h()) ||
            ((a.h()==b.h()) && (a.l() < b.l())));
}

bool
ocl::be::operator<=(const kernel_key& a, const kernel_key& b)
{
    return ((a.h() < b.h()) ||
            ((a.h()==b.h()) && (a.l() <= b.l())));
}

bool
ocl::be::operator==(const kernel_key& a, const kernel_key& b)
{
    return ((a.h()==b.h()) && (a.l() == b.l()));
}

bool
ocl::be::operator!=(const kernel_key& a, const kernel_key& b)
{
    return ((a.h()!=b.h()) && (a.l() != b.l()));
}

bool
ocl::be::operator>=(const kernel_key& a, const kernel_key& b)
{
    return ((a.h() > b.h()) ||
            ((a.h()==b.h()) && (a.l() >= b.l())));
}

bool
ocl::be::operator>(const kernel_key& a, const kernel_key& b)
{
    return ((a.h() > b.h()) ||
            ((a.h()==b.h()) && (a.l() > b.l())));
}

std::ostream&
ocl::be::operator<<(std::ostream& s, const print_kernel_key& k)
{
    const void* c=reinterpret_cast<const void*>(k._k.h());
    s << c;
    const std::string& l=k._k.l();
    if (!l.empty()) {
        const size_t max_size=15;
        size_t ms=std::min(max_size, l.length());
        s << " [ ";
        for (size_t i=0; i<ms; ++i ){
            char c=l[i];
            switch (c) {
            case '\n':
                s << "\\n";
                break;
            case '\t':
                s << "\\t";
                break;
            case '\r':
                s << "\\r";
                break;
            default:
                s<< c;
            }
        }
        if (l.length() > max_size) {
            s << "...";
        }
        s << " ]";
    }
    return s;
}
