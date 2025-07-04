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
#include "ocl/expr_custom.h"

ocl::impl::__cf_body::
__cf_body(const std::string& n, const std::string& b)
    : _name(n), _body(b)
{
}

ocl::impl::__cf_body::__cf_body(const __cf_body& r)
    : _name(r._name), _body(r._body)
{
}

ocl::impl::__cf_body::__cf_body(__cf_body&& r)
    : _name(std::move(r._name)), _body(std::move(r._body))
{
}

ocl::impl::__cf_body&
ocl::impl::__cf_body::operator=(const __cf_body& r)
{
    if (this != &r) {
        _name = r._name;
        _body = r._body;
    }
    return *this;
}

ocl::impl::__cf_body&
ocl::impl::__cf_body::operator=(__cf_body&& r)
{
    _name = std::move(r._name);
    _body = std::move(r._body);
    return *this;
}

ocl::impl::__cf_body::~__cf_body()
{
}

ocl::impl::__ck_body::__ck_body(const std::string& n, const std::string& b,
                                size_t s)
    : __cf_body(n, b), _s(s)
{
}

ocl::impl::__ck_body::__ck_body(const std::string& n, const std::string& b)
    : __cf_body(n, b), _s()
{
}

ocl::impl::__ck_body::__ck_body(const __ck_body& r)
    : __cf_body(r), _s(r._s)
{
}

ocl::impl::__ck_body::__ck_body(__ck_body&& r)
    : __cf_body(std::move(r)), _s(std::move(r._s))
{
}

ocl::impl::__ck_body&
ocl::impl::__ck_body::operator=(const __ck_body& r)
{
    if (this != &r) {
        __cf_body::operator=(r);
        _s = r._s;
    }
    return *this;
}

ocl::impl::__ck_body&
ocl::impl::__ck_body::operator=(__ck_body&& r)
{
    __cf_body::operator=(std::move(r));
    _s = std::move(r._s);
    return *this;
}

ocl::impl::__ck_body::~__ck_body()
{
}

std::string
ocl::impl::
decl_buffer_args_local_mem(const std::string_view& tname,
                           unsigned& arg_num,
                           bool ro)
{
    static_cast<void>(ro);
    std::ostringstream s;
    s << "    __local "
      << tname
      << "* arg" << arg_num << ",\n";
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
decl_buffer_args_local_mem(const char* tname,
                            unsigned& arg_num,
                            bool ro)
{
    return decl_buffer_args_local_mem(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
decl_buffer_args_local_mem(const std::string& tname,
                           unsigned& arg_num,
                           bool ro)
{
    return decl_buffer_args_local_mem(std::string_view(tname), arg_num, ro);
}

void
ocl::impl::
bind_buffer_args_local_mem(const std::string_view& tname,
                           size_t bytes,
                           size_t elements,
                           size_t wg_size,
                           unsigned& buf_num,
                           be::kernel& k)
{
    // bind p.bytes() * wgs bytes local memory to k
    if (be::data::instance()->debug() != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": "
          << kn << ": " << " binding local_mem_per_workitem<"
          << tname << "> with "
          << elements
          << " elements and "
          << wg_size << " workitems ("
          << bytes << " bytes) "
          << "to arg " << buf_num << '\n';
        be::data::debug_print(s.str());
    }
    k.set_arg(buf_num, bytes, static_cast<const void*>(0));
    ++buf_num;
}

void
ocl::impl::
bind_buffer_args_local_mem(const char* tname,
                           size_t bytes,
                           size_t elements,
                           size_t wg_size,
                           unsigned& buf_num,
                           be::kernel& k)
{
    return bind_buffer_args_local_mem(std::string_view(tname),
                                      bytes, elements, wg_size, buf_num, k);
}

void
ocl::impl::
bind_buffer_args_local_mem(const std::string& tname,
                           size_t bytes,
                           size_t elements,
                           size_t wg_size,
                           unsigned& buf_num,
                           be::kernel& k)
{
    return bind_buffer_args_local_mem(std::string_view(tname),
                                      bytes, elements, wg_size, buf_num, k);
}


std::string
ocl::impl::concat_args_local_mem(var_counters& c)
{
    std::ostringstream s;
    s << "arg" << c._buf_num;
    ++c._var_num;
    ++c._buf_num;
    return s.str();
}
