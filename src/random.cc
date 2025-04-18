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
#include "ocl/random.h"

ocl::dvec<std::uint32_t>
ocl::random_base::
fill_with_global_id(std::size_t s, be::data_ptr p)
{
    ocl::dvec<std::uint32_t> r(p, s);
    if (s) {
        const char* kname="get_gid";
        const char* ksrc=
            "void get_gid(ulong n, __global uint* d)\n"
            "{\n"
            "    ulong gid=get_global_id(0);\n"
            "    if (gid < n) {\n"
            "        d[gid]=gid;\n"
            "    }\n"
            "}\n";
        auto ck=custom_kernel<std::uint32_t>(kname, ksrc, r);
        execute_custom(ck, s, p);
    }
    return r;
}

ocl::rand::rand(std::size_t s)
    : _next(fill_with_global_id(s, be::data::instance()))
{
}

ocl::rand::rand(std::size_t s, be::data_ptr p)
    : _next(fill_with_global_id(s, p))
{
}

void
ocl::rand::rand::seed_times_global_id(uint32_t s)
{
    _next=fill_with_global_id(_next.size(), _next.backend_data())*s;
}

ocl::dvec<int32_t>
ocl::rand::next()
{
    const char* kname="rand_next";
    const char* ksrc=
        "void\n"
        "rand_next(ulong n, __global int* r, __global uint* s)\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        uint next = s[gid];\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        int result = (int)((next >> 16) & 2047);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int)((next >> 16) & 1023);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int) ((next >> 16) & 1023);\n"
        "        s[gid] = next;\n"
        "        r[gid] = result;\n"
        "    }\n"
        "}\n";
    auto ck=custom_kernel<std::int32_t>(kname, ksrc, _next);
    dvec<std::int32_t> r(ck);
    return r;
}

ocl::dvec<float>
ocl::rand::nextf()
{
    const char* kname="rand_nextf";
    const char* ksrc=
        "void\n"
        "rand_nextf(ulong n, __global float* r, __global uint* s)\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        uint next = s[gid];\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        int result = (int)((next >> 16) & 2047);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int)((next >> 16) & 1023);\n"
        "        next *= 1103515245;\n"
        "        next += 12345;\n"
        "        result <<= 10;\n"
        "        result ^= (int) ((next >> 16) & 1023);\n"
        "        result &= 0xffffff; \n"
        "        float fresult= convert_float_rtz(result)* 0x1p-24f;\n"
        "        s[gid] = next;\n"
        "        r[gid] = fresult;\n"
        "    }\n"
        "}\n";
    auto ck=custom_kernel<float>(kname, ksrc, _next);
    dvec<float> r(ck);
    return r;
}

const std::uint64_t ocl::rand48::A=0x5DEECE66Dul;
const std::uint64_t ocl::rand48::C=0xBL;
const std::uint64_t ocl::rand48::M=(1ULL<<48);
const std::uint64_t ocl::rand48::MM=(M-1);

void
ocl::rand48::
next()
{
    _state= ((_state * A + C) & MM);
}


ocl::rand48::
rand48(size_t s, be::data_ptr p)
    : _state(
        (((cvt<dvec<uint64_t> >(fill_with_global_id(s, p)) << 16)
          | 0x330E)) & MM)
{
}

void
ocl::rand48::rand48::seed_times_global_id(uint32_t seed_val)
{
    size_t s=_state.size();
    auto p= _state.backend_data();
    _state=
        ((((cvt<dvec<uint64_t> >(fill_with_global_id(s, p))*seed_val) << 16)
          | 0x330E)) & MM;
}

ocl::dvec<std::int32_t>
ocl::rand48::
lrand48()
{
    next();
    // return cvt<dvec<std::int32_t> >(_state) & (0x7fffffff);
    const unsigned shift=48-32+1;
    return cvt<dvec<std::int32_t> >(_state >> shift);
}

inline
ocl::dvec<std::int32_t>
ocl::rand48::
mrand48()
{
    next();
    const unsigned shift=48-32;
    return as<dvec<std::int32_t> >(cvt<dvec<std::uint32_t> >(_state>>shift));
}

ocl::dvec<float>
ocl::rand48::
drand48()
{
    next();
    const unsigned shift=48-24;
    return cvt<dvec<float>>(_state >>shift) * 0x1p-24f;
    // return cvt<dvec<float>>(_state & 0xffffff) * 0x1p-24f;
}

void
ocl::rand48::
seed(const dvec<uint64_t>& gid)
{
    _state = ((gid * 65536) | 0x330E) & MM;
}

ocl::dvec<float>
ocl::rand48::
nextf()
{
    return drand48();
}

ocl::dvec<float>
ocl::
uniform_float_random_vector(rand& rnd, float min_val, float max_val)
{
    const float range=max_val - min_val;
    return ((rnd.nextf() * range) + min_val);
}

ocl::dvec<float>
ocl::
uniform_float_random_vector(rand48& rnd, float min_val, float max_val)
{
    const float range=max_val - min_val;
    return ((rnd.nextf() * range) + min_val);
}
