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
#include "ocl/dvec_func.h"

std::string
ocl::dop::names::f_sqrt_base::name(const std::string_view& tname)
{
    return name(std::string(tname));
}

std::string
ocl::dop::names::f_sqrt_base::name(const std::string& tname)
{
    return std::string("__sqrt_") + tname;
}

std::string
ocl::dop::names::f_sqrt_base::body(const std::string_view& tname)
{
    return body(std::string(tname));
}

std::string
ocl::dop::names::f_sqrt_base::body(const std::string& tname)
{
    std::string inl="static inline ";
    std::string fbody =
        inl +
        tname + " __sqrt_" + tname + "(" +
        tname + " a)\n"
        "{\n"
        "#if F32_CORRECTLY_ROUNDED_DIVIDE_SQRT>0\n"
        "    return sqrt(a);\n"
        "#else\n"
        "    " + tname + " r=rsqrt(a);\n"
        "    " + tname + " rah, ral;\n"
        "    rah=a*r;\n"
        "    ral=fma(-a, r, rah);\n"
        "    " + tname + " th= fma(r, rah, -1.0f);\n"
        "    th=fma(r, ral, th);\n"
        "    r= fma(-0.5f*r*a, th, r*a);\n"
        "    r= isnan(r) ? a*r : r;\n"
        "    r= a==0 ? a : r;\n"
        "    return r;\n"
        "#endif\n"
        "}\n";
    return fbody;
}

std::string
ocl::dop::names::f_sel_base::
body(const std::string& s, const std::string& on_true,
     const std::string& on_false)
{
    std::string r="(( ";
    r += s + ") ? (" + on_true + ") : (" + on_false + "))";
    return r;
}

ocl::impl::__ck_body
ocl::impl::
gen_all_of(const std::string_view& tname)
{
    std::ostringstream s;
    s << "all_of_" << tname;
    std::string kname=s.str();
    s.str("");
    s <<"void " << kname << "(\n"
        "    ulong n,\n"
        "    __global " << tname << "* d,\n"
        "    __global const " << tname << "* s,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? s[gid] : 1;\n"
        "    t[lid]=v != 0 ? 1: 0;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            int vi= pos < lsz ? t[pos] : 1;\n"
        "            t[lid] &= vi;\n"
        "        }\n"
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    }\n"
        "    if (lid == 0) {\n"
        "        ulong grp_id=get_group_id(0);\n"
        "        d[grp_id]=t[0];\n"
        "    }\n"
        "    if (gid == 0) {\n"
        "        ulong grps=get_num_groups(0);\n"
        "        dcnt[0]=grps;\n"
        "    }\n"
        "}\n";
    return __ck_body(kname, s.str());
}

ocl::impl::__ck_body
ocl::impl::
gen_none_of(const std::string_view& tname)
{
    std::ostringstream s;
    s << "none_of_" << tname;
    std::string kname=s.str();
    s.str("");
    s <<"void " << kname << "(\n"
        "    ulong n,\n"
        "    __global " << tname << "* d,\n"
        "    __global const " << tname << "* s,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? s[gid] : 0;\n"
        "    t[lid]=v != 0 ? 1: 0;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            int vi= pos < lsz ? t[pos] : 0;\n"
        "            t[lid] |= vi;\n"
        "        }\n"
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    }\n"
        "    if (lid == 0) {\n"
        "        ulong grp_id=get_group_id(0);\n"
        "        d[grp_id]=t[0];\n"
        "    }\n"
        "    if (gid == 0) {\n"
        "        ulong grps=get_num_groups(0);\n"
        "        dcnt[0]=grps;\n"
        "    }\n"
        "}\n";
    return __ck_body(kname, s.str());
}

ocl::impl::__ck_body
ocl::impl::
gen_any_of(const std::string_view& tname)
{
    std::ostringstream s;
    s << "any_of_" << tname;
    std::string kname=s.str();
    s.str("");
    s <<"void " << kname << "(\n"
        "    ulong n,\n"
        "    __global " << tname << "* d,\n"
        "    __global const " << tname << "* s,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? s[gid] : 0;\n"
        "    t[lid]=v != 0 ? 1: 0;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            int vi= pos < lsz ? t[pos] : 0;\n"
        "            t[lid] |= vi;\n"
        "        }\n"
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    }\n"
        "    if (lid == 0) {\n"
        "        ulong grp_id=get_group_id(0);\n"
        "        d[grp_id]=t[0];\n"
        "    }\n"
        "    if (gid == 0) {\n"
        "        ulong grps=get_num_groups(0);\n"
        "        dcnt[0]=grps;\n"
        "    }\n"
        "}\n";
    return __ck_body(kname, s.str());
}

ocl::impl::__ck_body
ocl::impl::
gen_hsum(const std::string_view& tname)
{
    std::ostringstream s;
    s << "hsum_" << tname;
    std::string kname=s.str();
    s.str("");
    s <<"void " << kname << "(\n"
        "    ulong n,\n"
        "    __global " << tname << "* d,\n"
        "    __global const " << tname << "* s,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? s[gid] : 0;\n"
        "    t[lid]=v;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            t[lid] += t[pos];\n"
        "        }\n"
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    }\n"
        "    if (lid == 0) {\n"
        "        ulong grp_id=get_group_id(0);\n"
        // "        printf(\"%lu %f %u\\n\", grp_id, t[0], lsz);\n"
        "        d[grp_id]=t[0];\n"
        "    }\n"
        "    if (gid == 0) {\n"
        "        ulong grps=get_num_groups(0);\n"
        "        dcnt[0]=grps;\n"
        "    }\n"
        "}\n";
    return __ck_body(kname, s.str());
}

ocl::impl::__ck_body
ocl::impl::
gen_dot_product(const std::string_view& tname)
{
    std::ostringstream s;
    s << "dot_product_" << tname;
    std::string kname=s.str();
    s.str("");
    s <<"void " << kname << "(\n"
        "    ulong n,\n"
        "    __global " << tname << "* d,\n"
        "    __global const " << tname << "* a,\n"
        "    __global const " << tname << "* b,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy a[gid]*b[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? (a[gid]*b[gid]) : 0;\n"
        "    t[lid]=v;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            t[lid] += t[pos];\n"
        "        }\n"
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    }\n"
        "    if (lid == 0) {\n"
        "        ulong grp_id=get_group_id(0);\n"
        "        d[grp_id]=t[0];\n"
        "    }\n"
        "    if (gid == 0) {\n"
        "        ulong grps=get_num_groups(0);\n"
        "        dcnt[0]=grps;\n"
        "    }\n"
        "}\n";
    return __ck_body(kname, s.str());
}

ocl::impl::__ck_body
ocl::impl::even_elements(const std::string_view& tname)
{
    std::ostringstream s;
    s << "even_elements_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << tname << "* src\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        ulong sgid=2*gid;\n"
        "        res[gid] = src[sgid];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}

ocl::impl::__ck_body
ocl::impl::odd_elements(const std::string_view& tname)
{
    std::ostringstream s;
    s << "odd_elements_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << tname << "* src\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        ulong sgid=2*gid+1;\n"
        "        res[gid] = src[sgid];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}

ocl::impl::__ck_body
ocl::impl::combine_even_odd(const std::string_view& tname)
{
    std::ostringstream s;
    s << "combine_even_odd_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << tname << "* e,\n"
        "__global const " << tname << "* o\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        ulong sgid=gid>>1;\n"
        "        if ((gid & 1) == 0) {\n"
        "           res[gid] = e[sgid];\n"
        "        } else {\n"
        "           res[gid] = o[sgid];\n"
        "        }\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}

ocl::impl::__ck_body
ocl::impl::select_even_odd(const std::string_view& tname)
{
    std::ostringstream s;
    s << "select_even_odd_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << tname << "* e,\n"
        "__global const " << tname << "* o\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        if ((gid & 1) == 0) {\n"
        "           res[gid] = e[gid];\n"
        "        } else {\n"
        "           res[gid] = o[gid];\n"
        "        }\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}

ocl::impl::__ck_body
ocl::impl::copy_even_to_odd(const std::string_view& tname)
{
    std::ostringstream s;
    s << "copy_even_to_odd_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << tname << "* src\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        ulong sgid= gid & ~1ul;\n"
        "        res[gid] = src[sgid];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}

ocl::impl::__ck_body
ocl::impl::copy_odd_to_even(const std::string_view& tname)
{
    std::ostringstream s;
    s << "copy_even_to_odd_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << tname << "* src\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    ulong sgid= gid|1ul;\n"
        "    if (sgid < n) {\n"
        "        res[gid] = src[sgid];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}


ocl::impl::__ck_body
ocl::impl::permute(const std::string_view& tname,
                   const std::string_view& iname)
{
    std::ostringstream s;
    s << "permute_"  << iname << '_' << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << iname << "* idx,\n"
        "__global const " << tname << "* src\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        " << iname << " sidx= idx[gid];\n"
        "        int zero= sidx < 0;\n"
        "        res[gid] = zero ? 0 : src[sidx];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}

ocl::impl::__ck_body
ocl::impl::permute2(const std::string_view& tname,
                    const std::string_view& iname)
{
    std::ostringstream s;
    s << "permute2_"  << iname << '_' << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << iname << "* idx,\n"
        "__global const " << tname << "* src0,\n"
        "__global const " << tname << "* src1\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        " << iname << " sidx= idx[gid];\n"
        "        " << iname << " in= (" << iname << ")n;\n"
        "        int zero= sidx < 0;\n"
        "        __global const " << tname << "* src= sidx < in ? src0 : src1;\n"
        "        sidx = sidx < in ? sidx : sidx - in;\n"
        "        res[gid] = zero ? 0 : src[sidx];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}
