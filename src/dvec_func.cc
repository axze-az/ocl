#include "ocl/dvec_func.h"

std::string
ocl::dop::names::f_sqrt_base::name(const char* tname)
{
    return name(std::string(tname));
}

std::string
ocl::dop::names::f_sqrt_base::name(const std::string& tname)
{
    return std::string("__sqrt_") + tname;
}

std::string
ocl::dop::names::f_sqrt_base::body(const char* tname)
{
    return body(std::string(tname));
}

std::string
ocl::dop::names::f_sqrt_base::body(const std::string& tname)
{
    std::string inl="inline ";
    std::string fbody =
        inl +
        tname + " __sqrt_" + tname + "(" +
        tname + " a)\n"
        "{\n"
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
        // "    __global const " << tname << "* s,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? d[gid] : 1;\n"
        "    t[lid]=v != 0 ? 1: 0;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            int vi= pos < lsz ? t[pos] : 1;\n"
        "            t[lid] &= vi;\n"
        "        }\n"
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
        // "    __global const " << tname << "* s,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? d[gid] : 0;\n"
        "    t[lid]=v != 0 ? 1: 0;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            int vi= pos < lsz ? t[pos] : 0;\n"
        "            t[lid] |= vi;\n"
        "        }\n"
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
        // "    __global const " << tname << "* s,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? d[gid] : 0;\n"
        "    t[lid]=v != 0 ? 1: 0;\n"
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        "    // loop over t[0, lsz)\n"
        "    for (uint stride=lsz>>1; stride>0; stride >>=1) {\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            int vi= pos < lsz ? t[pos] : 0;\n"
        "            t[lid] |= vi;\n"
        "        }\n"
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
