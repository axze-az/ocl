#include "ocl/dvec.h"

namespace ocl {
    namespace test {
        void
        test_xxx_of();
    }

    namespace impl {
        __ck_body
        gen_all_of(const std::string_view& tname);

        __ck_body
        gen_any_of(const std::string_view& tname);

        __ck_body
        gen_none_of(const std::string_view& tname);

        template <typename _T>
        typename dvec<_T>::mask_value_type
        xxx_of(const __ck_body& nb, const dvec<_T>& z);

    }
    
    template <typename _T>
    bool
    all_of(const dvec<_T>& v);

    template <typename _T>
    bool
    none_of(const dvec<_T>& v);

    template <typename _T>
    bool
    any_of(const dvec<_T>& v);

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
        "    __global " << tname << "* ds,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? ds[gid] : 1;\n"
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
        "        ds[grp_id]=t[0];\n"
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
        "    __global " << tname << "* ds,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? ds[gid] : 0;\n"
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
        "        ds[grp_id]=t[0];\n"
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
        "    __global " << tname << "* ds,\n"
        "    __global ulong* dcnt,\n"
        "    __local " << tname << "* t\n"
        ")\n"
        "{\n"
        "    ulong gid= get_global_id(0);\n"
        "    uint lid= get_local_id(0);\n"
        "    uint lsz= get_local_size(0);\n"
        "    // copy s[gid] into t[lid]\n"
        "    " << tname << " v= gid < n ? ds[gid] : 0;\n"
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
        "        ds[grp_id]=t[0];\n"
        "    }\n"
        "    if (gid == 0) {\n"
        "        ulong grps=get_num_groups(0);\n"
        "        dcnt[0]=grps;\n"
        "    }\n"
        "}\n";
    return __ck_body(kname, s.str());
}

template <typename _T>
typename ocl::dvec<_T>::mask_value_type
ocl::impl::xxx_of(const __ck_body& nb, const dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    typename dvec<_T>::mask_type nz= v != _T(0);
    dvec<uint64_t> dcnt(1);
    uint64_t hdcnt=nz.size();
    do {
        auto ck=custom_kernel_with_size<type>(nb.name(), nb.body(),
                                              hdcnt, dcnt,
                                              local_mem_per_workitem<type>(1));
        nz=ck;
        dcnt.copy_to_host(&hdcnt);
    } while (hdcnt>1);
    // copy only one element from nz
    type r;
    nz.copy_to_host(&r, 0, 1);
    return r;
}

template <typename _T>
bool
ocl::all_of(dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    const auto tname=be::type_2_name<type>::v();
    auto nb=impl::gen_all_of(tname);
    auto r=impl::xxx_of(nb, v);
    return r!=0;
}

template <typename _T>
bool
ocl::none_of(dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    const auto tname=be::type_2_name<type>::v();
    auto nb=impl::gen_none_of(tname);
    auto r=impl::xxx_of(nb, v);
    return r==0;
}

template <typename _T>
bool
ocl::any_of(dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    const auto tname=be::type_2_name<type>::v();
    auto nb=impl::gen_any_of(tname);
    auto r=impl::xxx_of(nb, v);
    return r!=0;
}

void
ocl::test::test_xxx_of()
{
    for (int i=0; i<2; ++i) {
        dvec<float> v0({2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f});
        dvec<float>::mask_type t00= v0==0.0f;
        std::cout << "all_of(v0==0.0f)=false  " << all_of(t00) << std::endl;
        std::cout << "any_of(v0==0.0f)=false  " << any_of(t00) << std::endl;
        std::cout << "none_of(v0==0.0f)=true  " << none_of(t00) << std::endl;
        dvec<float>::mask_type t01= v0==2.4f;
        std::cout << "all_of(v0==2.4f)=true   " << all_of(t01) << std::endl;
        std::cout << "any_of(v0==2.4f)=true   " << any_of(t01) << std::endl;
        std::cout << "none_of(v0==2.4f)=false " << none_of(t01) << std::endl;
        
        dvec<float> v1({2.4f, 2.4f, 2.4f, 2.4f, 0.0f, 0.0f, 0.0f});
        dvec<float>::mask_type t10= v1==0.0f;
        std::cout << "all_of(v1==0.0f)=false  " << all_of(t10) << std::endl;
        std::cout << "any_of(v1==0.0f)=true   " << any_of(t10) << std::endl;
        std::cout << "none_of(v1==0.0f)=false " << none_of(t10) << std::endl;
        dvec<float>::mask_type t11= v1==2.4f;
        std::cout << "all_of(v0==2.4f)=false   " << all_of(t11) << std::endl;
        std::cout << "any_of(v0==2.4f)=true    " << any_of(t11) << std::endl;
        std::cout << "none_of(v0==2.4f)=false  " << none_of(t11) << std::endl;
    }
}

int main()
{
    ocl::test::test_xxx_of();
    return 0;
}
