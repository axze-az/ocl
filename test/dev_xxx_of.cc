#include "ocl/dvec.h"

namespace ocl {
    namespace test {
        void
        test_all_of();
    }

    namespace impl {
        std::string
        gen_all_of(const std::string& tname);
    }
    
    template <typename _T>
    bool
    all_of(const dvec<_T>& v);
}

std::string
gen_all_of(const std::string& tname)
{
    std::ostringstream s;
    s <<"__kernel void all_of_"<< tname << "(ulong n,\n"
        "                     __global " << tname << "* ds,\n"
        "                     __global ulong* dcnt,\n"
        "                     __local " << tname << "* t)\n"
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
    return s.str();
}

template <typename _T>
bool
ocl::all_of(dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    typename dvec<_T>::mask_type nz= v != _T(0);
    dvec<uint64_t> dcnt(1);
    uint64_t hdcnt=nz.size();
    const auto tname=be::type_2_name<type>::v();
    const std::string k_name=std::string("all_of_") + tname;
    const std::string k_body=gen_all_of(tname);
    do {
        auto ck=custom_kernel_with_size<uint64_t>(k_name, k_body,
                                                  hdcnt, dcnt,
                                                  local_mem_per_workitem<type>(1));
        nz=ck;
        dcnt.copy_to_host(&hdcnt);
    } while (hdcnt>1);
    // copy only one element from nz
    type r;
    nz.copy_to_host(&r, 0, 1);
    return r != 0;
}

void
ocl::test::test_all_of()
{
    const
        dvec<float> v0({2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f});
    dvec<float>::mask_type t0= v0==0.0f;
    std::cout << all_of(t0) << std::endl;
    dvec<float>::mask_type t1= v0==2.4f;
    std::cout << all_of(t1) << std::endl;
}

int main()
{
    ocl::test::test_all_of();
    return 0;
}
