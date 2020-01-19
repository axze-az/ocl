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
        "                     __global ulong* dcnt,\n"
        "                     __global " << tname << "* ds,\n"
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
        "#if 1\n"
        "        uint pos= lid + stride;\n"
        "        " << tname <<
        " vi= (lid < stride) & (pos < lsz) ? t[pos] : t[lid];\n"
        "        t[lid] &= vi;\n"
        "#else\n"
        "        if (lid < stride) {\n"
        "            uint pos=lid + stride;\n"
        "            int vi= pos < lsz ? t[pos] : 1;\n"
        "            t[lid] &= vi;\n"
        "        }\n"
        "#endif\n"
        "        barrier(CLK_LOCAL_MEM_FENCE);\n"
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
    const std::string k_name="all_of_" + tname;
    const std::string k_body=gen_all_of(tname);
    do {
        auto ck=custom_kernel_with_size(k_name, k_body,
                                        hdcnt, dcnt, nz,
                                        local_mem_per_workitem<type>(1));
        nz=ck;
        dcnt.copy_to_host(&hdcnt);
    } while (hdcnt>1);
    // copy only one element from nz
    type r;
    dcnt.copy_to_host(&r, 0, 1);
    return r != 0;
}

void
ocl::test::test_all_of()
{
}

int main()
{
    return 0;
}
