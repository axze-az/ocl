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
                                        hdcnt, dcnt, nz);
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
