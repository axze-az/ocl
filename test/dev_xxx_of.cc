#include "ocl/dvec.h"

namespace ocl {
    namespace test {
        void
        test_xxx_of();
    }

}

void
ocl::test::test_xxx_of()
{
    for (int i=65; i<66; ++i) {
        std::cout << "testing size " << i << std::endl;
        std::vector h0(i, 2.4f);
        dvec<float> v0(h0);
        dvec<float>::mask_type t00= v0==0.0f;
        std::cout << "all_of(v0==0.0f)=false  " << all_of(t00) << std::endl;
        std::cout << "any_of(v0==0.0f)=false  " << any_of(t00) << std::endl;
        std::cout << "none_of(v0==0.0f)=true  " << none_of(t00) << std::endl;
        dvec<float>::mask_type t01= v0==2.4f;
        std::cout << "all_of(v0==2.4f)=true   " << all_of(t01) << std::endl;
        std::cout << "any_of(v0==2.4f)=true   " << any_of(t01) << std::endl;
        std::cout << "none_of(v0==2.4f)=false " << none_of(t01) << std::endl;
        h0.back() = 0.0f;
        dvec<float> v1(h0);
        dvec<float>::mask_type t10= v1==0.0f;
        std::cout << "all_of(v1==0.0f)=false  " << all_of(t10) << std::endl;
        std::cout << "any_of(v1==0.0f)=true   " << any_of(t10) << std::endl;
        std::cout << "none_of(v1==0.0f)=false " << none_of(t10) << std::endl;
        dvec<float>::mask_type t11= v1==2.4f;
        std::cout << "all_of(v0==2.4f)=false   " << all_of(t11) << std::endl;
        std::cout << "any_of(v0==2.4f)=true    " << any_of(t11) << std::endl;
        std::cout << "none_of(v0==2.4f)=false  " << none_of(t11) << std::endl;
        std::cout << std::endl;
    }
}

int main()
{
    ocl::test::test_xxx_of();
    return 0;
}
