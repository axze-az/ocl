#include "ocl/vector.h"
#include <cftal/math/func_constants.h>
#include <cftal/cast.h>
#include <vector>

namespace ocl { namespace test {
        void subnormals();
    }
}

void ocl::test::subnormals()
{
    using fc_t=cftal::math::func_constants<float>;
    const float msub=fc_t::max_denormal();
    const int32_t maxi=cftal::as<int32_t>(msub);
    std::vector<float> vh;
    for (int32_t i=0; i<=maxi; ++i) {
        float f=cftal::as<float>(i);
        vh.push_back(f);
    }
    std::cout << "created a vector with " << vh.size() << " elements\n";
    vector<float> v0(vh);
    vector<float> v1=(v0*2.0f)*0.5f;
    std::vector<float> vr(v1);
    size_t cnt=0;
    for (size_t i=0; i< vr.size(); ++i) {
        if (vr[i] != 0.0f)
            ++cnt;
    }
    std::cout << cnt << " elements after operations are not equal zero"
              << std::endl;
}

int main()
{
    ocl::test::subnormals();
    return 0;
}
