#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"
#include "ocl/test/tools.h"
#include <set>

namespace ocl {

    namespace test {
        void
        test_select();
    }

}

void
ocl::test::test_select()
{
    const dvec<float> v0{2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f};
    const dvec<float> v1{1.2f, 1.2f, 1.2f, 1.2f, 1.2f, 1.2f, 1.2f, 1.2f};
    dump(v0, "v0:");
    dump(v1, "v1:");
    for (int i=0; i<2; ++i) {
        dvec<float>::mask_type vm= v0> v1;
        dvec<float> v2=select(vm, v1, v0);
        dump(v2, "v2: v0 > v1 ? v1 : v0");
        v2=select(v0 > v1, v1, v0);
        dump(v2, "v2: v0 > v1 ? v1 : v0");
        v2=select((v0 > v1) | (v0==v1), 3.0f*v1, v0+2.0f);
        dump(v2, "v2: (v0 > v1) | (v0==v1) ? 3.0f*v1 : v0+2.0f");
    }
}

int main()
{
    ocl::test::test_select();
}
