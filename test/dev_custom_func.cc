#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"
#include "ocl/test/tools.h"
#include <set>

namespace ocl {

    namespace test {
        void
        test_custom_func();
    }

}

void
ocl::test::test_custom_func()
{
    const
    dvec<float> v0({2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f});
    dump(v0, "v0");
    const char* fbody0=
        "float add(float v0, float v1, float v2)\n"
        "{\n"
        "    return v0 + v1 + v2;\n"
        "}\n";
    const char* fname0="add";
    auto e=custom_func<float>(fname0, fbody0, v0, v0);
    std::cout << ocl::be::demangle(typeid(e).name()) << std::endl;
    auto t1=custom_func<float>(fname0, fbody0, v0, v0, 2.0f) +
        custom_func<float>(fname0, fbody0, v0, v0, 2.0f);
    std::set<std::string> fnames;
    std::cout << def_custom_func(fnames, t1);
    dvec<float> v1=t1;
    dump(v1, "v1: (v0+v0+2.0f)+(v0+v0+2.0f) = 2.0f*6.8f=13.6f");

    static const float ci[]={
        1.0f, 2.0f, 3.0f, 4.0f
    };
    const char* hname="horner";
    const char* hbody=
        "float horner(float v0, __arg_local const float* c)\n"
        "{\n"
        "    float r=v0*c[0];\n"
        "    for (int i=1; i<4; ++i) {\n"
        "        r=v0*r+c[i];\n"
        "    }\n"
        "    return r;\n"
        "}\n";
    auto h=custom_func<float>(hname, hbody, v0, ci);
    dvec<float> v2=h;
    dump(v2, "v2 after call to horner");
}

int main()
{
    ocl::test::test_custom_func();
}
