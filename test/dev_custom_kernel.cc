#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"
#include "ocl/test/tools.h"

namespace ocl {

    namespace impl {

        template <typename _T>
        struct get_arg_info {
            static
            constexpr
            size_t size() { return 1; }

        };

        template <>
        struct get_arg_info<void> {
        };


        template <typename _OP, typename _L, typename _R>
        struct get_arg_info<expr<_OP, _L, _R> > {
        };

        template <typename _T>
        struct get_arg_info<dvec<_T> > {
        };

    }

    namespace test {
        void
        test_custom_kernel();
    }

}

void
ocl::test::test_custom_kernel()
{
    for (std::size_t i=0; i<3; ++i) {
        const
            dvec<float> v0({2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f});
        dump(v0, "v0");
        const char* kbody0=
            "__kernel void add(ulong n, __global float* a0, float a1)\n"
            "{\n"
            "    ulong gid = get_global_id(0);\n"
            "    if (gid < n) {\n"
            "        a0[gid]= a1 + gid;\n"
            "    }\n"
            "}\n";
        const char* kname0="add";
        std::cout << "expecting 1 object\n" << dvec<float>::state();
        auto ck0=custom_kernel_with_size<float>(kname0, kbody0, 8, 1.5f);
        std::cout << "expecting 1 object\n" << dvec<float>::state();
        dvec<float> v1(v0);
        std::cout << "expecting 2 objects\n" << dvec<float>::state();
        dump(v0, "v0 after copy");
        v1 = ck0;
        dump(v1, "v1 after assignment 1.5f + gid");
        std::cout << "expecting 2 objects\n" << dvec<float>::state();

        const char* kbody1=
            "__kernel void add(ulong n, __global float* a0, float a1)\n"
            "{\n"
            "    ulong gid = get_global_id(0);\n"
            "    if (gid < n) {\n"
            "        a0[gid]= a1 - gid;\n"
            "    }\n"
            "}\n";
        const char* kname1="add";
        std::cout << "expecting 1 object\n" << dvec<float>::state();
        auto ck1=custom_kernel_with_size<float>(kname1, kbody1, 8, 1.5f);
        v1 = ck1;
        dump(v1, "v1 after assignment 1.5f - gid");
        std::cout << "expecting 2 objects\n" << dvec<float>::state();

        const char* kbody2=
            "__kernel void muladd(ulong n,\n"
            "                     __global float* a0,\n"
            "                     __global const float* a1,\n"
            "                     float a2)\n"
            "{\n"
            "    ulong gid = get_global_id(0);\n"
            "    if (gid < n) {\n"
            "        a0[gid]= a1[gid]+a2;\n"
            "    }\n"
            "}\n";
        const char* kname2="muladd";
        dvec<float> v2(v0);
        std::cout << "expecting 3 objects\n" << dvec<float>::state();
        auto ck2=custom_kernel<float>(kname2, kbody2, v1, 100.0f);
        std::cout << "expecting 3 objects\n" << dvec<float>::state();
        std::cout << be::demangle(typeid(ck2).name()) << std::endl;
        std::cout << be::demangle(typeid(ck2._r).name()) << std::endl;
        std::size_t s= eval_size(ck1._r);
        std::cout << s << std::endl;
        dvec<float> v3=ck2;
        dump(v3, "v3: v1 + 100");

        const char* kbody3=
            "__kernel void muladd(ulong n,\n"
            "                     __global float* a0,\n"
            "                     __global const float* a1,\n"
            "                     float a2,\n"
            "                     __global const float* a3)\n"
            "{\n"
            "    ulong gid = get_global_id(0);\n"
            "    if (gid < n) {\n"
            "        float v0;\n"
            "        float v1=a1[gid];\n"
            "        float v2=a2;\n"
            "        float v3=a3[gid];\n"
            "        a0[gid]= (v1*v2)+v3;\n"
            "    }\n"
            "}\n";
        const char* kname3="muladd";
        dvec<float> v4(v0);
        v1=v0;
        v4=dvec<float>(8, 2.0f);
        // 2.4 * 100 + 2.0
        dump(v4, "v4: 2.0");
        auto ck3=custom_kernel<float>(kname3, kbody3, v1, 100.0f, v4);
        dump(v4, "v4: 2.0 after assignment");
        std::cout << be::demangle(typeid(ck3).name()) << std::endl;
        dvec<float> v5=ck3;
        dump(v5, "v5: v1*100 + 2 = 242");
    }
}

int main()
{
    ocl::test::test_custom_kernel();
}
