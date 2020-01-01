#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"

namespace ocl {

    namespace impl {

        class __ck_body {
            std::string _name;
            std::string _body;
            std::optional<std::size_t> _s;
        public:
            __ck_body(const std::string& n,
                      const std::string& b,
                      std::size_t s)
                : _name(n), _body(b), _s(s) {}
            __ck_body(const std::string& n,
                      const std::string& b)
                : _name(n), _body(b), _s() {}
            const std::string& name() const { return _name; }
            const std::string& body() const { return _body; }
            const std::optional<std::size_t>& size() const {
                return _s;
            }
        };

        using ck_body = ignored_arg<__ck_body>;
    }

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

    namespace dop {

        // custom kernel function marker
        template <typename _T>
        struct custom_k {
            static
            std::string body(...) {
                return std::string();
            }
        };
        template <class _T>
        struct custom_k_arg {
        };

    }

    namespace impl {

        template <typename _RES, typename _T, typename _R>
        be::pgm_kernel_lock
        gen_kernel(_RES& res,
                   const expr<dop::custom_k<dvec<_T> >, impl::ck_body, _R>& r,
                   const void* addr,
                   be::data_ptr b, size_t lmem_size=0);
    }

    template <typename _T, typename _R>
    std::size_t
    eval_size(const expr<dop::custom_k<_T>, impl::ck_body, _R>& r)
    {
        if (r._l.size().has_value()) {
            return r._l.size().value();
        }
        std::size_t rs=eval_size(r._r);
        return rs;
    }

    template <typename _T, typename _R>
    be::data_ptr
    backend_data(const expr<dop::custom_k<_T>, impl::ck_body, _R>& r)
    {
        be::data_ptr p=backend_data(r);
        if (p==nullptr) {
            p=be::data::instance();
        }
        return p;
    }

#if 0
    // generate and execute an opencl kernel for an
    // expression
    template <typename _RES, typename _T, typename _L, typename _R>
    void execute(_RES& res,
                 const expr<dop::custom_k<_T>, impl::ck_body, _R>& r,
                 be::data_ptr b, size_t s) {
        std::cout << "match " << std::endl;
    }
#endif

    namespace impl {

        template <typename _T, typename _A0>
        auto
        custom_kernel_args(const _A0& a0) {
            return a0;
        }

        template <typename _T, typename _A0, typename ... _AX>
        auto
        custom_kernel_args(const _A0& a0, const _AX& ... ax)
        {
            auto tail=custom_kernel_args<_T>(ax...);
            expr<dop::custom_k_arg<_T>, _A0, typeof(tail)> e(a0, tail);
            return e;
        }
    }

    template <typename _T, typename ... _AX>
    auto
    custom_kernel_with_size(const std::string& name,
                            const std::string& body,
                            std::size_t s,
                            const _AX&... ax)
    {
        auto e0=impl::custom_kernel_args<dvec<_T>>(ax...);
        expr<dop::custom_k<dvec<_T> >, impl::ck_body, typeof(e0)>
            e(impl::ck_body(name, body, s), e0);
        return e;
    }

    template <typename _T, typename ... _AX>
    auto
    custom_kernel(const std::string& name,
                  const std::string& body,
                  const _AX&... ax)
    {
        auto e0=impl::custom_kernel_args<dvec<_T> >(ax...);
        expr<dop::custom_k<dvec<_T> >, impl::ck_body, typeof(e0)>
            e(impl::ck_body(name, body), e0);
        return e;
    }

    namespace test {
        void
        test_custom_kernel();
    }

}

template <typename _RES, typename _T, typename _R>
ocl::be::pgm_kernel_lock
ocl::impl::
gen_kernel(_RES& res,
           const expr<dop::custom_k<dvec<_T> >, impl::ck_body, _R>& r,
           const void* cookie,
           be::data_ptr b, size_t lmem_size)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    std::ostringstream s;
    s << "k_" << cookie;
    std::string k_name(s.str());
    s.str("");
    s << "arg_" << cookie;
    std::string k_arg_name(s.str());
    s.str("");
    impl::insert_headers(s);

#if USE_ARG_BUFFER == 0
    static_cast<void>(lmem_size);
    s << "__kernel " << r._l.body();
    k_name = r._l.name();
#else
    // the real kernel follows now
    s << "inline " << r._l.body();
#endif
    s << "\n";
#if USE_ARG_BUFFER>0
    unsigned decl_nb_args(0);
    // argument structure with the scalar arguments
    s << "struct " << k_arg_name << " {\n"
         "    ulong _n __attribute__((aligned("
         << alignof(int64_t) <<")));\n"
      << decl_non_buffer_args(r, decl_nb_args) << "};\n\n";
    // kernel argument
    unsigned buf_args(0);
    s << "__kernel void " << k_name
      << "\n(\n";
    s << decl_buffer_args(res, buf_args, false);
    s << decl_buffer_args(r, buf_args, true);
    if (lmem_size == 0) {
        s << spaces(4) << "__global const struct "
          << k_arg_name << "* pa\n)\n";
        s << "{\n";
    } else {
        s << spaces(4) << "__global const struct "
        << k_arg_name << "* pg\n)\n";
        const size_t uints_to_cpy=((lmem_size + 3) >>2);
        s << "{\n"
             "    /* copy arguments into __args: */\n"
             "    __local union {\n";
        s << "       struct " << k_arg_name << " _a;\n"
             "       uint _u[" << uints_to_cpy << "];\n"
             "    } __args;\n"
             "    {\n"
             "        uint lid= get_local_id(0);\n"
             "        uint lsz= get_local_size(0);\n"
             "        __global const uint* ps= (__global const uint*)pg;\n"
             "        __local uint* pd= __args._u;\n"
             "        const int cpy_size= " << uints_to_cpy << ";\n"
             "        __attribute__((opencl_unroll_hint()))\n"
             "        for (uint i= 0; i < cpy_size; i+= lsz) {\n"
             "            uint idx= i + lid;\n"
             "            if (idx < cpy_size) {\n"
             "                pd[idx] = ps[idx];\n"
             "            }\n"
             "        }\n"
             "        barrier(CLK_LOCAL_MEM_FENCE);\n"
             "    }\n"
             "    __local const struct "
          << k_arg_name << "* pa= &__args._a;\n";
    }
    var_counters c{0};
    s << "    " << r._l.name() << "("
      << "pa->_n, "
      << concat_args(res, c) << ", "
      << concat_args(r, c) << ");\n";
    s << "}\n";
#endif
    using namespace impl;
    std::string ss(s.str());
    if (b->debug() != 0) {
        std::ostringstream st;
        st << std::this_thread::get_id() << ": "
           << k_name << ": --- source code ------------------\n"
           << ss;
        be::data::debug_print(st.str());
    }
    be::program pgm=be::program::create_with_source(ss, b->dcq().c());
    try {
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=clc++");
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=CL1.1");
        pgm.build("-cl-std=CL1.1 -cl-mad-enable");
    }
    catch (const be::error& e) {
        std::cerr << "error info: " << e.what() << '\n';
        std::cerr << pgm.build_log() << std::endl;
        throw;
    }
    be::kernel k(pgm, k_name);
    if (b->debug() != 0) {
        std::ostringstream st;
        st << std::this_thread::get_id() << ": "
           << k_name << ": --- compiled with success --------\n";
        be::data::debug_print(st.str());
    }
    be::pgm_kernel_lock pkl(pgm, k);
    return pkl;
}


void
ocl::test::test_custom_kernel()
{
    const
    dvec<float> v0({2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f, 2.4f});
    {
        std::cout << "v0" << std::endl;
        std::vector<float> vh(v0);
        for (std::size_t i=0; i<vh.size(); ++i) {
            std::cout << vh[i];
            if ((i & 3) ==3)
                std::cout << '\n';
            else
                std::cout << ' ';
        }
    }

    const char* kbody0=
    "__kernel void add(ulong n, __global float* a0, float a1)\n"
    "{\n"
    "    ulong gid = get_global_id(0);\n"
    "    if (gid < n) {\n"
    "        a0[gid]= a1 + gid;\n"
    "    }\n"
    "}\n";
    const char* kname0="add";

    auto ck0=custom_kernel_with_size<float>(kname0, kbody0, 8, 1.5f);
    dvec<float> v1(v0);
    {
        std::cout << "v0 after copy" << std::endl;
        std::vector<float> vh(v0);
        for (std::size_t i=0; i<vh.size(); ++i) {
            std::cout << vh[i];
            if ((i & 3) ==3)
                std::cout << '\n';
            else
                std::cout << ' ';
        }
    }
    v1 = ck0;
    {
        std::vector<float> vh(v1);
        for (std::size_t i=0; i<vh.size(); ++i) {
            std::cout << vh[i];
            if ((i & 3) ==3)
                std::cout << '\n';
            else
                std::cout << ' ';
        }
    }
    const char* kbody1=
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
    const char* kname1="muladd";
    dvec<float> v2(v0);
    auto ck1=custom_kernel<float>(kname1, kbody1, v1, 100.0f);
    std::cout << be::demangle(typeid(ck1).name()) << std::endl;
    dvec<float> v3(v0);
    v3=ck1;
    {
        std::vector<float> vh(v3);
        for (std::size_t i=0; i<vh.size(); ++i) {
            std::cout << vh[i];
            if ((i & 3) ==3)
                std::cout << '\n';
            else
                std::cout << ' ';
        }
    }

    const char* kbody2=
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
    "        a0[gid]= v3;\n"
    "    }\n"
    "}\n";
    const char* kname2="muladd";
    dvec<float> v4(v0);
    {
        std::vector<float> vh(v4);
        for (std::size_t i=0; i<vh.size(); ++i) {
            std::cout << vh[i];
            if ((i & 3) ==3)
                std::cout << '\n';
            else
                std::cout << ' ';
        }
    }
    auto ck2=custom_kernel<float>(kname2, kbody2, v1, 100.0f, v4);
    std::cout << be::demangle(typeid(ck1).name()) << std::endl;
    dvec<float> v5(v0);
    v5=ck2;
    {
        std::vector<float> vh(v5);
        for (std::size_t i=0; i<vh.size(); ++i) {
            std::cout << vh[i];
            if ((i & 3) ==3)
                std::cout << '\n';
            else
                std::cout << ' ';
        }
    }
}

int main()
{
    ocl::test::test_custom_kernel();
}
