#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"
#include "ocl/test/tools.h"

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
        struct custom_arg {
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
        const _A0&
        custom_args(_A0&& a0) {
            return a0;
        }
#if 0
        template <typename _T, typename _A0, typename _A1>
        auto
        custom_args(_A0&& a0, _A1&& a1) {
            return make_expr<dop::custom_arg<_T> >(
                custom_args<_T>(std::forward<_A0&&>(a0)),
                custom_args<_T>(std::forward<_A1&&>(a1)));
        }
#endif
        template <typename _T, typename _A0, typename ... _AX>
        auto
        custom_args(_A0&& a0, _AX&& ... ax)
        {
            return make_expr<dop::custom_arg<_T> >(
                custom_args<_T>(std::forward<_A0&&>(a0)),
                custom_args<_T>(std::forward<_AX&&>(ax) ...));
        }
    }

    template <typename _T, typename ... _AX>
    auto
    custom_kernel_with_size(const std::string& name,
                            const std::string& body,
                            std::size_t s,
                            _AX&&... ax)
    {
        return make_expr<dop::custom_k<dvec<_T> > >(
            impl::ck_body(name, body, s),
            impl::custom_args<dvec<_T>>(
                std::forward<_AX&&>(ax) ...));
    }

    template <typename _T, typename ... _AX>
    auto
    custom_kernel(const std::string& name,
                  const std::string& body,
                  _AX&&... ax)
    {
        return make_expr<dop::custom_k<dvec<_T> > >(
            impl::ck_body(name, body),
            impl::custom_args<dvec<_T>>(
                std::forward<_AX&&>(ax) ...));
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
    std::cout << "expecting 3 objects\n" << dvec<float>::state();
    auto ck1=custom_kernel<float>(kname1, kbody1, v1, 100.0f);
    std::cout << "expecting 3 objects\n" << dvec<float>::state();
    std::cout << be::demangle(typeid(ck1).name()) << std::endl;
    std::cout << be::demangle(typeid(ck1._r).name()) << std::endl;
    std::size_t s= eval_size(ck1._r);
    std::cout << s << std::endl;
    dvec<float> v3(v0);
    v3=ck1;
    dump(v3, "v3: v1 + 100");

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
    "        a0[gid]= (v1*v2)+v3;\n"
    "    }\n"
    "}\n";
    const char* kname2="muladd";
    dvec<float> v4(v0);
    v1=v0;
    v4=dvec<float>(8, 2.0f);
    // 2.4 * 100 + 2.0
    dump(v4, "v4: 2.0");
    auto ck2=custom_kernel<float>(kname2, kbody2, v1, 100.0f, v4);
    dump(v4, "v4: 2.0 after assignment");
    std::cout << be::demangle(typeid(ck2).name()) << std::endl;
#if 0
    dvec<float> v5(v0);
    v5=ck2;
#else
    dvec<float> v5(ck2);
#endif
    dump(v5, "v5: v1*100 + 2 = 242");
}

int main()
{
    ocl::test::test_custom_kernel();
}
