#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"

namespace ocl {

    namespace impl {
        class ck_body {
            std::string _name;
            std::string _body;
        public:
            ck_body(const std::string& n,
                    const std::string& b)
                : _name(n), _body(b) {}
            const std::string& name() const { return _name; }
            const std::string& body() const { return _body; }
        };

        bool
        operator==(const ck_body& a, const ck_body& b);

        class ck_state {
            std::set<std::string> _f_names;
        public:
            ck_state() : _f_names() {}
        };
    }


    struct arg_info {
        // argument number
        int _arg_num;
        // buffer number
        int _buf_num;
        // scalar argument number
        int _scalar_num;
        // type of the argument
        std::string _type;
        // type as kernel argument
        std::string _kernel_arg_type;
        // type as function argument
        std::string _func_arg_type;
    public:
        // TODO: check if we really need strings here:
        arg_info(const std::string& type,
                    const std::string& kernel_arg_type,
                    const std::string& func_arg_type)
            : _arg_num(-1), _buf_num(-1), _scalar_num(-1),
                _type(type),
                _kernel_arg_type(kernel_arg_type),
                _func_arg_type(func_arg_type) {}
        arg_info(const arg_info& r) = default;
        arg_info(arg_info&& r)
            : _arg_num(std::move(r._arg_num)),
                _buf_num(std::move(r._buf_num)),
                _scalar_num(std::move(r._scalar_num)),
                _type(std::move(r._type)),
                _kernel_arg_type(std::move(r._kernel_arg_type)),
                _func_arg_type(std::move(r._func_arg_type)) {}
        arg_info& operator=(const arg_info& r) = default;
        arg_info& operator=(arg_info&& r) {
            _arg_num=std::move(r._arg_num);
            _buf_num=std::move(r._buf_num);
            _scalar_num=std::move(r._scalar_num);
            _type=std::move(r._type);
            _kernel_arg_type=std::move(r._kernel_arg_type);
            _func_arg_type=std::move(r._func_arg_type);
            return *this;
        }
    };

    using arg_info_list = std::vector<arg_info>;


    struct arg_info_state {
        // argument number
        int _arg_num;
        // buffer number
        int _buf_num;
        // scalar argument number
        int _scalar_num;
        arg_info_state() : _arg_num(0), _buf_num(0), _scalar_num(0) {}
    };

    namespace impl {

        template <typename _T>
        struct get_arg_info {
            static
            constexpr
            size_t size() { return 1; }

            static
            void
            collect(arg_info_list& va, arg_info_state& ais, bool ro) {
                arg_info& ai=
                    va.emplace_back(be::type_2_name<_T>::v(),
                                    be::type_2_name<_T>::v(),
                                    be::type_2_name<_T>::v());
                ai._arg_num=ais._arg_num;
                ai._scalar_num=ais._scalar_num;
                ++ais._arg_num;
                ++ais._scalar_num;
                static_cast<void>(ro);
            }
        };

        template <>
        struct get_arg_info<void> {
            void
            collect(arg_info_list& va, arg_info_state& ais, bool ro) {
                static_cast<void>(va);
                static_cast<void>(ais);
                static_cast<void>(ro);
            }
        };


        template <typename _OP, typename _L, typename _R>
        struct get_arg_info<expr<_OP, _L, _R> > {
            static
            void
            collect(arg_info_list& va, arg_info_state& ais, bool ro) {
                get_arg_info<_L>::collect(va, ais, ro);
                get_arg_info<_R>::collect(va, ais, ro);
            }
        };

        template <typename _T>
        struct get_arg_info<dvec<_T> > {
            static
            void
            collect(arg_info_list& va, arg_info_state& ais, bool ro) {
                const std::string karg="__global ";
                std::string cnst=ro ? "const " : "";
                arg_info& ai=
                    va.emplace_back(be::type_2_name<_T>::v(),
                                    karg+cnst+be::type_2_name<_T>::v()+"*",
                                    be::type_2_name<_T>::v());
                ai._arg_num=ais._arg_num;
                ai._buf_num=ais._buf_num;
                ++ais._arg_num;
                ++ais._buf_num;
            }
        };

    }

    template <typename _T>
    void
    get_arg_info(arg_info_list& va, arg_info_state& ais,
                 const _T& r, bool ro)
    {
        static_cast<void>(r);
        impl::get_arg_info<_T>::collect(va, ais, ro);
    }


    template <typename _R, typename _T>
    std::string
    gen_kernel(const std::string& k_name,
               _R& res,
               const _T& e)
    {
        std::ostringstream s;
        const char nl='\n';
        // the real kernel follows now
        s << "__kernel void " << k_name
        << "\n(\n";
        // element count:
        std::string element_count=spaces(4) +
            be::type_2_name<unsigned long>::v() + " n";
        // argument generation
        unsigned arg_num{0};
        s << eval_args(element_count, res, arg_num, false)
        << ",\n";
        s << eval_args("", e, arg_num, true);
        s << "\n)\n";
        // begin body
        s << "{\n";

        // global id
        s << spaces(4) << "ulong gid = get_global_id(0);\n";
        s << spaces(4) << "if (gid < n) {\n";
        // temporary variables
        unsigned var_num{1};
        s << eval_vars(e, var_num, true)
        << nl;

        // result variable
        unsigned res_num{0};
        s << eval_vars(res, res_num, false) << "= ";
        // the operations
        unsigned body_num{1};
        s << eval_ops(e, body_num) << ";\n";
        // write back
        res_num = 0;
        s << eval_results(res, res_num)
        << nl;
        // end if
        s << spaces(4) << "}\n";
        // end body
        s << "}\n";
        return s.str();
    }

    namespace be {
    }


    template <typename _T>
    std::string
    decl_func_args(const _T& t, unsigned& var_num) {
        static_cast<void>(t);
        std::ostringstream s;
        s << be::type_2_name<_T>::v() << " v" << var_num << ",\n";
        ++var_num;
        return spaces(4)+s.str();
    }

    template <typename _T>
    std::string
    decl_func_args(dvec<_T>& t, unsigned& var_num) {
        static_cast<void>(t);
        std::ostringstream s;
        s << "__global "
          << be::type_2_name<_T>::v() << "* v" << var_num << ",\n";
        ++var_num;
        return spaces(4)+s.str();
    }

    template <typename _T>
    std::string
    decl_func_args(const dvec<_T>& t, unsigned& var_num) {
        static_cast<void>(t);
        std::ostringstream s;
        s << "__global const "
          << be::type_2_name<_T>::v() << "* v" << var_num << ",\n";
        ++var_num;
        return spaces(4)+s.str();
    }

#if 0
    template <typename _T,
              template <typename _T1> class _OP,
              class _L, class _R>
    std::string
    decl_func_args(const expr<_OP<dvec<_T> >, _L, _R>& e, unsigned& var_num) {
        static_cast<void>(e);
        std::ostringstream s;
        s << be::type_2_name<_T>::v() << " v" << var_num << ",\n";
        ++var_num;
        return spaces(4)+s.str();
    }

    template <typename _T,
              template <typename _T1> class _OP,
              class _L, class _R>
    std::string
    decl_func_args(const expr<_OP<_T>, _L, _R>& e, unsigned& var_num) {
        static_cast<void>(e);
        std::ostringstream s;
        s << be::type_2_name<_T>::v() << " v" << var_num << ",\n";
        ++var_num;
        return spaces(4)+s.str();
    }
#endif

    template <typename _OP, typename _T, typename _R>
    std::string
    decl_func_args(const expr<_OP, _T, _R>& e, unsigned& var_num) {
        std::string l=decl_func_args(e._l, var_num);
        std::string r=decl_func_args(e._r, var_num);
        return l+r;
    }

    template <typename _OP, typename _T>
    std::string
    decl_func_args(const expr<_OP, _T, void>& e, unsigned& var_num) {
        std::string l=decl_func_args(e._l, var_num);
        return l;
    }

    namespace dop {

        // custom kernel function marker
        template <typename _T>
        struct ck_func {
        };
        // custome kernel function komma operator
        namespace names {
            struct komma { const char* operator()() const { return ", "; }};
        }
        template <class _T>
        struct ck_komma : public binary_func<names::komma, true> {};

        template <typename _T>
        struct ck_kernel {
        };

    }

    // scalar_kernel_func
    // kernel_func

    template <typename _T>
    std::string
    custom_kernel_func(const _T& e);

    template <typename _T, typename _R>
    std::string
    custom_kernel_func(
        const expr<dop::ck_func<dvec<_T> >, impl::ck_body, _R>& e);

    template <typename _T, typename _S>
    expr<dop::ck_func<dvec<_T> >, impl::ck_body, _S>
    custom_kernel(const std::string& name,
                  const std::string& body,
                  const _S& r);

    template <typename _T, typename _S1, typename _S2>
    expr<dop::ck_func<dvec<_T> >,
         impl::ck_body,
         expr<dop::ck_komma<_T>, _S1, _S2> >
    custom_kernel(const std::string& name,
                  const std::string& body,
                  const _S1& a0,
                  const _S2& a1);
#if 0
    template <typename _T, typename _S1, typename _S2>
    expr<dop::ck_func<dvec<_T> >,
         impl::ck_body,
         expr <dop::ck_komma<_T> , _S1, _S2> >
    custom_kernel(std::string& body);
#endif

    namespace test {
        void
        test_custom_kernel();
    }

    template <typename _T>
    class ckernel {
    public:
        template <typename _A0>
        auto
        args(const _A0& a0) {
            return a0;
        }

        template <typename _A0, typename ... _AX>
        auto
        args(const _A0& a0, const _AX& ... ax) {
            auto tail=args(ax...);
            expr<dop::ck_komma<_T>, _A0, typeof(tail)> e(a0, tail);
            return e;
        }
    };

}

template <typename _T>
std::string
ocl::
custom_kernel_func(const _T& e)
{
    static_cast<void>(e);
    return std::string();
}

template <typename _T, typename _R>
std::string
ocl::
custom_kernel_func(
    const expr<dop::ck_func< dvec<_T> >, impl::ck_body, _R>& e)
{
    static_cast<void>(e);
    unsigned arg_num=1;
    std::string arg0=spaces(4) + "ulong n,\n";
    std::string args=decl_func_args(e._r, arg_num);
    args = arg0 + args;
    std::ostringstream s;
    while (args.size() > 0 &&
           (args.back() == '\n' || args.back() == ',')) {
        args.pop_back();
    }
    const impl::ck_body& ck=e._l;
    s << "// custom_kernel: \n";
    s << be::type_2_name<_T>::v() << " " << ck.name() << "\n(\n"
      << args << "\n)\n{\n";
    s << ck.body() << "\n}\n";
    s << "// eval body: \n";
    arg_num=1;
    s << ck.name() << "pa->n, " << eval_ops(e._r, arg_num) << "\n";
    return s.str();
}

template <typename _T, typename _S>
ocl::expr<ocl::dop::ck_func<ocl::dvec<_T> >, ocl::impl::ck_body, _S>
ocl::custom_kernel(const std::string& fn,
                   const std::string& body,
                   const _S& r)
{
    const impl::ck_body ck(fn, body);
    expr<dop::ck_func<dvec<_T> >, impl::ck_body, _S> e(ck, r);
    return e;
}

template <typename _T, typename _S1, typename _S2>
ocl::expr<ocl::dop::ck_func<ocl::dvec<_T> >,
          ocl::impl::ck_body,
          ocl::expr<ocl::dop::ck_komma<_T>, _S1, _S2> >
ocl::custom_kernel(const std::string& fn,
                   const std::string& body,
                   const _S1& a0,
                   const _S2& a1)
{
    expr<dop::ck_komma<_T>, _S1, _S2> e(a0, a1);
    return custom_kernel<_T>(fn, body, e);
}

void
ocl::test::test_custom_kernel()
{
    ckernel<float> x;
    float v1=1.2f;
    dvec<float> v2(8, 2.4f); dvec<float> v3(8, 2.5f);
    auto ags=x.args(v2, v3, v1);
    std::cout << be::demangle(typeid(ags).name()) << "\n";
    std::cout << be::demangle(typeid(ags._l).name()) << "\n";

    const auto t1=v1+v2-v3;
    std::cout << be::demangle(typeid(t1).name()) << "\n";
    arg_info_list va;
    arg_info_state st;
    dvec<float> vr;
    // get_arg_info(va, st, size_t(0));
    get_arg_info(va, st, vr, false);
    get_arg_info(va, st, t1, true);
    for (size_t i=0; i<va.size(); ++i) {
        const auto& ai=va[i];
        std::cout << "arg " << i << " ----------------------\n";
        std::cout << ai._arg_num << '\n' << ai._buf_num << '\n'
                  << ai._scalar_num << '\n' << ai._type << '\n'
                  << ai._kernel_arg_type << '\n' << ai._func_arg_type
                  << '\n';
    }
    std::cout << gen_kernel("k_x", vr, t1);

#if 0
    // dvec<float> v1(8, 1.2f);
    float v1=1.2f;
    dvec<float> v2(8, 2.4f);
    auto k0=custom_kernel<float>(
        "k0",
        "return v1 > 0.0f ? -v1 : v1;", v1);
    std::cout << custom_kernel_func(k0);
    std::cout << be::demangle(typeid(k0).name()) << "\n";

    auto k1=custom_kernel<float>(
        "k1",
        "return v1 > 0.0f ? v1 : -v1;", v1+v2);
    std::cout << custom_kernel_func(k1);
    auto k2=custom_kernel<float>(
        "k2",
        "return v1 + v2", v1, v2);
    std::cout << custom_kernel_func(k2);
    std::cout << be::demangle(typeid(k2).name()) << "\n";
#endif
}

int main()
{
    ocl::test::test_custom_kernel();
}
