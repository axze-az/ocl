#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"

namespace ocl {
    namespace impl {
        using ck_body = std::string;
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
    decl_func_args(const dvec<_T>& t, unsigned& var_num) {
        static_cast<void>(t);
        std::ostringstream s;
        s << be::type_2_name<_T>::v() << " v" << var_num << ",\n";
        ++var_num;
        return spaces(4)+s.str();
    }

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
            struct komma { const char* operator()() const { return ","; }};
        }
        template <class _T>
        struct ck_komma : public binary_func<names::komma, true> {};
    }

    template <typename _T>
    std::string
    custom_kernel_func(const _T& e);

    template <typename _T, typename _R>
    std::string
    custom_kernel_func(
        const expr<dop::ck_func<dvec<_T> >, impl::ck_body, _R>& e);

    template <typename _T, typename _S>
    expr<dop::ck_func<dvec<_T> >, impl::ck_body, _S>
    custom_kernel(const std::string& body, const _S& r);

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
    std::string args=decl_func_args(e._r, arg_num);
    std::ostringstream s;
    while (args.size() > 0 &&
           (args.back() == '\n' || args.back() == ',')) {
        args.pop_back();
    }
    s << "// custom_kernel_func: \n";
    s << be::type_2_name<_T>::v() << " f\n(\n" << args << "\n)\n{\n";
    s << e._l << "\n}\n";
    s << "// eval body: \n";
    arg_num=1;
    s << "f(" << eval_ops(e._r, arg_num) << ")\n";
    return s.str();
}

template <typename _T, typename _S>
ocl::expr<ocl::dop::ck_func<ocl::dvec<_T> >, ocl::impl::ck_body, _S>
ocl::custom_kernel(const std::string& body, const _S& r)
{
    expr<dop::ck_func<dvec<_T> >, impl::ck_body, _S> e(body, r);
    return e;
}

void
ocl::test::test_custom_kernel()
{
    // dvec<float> v1(8, 1.2f);
    float v1=1.2f;
    dvec<float> v2(8, 2.4f);
    auto k0=custom_kernel<float>(
        "return v1 > 0.0f ? -v1 : v1;", v1);
    std::cout << custom_kernel_func(k0);
    std::cout << be::demangle(typeid(k0).name()) << "\n";

    auto k1=custom_kernel<float>(
        "return v1 > 0.0f ? v1 : -v1;", v1+v2);
    std::cout << custom_kernel_func(k1);
    auto k2=custom_kernel<float>(
        "return v1 > 0.0f ? 2.0f*v1 : -v1;", v1+v1);
    std::cout << custom_kernel_func(k2);
}

int main()
{
    ocl::test::test_custom_kernel();
}
