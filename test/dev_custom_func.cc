#include "ocl/ocl.h"
#include "ocl/dvec.h"
#include "ocl/be/devices.h"
#include "ocl/be/data.h"
#include "ocl/test/tools.h"

namespace ocl {

    template <typename _T, typename _R>
    std::string
    eval_ops(const expr<dop::custom_f<_T>, impl::cf_body, _R>& e,
             unsigned& arg_num) {
        std::cout << "called\n" << std::flush;
        std::ostringstream s;
        s << e._l.name() << "(";
        s << ")";
        return s.str();
    };

    template <class _T, typename _R>
    std::string
    eval_vars(const expr<dop::custom_f<_T>, impl::cf_body, _R>& e,
              unsigned& arg_num, bool read) {

        std::cout << "called\n" << std::flush;
        return eval_vars(e._r, arg_num, read);
    };


    namespace impl {

        template <typename _T, typename _A0>
        auto
        custom_func_args(_A0&& a0) {
            return a0;
        }

        template <typename _T, typename _A0, typename _A1>
        auto
        custom_func_args(_A0&& a0, _A1&& a1) {
            return make_expr<dop::custom_arg<_T> >(
                custom_func_args<_T>(std::forward<_A0&&>(a0)),
                custom_func_args<_T>(std::forward<_A1&&>(a1)));
        }

        template <typename _T, typename _A0, typename ... _AX>
        auto
        custom_func_args(_A0&& a0, _AX&& ... ax)
        {
            return make_expr<dop::custom_arg<_T> >(
                a0,
                custom_func_args<_T>(std::forward<_AX&&>(ax) ...));
        }
    }


    template <typename _T, typename ... _AX>
    auto
    custom_func(const std::string& name,
                const std::string& body,
                _AX&... ax)
    {
        return make_expr<dop::custom_f<dvec<_T> > >(
            impl::cf_body(name, body),
            impl::custom_func_args<dvec<_T>>(
                std::forward<_AX&&>(ax) ...));
    }

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
        "float add(float v0, float v1)\n"
        "{\n"
        "    return v0 + v1;\n"
        "}\n";
    const char* fname0="add";
    auto e=custom_func<float>(fname0, fbody0, v0, v0);
    std::cout << ocl::be::demangle(typeid(e).name()) << std::endl;
    dvec<float> v1=custom_func<float>(fname0, fbody0, v0, v0);
    dump(v1, "v1: v0+v0 = 4.8f");
}

int main()
{
    ocl::test::test_custom_func();
}
