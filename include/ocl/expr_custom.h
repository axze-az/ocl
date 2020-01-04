#if !defined (__OCL_EXPR_CUSTOM_H__)
#define __OCL_EXPR_CUSTOM_H__ 1

#include <ocl/config.h>
#include <ocl/expr.h>
#include <ocl/be/data.h>

namespace ocl {

    namespace impl {

        // body of a custom function consisting of name and body
        class __cf_body {
            std::string _name;
            std::string _body;
        public:
            __cf_body(const std::string& n,
                      const std::string& b,
                      std::size_t s)
                : _name(n), _body(b) {}
            __cf_body(const std::string& n,
                      const std::string& b)
                : _name(n), _body(b) {}
            const std::string& name() const { return _name; }
            const std::string& body() const { return _body; }
        };

        // allow encapsulation into expressions: 
        using cf_body = ignored_arg<__cf_body>;
        
        // body of a custom kernel consisting of name, body and
        // size if wanted
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

        // allow encapsulation into expressions: 
        using ck_body = ignored_arg<__ck_body>;


    }

    namespace dop {

        // custom kernel marker
        template <typename _T>
        struct custom_k {
        };

        // argument for a custom kernel or a custom functions
        template <class _T>
        struct custom_arg {
        };

        // custom function marker
        template <typename _T>
        struct custom_f {
        };
    }

    namespace impl {

        // read argument for custom functions and kernels
        template <typename _T, typename _A0>
        const _A0&
        custom_args(_A0&& a0) {
            return a0;
        }

        // read arguments for custom functions and kernels
        template <typename _T, typename _A0, typename ... _AX>
        auto
        custom_args(_A0&& a0, _AX&& ... ax)
        {
            return make_expr<dop::custom_arg<_T> >(
                custom_args<_T>(std::forward<_A0&&>(a0)),
                custom_args<_T>(std::forward<_AX&&>(ax) ...));
        }
    }
    
    // overload required to read out the size from a ck_body object
    template <typename _T, typename _R>
    std::size_t
    eval_size(const expr<dop::custom_k<_T>, impl::ck_body, _R>& r);

    // overload to read out the backend data from a custom kernel
    template <typename _T, typename _R>
    be::data_ptr
    backend_data(const expr<dop::custom_k<_T>, impl::ck_body, _R>& r);
    
}

template <typename _T, typename _R>
std::size_t
ocl::eval_size(const expr<dop::custom_k<_T>, impl::ck_body, _R>& r)
{
    if (r._l.size().has_value()) {
        return r._l.size().value();
    }
    std::size_t rs=eval_size(r._r);
    return rs;
}

// overload to read out the backend data from a custom kernel
template <typename _T, typename _R>
ocl::be::data_ptr
ocl::backend_data(const expr<dop::custom_k<_T>, impl::ck_body, _R>& r)
{
    be::data_ptr p=backend_data(r._r);
    if (p==nullptr) {
        p=be::data::instance();
    }
    return p;
}

// local variables:
// mode: c++
// end:
#endif
