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
        template <typename _OP>
        struct custom_k {
        };

        // argument for a custom kernel or a custom functions
        template <class _OP>
        struct custom_arg {
        };

        // custom function marker
        template <typename _OP>
        struct custom_f {
        };
    }

    namespace impl {

        // read argument for custom functions and kernels
        template <typename _OP, typename _A0>
        const _A0&
        custom_args(_A0&& a0) {
            return a0;
        }

        // read arguments for custom functions and kernels
        template <typename _OP, typename _A0, typename ... _AX>
        auto
        custom_args(_A0&& a0, _AX&& ... ax)
        {
            return make_expr<dop::custom_arg<_OP> >(
                custom_args<_OP>(std::forward<_A0&&>(a0)),
                custom_args<_OP>(std::forward<_AX&&>(ax) ...));
        }
    }

    // support for custom functions: definition of the body
    template <typename _T>
    std::string
    def_custom_func(std::set<std::string>& fnames, const _T& l);

    // overload for any expressions
    template <typename _OP, typename _L, typename _R>
    std::string
    def_custom_func(std::set<std::string>& fnames,
                    const expr<_OP, _L, _R>& e);

    // overload for expressions with only one argument
    template <typename _OP, typename _L>
    std::string
    def_custom_func(std::set<std::string>& fnames,
                    const expr<_OP, _L, void>& e);
    
    // overload for custom functions
    template <typename _OP, typename _R>
    std::string
    def_custom_func(std::set<std::string>& fnames,
                    const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e );
    
    // eval_ops overload for custom functions
    template <typename _OP, typename _R>
    std::string
    eval_ops(const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e,
             unsigned& arg_num);

    // eval_ops overload for custom arguments
    template <typename _OP, typename _L, typename _R>
    std::string
    eval_ops(const expr<dop::custom_arg<_OP>, _L, _R>& e,
             unsigned& arg_num);

    // eval_vars for custom functions
    template <class _OP, typename _R>
    std::string
    eval_vars(const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e,
              unsigned& arg_num, bool read);
    
    // overload required to read out the size from a ck_body object
    template <typename _OP, typename _R>
    std::size_t
    eval_size(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r);

    // overload to read out the backend data from a custom kernel
    template <typename _OP, typename _R>
    be::data_ptr
    backend_data(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r);

}

template <typename _T>
std::string
ocl::def_custom_func(std::set<std::string>& fnames, const _T& l)
{
    static_cast<void>(fnames);
    static_cast<void>(l);
    return std::string();
}

template <typename _OP, typename _L, typename _R>
std::string
ocl::def_custom_func(std::set<std::string>& fnames,
                     const expr<_OP, _L, _R>& e)
{
    std::string l=def_custom_func(fnames, e._l);
    std::string r=def_custom_func(fnames, e._r);
    return l+r;
}

template <typename _OP, typename _L>
std::string
ocl::def_custom_func(std::set<std::string>& fnames,
                     const expr<_OP, _L, void>& e)
{
    return def_custom_func(fnames, e._l);
}

template <typename _OP, typename _R>
std::string
ocl::
def_custom_func(std::set<std::string>& fnames,
                const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e )
{
    const std::string& fn=e._l.name();
    std::string s;
    if (fnames.find(fn) == fnames.end()) {
        s = e._l.body() + '\n';
        fnames.insert(fn);
    }
    return s;
}
    
template <typename _OP, typename _R>
std::string
ocl::
eval_ops(const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e,
         unsigned& arg_num)
{
    std::ostringstream s;
    s << e._l.name() << "(";
    s << eval_ops(e._r, arg_num);
    s << ")";
    return s.str();
};

template <typename _OP, typename _L, typename _R>
std::string
ocl::
eval_ops(const expr<dop::custom_arg<_OP>, _L, _R>& e,
         unsigned& arg_num) {
    std::string l=eval_ops(e._l, arg_num);
    std::string r=eval_ops(e._r, arg_num);
    return l + ", " + r;
};
    
template <class _OP, typename _R>
std::string
ocl::
eval_vars(const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e,
          unsigned& arg_num, bool read)
{
    return eval_vars(e._r, arg_num, read);
};

template <typename _OP, typename _R>
std::size_t
ocl::
eval_size(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r)
{
    if (r._l.size().has_value()) {
        return r._l.size().value();
    }
    std::size_t rs=eval_size(r._r);
    return rs;
}

// overload to read out the backend data from a custom kernel
template <typename _OP, typename _R>
ocl::be::data_ptr
ocl::
backend_data(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r)
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
