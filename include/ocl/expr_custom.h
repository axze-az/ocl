#if !defined (__OCL_EXPR_CUSTOM_H__)
#define __OCL_EXPR_CUSTOM_H__ 1

#include <ocl/config.h>
#include <ocl/expr.h>
#include <ocl/be/data.h>
#include <ocl/be/kernel_functions.h>
#include <ocl/be/type_2_name.h>
#include <ocl/be/devices.h>

namespace ocl {

    // determines the amount of local memory per workitem
    template <typename _T>
    class local_mem_per_workitem {
        unsigned _e;
    public:
        local_mem_per_workitem(unsigned e) : _e(e) {};
        unsigned elements() const { return _e; }
        unsigned bytes() const { return _e * sizeof(_T); }
    };

    namespace impl {

        // body of a custom function consisting of name and body
        class __cf_body {
            std::string _name;
            std::string _body;
        public:
            __cf_body(const std::string& n,
                      const std::string& b);
            __cf_body(const __cf_body& r);
            __cf_body(__cf_body&& r);
            __cf_body&
            operator=(const __cf_body& r);
            __cf_body&
            operator=(__cf_body&& r);
            ~__cf_body();
            const std::string& name() const { return _name; }
            const std::string& body() const { return _body; }
        };

        // allow encapsulation into expressions:
        using cf_body = ignored_arg<__cf_body>;

        // body of a custom kernel consisting of name, body and
        // size if wanted
        class __ck_body : private __cf_body {
            using base_type = __cf_body;
            std::optional<std::size_t> _s;
        public:
            __ck_body(const std::string& n,
                      const std::string& b,
                      std::size_t s);
            __ck_body(const std::string& n,
                      const std::string& b);
            __ck_body(const __ck_body& r);
            __ck_body(__ck_body&& r);
            __ck_body&
            operator=(const __ck_body& r);
            __ck_body&
            operator=(__ck_body&& r);
            ~__ck_body();
            using base_type::name;
            using base_type::body;
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
    def_custom_func(be::kernel_functions& fnames, const _T& l);

    // overload for any expressions
    template <typename _OP, typename _L, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<_OP, _L, _R>& e);

    // overload for expressions with only one argument
    template <typename _OP, typename _L>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<_OP, _L, void>& e);

    // overload for custom functions
    template <typename _OP, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
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

    // decl_non_buffer_args specialized for dvecs
    template <typename _T>
    std::string
    decl_non_buffer_args(const local_mem_per_workitem<_T>& p,
                         unsigned& arg_num);

    // declare buffer arguments, must increment arg_num if something
    // generated
    template <typename _T>
    std::string
    decl_buffer_args(const local_mem_per_workitem<_T>& p,
                     unsigned& arg_num, bool read_only);

    namespace impl {
        std::string
        decl_buffer_args_local_mem(const std::string_view& tname,
                                   unsigned& arg_num,
                                   bool ro);
        std::string
        decl_buffer_args_local_mem(const char* tname,
                                   unsigned& arg_num,
                                   bool ro);
        std::string
        decl_buffer_args_local_mem(const std::string& tname,
                                   unsigned& arg_num,
                                   bool ro);
    }

    // bind non buffer arguments
    template <typename _T>
    void
    bind_non_buffer_args(const local_mem_per_workitem<_T>& p,
                         be::argument_buffer& a);

    // bind buffer arguments
    template <typename _T>
    void
    bind_buffer_args(const local_mem_per_workitem<_T>& p,
                     unsigned& buf_num,
                     be::kernel& k, unsigned wgs);

    namespace impl {
        void
        bind_buffer_args_local_mem(const std::string_view& tname,
                                   size_t bytes,
                                   size_t elements,
                                   size_t wg_size,
                                   unsigned& buf_num,
                                   be::kernel& k);

        void
        bind_buffer_args_local_mem(const char* tname,
                                   size_t bytes,
                                   size_t elements,
                                   size_t wg_size,
                                   unsigned& buf_num,
                                   be::kernel& k);

        void
        bind_buffer_args_local_mem(const std::string& tname,
                                   size_t bytes,
                                   size_t elements,
                                   size_t wg_size,
                                   unsigned& buf_num,
                                   be::kernel& k);

    }

    // concat_args specialized for local memory
    template <typename _T>
    std::string
    concat_args(const local_mem_per_workitem<_T>& p,
                var_counters& c);

    namespace impl {
        std::string
        concat_args_local_mem(var_counters& s);
    }
}

template <typename _T>
std::string
ocl::def_custom_func(be::kernel_functions& fnames, const _T& l)
{
    static_cast<void>(fnames);
    static_cast<void>(l);
    return std::string();
}

template <typename _OP, typename _L, typename _R>
std::string
ocl::def_custom_func(be::kernel_functions& fnames,
                     const expr<_OP, _L, _R>& e)
{
    std::string l=def_custom_func(fnames, e._l);
    std::string r=def_custom_func(fnames, e._r);
    return l+r;
}

template <typename _OP, typename _L>
std::string
ocl::def_custom_func(be::kernel_functions& fnames,
                     const expr<_OP, _L, void>& e)
{
    return def_custom_func(fnames, e._l);
}

template <typename _OP, typename _R>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e )
{
    const std::string& fn=e._l.name();
    std::string s;
    if (fnames.insert(fn) == true) {
        s = e._l.body() + '\n';
    }
    return s;
}

template <typename _OP, typename _R>
std::string
ocl::
eval_ops(const expr<dop::custom_f<_OP>, impl::cf_body, _R>& e,
         unsigned& arg_num)
{
    std::string s= e._l.name() + '(';
    s += eval_ops(e._r, arg_num);
    s += ')';
    return s;
};

template <typename _OP, typename _L, typename _R>
std::string
ocl::
eval_ops(const expr<dop::custom_arg<_OP>, _L, _R>& e,
         unsigned& arg_num)
{
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

template <typename _T>
std::string
ocl::decl_non_buffer_args(const local_mem_per_workitem<_T>& p,
                          unsigned& arg_num)
{
    static_cast<void>(p);
    static_cast<void>(arg_num);
    return std::string();
}

template <typename _T>
std::string
ocl::decl_buffer_args(const local_mem_per_workitem<_T>& p,
                      unsigned& arg_num, bool read_only)
{
    return impl::decl_buffer_args_local_mem(be::type_2_name<_T>::v(),
                                            arg_num, read_only);
}

template <typename _T>
void
ocl::bind_buffer_args(const local_mem_per_workitem<_T>& p,
                      unsigned& buf_num,
                      be::kernel& k, unsigned wgs)
{
    impl::bind_buffer_args_local_mem(be::type_2_name<_T>::v(),
                                     p.bytes()*wgs,
                                     p.elements(), wgs,
                                     buf_num, k);
}

template <typename _T>
void
ocl::bind_non_buffer_args(const local_mem_per_workitem<_T>& t,
                          be::argument_buffer& a)
{
    static_cast<void>(t);
    static_cast<void>(a);
}

template <typename _T>
std::string
ocl::concat_args(const local_mem_per_workitem<_T>& r, var_counters& c)
{
    static_cast<void>(r);
    return impl::concat_args_local_mem(c);
}

// local variables:
// mode: c++
// end:
#endif
