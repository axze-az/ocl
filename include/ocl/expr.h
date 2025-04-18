//
// Copyright (C) 2010-2025 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#if !defined (__OCL_EXPR_H__)
#define __OCL_EXPR_H__ 1

#include <ocl/config.h>
#include <ocl/types.h>
#include <ocl/be/data.h>
#include <ocl/be/type_2_name.h>
#include <ocl/be/argument_buffer.h>
#include <iostream>
#include <sstream>
#include <thread>
#include <string_view>

namespace ocl {

    // default expression traits for simple/unspecialized
    // types
    template <class _T>
    struct expr_traits {
        using type = const _T;
    };

    // the expression template
    template <class _OP, class _L, class _R>
    struct expr {
        typename expr_traits<_L>::type _l;
        typename expr_traits<_R>::type _r;
        constexpr expr(const _L& l, const _R& r);
    };

    template <class _OP, class _L>
    struct expr<_OP, _L, void> {
        typename expr_traits<_L>::type _l;
        constexpr expr(const _L& l) : _l{l} {};
    };

    template <class _OP, class _L, class _R>
    expr<_OP, _L, _R>
    make_expr(const _L& l, const _R& r) {
        return expr<_OP, _L, _R>(l, r);
    }

    template <class _OP, class _L>
    expr<_OP, _L, void>
    make_expr(const _L& l) {
        return expr<_OP, _L, void>(l);
    }

    namespace impl {
        // a template to allow encapsulation of payload into expressions
        template <typename _T>
        struct ignored_arg : public _T {
            using _T::_T;
        };

        template <typename _T, size_t _N>
        struct array_ptr {
            const _T* _p;
            constexpr
            array_ptr(const _T(&r)[_N]) : _p(r) {}
        };
    }

    // specialization of expr_traits for arrays
    template <typename _T, size_t _N>
    struct expr_traits<_T[_N]> {
        using type = impl::array_ptr<_T, _N>;
    };

    struct spaces : public std::string {
        spaces(unsigned int n) : std::string(n, ' ') {}
    };
    // return back end data:
    template <class _T>
    be::data_ptr backend_data(const _T& t);
    // evaluate the size of an expression
    template <class _T>
    constexpr
    std::size_t eval_size(const _T& t);

    // generate the declarations for the members of the structure
    // with the scalar arguments, must increment arg_num if something
    // was generated
    template <typename _T>
    std::string
    decl_non_buffer_args(const _T& p, unsigned& arg_num);

    namespace impl {
        std::string
        decl_non_buffer_args_t(const std::string_view& tname,
                               size_t alignment,
                               unsigned& arg_num);
        std::string
        decl_non_buffer_args_t(const char* tname,
                               size_t alignment,
                               unsigned& arg_num);
        std::string
        decl_non_buffer_args_t(const std::string& tname,
                               size_t alignment,
                               unsigned& arg_num);
    }

    // generate the declarations for the member of the array
    template <typename _T, size_t _N>
    std::string
    decl_non_buffer_args(const impl::array_ptr<_T, _N>& r, unsigned& arg_num);

    namespace impl {
        std::string
        decl_non_buffer_args_array_ptr(const std::string_view& tname,
                                       size_t n,
                                       size_t alignment,
                                       unsigned& arg_num);
        std::string
        decl_non_buffer_args_array_ptr(const char* tname,
                                       size_t n,
                                       size_t alignment,
                                       unsigned& arg_num);
        std::string
        decl_non_buffer_args_array_ptr(const std::string& tname,
                                       size_t n,
                                       size_t alignment,
                                       unsigned& arg_num);
    }

    // ignored_arg are really ignored
    template <typename _T>
    std::string
    decl_non_buffer_args(const impl::ignored_arg<_T>& r, unsigned& arg_num);

    // declare buffer arguments, must increment arg_num if something
    // generated
    template <typename _T>
    std::string
    decl_buffer_args(const _T& p, unsigned& arg_num, bool read_only);

    struct var_counters {
        unsigned _var_num;
        unsigned _buf_num;
        unsigned _scalar_num;
        var_counters() = default;
        var_counters(unsigned buf_offs)
            : _var_num(buf_offs), _buf_num(buf_offs), _scalar_num(0) {};
    };

    template <typename _T>
    std::string
    concat_args(const _T& r, var_counters& c);

    namespace impl {
        std::string
        concat_args_t(var_counters& c);
    }

    // bind non buffer arguments
    template <typename _T>
    void
    bind_non_buffer_args(const _T& t, be::argument_buffer& a);

    // bind non buffer arguments
    template <typename _T, size_t _N>
    void
    bind_non_buffer_args(const impl::array_ptr<_T, _N>& t,
                         be::argument_buffer& a);

    // bind ignored arguments
    template <typename _T>
    void
    bind_non_buffer_args(const impl::ignored_arg<_T>& t,
                         be::argument_buffer& a);

    // bind buffer arguments
    template <typename _T>
    void
    bind_buffer_args(const _T& t, unsigned& buf_num,
                     be::kernel& k, unsigned wgs);

    // eval_args, returns the opencl source code
    // for all arguments of generated from r,
    // increments arg_num.
    template <class _T>
    std::string
    eval_args(const _T& r, unsigned& arg_num, bool ro);

    namespace impl {
        std::string
        eval_args_t(const std::string_view& tname,
                    unsigned& arg_num,
                    bool ro);
        std::string
        eval_args_t(const char* tname,
                    unsigned& arg_num,
                    bool ro);
        std::string
        eval_args_t(const std::string& tname,
                    unsigned& arg_num,
                    bool ro);
    }

    // eval_args for ignored args
    template <class _T>
    std::string
    eval_args(const impl::ignored_arg<_T>& t, unsigned& arg_num, bool ro);

    // eval_args for array_ptr<_T, _N>
    template <class _T, size_t _N>
    std::string
    eval_args(const impl::array_ptr<_T, _N>& t,unsigned& arg_num, bool ro);

    namespace impl {
        std::string
        eval_args_array_ptr(const std::string_view& tname,
                            unsigned& arg_num,
                            bool ro);
        std::string
        eval_args_array_ptr(const char* tname,
                            unsigned& arg_num,
                            bool ro);
        std::string
        eval_args_array_ptr(const std::string& tname,
                            unsigned& arg_num,
                            bool ro);
    }

    // eval_vars
    template <class _T>
    std::string
    eval_vars(const _T& r, unsigned& arg_num, bool read);

    namespace impl {
        std::string
        eval_vars_t(const std::string_view& tname,
                    unsigned& arg_num,
                    bool ro);
        std::string
        eval_vars_t(const char* tname,
                    unsigned& arg_num,
                    bool ro);

        std::string
        eval_vars_t(const std::string& tname,
                    unsigned& arg_num,
                    bool ro);
    }

    // eval_vars
    template <class _T, size_t _N>
    std::string
    eval_vars(const impl::array_ptr<_T, _N>& r,
              unsigned& arg_num, bool read);

    namespace impl {
        std::string
        eval_vars_array_ptr(const std::string_view& tname,
                            unsigned& arg_num,
                            bool ro);


        std::string
        eval_vars_array_ptr(const char* tname,
                            unsigned& arg_num,
                            bool ro);

        std::string
        eval_vars_array_ptr(const std::string& tname,
                            unsigned& arg_num,
                            bool ro);
    }

    // eval_ops
    template <class _T>
    std::string eval_ops(const _T& r, unsigned& arg_num);

    namespace impl {
        std::string
        eval_ops_t(unsigned& arg_num);
    }

    // eval_results, unimplemented
    template <class _T>
    std::string eval_results(_T& r, unsigned& res_num);

    // backend_data specialized for expr
    template <class _OP, class _L, class _R>
    be::data_ptr backend_data(const expr<_OP, _L, _R>& a);
    // and for unary expressions
    template <class _OP, class _L>
    be::data_ptr backend_data(const expr<_OP, _L, void>& a);

    // eval_size specialized for expr<>
    template <class _OP, class _L, class _R>
    std::size_t eval_size(const expr<_OP, _L, _R>& a);

    template <class _OP, class _L>
    std::size_t eval_size(const expr<_OP, _L, void>& a);

    // decl_non_buffer_args specialized for expressions
    template <class _OP, class _L, class _R>
    std::string
    decl_non_buffer_args(const expr<_OP, _L, _R>& r,
                         unsigned& arg_num);

    template <class _OP, class _L>
    std::string
    decl_non_buffer_args(const expr<_OP, _L, void>& r,
                         unsigned& arg_num);

    // decl_non_buffer_args specialized for expressions
    template <class _OP, class _L, class _R>
    std::string
    decl_buffer_args(const expr<_OP, _L, _R>& r,
                     unsigned& arg_num, bool read_only);

    template <class _OP, class _L>
    std::string
    decl_buffer_args(const expr<_OP, _L, void>& r,
                     unsigned& arg_num, bool read_only);

    template <typename _T>
    std::string
    concat_args(const impl::ignored_arg<_T>& r, var_counters& c) {
        return std::string();
    }

    template <class _OP, class _L, class _R>
    std::string
    concat_args(const expr<_OP, _L, _R>& e, var_counters& c);

    template <class _OP, class _L>
    std::string
    concat_args(const expr<_OP, _L, void>& e, var_counters& c);

    // bind_non_buffer_args specialized for expr<>
    template <class _OP, class _L, class _R>
    void
    bind_non_buffer_args(const expr<_OP, _L, _R>& e,
                         be::argument_buffer& a);
    template <class _OP, class _L>
    void
    bind_non_buffer_args(const expr<_OP, _L, void>& e,
                         be::argument_buffer& a);

    // bind_buffer_args specialized for expr<>
    template <class _OP, class _L, class _R>
    void
    bind_buffer_args(const expr<_OP, _L, _R>& r,
                     unsigned& buf_num,
                     be::kernel& k,
                     unsigned wgs);
    template <class _OP, class _L>
    void
    bind_buffer_args(const expr<_OP, _L, void>& r,
                     unsigned& buf_num,
                     be::kernel& k,
                     unsigned wgs);

    // eval_args specialized for expr<>
    template <class _OP, class _L, class _R>
    std::string
    eval_args(const expr<_OP, _L, _R>& r,
              unsigned& arg_num,
              bool ro);

    template <class _OP, class _L>
    std::string
    eval_args(const expr<_OP, _L, void>& r,
              unsigned& arg_num,
              bool ro);

    // eval_vars specialized for expr<>
    template <class _OP, class _L, class _R>
    std::string
    eval_vars(const expr<_OP, _L, _R>& a, unsigned& arg_num,
              bool read);

    template <class _OP, class _L>
    std::string
    eval_vars(const expr<_OP, _L, void>& a, unsigned& arg_num,
              bool read);

    // eval_ops specialized for expr<>
    template <class _OP, class _L, class _R>
    std::string
    eval_ops(const expr<_OP, _L, _R>& a, unsigned& arg_num);

    template <class _OP, class _L>
    std::string
    eval_ops(const expr<_OP, _L, void>& a, unsigned& arg_num);

}

template <class _T>
inline
ocl::be::data_ptr
ocl::backend_data(const _T& t)
{
    static_cast<void>(t);
    return nullptr;
}

template <class _T>
inline
constexpr
std::size_t
ocl::eval_size(const _T& t)
{
    static_cast<void>(t);
    return 1;
}


template <class _T>
std::string
ocl::decl_non_buffer_args(const _T& r, unsigned& arg_num)
{
    static_cast<void>(r);
    return impl::decl_non_buffer_args_t(be::type_2_name<_T>::v(),
                                        alignof(_T),
                                        arg_num);
}

template <class _T, std::size_t _N>
std::string
ocl::
decl_non_buffer_args(const impl::array_ptr<_T, _N>& r, unsigned& arg_num)
{
    static_cast<void>(r);
    return impl::decl_non_buffer_args_array_ptr(be::type_2_name<_T>::v(),
                                                _N, alignof(_T),
                                                arg_num);
}

template <class _T>
std::string
ocl::decl_non_buffer_args(const impl::ignored_arg<_T>& r, unsigned& arg_num)
{
    static_cast<void>(r);
    static_cast<void>(arg_num);
    return std::string();
}

template <class _T>
std::string
ocl::decl_buffer_args(const _T& r, unsigned& arg_num, bool read_only)
{
    static_cast<void>(r);
    static_cast<void>(arg_num);
    static_cast<void>(read_only);
    return std::string();
}

template <class _T>
std::string
ocl::concat_args(const _T& r, var_counters& c)
{
    static_cast<void>(r);
    return impl::concat_args_t(c);
}

template <typename _T>
void
ocl::
bind_non_buffer_args(const _T& t, be::argument_buffer& a)
{
    a.insert(t);
}

template <typename _T, std::size_t _N>
void
ocl::
bind_non_buffer_args(const impl::array_ptr<_T, _N>& t,
                     be::argument_buffer& a)
{
    const _T* p=t._p;
    for (size_t i=0; i<_N; ++i) {
        a.insert(p[i]);
    }
}

template <typename _T>
void
ocl::
bind_non_buffer_args(const impl::ignored_arg<_T>& t, be::argument_buffer& a)
{
    static_cast<void>(t);
    static_cast<void>(a);
}

template <typename _T>
void
ocl::bind_buffer_args(const _T& t, unsigned& buf_num,
                      be::kernel& k, unsigned wgs)
{
    static_cast<void>(t);
    static_cast<void>(buf_num);
    static_cast<void>(k);
    static_cast<void>(wgs);
}

template <class _T>
std::string
ocl::
eval_args(const _T& r, unsigned& arg_num, bool ro)
{
    static_cast<void>(r);
    return impl::eval_args_t(be::type_2_name<_T>::v(),
                             arg_num, ro);
}

template <class _T>
std::string
ocl::
eval_args(const impl::ignored_arg<_T>& t, unsigned& arg_num, bool ro)
{
    static_cast<void>(t);
    static_cast<void>(arg_num);
    static_cast<void>(ro);
    return std::string();
}

template <class _T, std::size_t _N>
std::string
ocl::
eval_args(const impl::array_ptr<_T, _N>& r, unsigned& arg_num, bool ro)
{
    static_cast<void>(r);
    return impl::eval_args_array_ptr(be::type_2_name<_T>::v(),
                                     arg_num, ro);
}

template <class _T>
std::string ocl::eval_vars(const _T& r, unsigned& arg_num,
                           bool read)
{
    static_cast<void>(r);
    return impl::eval_vars_t(be::type_2_name<_T>::v(),
                             arg_num, read);
}

template <class _T, std::size_t _N>
std::string
ocl::
eval_vars(const impl::array_ptr<_T, _N>& r, unsigned& arg_num,
          bool read)
{
    static_cast<void>(r);
    return impl::eval_vars_array_ptr(be::type_2_name<_T>::v(),
                                     arg_num, read);
}

template <class _T>
std::string ocl::eval_ops(const _T& r, unsigned& arg_num)
{
    static_cast<void>(r);
    return impl::eval_ops_t(arg_num);
}

template <class _OP, class _L, class _R>
inline
constexpr
ocl::expr<_OP, _L, _R>::expr(const _L& l, const _R& r)
    :  _l(l), _r(r)
{
}

template <class _OP, class _L, class _R>
ocl::be::data_ptr
ocl::backend_data(const expr<_OP, _L, _R>& a)
{
    be::data_ptr pl=backend_data(a._l);
    if (pl != nullptr) {
        return pl;
    }
    return backend_data(a._r);
}

template <class _OP, class _L>
ocl::be::data_ptr
ocl::backend_data(const expr<_OP, _L, void>& a)
{
    return backend_data(a._l);
}

template <class _OP, class _L, class _R>
std::size_t ocl::eval_size(const expr<_OP, _L, _R>& a)
{
    std::size_t l=eval_size(a._l);
    std::size_t r=eval_size(a._r);
    return std::max(l, r);
}

template <class _OP, class _L>
std::size_t ocl::eval_size(const expr<_OP, _L, void>& a)
{
    std::size_t l=eval_size(a._l);
    return l;
}

template <class _OP, class _L, class _R>
std::string
ocl::
decl_non_buffer_args(const expr<_OP, _L, _R>& e, unsigned& arg_num)
{
    std::string l=decl_non_buffer_args(e._l, arg_num);
    std::string r=decl_non_buffer_args(e._r, arg_num);
    return l+r;
}

template <class _OP, class _L>
std::string
ocl::
decl_non_buffer_args(const expr<_OP, _L, void>& e, unsigned& arg_num)
{
    return decl_non_buffer_args(e._l, arg_num);
}

template <class _OP, class _L, class _R>
std::string
ocl::
decl_buffer_args(const expr<_OP, _L, _R>& e,
                 unsigned& arg_num, bool read_only)
{
    std::string l=decl_buffer_args(e._l, arg_num, read_only);
    std::string r=decl_buffer_args(e._r, arg_num, read_only);
    return l+r;
}

template <class _OP, class _L>
std::string
ocl::
decl_buffer_args(const expr<_OP, _L, void>& e,
                 unsigned& arg_num, bool read_only)
{
    return decl_buffer_args(e._l, arg_num, read_only);
}

template <class _OP, class _L>
std::string
ocl::
concat_args(const expr<_OP, _L, void>& e, var_counters& c)
{
    return concat_args(e._l, c);
}

template <class _OP, class _L, class _R>
std::string
ocl::
concat_args(const expr<_OP, _L, _R>& e, var_counters& c)
{
    std::string l=concat_args(e._l, c);
    std::string r=concat_args(e._r, c);
    if (l.empty())
        return r;
    if (r.empty())
        return l;
    return l+", " + r;
}

template <class _OP, class _L, class _R>
std::string
ocl::eval_args(const expr<_OP, _L, _R>& e, unsigned& arg_num, bool ro)
{
    std::string l=eval_args(e._l, arg_num, ro);
    std::string r=eval_args(e._r, arg_num, ro);
    if (l.empty())
        return r;
    if (r.empty())
        return l;
    return l + ",\n" + r;
}

template <class _OP, class _L>
std::string
ocl::eval_args(const expr<_OP, _L, void>& r, unsigned& arg_num, bool ro)
{
    std::string left(eval_args(r._l, arg_num, ro));
    return left;
}

template <class _OP, class _L, class _R>
std::string
ocl::
eval_vars(const expr<_OP, _L, _R>& a, unsigned& arg_num, bool read)
{
    std::string l=eval_vars(a._l, arg_num, read);
    std::string r=eval_vars(a._r, arg_num, read);
    if (l.empty())
        return r;
    if (r.empty())
        return l;
    return l + '\n' + r;
}

template <class _OP, class _L>
std::string
ocl::
eval_vars(const expr<_OP, _L, void>& a, unsigned& arg_num, bool read)
{
    auto l=eval_vars(a._l, arg_num, read);
    return l;
}

template <class _OP, class _L, class _R>
std::string ocl::eval_ops(const expr<_OP, _L, _R>& a, unsigned& arg_num)
{
    auto l=eval_ops(a._l, arg_num);
    auto r=eval_ops(a._r, arg_num);
    std::string t(_OP::body(l, r));
    return std::string("(") + t + std::string(")");
}

template <class _OP, class _L>
std::string ocl::eval_ops(const expr<_OP, _L, void>& a, unsigned& arg_num)
{
    auto l=eval_ops(a._l, arg_num);
    std::string t(_OP::body(l));
    return std::string("(") + t + std::string(")");
}

// bind_non_buffer_args specialized for expr<>
template <class _OP, class _L, class _R>
void
ocl::bind_non_buffer_args(const expr<_OP, _L, _R>& e,
                          be::argument_buffer& a)
{
    bind_non_buffer_args(e._l, a);
    bind_non_buffer_args(e._r, a);
}

template <class _OP, class _L>
void
ocl::bind_non_buffer_args(const expr<_OP, _L, void>& e,
                          be::argument_buffer& a)
{
    bind_non_buffer_args(e._l, a);
}

// bind_buffer_args specialized for expr<>
template <class _OP, class _L, class _R>
void
ocl::bind_buffer_args(const expr<_OP, _L, _R>& e,
                      unsigned& buf_num,
                      be::kernel& k,
                      unsigned wgs)
{
    bind_buffer_args(e._l, buf_num, k, wgs);
    bind_buffer_args(e._r, buf_num, k, wgs);
}

template <class _OP, class _L>
void
ocl::bind_buffer_args(const expr<_OP, _L, void>& e,
                      unsigned& buf_num,
                      be::kernel& k,
                      unsigned wgs)
{
    bind_buffer_args(e._l, buf_num, k, wgs);
}

// Local variables:
// mode: c++
// end:
#endif
