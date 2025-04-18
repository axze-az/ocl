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
#if !defined (__OCL_DVEC_OP_H__)
#define __OCL_DVEC_OP_H__ 1

#include <ocl/config.h>
#include <ocl/dvec_t.h>

namespace ocl {

    // backend_data specialized for dvec t
    template <class _T>
    be::data_ptr
    backend_data(const dvec<_T>& t);
    // eval_size specialized for dvec
    template <class _T>
    std::size_t eval_size(const dvec<_T>& t);
    // decl_non_nuffer_args specialized for be::buffer
    std::string
    decl_non_buffer_args(const be::buffer& b, unsigned& arg_num);
    // decl_non_buffer_args specialized for dvecs
    template <typename _T>
    std::string
    decl_non_buffer_args(const dvec<_T>& t, unsigned& arg_num);

    // decl_buffer_args specialized for dvecs
    template <typename _T>
    std::string
    decl_buffer_args(const dvec<_T>& t, unsigned& arg_num, bool ro);

    namespace impl {
        std::string
        decl_buffer_args_dvec_t(const std::string_view& tname,
                                unsigned& arg_num,
                                bool ro);
        std::string
        decl_buffer_args_dvec_t(const char* tname,
                                unsigned& arg_num,
                                bool ro);
        std::string
        decl_buffer_args_dvec_t(const std::string& tname,
                                unsigned& arg_num,
                                bool ro);
    }

    // concat_args specialized for dvecs
    template <typename _T>
    std::string
    concat_args(const dvec<_T>& t, var_counters& c);

    namespace impl {
        std::string
        concat_args_dvec_t(var_counters& c);
    }

    // bind non buffer arguments
    template <typename _T>
    void
    bind_non_buffer_args(const dvec<_T>& t, be::argument_buffer& a);

    // bind buffer arguments to kernel with workgroup size wgs
    template <typename _T>
    void
    bind_buffer_args(const dvec<_T>& t, unsigned& buf_num,
                     be::kernel& k, unsigned wgs);

    // bind buffer arguments
    template <typename _T>
    void
    bind_buffer_args(dvec<_T>& t, unsigned& buf_num,
                     be::kernel& k, unsigned wgs);

    namespace impl {
        void
        bind_buffer_args_dvec_t(const dvec_base& r,
                                const std::string_view& tname,
                                unsigned& buf_num,
                                be::kernel& k,
                                bool is_const,
                                size_t elements);
        void
        bind_buffer_args_dvec_t(const dvec_base& r,
                                const char* tname,
                                unsigned& buf_num,
                                be::kernel& k,
                                bool ist_const,
                                size_t elements);
        void
        bind_buffer_args_dvec_t(const dvec_base& r,
                                const std::string& tname,
                                unsigned& buf_num,
                                be::kernel& k,
                                bool is_const,
                                size_t elements);
    }

    // store results
    template <typename _T>
    std::string
    store_result(dvec<_T>& t, var_counters& c);

    namespace impl {
        std::string
        store_result_dvec_t(var_counters& c);
    }

    // eval_args specialized for dvec
    template <class _T>
    std::string
    eval_args(const dvec<_T>& r,
              unsigned& arg_num, bool ro);

    namespace impl {
        std::string
        eval_args_dvec_t(const std::string_view& tname,
                         unsigned& arg_num,
                         bool ro);
        std::string
        eval_args_dvec_t(const char* tname,
                         unsigned& arg_num,
                         bool ro);
        std::string
        eval_args_dvec_t(const std::string& tname,
                         unsigned& arg_num,
                         bool ro);
    }

    // eval_vars specialized for dvec
    template <class _T>
    std::string
    eval_vars(const dvec<_T>& r, unsigned& arg_num, bool read);

    namespace impl {
        std::string
        eval_vars_dvec_t(const std::string_view& tname,
                         unsigned& arg_num,
                         bool ro);
        std::string
        eval_vars_dvec_t(const char* tname,
                         unsigned& arg_num,
                         bool ro);
        std::string
        eval_vars_dvec_t(const std::string& tname,
                         unsigned& arg_num,
                         bool ro);
    }

    // store the results into a dvec
    template <class _T>
    std::string
    eval_results(dvec<_T>& r, unsigned& res_num);

    namespace impl {
        std::string
        eval_results_dvec_t(unsigned& res_num);
    }

    namespace dop {

        struct unary_func_base {
            // generate the body of an unary_func object
            static
            std::string
            body(const std::string& l, bool is_operator,
                 const char* name);
            // generate the body of an unary_func object
            static
            std::string
            body(const std::string& l, bool is_operator,
                 const std::string& name);
        };

        template <typename _P, bool _OP=false>
        struct unary_func : private unary_func_base {
            static
            std::string body(const std::string& l) {
                _P p;
                return unary_func_base::body(l, _OP, p());
            }
        };

        struct binary_func_base {
            // generate the body of a binary_func object
            static
            std::string
            body(const std::string& l, const std::string& r,
                 bool is_operator, const char* name);
            // generate the body of a binary_func object
            static
            std::string
            body(const std::string& l, const std::string& r,
                 bool is_operator, const std::string& name);
        };

        template <typename _P, bool _OP = false>
        struct binary_func : private binary_func_base {
            static
            std::string body(const std::string& l, const std::string& r) {
                _P p;
                return binary_func_base::body(l, r, _OP, p());
            }
        };

        namespace names {
            struct neg {
                constexpr
                const char* operator()() const { return "-"; }
            };
            struct bit_not {
                constexpr
                const char* operator()() const { return "~"; }
            };
        }

        template <class _T>
        struct neg : public unary_func<names::neg, true>{};

        template <class _T>
        struct bit_not : public unary_func<names::bit_not, true>{};


        namespace names {

            struct add {
                constexpr
                const char* operator()() const { return "+"; }
            };
            struct sub {
                constexpr
                const char* operator()() const { return "-"; }
            };
            struct mul {
                constexpr
                const char* operator()() const { return "*"; }
            };
            struct div {
                constexpr
                const char* operator()() const { return "/"; }
            };
            struct bit_and {
                constexpr
                const char* operator()() const { return "&"; }
            };
            struct bit_or {
                constexpr
                const char* operator()() const { return "|"; }
            };
            struct bit_xor {
                constexpr
                const char* operator()() const { return "^"; }
            };
            struct shl {
                constexpr
                const char* operator()() const { return "<<"; }
            };
            struct shr {
                constexpr
                const char* operator()() const { return ">>"; }
            };
            struct lt {
                constexpr
                const char* operator()() const { return "<"; }
            };
            struct le {
                constexpr
                const char* operator()() const { return "<="; }
            };
            struct eq {
                constexpr
                const char* operator()() const { return "=="; }
            };
            struct ne {
                constexpr
                const char* operator()() const { return "!="; }
            };
            struct ge {
                constexpr
                const char* operator()() const { return ">="; }
            };
            struct gt {
                constexpr
                const char* operator()() const { return ">"; }
            };
        }

        template <class _T>
        struct add : public binary_func<names::add, true> {};

        template <class _T>
        struct sub : public binary_func<names::sub, true> {};

        template <class _T>
        struct mul : public binary_func<names::mul, true> {};

        template <class _T>
        struct div : public binary_func<names::div, true> {};

        namespace names {

            struct div_base {
                // function name
                static
                std::string name(const char* tname);
                // function name
                static
                std::string name(const std::string& tname);
                // function body
                static
                std::string body(const char* tname);
                static
                std::string body(const std::string& tname);
            };

            template <typename _T>
            struct div_fix : private div_base {
                static
                std::string
                func_name() {
                    return name(be::type_2_name<_T>::v());
                }
                static
                std::string
                func_body() {
                    return body(be::type_2_name<_T>::v());
                }
                std::string
                operator()() const {
                    return func_name();
                }
            };
        }

        template <>
        struct div<dvec<float> >
            : public binary_func<names::div_fix<float>, false> {
        };

        template <std::size_t _N>
        struct div<dvec<cftal::vec<float, _N> > > :
            public binary_func<names::div_fix<cftal::vec<float, _N> >,
                               false> {
        };

        template <class _T>
        struct bit_and : public binary_func<names::bit_and, true> {};

        template <class _T>
        struct bit_or : public binary_func<names::bit_or, true> {};

        template <class _T>
        struct bit_xor : public binary_func<names::bit_xor, true> {};

        template <class _T>
        struct shl : public binary_func<names::shl, true> {};

        template <class _T>
        struct shr : public binary_func<names::shr, true> {};

        template <class _T>
        struct lt : public binary_func<names::lt, true> {};
        template <class _T>
        struct le : public binary_func<names::le, true> {};
        template <class _T>
        struct eq : public binary_func<names::eq, true> {};
        template <class _T>
        struct ne : public binary_func<names::ne, true> {};
        template <class _T>
        struct ge : public binary_func<names::ge, true> {};
        template <class _T>
        struct gt : public binary_func<names::gt, true> {};

    }

#define DEF_UNARY_FUNC(fname, op_name)                          \
    /* fname (V) */                                             \
    template <class _T>                                         \
    inline                                                      \
    expr<dop:: op_name <dvec<_T> >, dvec<_T>, void>             \
    fname (const dvec<_T>& t) {                                 \
        return expr<dop:: op_name <dvec<_T> >,                  \
                    dvec<_T>, void>(t);                         \
    }                                                           \
                                                                \
    /* fname (expr) */                                          \
    template <class _T,                                         \
              template <class _T1> class _OP,                   \
              class _L, class _R>                               \
    inline                                                      \
    expr<dop:: op_name <dvec<_T> >,                             \
         expr<_OP<dvec<_T> >, _L, _R>,                          \
         void>                                                  \
    fname (const expr<_OP<dvec<_T> >, _L, _R>& v) {             \
        return expr<dop:: op_name <dvec<_T> >,                  \
                    expr<_OP<dvec<_T> >, _L, _R>,               \
                    void>(v);                                   \
    }

    DEF_UNARY_FUNC(operator-, neg)
    DEF_UNARY_FUNC(operator~, bit_not)

    // unary plus
    template <class _T>
    inline
    const
    _T& operator+(const _T& v) {
        return v;
    }

#define BINARY_FUNC(name, op_name)                                      \
    /* name(V, V) */                                                    \
    template <class _T>                                                 \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, dvec<_T>, dvec<_T> >                 \
    name (const dvec<_T>& a, const dvec<_T>& b) {                       \
        return expr<dop:: op_name<dvec<_T> >,                           \
                    dvec<_T>, dvec<_T> >(a,b);                          \
    }                                                                   \
    /* name(V, _T) */                                                   \
    template <class _T, class _S>                                       \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, dvec<_T>, _S>                        \
    name (const dvec<_T>& a, const _S& b) {                             \
        return expr<dop:: op_name<dvec<_T> >, dvec<_T>, _S>(a,b);       \
    }                                                                   \
    /* name(_T, V) */                                                   \
    template <class _T, class _S>                                       \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, _S, dvec<_T> >                       \
    name (const _S& a, const dvec<_T>& b) {                             \
        return expr<dop:: op_name<dvec<_T> >, _S, dvec<_T> >(a,b);      \
    }                                                                   \
    /* name(V, expr) */                                                 \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >,                                      \
         dvec<_T>,                                                      \
         expr<_OP<dvec<_T> >, _L, _R> >                                 \
    name (const dvec<_T>& a,                                            \
          const expr<_OP<dvec<_T> >, _L, _R>& b) {                      \
        return expr<dop:: op_name<dvec<_T> >,                           \
                    dvec<_T>,                                           \
                    expr<_OP<dvec<_T>>, _L, _R> >(a, b);                \
    }                                                                   \
    /* name(_S, expr) */                                                \
    template <class _T, class _S,                                       \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, _S,                                  \
         expr<_OP<dvec<_T> >, _L, _R> >                                 \
    name (const _S& a,                                                  \
          const expr<_OP<dvec<_T> >, _L, _R>& b) {                      \
        return expr<dop:: op_name<dvec<_T> >,                           \
                    _S, expr<_OP<dvec<_T> >, _L, _R> >(a, b);           \
    }                                                                   \
    /* name(expr, V) */                                                 \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T>>,                                       \
         expr<_OP<dvec<_T> >, _L, _R>, dvec<_T> >                       \
    name (const expr<_OP<dvec<_T> >, _L, _R>& a,                        \
          const dvec<_T>& b) {                                          \
        return expr<dop:: op_name<dvec<_T> >,                           \
                    expr<_OP<dvec<_T> >, _L, _R>,                       \
                    dvec<_T> >(a, b);                                   \
    }                                                                   \
    /* name(expr, _S) */                                                \
    template <class _T, class _S,                                       \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >,                                      \
         expr<_OP<dvec<_T> >, _L, _R>, _S>                              \
    name (const expr<_OP<dvec<_T> >,                                    \
          _L, _R>& a, const _S& b) {                                    \
        return expr<dop:: op_name<dvec<_T> >,                           \
                    expr<_OP<dvec<_T> >, _L, _R>, _S>(a, b);            \
    }                                                                   \
    /* name(expr, expr)  */                                             \
    template <class _T,                                                 \
              template <class _V> class _OP1, class _L1, class _R1,     \
              template <class _V> class _OP2, class _L2, class _R2>     \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >,                                      \
         expr<_OP1<dvec<_T> >, _L1, _R1>,                               \
         expr<_OP2<dvec<_T> >, _L2, _R2> >                              \
    name(const expr<_OP1<dvec<_T> >, _L1, _R1>& a,                      \
         const expr<_OP2<dvec<_T> >, _L2, _R2>& b) {                    \
        return expr<dop:: op_name<dvec<_T> >,                           \
                    expr<_OP1<dvec<_T> >, _L1, _R1>,                    \
                    expr<_OP2<dvec<_T> >, _L2, _R2> > (a, b);           \
    }                                                                   \

#define DEFINE_OCLVEC_OPERATOR(op, eq_op, op_name)                      \
    BINARY_FUNC(operator op, op_name)                                   \
    /* operator eq_op V */                                              \
    template <class _T>                                                 \
    inline                                                              \
    dvec<_T>& operator eq_op(dvec<_T>& a, const dvec<_T>& r) {          \
        a = a op r;                                                     \
        return a;                                                       \
    }                                                                   \
    /* operator eq_op _T */                                             \
    template <class _T>                                                 \
    inline                                                              \
    dvec<_T>& operator eq_op(dvec<_T>& a, const _T& r) {                \
        a = a op r;                                                     \
        return a;                                                       \
    }                                                                   \
    /* operator eq_op expr */                                           \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    dvec<_T>&                                                           \
    operator eq_op(dvec<_T>& a,                                         \
                   const expr<_OP<dvec<_T> >, _L, _R>& r) {             \
        a = a op r;                                                     \
        return a;                                                       \
    }

#define DEFINE_OCLVEC_OPERATORS()          \
    DEFINE_OCLVEC_OPERATOR(+, +=, add)     \
    DEFINE_OCLVEC_OPERATOR(-, -=, sub)     \
    DEFINE_OCLVEC_OPERATOR(*, *=, mul)     \
    DEFINE_OCLVEC_OPERATOR(/, /=, div)     \
    DEFINE_OCLVEC_OPERATOR(&, &=, bit_and) \
    DEFINE_OCLVEC_OPERATOR(|, |=, bit_or)  \
    DEFINE_OCLVEC_OPERATOR(^, ^=, bit_xor) \
    DEFINE_OCLVEC_OPERATOR(<<, <<=, shl)   \
    DEFINE_OCLVEC_OPERATOR(>>, >>=, shr)

    DEFINE_OCLVEC_OPERATORS();
#undef DEFINE_OCLVEC_OPERATORS

    // TODO: more overloads also for (vec, expr), (expr, vec), (expr, expr)
#define DEFINE_OCLVEC_CMP_OPERATOR(op, op_name )                        \
    template <typename _T>                                              \
    expr<dop:: op_name<typename dvec<_T>::mask_type >,                \
         _T, dvec<_T> >                                               \
    operator op(const _T& a, const dvec<_T>& b) {                     \
        return expr<dop:: op_name <typename dvec<_T>::mask_type>,     \
                    _T, dvec<_T> >(a, b);                             \
    }                                                                   \
                                                                        \
    template <typename _T>                                              \
    expr<dop:: op_name<typename dvec<_T>::mask_type >,                \
         dvec<_T>, dvec<_T> >                                       \
    operator op(const dvec<_T>& a, const dvec<_T>& b) {             \
        return expr<dop:: op_name <typename dvec<_T>::mask_type>,     \
                    dvec<_T>, dvec<_T> >(a, b);                     \
    }                                                                   \
                                                                        \
    template <typename _T>                                              \
    expr<dop:: op_name<typename dvec<_T>::mask_type >,                \
         dvec<_T>, _T>                                                \
    operator op(const dvec<_T>& a, const _T& b) {                     \
        return expr<dop:: op_name <typename dvec<_T>::mask_type>,     \
                    dvec<_T>, _T>(a, b);                              \
    }

    DEFINE_OCLVEC_CMP_OPERATOR(<, lt)
    DEFINE_OCLVEC_CMP_OPERATOR(<=, le)
    DEFINE_OCLVEC_CMP_OPERATOR(==, eq)
    DEFINE_OCLVEC_CMP_OPERATOR(!=, ne)
    DEFINE_OCLVEC_CMP_OPERATOR(>=, ge)
    DEFINE_OCLVEC_CMP_OPERATOR(>, gt)

#undef DEFINE_OCLVEC_CMP_OPERATOR


    // overload for float vectors with incorrectly rounded division
    template <typename _L, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::div<dvec<float>>, _L, _R>& e );

    // overload for float vectors with incorrectly rounded division
    template <std::size_t _N, typename _L, typename _R>
    std::string
    def_custom_func(
        be::kernel_functions& fnames,
        const expr<dop::div<dvec<cftal::vec<float, _N>>>, _L, _R>& e );

}

template <class _T>
inline
std::size_t ocl::eval_size(const dvec<_T>& v)
{
    return v.size();
}

template <class _T>
inline
ocl::be::data_ptr
ocl::backend_data(const dvec<_T>& v)
{
    return v.backend_data();
}

inline
std::string
ocl::decl_non_buffer_args(const be::buffer& r, unsigned& arg_num)
{
    static_cast<void>(r);
    static_cast<void>(arg_num);
    return std::string();
}

template <typename _T>
std::string
ocl::decl_non_buffer_args(const dvec<_T>& r, unsigned& arg_num)
{
    static_cast<void>(r);
    static_cast<void>(arg_num);
    return std::string();
}

template <typename _T>
std::string
ocl::decl_buffer_args(const dvec<_T>& r, unsigned& arg_num, bool ro)
{
    static_cast<void>(r);
    return impl::decl_buffer_args_dvec_t(be::type_2_name<_T>::v(),
                                         arg_num, ro);
}

template <typename _T>
std::string
ocl::concat_args(const dvec<_T>& r, var_counters& c)
{
    static_cast<void>(r);
    return impl::concat_args_dvec_t(c);
}

template <typename _T>
void
ocl::bind_non_buffer_args(const dvec<_T>& t, be::argument_buffer& a)
{
    static_cast<void>(t);
    static_cast<void>(a);
}

template <typename _T>
void
ocl::
bind_buffer_args(const dvec<_T>& r, unsigned& buf_num,
                 be::kernel& k, unsigned wgs)
{
    static_cast<void>(wgs);
    impl::bind_buffer_args_dvec_t(r,
                                  be::type_2_name<_T>::v(),
                                  buf_num, k,
                                  true, r.size());
}

template <typename _T>
void
ocl::
bind_buffer_args(dvec<_T>& r, unsigned& buf_num,
                 be::kernel& k, unsigned wgs)
{
    static_cast<void>(wgs);
    impl::bind_buffer_args_dvec_t(r,
                                  be::type_2_name<_T>::v(),
                                  buf_num, k,
                                  false, r.size());
}

template <typename _T>
std::string
ocl::store_result(dvec<_T>& r, var_counters& c)
{
    static_cast<void>(r);
    return impl::store_result_dvec_t(c);
}

template <class _T>
std::string
ocl::eval_args(const dvec<_T>& r, unsigned& arg_num, bool ro)
{
    static_cast<void>(r);
    return impl::eval_args_dvec_t(be::type_2_name<_T>::v(),
                                  arg_num, ro);
}

template <class _T>
std::string
ocl::eval_vars(const dvec<_T>& r, unsigned& arg_num, bool read)
{
    static_cast<void>(r);
    return impl::eval_vars_dvec_t(be::type_2_name<_T>::v(),
                                  arg_num, read);
}

template <class _T>
std::string ocl::eval_results(dvec<_T>& r,
                              unsigned& res_num)
{
    static_cast<void>(r);
    return impl::eval_results_dvec_t(res_num);
}

template <typename _L, typename _R>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::div<dvec<float>>, _L, _R>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::div_fix<float>;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.insert(fn) == true) {
        s = d_t::func_body() + '\n';
    }
    return s;
}

// overload for float vectors with incorrectly rounded division
template <std::size_t _N, typename _L, typename _R>
std::string
ocl::
def_custom_func(
    be::kernel_functions& fnames,
    const expr<dop::div<dvec<cftal::vec<float, _N>>>, _L, _R>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::div_fix<cftal::vec<float, _N> >;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.insert(fn) == true) {
        s = d_t::func_body() + '\n';
    }
    return s;
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_VECTOR_H__
