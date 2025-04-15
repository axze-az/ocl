#if !defined (__OCL_DVEC_FUNC_H__)
#define __OCL_DVEC_FUNC_H__ 1

#include <ocl/config.h>
#include <ocl/dvec_op.h>

namespace ocl {

    namespace dop {
        namespace names {

            struct f_fabs {
                constexpr
                const char* operator()() const { return "fabs"; }
            };
            struct f_abs {
                constexpr
                const char* operator()() const { return "abs"; }
            };
            struct f_rint {
                constexpr
                const char* operator()() const { return "rint"; }
            };
            struct f_isinf {
                constexpr
                const char* operator()() const { return "isinf"; }
            };
            struct f_isnan {
                constexpr
                const char* operator()() const { return "isnan"; }
            };
        };

        template <class _T>
        struct abs_f : public unary_func<names::f_abs, false>{};

        template <>
        struct abs_f< dvec<float> >
            : public unary_func<names::f_fabs, false> {
        };
        template <std::size_t _N>
        struct abs_f< dvec<cftal::vec<float, _N> > >
            : public unary_func<names::f_fabs, false> {
        };

        template <>
        struct abs_f< dvec<double> >
            : public unary_func<names::f_fabs, false> {
        };
        template <std::size_t _N>
        struct abs_f< dvec<cftal::vec<double, _N> > >
            : public unary_func<names::f_fabs, false> {
        };


        template <class _T>
        struct rint_f  {};

        template <>
        struct rint_f< dvec<float> >
            : public unary_func<names::f_rint, false> {
        };
        template <std::size_t _N>
        struct rint_f< dvec<cftal::vec<float, _N> > >
            : public unary_func<names::f_rint, false> {
        };

        template <>
        struct rint_f< dvec<double> >
            : public unary_func<names::f_rint, false> {
        };
        template <std::size_t _N>
        struct rint_f< dvec<cftal::vec<double, _N> > >
            : public unary_func<names::f_rint, false> {
        };

        template <class _T>
        struct isinf_f  {};

        template <>
        struct isinf_f< dvec<float> >
            : public unary_func<names::f_isinf, false> {
        };
        template <std::size_t _N>
        struct isinf_f< dvec<cftal::vec<float, _N> > >
            : public unary_func<names::f_isinf, false> {
        };

        template <>
        struct isinf_f< dvec<double> >
            : public unary_func<names::f_isinf, false> {
        };
        template <std::size_t _N>
        struct isinf_f< dvec<cftal::vec<double, _N> > >
            : public unary_func<names::f_isinf, false> {
        };


        template <class _T>
        struct isnan_f  {};

        template <>
        struct isnan_f< dvec<float> >
            : public unary_func<names::f_isnan, false> {
        };
        template <std::size_t _N>
        struct isnan_f< dvec<cftal::vec<float, _N> > >
            : public unary_func<names::f_isnan, false> {
        };

        template <>
        struct isnan_f< dvec<double> >
            : public unary_func<names::f_isnan, false> {
        };
        template <std::size_t _N>
        struct isnan_f< dvec<cftal::vec<double, _N> > >
            : public unary_func<names::f_isnan, false> {
        };

        namespace names {

            struct f_sqrt {
                constexpr
                const char* operator()() const { return "sqrt"; }
            };

            struct f_sqrt_base {
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
            struct f_sqrt_fix : private f_sqrt_base {
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

        template <class _T>
        struct sqrt_f : public unary_func<names::f_sqrt, false> {};

        template <>
        struct sqrt_f<dvec<float> >
            : public unary_func<names::f_sqrt_fix<float>, false> {
        };

        template <std::size_t _N>
        struct sqrt_f<dvec<cftal::vec<float, _N> > > :
            public unary_func<names::f_sqrt_fix<cftal::vec<float, _N> >,
                               false> {
        };

        namespace names {
            struct f_rsqrt {
                constexpr
                const char* operator()() const { return "exp"; }
            };
        }

        template <class _T>
        struct rsqrt_f : public unary_func<names::f_rsqrt, false>{};

        namespace names {
            struct f_exp {
                constexpr
                const char* operator()() const { return "exp"; }
            };
            struct f_expm1 {
                constexpr
                const char* operator()() const { return "expm1"; }
            };
            struct f_exp2 {
                constexpr
                const char* operator()() const { return "exp2"; }
            };
            struct f_exp10 {
                constexpr
                const char* operator()() const { return "exp10"; }
            };
        }

        template <class _T>
        struct exp_f : public unary_func<names::f_exp, false>{};

        template <class _T>
        struct expm1_f : public unary_func<names::f_expm1, false>{};

        template <class _T>
        struct exp2_f : public unary_func<names::f_exp2, false>{};

        template <class _T>
        struct exp10_f : public unary_func<names::f_exp10, false>{};

        namespace names {
            struct f_log {
                constexpr
                const char* operator()() const { return "log"; }
            };
            struct f_log1p {
                constexpr
                const char* operator()() const { return "log1p"; }
            };
            struct f_log2 {
                constexpr
                const char* operator()() const { return "log2"; }
            };
            struct f_log10 {
                constexpr
                const char* operator()() const { return "log10"; }
            };
        }

        template <class _T>
        struct log_f : public unary_func<names::f_log, false>{};

        template <class _T>
        struct log1p_f : public unary_func<names::f_log1p, false>{};

        template <class _T>
        struct log2_f : public unary_func<names::f_log2, false>{};

        template <class _T>
        struct log10_f : public unary_func<names::f_log10, false>{};

        namespace names {
            struct f_sinh {
                constexpr
                const char* operator()() const { return "sinh"; }
            };
            struct f_cosh {
                constexpr
                const char* operator()() const { return "cosh"; }
            };
            struct f_tanh {
                constexpr
                const char* operator()() const { return "tanh"; }
            };
        }

        template <class _T>
        struct sinh_f : public unary_func<names::f_sinh, false>{};

        template <class _T>
        struct cosh_f : public unary_func<names::f_cosh, false>{};

        template <class _T>
        struct tanh_f : public unary_func<names::f_tanh, false>{};

        namespace names {
            struct f_sin {
                constexpr
                const char* operator()() const { return "sin"; }
            };
            struct f_cos {
                constexpr
                const char* operator()() const { return "cos"; }
            };
            struct f_tan {
                constexpr
                const char* operator()() const { return "tan"; }
            };
        }

        template <class _T>
        struct sin_f : public unary_func<names::f_sin, false>{};

        template <class _T>
        struct cos_f : public unary_func<names::f_cos, false>{};

        template <class _T>
        struct tan_f : public unary_func<names::f_tan, false>{};

        namespace names {

            struct f_min {
                constexpr
                const char* operator()() const { return  "min"; }
            };
            struct f_max {
                constexpr
                const char* operator()() const { return  "max"; }
            };

            struct f_pow {
                constexpr
                const char* operator()() const { return  "pow"; }
            };
        };

        template <class _T>
        struct max_f : public binary_func<names::f_max> {};

        template <class _T>
        struct min_f : public binary_func<names::f_min> {};

        template <class _T>
        struct pow_f : public binary_func<names::f_pow> {};

        template <class _D>
        struct cvt {
            template <typename _S>
            static
            std::string body(const std::string& l) {
#if 1
                // rusticl on rx460 workaround for now because
                // convert_ulong_rte(uint) is undefined
                using namespace std;
                constexpr const bool cast_only=
                    is_floating_point_v<_D> ||
                    (is_integral_v<_D> && is_integral_v<_S>);
                string res("((");
                res += be::type_2_name<_D>::v();
                res +=")";
                if (cast_only) {
                    res +=l;
                } else {
                    res += "rint";
                    res += "(";
                    res += l;
                    res += ")";
                }
                res +=")";
#else
                std::string res("convert_");
                res += be::type_2_name<_D>::v();
                res += "_rte(";
                res += l;
                res += ")";
#endif
                return res;
            }
        };

        template <class _D>
        struct cvt<dvec<_D> > {
            template <typename _S>
            static
            std::string body(const std::string& l) {
                return cvt<_D>::template body<_S>(l);
            }
        };

        template <class _D>
        struct cvt_rz {
            static
            std::string body(const std::string& l) {
                std::string res("convert_");
                res += be::type_2_name<_D>::v();
                res += "_rtz(";
                res += l;
                res += ")";
                return res;
            }
        };

        template <class _D>
        struct cvt_rz<dvec<_D> > {
            static
            std::string body(const std::string& l) {
                return cvt_rz<_D>::body(l);
            }
        };

        template <class _D>
        struct as {
            static
            std::string body(const std::string& l);
        };

        template <class _D>
        struct as<dvec<_D> > {
            static
            std::string body(const std::string& l) {
                std::string res("(");
                res += "as_";
                res += be::type_2_name<_D>::v();
                res += "(";
                res += l;
                res += "))";
                return res;
            }
        };

        // place holder for the arguments of select
        template <class _D>
        struct sel_data {
        };

        namespace names {
            struct f_sel_base {
                static
                std::string
                body(const std::string& s,
                     const std::string& on_true,
                     const std::string& on_false);
            };
        }

        template <class _D>
        struct f_sel : public names::f_sel_base {
        };
    }

    // convert with round nearest even
    template <class _D, class _S>
    inline
    expr<dop::cvt<_D>, dvec<_S>, void>
    cvt(const dvec<_S>& s) {
        return expr<dop::cvt<_D>, dvec<_S>, void>(s);
    }

    template <class _D, class _S,
              template <class _T> class _DOP,
              class _L, class _R>
    inline
    expr<dop::cvt<_D>, expr<_DOP<dvec<_S> >, _L, _R>, void>
    cvt(const expr<_DOP<dvec<_S> >, _L, _R>& s) {
        return expr<dop::cvt<_D>, expr<_DOP<dvec<_S> >, _L, _R> , void>(s);
    }

    template <class _D, class _S>
    std::string
    eval_ops(const expr<dop::cvt<_D>, dvec<_S>, void>& a,
             unsigned& arg_num) {
        auto l=eval_ops(a._l, arg_num);
        std::string t=dop::cvt<_D>::template body<_S>(l);
        return std::string("(") + t + std::string(")");
    }

    template <class _D, class _S,
              template <class _T> class _DOP,
              class _L, class _R>
    std::string
    eval_ops(const expr<dop::cvt<_D>, expr<_DOP<dvec<_S> >, _L, _R>, void>& a,
             unsigned& arg_num) {
        auto l=eval_ops(a._l, arg_num);
        std::string t=dop::cvt<_D>::template body<_S>(l);
        return std::string("(") + t + std::string(")");
    }

    // convert with round to zero
    template <class _D, class _S>
    inline
    expr<dop::cvt_rz<_D>, dvec<_S>, void>
    cvt_rz(const dvec<_S>& s) {
        return expr<dop::cvt_rz<_D>, dvec<_S>, void>(s);
    }

    template <class _D, class _S,
              template <class _T> class _DOP,
              class _L, class _R>
    inline
    expr<dop::cvt_rz<_D>, expr<_DOP<dvec<_S> >, _L, _R>, void>
    cvt_rz(const expr<_DOP<dvec<_S> >, _L, _R>& s) {
        return expr<dop::cvt_rz<_D>, expr<_DOP<dvec<_S> >, _L, _R> , void>(s);
    }

    template <class _D, class _S>
    inline
    expr<dop::as<_D>, dvec<_S>, void>
    as(const dvec<_S>& s) {
        return expr<dop::as<_D>, dvec<_S>, void>(s);
    }

    template <class _D, class _S,
              template <class _T> class _DOP,
              class _L, class _R>
    inline
    expr<dop::as<_D>, expr<_DOP<dvec<_S> >, _L, _R>, void>
    as(const expr<_DOP<dvec<_S> >, _L, _R>& s) {
        return expr<dop::as<_D>, expr<_DOP<dvec<_S> >, _L, _R> , void>(s);
    }

    DEF_UNARY_FUNC(rint, rint_f)
    DEF_UNARY_FUNC(isinf, isinf_f)
    DEF_UNARY_FUNC(isnan, isnan_f)

    DEF_UNARY_FUNC(abs, abs_f)
    DEF_UNARY_FUNC(sqrt, sqrt_f)
    DEF_UNARY_FUNC(rsqrt, rsqrt_f)

    DEF_UNARY_FUNC(exp, exp_f)
    DEF_UNARY_FUNC(expm1, expm1_f)
    DEF_UNARY_FUNC(exp2, exp2_f)
    DEF_UNARY_FUNC(exp10, exp10_f)

    DEF_UNARY_FUNC(log, log_f)
    DEF_UNARY_FUNC(log1p, log1p_f)
    DEF_UNARY_FUNC(log2, log2_f)
    DEF_UNARY_FUNC(log10, log10_f)

    DEF_UNARY_FUNC(sinh, sinh_f)
    DEF_UNARY_FUNC(cosh, cosh_f)
    DEF_UNARY_FUNC(tanh, tanh_f)

    DEF_UNARY_FUNC(sin, sin_f)
    DEF_UNARY_FUNC(cos, cos_f)
    DEF_UNARY_FUNC(tan, tan_f)

    BINARY_FUNC(max, max_f)
    BINARY_FUNC(min, min_f)
    BINARY_FUNC(pow, pow_f)

    template <typename _T, typename _U>
    auto
    select(const dvec<_U>& m, const _T& ot, const _T& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }
    template <typename _T, typename _U>
    auto
    select(const dvec<_U>& m, const dvec<_T>& ot, const dvec<_T>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }
    template <typename _T, typename _U, typename _R>
    auto
    select(const dvec<_U>& m, const dvec<_T>& ot, const _R& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    template <typename _T, typename _U, typename _L>
    auto
    select(const dvec<_U>& m, const _L& ot, const dvec<_T>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    template <typename _T,
              template <typename _V> class _OPU,
              typename _U, typename _UL, typename _UR>
    auto
    select(const expr<_OPU<dvec<_U> >, _UL, _UR> & m,
           const _T ot, const _T& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }
    template <typename _T,
              template <typename _V> class _OPU,
              typename _U, typename _UL, typename _UR>
    auto
    select(const expr<_OPU<dvec<_U> >, _UL, _UR> & m,
           const dvec<_T>& ot, const dvec<_T>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }
    template <typename _T, template <typename _V> class _OPU,
              typename _U, typename _UL, typename _UR,
              typename _R>
    auto
    select(const expr<_OPU<dvec<_U> >, _UL, _UR> & m,
           const dvec<_T>& ot, const _R& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    template <typename _T, template <typename _V> class _OPU,
              typename _U, typename _UL, typename _UR,
              typename _L>
    auto
    select(const expr<_OPU<dvec<_U> >, _UL, _UR> & m,
           const _L& ot, const dvec<_T>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    template <typename _T,
              template <typename _V1> class _OPL,
              typename _LL, typename _LR,
              template <typename _V2> class _OPR,
              typename _RL, typename _RR,
              typename _U>
    auto
    select(const dvec<_U>& m,
           const expr<_OPL<dvec<_T> >, _LL, _LR>& ot,
           const expr<_OPR<dvec<_T> >, _RL, _RR>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }
    template <typename _T, typename _U,
              template <typename _V1> class _OPL,
              typename _LL, typename _LR,
              typename _R>
    auto
    select(const dvec<_U>& m,
           const expr<_OPL<dvec<_T> >, _LL, _LR>& ot,
           const _R& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    template <typename _T, typename _U,
              template <typename _V2> class _OPR,
              typename _RL, typename _RR,
              typename _L>
    auto
    select(const dvec<_U>& m,
           const _L& ot,
           const expr<_OPR<dvec<_T> >, _RL, _RR>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    template <typename _T,
              template <typename _V> class _OPU,
              typename _U, typename _UL, typename _UR,
              template <typename _V1> class _OPL,
              typename _LL, typename _LR,
              template <typename _V2> class _OPR,
              typename _RL, typename _RR>
    auto
    select(const expr<_OPU<dvec<_U> >, _UL, _UR> & m,
           const expr<_OPL<dvec<_T> >, _LL, _LR>& ot,
           const expr<_OPR<dvec<_T> >, _RL, _RR>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }
    template <typename _T, template <typename _V> class _OPU,
              typename _U, typename _UL, typename _UR,
              template <typename _V1> class _OPL,
              typename _LL, typename _LR,
              typename _R>
    auto
    select(const expr<_OPU<dvec<_U> >, _UL, _UR> & m,
           const expr<_OPL<dvec<_T> >, _LL, _LR>& ot,
           const _R& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    template <typename _T, template <typename _V> class _OPU,
              typename _U, typename _UL, typename _UR,
              typename _L,
              template <typename _V2> class _OPR,
              typename _RL, typename _RR>
    auto
    select(const expr<_OPU<dvec<_U> >, _UL, _UR> & m,
           const _L& ot,
           const expr<_OPR<dvec<_T> >, _RL, _RR>& of) {
        return make_expr<dop::f_sel<dvec<_T> > >(
            m, make_expr<dop::sel_data<dvec<_T> > >(ot, of));
    }

    // eval_ops specialized for expr<>
    template <class _T, class _L, class _R>
    std::string
    eval_ops(const expr<dop::f_sel<dvec<_T> >, _L, _R>& e, unsigned& arg_num) {
        std::string m=eval_ops(e._l, arg_num);
        std::string ot=eval_ops(e._r._l, arg_num);
        std::string of=eval_ops(e._r._r, arg_num);
        return dop::f_sel<dvec<_T> >::body(m, ot, of);
    }

    // overload for float vectors with incorrectly rounded sqrt
    template <typename _L>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::sqrt_f<dvec<float>>, _L, void>& e );

    // overload for float vectors with incorrectly rounded sqrt
    template <std::size_t _N, typename _L>
    std::string
    def_custom_func(
        be::kernel_functions& fnames,
        const expr<dop::sqrt_f<dvec<cftal::vec<float, _N>>>, _L, void>& e );

    namespace impl {
        __ck_body
        gen_all_of(const std::string_view& tname);

        __ck_body
        gen_none_of(const std::string_view& tname);

        __ck_body
        gen_any_of(const std::string_view& tname);

        template <typename _T>
        typename dvec<_T>::mask_value_type
        dvec_xxx_of(const __ck_body& nb, const dvec<_T>& z);
    }

    template <typename _T>
    bool
    all_of(const dvec<_T>& v);

    template <typename _T>
    bool
    none_of(const dvec<_T>& v);

    template <typename _T>
    bool
    any_of(const dvec<_T>& v);

    namespace impl {
        __ck_body
        gen_hadd(const std::string_view& tname);
    }

    template <typename _T>
    _T
    hadd(const dvec<_T>& v);

    namespace impl {
        __ck_body
        gen_dot_product(const std::string_view& tname);
    }

    template <typename _T>
    _T
    dot_product(const dvec<_T>& a, const dvec<_T>& b);

    namespace impl {
        // worker function for evem_elements(v)
        __ck_body
        even_elements(const std::string_view& tname);
    }

    // returns only the even elements of s
    template <typename _T>
    dvec<_T>
    even_elements(const dvec<_T>& s);

    namespace impl {
        // worker function for odd_elements(v)
        __ck_body
        odd_elements(const std::string_view& tname);
    }

    // returns only the odd elements of s
    template <typename _T>
    dvec<_T>
    odd_elements(const dvec<_T>& s);

    namespace impl {
        // worker function for combine_even_odd(e, o)
        __ck_body
        combine_even_odd(const std::string_view& tname);
    }

    // return the interleaved elemnent of e and o
    template <typename _T>
    dvec<_T>
    combine_even_odd(const dvec<_T>& e, const dvec<_T>& o);

    namespace impl {
        // worker function for select_even_odd
        __ck_body
        select_even_odd(const std::string_view& tname);
    }

    template <typename _T>
    dvec<_T>
    select_even_odd(const dvec<_T>& e, const dvec<_T>& o);


    namespace impl {
        // worker function for copy_even_to_odd(v)
        __ck_body
        copy_even_to_odd(const std::string_view& tname);
    }

    // copy the even elements of s to the odd elements
    template <typename _T>
    dvec<_T>
    copy_even_to_odd(const dvec<_T>& s);

    namespace impl {
        // worker function for copy_odd_to_even(v)
        __ck_body
        copy_odd_to_even(const std::string_view& tname);
    }

    // copy the odd elements of s to the even elements
    template <typename _T>
    dvec<_T>
    copy_odd_to_even(const dvec<_T>& s);


    namespace impl {
        // worker function for permute(i, v)
        __ck_body
        permute(const std::string_view& tname,
                const std::string_view& iname);
    }

    // permute the vector using idx
    template <typename _T, typename _I>
    dvec<_T>
    permute(const dvec<_I>& i, const dvec<_T>& s);

    namespace impl {
        // worker function for permute(i, v, v)
        __ck_body
        permute2(const std::string_view& tname,
                 const std::string_view& iname);
    }

    template <typename _T, typename _I>
    dvec<_T>
    permute(const dvec<_I>& i, const dvec<_T>& s1, const dvec<_T>& s2);

}

// overload for float vectors with incorrectly rounded sqrt
template <typename _L>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::sqrt_f<dvec<float>>, _L, void>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::f_sqrt_fix<float>;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.insert(fn) == true) {
        s = d_t::func_body() + '\n';
    }
    return s;
}

// overload for float vectors with incorrectly rounded sqrt
template <std::size_t _N, typename _L>
std::string
ocl::
def_custom_func(
    be::kernel_functions& fnames,
    const expr<dop::sqrt_f<dvec<cftal::vec<float, _N>>>, _L, void>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::f_sqrt_fix<cftal::vec<float, _N> >;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.insert(fn) == true) {
        s = d_t::func_body() + '\n';
    }
    return s;
}

template <typename _T>
typename ocl::dvec<_T>::mask_value_type
ocl::impl::dvec_xxx_of(const __ck_body& nb, const dvec<_T>& v)
{
    auto p=v.backend_data();
    using type= typename dvec<_T>::mask_value_type;
    typename dvec<_T>::mask_type nz= v != _T(0);
    dvec<uint64_t> dcnt(p, 1);
    uint64_t hdcnt=nz.size();
    // Note: in custom kernels the left hand side may be
    // read also from the kernel:
    auto k=custom_kernel<type>(nb.name(), nb.body(),
                               nz, dcnt, local_mem_per_workitem<type>(1));
    do {
        execute_custom(k, hdcnt, p);
        dcnt.copy_to_host(&hdcnt);
    } while (hdcnt>1);
    // copy only one element from nz
    type r;
    nz.copy_to_host(&r, 0, 1);
    return r;
}

template <typename _T>
bool
ocl::all_of(const dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    const auto tname=be::type_2_name<type>::v();
    auto nb=impl::gen_all_of(tname);
    auto r=impl::dvec_xxx_of(nb, v);
    return r!=0;
}

template <typename _T>
bool
ocl::none_of(const dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    const auto tname=be::type_2_name<type>::v();
    auto nb=impl::gen_none_of(tname);
    auto r=impl::dvec_xxx_of(nb, v);
    return r==0;
}

template <typename _T>
bool
ocl::any_of(const dvec<_T>& v)
{
    using type= typename dvec<_T>::mask_value_type;
    const auto tname=be::type_2_name<type>::v();
    auto nb=impl::gen_any_of(tname);
    auto r=impl::dvec_xxx_of(nb, v);
    return r!=0;
}

template <typename _T>
_T
ocl::hadd(const dvec<_T>& v)
{
    auto p=v.backend_data();
    dvec<uint64_t> dcnt(p, 1);
    uint64_t hdcnt=v.size();
    // create a working copy:
    dvec<_T> vc(p, hdcnt);
    const auto tname=be::type_2_name<_T>::v();
    auto nb=impl::gen_hadd(tname);
    // Note: in custom kernels the left hand side may be
    // read also from the kernel:
    auto k=custom_kernel<_T>(nb.name(), nb.body(),
                             vc, dcnt, local_mem_per_workitem<_T>(1));
    do {
        execute_custom(k, hdcnt, p);
        dcnt.copy_to_host(&hdcnt);
    } while (hdcnt>1);
    // copy only one element from vc
    _T r;
    vc.copy_to_host(&r, 0, 1);
    return r;
}

template <typename _T>
ocl::dvec<_T>
ocl::even_elements(const dvec<_T>& s)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::even_elements(tname);
    size_t n=(s.size() +1) >> 1;
    dvec<_T> r=custom_kernel_with_size<_T>(ckb.name(), ckb.body(), n, s);
    return r;
}

template <typename _T>
ocl::dvec<_T>
ocl::odd_elements(const dvec<_T>& s)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::odd_elements(tname);
    size_t n=s.size() >> 1;
    dvec<_T> r=custom_kernel_with_size<_T>(ckb.name(), ckb.body(), n, s);
    return r;
}

template <typename _T>
ocl::dvec<_T>
ocl::combine_even_odd(const dvec<_T>& e, const dvec<_T>& o)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::combine_even_odd(tname);
    size_t n=e.size() + o.size();
    dvec<_T> r=custom_kernel_with_size<_T>(ckb.name(), ckb.body(), n, e, o);
    return r;
}

template <typename _T>
ocl::dvec<_T>
ocl::select_even_odd(const dvec<_T>& e, const dvec<_T>& o)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::select_even_odd(tname);
    dvec<_T> r=custom_kernel<_T>(ckb.name(), ckb.body(), e, o);
    return r;
}

template <typename _T>
ocl::dvec<_T>
ocl::copy_even_to_odd(const dvec<_T>& s)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::copy_even_to_odd(tname);
    dvec<_T> r=custom_kernel<_T>(ckb.name(), ckb.body(), s);
    return r;
}

template <typename _T>
ocl::dvec<_T>
ocl::copy_odd_to_even(const dvec<_T>& s)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::copy_odd_to_even(tname);
    dvec<_T> r=custom_kernel<_T>(ckb.name(), ckb.body(), s);
    return r;
}

template <typename _T, typename _I>
ocl::dvec<_T>
ocl::permute(const dvec<_I>& idx, const dvec<_T>& s)
{
    const auto tname=be::type_2_name<_T>::v();
    const auto iname=be::type_2_name<_I>::v();
    impl::__ck_body ckb=impl::permute(tname, iname);
    dvec<_T> r=custom_kernel<_T>(ckb.name(), ckb.body(), idx, s);
    return r;
}

template <typename _T, typename _I>
ocl::dvec<_T>
ocl::permute(const dvec<_I>& idx, const dvec<_T>& s0, const dvec<_T>& s1)
{
    const auto tname=be::type_2_name<_T>::v();
    const auto iname=be::type_2_name<_I>::v();
    impl::__ck_body ckb=impl::permute2(tname, iname);
    dvec<_T> r=custom_kernel<_T>(ckb.name(), ckb.body(), idx, s0, s1);
    return r;
}

// local variables:
// mode: c++
// end:
#endif
