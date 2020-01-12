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

    // fetch_args specialized for dvecs
    template <typename _T>
    std::string
    fetch_args(const dvec<_T>& t, var_counters& c);

    // fetch_args specialized for dvecs
    template <typename _T>
    std::string
    concat_args(const dvec<_T>& t, var_counters& c);

    // bind non buffer arguments
    template <typename _T>
    void
    bind_non_buffer_args(const dvec<_T>& t, be::argument_buffer& a);

    // bind buffer arguments
    template <typename _T>
    void
    bind_buffer_args(const dvec<_T>& t, unsigned& buf_num, be::kernel& k);

    // bind buffer arguments
    template <typename _T>
    void
    bind_buffer_args(dvec<_T>& t, unsigned& buf_num, be::kernel& k);

    // fetch_args specialized for dvecs
    template <typename _T>
    std::string
    store_result(dvec<_T>& t, var_counters& c);

    // eval_args specialized for dvec
    template <class _T>
    std::string
    eval_args(const std::string& p, const dvec<_T>& r,
              unsigned& arg_num, bool ro);
    // eval_vars specialized for dvec
    template <class _T>
    std::string
    eval_vars(const dvec<_T>& r, unsigned& arg_num, bool read);

    // store the results into a dvec
    template <class _T>
    std::string
    eval_results(dvec<_T>& r, unsigned& res_num);

    // bind_args for non const arguments
    template <class _T>
    void
    bind_args(be::kernel& k, dvec<_T>& r,  unsigned& arg_num);
    // bind_args for const arguments
    template <class _T>
    void
    bind_args(be::kernel& k, const dvec<_T>& r,  unsigned& arg_num);

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
            // generate the body of an binary_func object
            static
            std::string
            body(const std::string& l, const std::string& r,
                 bool is_operator, const char* name);
            // generate the body of an binary_func object
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
            
            struct f_fabs {
                constexpr
                const char* operator()() const { return "fabs"; }
            };
            struct f_abs {
                constexpr
                const char* operator()() const { return "abs"; }
            };
            struct f_sqrt {
                constexpr
                const char* operator()() const { return "sqrt"; }
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
        struct sqrt_f : public unary_func<names::f_sqrt, false> {};

        namespace names {

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
            static
            std::string body(const std::string& l) {
                std::string res("convert_");
                res += be::type_2_name<_D>::v();
                res += "_rte(";
                res += l;
                res += ")";
                return res;
            }
        };

        template <class _D>
        struct cvt<dvec<_D> > {
            static
            std::string body(const std::string& l) {
                std::string res("convert_");
                res += be::type_2_name<_D>::v();
                res += "_rte(";
                res += l;
                res += ")";
                return res;
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
    }

    template <class _D, class _S>
    inline
    expr<dop::cvt<_D>, _S, void>
    cvt(const _S& s) {
        return expr<dop::cvt<_D>, _S, void>(s);
    }

    template <class _D, class _S>
    inline
    expr<dop::as<_D>, _S, void>
    as(const _S& s) {
        return expr<dop::as<_D>, _S, void>(s);
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
    DEF_UNARY_FUNC(abs, abs_f)
    DEF_UNARY_FUNC(sqrt, sqrt_f)
    DEF_UNARY_FUNC(exp, exp_f)
    DEF_UNARY_FUNC(expm1, expm1_f)
    DEF_UNARY_FUNC(exp2, exp2_f)
    DEF_UNARY_FUNC(exp10, exp10_f)

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

    BINARY_FUNC(max, max_f)
    BINARY_FUNC(min, min_f)
    BINARY_FUNC(pow, pow_f)
    
    // overload for float vectors with incorrectly rounded division
    template <typename _L, typename _R>
    std::string
    def_custom_func(std::set<std::string>& fnames,
                    const expr<dop::div<dvec<float>>, _L, _R>& e );

    // overload for float vectors with incorrectly rounded division
    template <std::size_t _N, typename _L, typename _R>
    std::string
    def_custom_func(
        std::set<std::string>& fnames,
        const expr<dop::div<dvec<cftal::vec<float, _N>>>, _L, _R>& e );

    // overload for float vectors with incorrectly rounded sqrt
    template <typename _L>
    std::string
    def_custom_func(std::set<std::string>& fnames,
                    const expr<dop::sqrt_f<dvec<float>>, _L, void>& e );

    // overload for float vectors with incorrectly rounded sqrt
    template <std::size_t _N, typename _L>
    std::string
    def_custom_func(
        std::set<std::string>& fnames,
        const expr<dop::sqrt_f<dvec<cftal::vec<float, _N>>>, _L, void>& e );
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
    std::ostringstream s;
    s << spaces(4) << "__global ";
    if (ro) {
        s << "const ";
    }
    s << be::type_2_name<_T>::v()
      << "* arg" << arg_num << ",\n";
    ++arg_num;
    return s.str();
}

template <typename _T>
std::string
ocl::fetch_args(const dvec<_T>& r, var_counters& c)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(8) << "const " << be::type_2_name<_T>::v()
      << " v" << c._var_num
      << " = arg" << c._buf_num << "[gid];\n";
    ++c._var_num;
    ++c._buf_num;
    return s.str();
}

template <typename _T>
std::string
ocl::concat_args(const dvec<_T>& r, var_counters& c)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << "arg" << c._buf_num;
    ++c._var_num;
    ++c._buf_num;
    return s.str();
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
bind_buffer_args(const dvec<_T>& r, unsigned& buf_num, be::kernel& k)
{
    if (r.backend_data()->debug() != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": "
          << kn << ": " << &r << ": binding const dvec<"
          << be::type_2_name<_T>::v()<< "> with "
          << r.size()
          << " elements to arg " << buf_num << '\n';
        be::data::debug_print(s.str());
    }
    k.set_arg(buf_num, r.buf());
    ++buf_num;
}

template <typename _T>
void
ocl::
bind_buffer_args(dvec<_T>& r, unsigned& buf_num, be::kernel& k)
{
    if (r.backend_data()->debug() != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": "
          << kn << ": " << &r << ": binding dvec<"
          << be::type_2_name<_T>::v()<< "> with "
          << r.size()
          << " elements to arg " << buf_num << '\n';
        be::data::debug_print(s.str());
    }
    k.set_arg(buf_num, r.buf());
    ++buf_num;
}

template <typename _T>
std::string
ocl::store_result(dvec<_T>& r, var_counters& c)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(8)
      << "arg" << c._buf_num
      << "[gid] =";
    ++c._var_num;
    ++c._buf_num;
    return s.str();
}

template <class _T>
std::string
ocl::eval_args(const std::string& p, const dvec<_T>& r, unsigned& arg_num,
               bool ro)
{
    static_cast<void>(r);
    std::ostringstream s;
    if (!p.empty()) {
        s << p << ",\n";
    }
    s << spaces(4) << "__global " ;
    if (ro) {
        s<< "const ";
    }
    s << be::type_2_name<_T>::v()
      << "* arg"  << arg_num;
    ++arg_num;
    return s.str();
}

template <class _T>
std::string
ocl::eval_vars(const dvec<_T>& r, unsigned& arg_num, bool read)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(8) << be::type_2_name<_T>::v()
      << " v" << arg_num;
    if (read== true) {
        s << " = arg"
          << arg_num << "[gid];";
    }
    std::string a(s.str());
    ++arg_num;
    return a;
}

template <class _T>
std::string ocl::eval_results(dvec<_T>& r,
                              unsigned& res_num)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(8) << "arg" << res_num << "[gid]="
      << " v" << res_num << ';';
    ++res_num;
    return s.str();
}

template <class _T>
void
ocl::bind_args(be::kernel& k, dvec<_T>& r, unsigned& arg_num)
{
    if (r.backend_data()->debug() != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": "
          << kn << ": binding dvec<"
          << be::type_2_name<_T>::v()<< "> with "
          << r.size()
          << " elements to arg " << arg_num << '\n';
        be::data::debug_print(s.str());
    }
    k.set_arg(arg_num, r.buf());
    ++arg_num;
}

template <class _T>
void
ocl::bind_args(be::kernel& k, const dvec<_T>& r, unsigned& arg_num)
{
    if (r.backend_data()->debug() != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": "
          << kn << ": binding const dvec<"
          << be::type_2_name<_T>::v()<< "> with "
          << r.size()
          << " elements to arg " << arg_num << '\n';
        be::data::debug_print(s.str());
    }
    k.set_arg(arg_num, r.buf());
    ++arg_num;
}

template <typename _L, typename _R>
std::string
ocl::def_custom_func(std::set<std::string>& fnames,
                     const expr<dop::div<dvec<float>>, _L, _R>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::div_fix<float>;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.find(fn) == fnames.end()) {
        s = d_t::func_body() + '\n';
        fnames.insert(fn);
    }
    return s;
}

// overload for float vectors with incorrectly rounded division
template <std::size_t _N, typename _L, typename _R>
std::string
ocl::def_custom_func(
    std::set<std::string>& fnames,
    const expr<dop::div<dvec<cftal::vec<float, _N>>>, _L, _R>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::div_fix<cftal::vec<float, _N> >;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.find(fn) == fnames.end()) {
        s = d_t::func_body() + '\n';
        fnames.insert(fn);
    }
    return s;
}

// overload for float vectors with incorrectly rounded sqrt
template <typename _L>
std::string
ocl::def_custom_func(std::set<std::string>& fnames,
                     const expr<dop::sqrt_f<dvec<float>>, _L, void>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::f_sqrt_fix<float>;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.find(fn) == fnames.end()) {
        s = d_t::func_body() + '\n';
        fnames.insert(fn);
    }
    return s;
}

// overload for float vectors with incorrectly rounded sqrt
template <std::size_t _N, typename _L>
std::string
ocl::def_custom_func(
    std::set<std::string>& fnames,
    const expr<dop::sqrt_f<dvec<cftal::vec<float, _N>>>, _L, void>& e )
{
    static_cast<void>(e);
    using d_t=dop::names::f_sqrt_fix<cftal::vec<float, _N> >;
    const std::string fn=d_t::func_name();
    std::string s;
    if (fnames.find(fn) == fnames.end()) {
        s = d_t::func_body() + '\n';
        fnames.insert(fn);
    }
    return s;
}


// Local variables:
// mode: c++
// end:
#endif // __OCL_VECTOR_H__
