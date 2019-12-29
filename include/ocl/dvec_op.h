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

        template <typename _P, bool _OP=false>
        struct unary_func {
            static const _P m_p;
            static
            std::string body(const std::string& l) {
                std::string res;
                res += m_p();
                if (_OP == false)
                    res += '(';
                res += l;
                if (_OP == false)
                    res += ')';
                return res;
            };
        };

        template <typename _P, bool _OP>
        const _P unary_func<_P, _OP>::m_p=_P();

        template <typename _P, bool _OP = false>
        struct binary_func {
            static const _P m_p;
            static
            std::string body(const std::string& l,
                             const std::string& r) {
                std::string res;
                if (_OP == false) {
                    res += m_p();
                    res += "(";
                }
                res += l;
                if (_OP == false)
                    res += ", ";
                else
                    res += m_p();
                res += r;
                if (_OP == false)
                    res += ")";
                return res;
            }
        };

        template <typename _P, bool _OP>
        const _P binary_func<_P, _OP>::m_p=_P();

        namespace names {
            struct neg {
                const char* operator()() const { return "-"; }
            };
            struct bit_not {
                const char* operator()() const { return "~"; }
            };
            struct fabs {
                const char* operator()() const { return "fabs"; }
            };
            struct abs {
                const char* operator()() const { return "abs"; }
            };
        };

        template <class _T>
        struct neg : public unary_func<names::neg, true>{};

        template <class _T>
        struct bit_not : public unary_func<names::bit_not, true>{};

        template <class _T>
        struct abs : public unary_func<names::abs, false>{};

        template <>
        struct abs< dvec<float> > {
            static
            std::string body(const std::string& l) {
                std::string res("fabs(");
                res += l;
                res += ")";
                return res;
            }
        };
        // public unary_func<names::fabs, false>{};

        template <>
        struct abs< dvec<double> >  {
            static
            std::string body(const std::string& l) {
                std::string res("fabs(");
                res += l;
                res += ")";
                return res;
            }
        };
        // public unary_func<names::fabs, false>{};


        namespace names {

            struct add { const char* operator()() const { return "+"; }};
            struct sub { const char* operator()() const { return "-"; }};
            struct mul { const char* operator()() const { return "*"; }};
            struct div { const char* operator()() const { return "/"; }};

            struct bit_and { const char* operator()() const { return "&"; }};
            struct bit_or { const char* operator()() const { return "|"; }};
            struct bit_xor { const char* operator()() const { return "^"; }};

            struct shl { const char* operator()() const { return "<<"; }};
            struct shr { const char* operator()() const { return ">>"; }};

            struct lt { const char* operator()() const { return "<"; }};
            struct le { const char* operator()() const { return "<="; }};
            struct eq { const char* operator()() const { return "=="; }};
            struct ne { const char* operator()() const { return "!="; }};
            struct ge { const char* operator()() const { return ">="; }};
            struct gt { const char* operator()() const { return ">"; }};
        }


        template <class _T>
        struct add : public binary_func<names::add, true> {};

        template <class _T>
        struct sub : public binary_func<names::sub, true> {};

        template <class _T>
        struct mul : public binary_func<names::mul, true> {};

        template <class _T>
        struct div : public binary_func<names::div, true> {};

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

            struct min{ const char* operator()() const { return  "min"; }};
            struct max{ const char* operator()() const { return  "max"; }};
        };


        template <class _T>
        struct max_func : public binary_func<names::max> {};

        template <class _T>
        struct min_func : public binary_func<names::min> {};

        template <class _D>
        struct cvt_to {
            static
            std::string body(const std::string& l) {
                std::string res("((");
                // I AM BUGGY:
                // res += "int"; // impl::type_2_name<_D>::v();
                res += be::type_2_name<_D>::v();
                res += ")";
                res += l;
                res += ")";
                std::cout<< res << std::endl;
                return res;
            }
        };

        template <class _D>
        struct cvt_to<dvec<_D> > {
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
                // std::cout<< res << std::endl;
                return res;
            }
        };

    }

    template <class _D, class _S>
    inline
    expr<dop::cvt_to<_D>, _S, void>
    cvt_to(const _S& s) {
        return expr<dop::cvt_to<_D>, _S, void>(s);
    }

    template <class _D, class _S>
    inline
    expr<dop::as<_D>, _S, void>
    as(const _S& s) {
        return expr<dop::as<_D>, _S, void>(s);
    }

    // abs(V)
    template <class _T>
    inline
    expr<dop::abs<dvec<_T> >, dvec<_T>, void>
    abs(const dvec<_T>& t) {
        return expr<dop::abs<dvec<_T> >, dvec<_T>, void>(t);
    }

    // abs(expr)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R>
    inline
    expr<dop::abs<dvec<_T> >,
         expr<_OP<dvec<_T> >, _L, _R>,
         void>
    abs(const expr<_OP<dvec<_T> >, _L, _R>& v) {
        return expr<dop::abs<dvec<_T> >,
                    expr<_OP<dvec<_T> >, _L, _R>,
                    void>(v);
    }

    // min(V)
    template <class _T, class _S>
    inline
    expr<dop::min_func<dvec<_T> >,
         dvec<_T>, _S>
    min(const dvec<_T>& a, const _S& b)
    {
        return expr<dop::min_func<dvec<_T> >,
                    dvec<_T>, _S >(a, b);
    }

    // min(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<dop::min_func<dvec<_T> >,
         expr<_OP<dvec<_T> >, _L, _R>,
         _S >
    min(const expr<_OP<dvec<_T> >, _L, _R>& a, const _S& b)
    {
        return expr<dop::min_func<dvec<_T> >,
                    expr<_OP<dvec<_T> >, _L, _R>,
                    _S >(a, b);
    }

    // min(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<dop::min_func<dvec<_T> >,
         _S,
         expr<_OP<dvec<_T> >, _L, _R> >
    min(const _S& b, const expr<_OP<dvec<_T> >, _L, _R>& a)
    {
        return expr<dop::min_func<dvec<_T> >,
                    _S,
                    expr<_OP<dvec<_T> >, _L, _R> >(a, b);
    }

    // max(V)
    template <class _T, class _S>
    inline
    expr<dop::max_func<dvec<_T> >,
         dvec<_T>, _S>
    max(const dvec<_T>& a, const _S& b)
    {
        return expr<dop::max_func<dvec<_T> >,
                    dvec<_T>, _S >(a, b);
    }

    // max(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<dop::max_func<dvec<_T> >,
         expr<_OP<dvec<_T> >, _L, _R>,
         _S >
    max(const expr<_OP<dvec<_T> >, _L, _R>& a, const _S& b)
    {
        return expr<dop::max_func<dvec<_T> >,
                    expr<_OP<dvec<_T> >, _L, _R>,
                    _S >(a, b);
    }

    // max(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<dop::max_func<dvec<_T> >,
         _S,
         expr<_OP<dvec<_T> >, _L, _R> >
    max(const _S& b, const expr<_OP<dvec<_T> >, _L, _R>& a)
    {
        return expr<dop::max_func<dvec<_T> >,
                    _S,
                    expr<_OP<dvec<_T> >, _L, _R> >(a, b);
    }

    // unary plus
    template <class _T>
    inline
    const
    _T& operator+(const _T& v) {
        return v;
    }

    // unary minus V
    template <class _T>
    inline
    expr<dop::neg<dvec<_T> >, dvec<_T>, void>
    operator-(const dvec<_T>& v) {
        return expr<dop::neg<dvec<_T> >, dvec<_T>, void>(v);
    };
    // unary minus expr
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R>
    inline
    expr<dop::neg<dvec<_T> >,
         expr<_OP<dvec<_T> >, _L, _R>,
         void>
    operator-(const expr<_OP<dvec<_T> >, _L, _R>& v) {
        return expr<dop::neg<dvec<_T> >,
                    expr<_OP<dvec<_T> >, _L, _R>,
                    void>(v);
    }

    // unary not V
    template <class _T>
    inline
    expr<dop::bit_not<dvec<_T> >, dvec<_T>, void>
    operator~(const dvec<_T>& v) {
        return expr<dop::bit_not<dvec<_T> >, dvec<_T>, void>(v);
    };
    // unary not expr
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R>
    inline
    expr<dop::bit_not<dvec<_T> >,
         expr<_OP<dvec<_T> >, _L, _R>,
         void>
    operator~(const expr<_OP<dvec<_T> >, _L, _R>& v) {
        return expr<dop::bit_not<dvec<_T> >,
                    expr<_OP<dvec<_T> >, _L, _R>,
                    void>(v);
    }


#define DEFINE_OCLVEC_OPERATOR(op, eq_op, op_name)                      \
    /* operator op(V, V) */                                             \
    template <class _T>                                                 \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, dvec<_T>, dvec<_T> >           \
    operator op (const dvec<_T>& a, const dvec<_T>& b) {            \
        return expr<dop:: op_name<dvec<_T> >,                         \
                    dvec<_T>, dvec<_T> >(a,b);                      \
    }                                                                   \
    /* operator op(V, _T) */                                            \
    template <class _T, class _S>                                       \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, dvec<_T>, _S>                    \
    operator op (const dvec<_T>& a, const _S& b) {                    \
        return expr<dop:: op_name<dvec<_T> >, dvec<_T>, _S>(a,b);   \
    }                                                                   \
    /* operator op(_T, V) */                                            \
    template <class _T, class _S>                                       \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, _S, dvec<_T> >                   \
    operator op (const _S& a, const dvec<_T>& b) {                    \
        return expr<dop:: op_name<dvec<_T> >, _S, dvec<_T> >(a,b);  \
    }                                                                   \
    /* operator op(V, expr) */                                          \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >,                                    \
         dvec<_T>,                                                    \
         expr<_OP<dvec<_T> >, _L, _R> >                               \
    operator op (const dvec<_T>& a,                                   \
                 const expr<_OP<dvec<_T> >, _L, _R>& b) {             \
        return expr<dop:: op_name<dvec<_T> >,                         \
                    dvec<_T>,                                         \
                    expr<_OP<dvec<_T>>, _L, _R> >(a, b);              \
    }                                                                   \
    /* operator op(_S, expr) */                                         \
    template <class _T, class _S,                                       \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >, _S,                                \
         expr<_OP<dvec<_T> >, _L, _R> >                               \
    operator op (const _S& a,                                           \
                 const expr<_OP<dvec<_T> >, _L, _R>& b) {             \
        return expr<dop:: op_name<dvec<_T> >,                         \
                    _S, expr<_OP<dvec<_T> >, _L, _R> >(a, b);         \
    }                                                                   \
    /* operator op(expr, V) */                                          \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T>>,                                     \
         expr<_OP<dvec<_T> >, _L, _R>, dvec<_T> >                   \
    operator op (const expr<_OP<dvec<_T> >, _L, _R>& a,               \
                 const dvec<_T>& b) {                                 \
        return expr<dop:: op_name<dvec<_T> >,                         \
                    expr<_OP<dvec<_T> >, _L, _R>,                     \
                    dvec<_T> >(a, b);                                 \
    }                                                                   \
    /* operator op(expr, _S) */                                         \
    template <class _T, class _S,                                       \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >,                                    \
         expr<_OP<dvec<_T> >, _L, _R>, _S>                            \
    operator op (const expr<_OP<dvec<_T> >,                           \
                 _L, _R>& a, const _S& b) {                             \
        return expr<dop:: op_name<dvec<_T> >,                         \
                    expr<_OP<dvec<_T> >, _L, _R>, _S>(a, b);          \
    }                                                                   \
    /* operator op(expr, expr)  */                                      \
    template <class _T,                                                 \
              template <class _V> class _OP1, class _L1, class _R1,     \
              template <class _V> class _OP2, class _L2, class _R2>     \
    inline                                                              \
    expr<dop:: op_name<dvec<_T> >,                                    \
         expr<_OP1<dvec<_T> >, _L1, _R1>,                             \
         expr<_OP2<dvec<_T> >, _L2, _R2> >                            \
    operator op(const expr<_OP1<dvec<_T> >, _L1, _R1>& a,             \
                const expr<_OP2<dvec<_T> >, _L2, _R2>& b) {           \
        return expr<dop:: op_name<dvec<_T> >,                         \
                    expr<_OP1<dvec<_T> >, _L1, _R1>,                  \
                    expr<_OP2<dvec<_T> >, _L2, _R2> > (a, b);         \
    }                                                                   \
    /* operator eq_op V */                                              \
    template <class _T>                                                 \
    inline                                                              \
    dvec<_T>& operator eq_op(dvec<_T>& a, const dvec<_T>& r) {    \
        a = a op r;                                                     \
        return a;                                                       \
    }                                                                   \
    /* operator eq_op _T */                                             \
    template <class _T>                                                 \
    inline                                                              \
    dvec<_T>& operator eq_op(dvec<_T>& a, const _T& r) {            \
        a = a op r;                                                     \
        return a;                                                       \
    }                                                                   \
    /* operator eq_op expr */                                           \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    dvec<_T>&                                                         \
    operator eq_op(dvec<_T>& a,                                       \
                   const expr<_OP<dvec<_T> >, _L, _R>& r) {           \
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
        std::cout << "binding const dvec<"
                  << be::type_2_name<_T>::v()
                  << "> with "
                  << r.size()
                  << " elements to arg " << buf_num
                  << std::endl;
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
        std::cout << "binding dvec<"
                  << be::type_2_name<_T>::v()
                  << "> with "
                  << r.size()
                  << " elements to arg " << buf_num
                  << std::endl;
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
        std::cout << "binding dvec<"
                  << be::type_2_name<_T>::v()
                  << "> with "
                  << r.size()
                  << " elements to arg " << arg_num
                  << std::endl;
    }
    k.set_arg(arg_num, r.buf());
    ++arg_num;
}

template <class _T>
void
ocl::bind_args(be::kernel& k, const dvec<_T>& r, unsigned& arg_num)
{
    if (r.backend_data()->debug() != 0) {
        std::cout << "binding const dvec<"
                  << be::type_2_name<_T>::v()
                  << "> with "<< r.size()
                  << " elements to arg " << arg_num
                  << std::endl;
    }
    k.set_arg(arg_num, r.buf());
    ++arg_num;
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_VECTOR_H__
