#if !defined (__OCL_VECTOR_H__)
#define __OCL_VECTOR_H__ 1

#include <ocl/config.h>
#include <ocl/expr_kernel.h>
#include <ocl/impl_type_2_name.h>
#include <initializer_list>
// #include <vexcl/vexcl.hpp>
#include <atomic>

namespace ocl {

    // vector base class wrapping an opencl buffer and a
    // (shared) pointer to opencl backend data
    class vector_base {
        // shared pointer to the backend data
        impl::be_data_ptr _bed;
        // backend buffer object
        impl::buffer _b;
    protected:
        // destructor
        ~vector_base();
        // default constructor
        vector_base();
        // constructor, with size
        explicit vector_base(std::size_t s);
        // copy constructor
        vector_base(const vector_base& r);
        // move constructor
        vector_base(vector_base&& r);
        // assignment operator
        vector_base& operator=(const vector_base& r);
        // move assignment operator
        vector_base& operator=(vector_base&& r);
        // swap two vector base objects
        vector_base& swap(vector_base& r);
        // device device copy
        void copy_on_device(const vector_base& r);
        // host device copy
        void copy_from_host(const void* src);
        // device host copy
        void copy_to_host(void* dst)
            const;
    public:
        // return the size of the vector in bytes
        std::size_t buffer_size() const;
        // return the underlying opencl buffer
        const impl::buffer& buf() const;
        // return the opencl backend information
        impl::be_data_ptr
        backend_data();
        // return the opencl backend information
        const impl::be_data_ptr
        backend_data() const;
    };

    namespace impl {

        template <typename _T>
        struct vector_select_mask_value {
            using type = _T;
        };

        template <>
        struct vector_select_mask_value<double> {
            using type = std::int64_t;
        };

        template <>
        struct vector_select_mask_value<float> {
            using type = std::int32_t;
        };

        template <typename _T>
        using vector_select_mask_value_t =
            typename vector_select_mask_value<_T>::type;

    }

    // vector: representation of data on the acceleration device
    template <class _T>
    class vector : public vector_base {
        using base_type = vector_base;
    public:
        using value_type = _T;
        using mask_value_type = impl::vector_select_mask_value_t<_T>;
        using mask_type = vector<mask_value_type>;
        // using base_type::backend_data;
        // using base_type::buf;
        ~vector() {}
        // size of the vector
        std::size_t size() const;
        // default constructor.
        vector() : base_type{} {}
        // constructor from memory buffer
        vector(std::size_t n, const _T* s);
        // constructor with size and initializer
        vector(std::size_t n, const _T& i);
        // constructor from initializer list
        vector(std::initializer_list<_T> l);
        // constructor with size and initializer
        template <typename _U>
        vector(std::size_t n, const _U& i);
        // copy constructor
        vector(const vector& v);
        // move constructor
        vector(vector&& v);
        // construction from std::vector, forces move of data
        // from host to device
        vector(const std::vector<_T>& v);
        // assignment operator from vector
        vector& operator=(const vector& v);
        // move assignment
        vector& operator=(vector&& v);
        // assignment from scalar
        vector& operator=(const _T& i);
        // template constructor for evaluation of expressions
        template <template <class _V> class _OP,
                  class _L, class _R>
        vector(const expr<_OP<vector<_T> >, _L, _R>& r);
        // conversion operator to std::vector, forces move of
        // data to host
        explicit operator std::vector<_T> () const;
    };

    template <class _T>
    struct expr_traits<vector<_T> > {
        using type = const vector<_T>&;
    };

    // backend_data specialized for vector t
    template <class _T>
    impl::be_data_ptr
    backend_data(const vector<_T>& t);
    // eval_size specialized for vector
    template <class _T>
    std::size_t eval_size(const vector<_T>& t);
    // eval_args specialized for vector
    template <class _T>
    std::string
    eval_args(const std::string& p, const vector<_T>& r,
              unsigned& arg_num, bool ro);
    // eval_vars specialized for vector
    template <class _T>
    std::string
    eval_vars(const vector<_T>& r, unsigned& arg_num, bool read);

    // store the results into a vector
    template <class _T>
    std::string
    eval_results(vector<_T>& r, unsigned& res_num);

    // bind_args for non const arguments
    template <class _T>
    void
    bind_args(impl::kernel& k, vector<_T>& r,  unsigned& arg_num);
    // bind_args for const arguments
    template <class _T>
    void
    bind_args(impl::kernel& k, const vector<_T>& r,  unsigned& arg_num);

    namespace ops {

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
        struct abs< vector<float> > {
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
        struct abs< vector<double> >  {
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
                res += impl::type_2_name<_D>::v();
                res += ")";
                res += l;
                res += ")";
                std::cout<< res << std::endl;
                return res;
            }
        };

        template <class _D>
        struct cvt_to<vector<_D> > {
            static
            std::string body(const std::string& l) {
                std::string res("convert_");
                res += impl::type_2_name<_D>::v();
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
        struct as<vector<_D> > {
            static
            std::string body(const std::string& l) {
                std::string res("(");
                res += "as_";
                res += impl::type_2_name<_D>::v();
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
    expr<ops::cvt_to<_D>, _S, void>
    cvt_to(const _S& s) {
        return expr<ops::cvt_to<_D>, _S, void>(s);
    }

    template <class _D, class _S>
    inline
    expr<ops::as<_D>, _S, void>
    as(const _S& s) {
        return expr<ops::as<_D>, _S, void>(s);
    }

    // abs(V)
    template <class _T>
    inline
    expr<ops::abs<vector<_T> >, vector<_T>, void>
    abs(const vector<_T>& t) {
        return expr<ops::abs<vector<_T> >, vector<_T>, void>(t);
    }

    // abs(expr)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R>
    inline
    expr<ops::abs<vector<_T> >,
         expr<_OP<vector<_T> >, _L, _R>,
         void>
    abs(const expr<_OP<vector<_T> >, _L, _R>& v) {
        return expr<ops::abs<vector<_T> >,
                    expr<_OP<vector<_T> >, _L, _R>,
                    void>(v);
    }

    // min(V)
    template <class _T, class _S>
    inline
    expr<ops::min_func<vector<_T> >,
         vector<_T>, _S>
    min(const vector<_T>& a, const _S& b)
    {
        return expr<ops::min_func<vector<_T> >,
                    vector<_T>, _S >(a, b);
    }

    // min(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<ops::min_func<vector<_T> >,
         expr<_OP<vector<_T> >, _L, _R>,
         _S >
    min(const expr<_OP<vector<_T> >, _L, _R>& a, const _S& b)
    {
        return expr<ops::min_func<vector<_T> >,
                    expr<_OP<vector<_T> >, _L, _R>,
                    _S >(a, b);
    }

    // min(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<ops::min_func<vector<_T> >,
         _S,
         expr<_OP<vector<_T> >, _L, _R> >
    min(const _S& b, const expr<_OP<vector<_T> >, _L, _R>& a)
    {
        return expr<ops::min_func<vector<_T> >,
                    _S,
                    expr<_OP<vector<_T> >, _L, _R> >(a, b);
    }

    // max(V)
    template <class _T, class _S>
    inline
    expr<ops::max_func<vector<_T> >,
         vector<_T>, _S>
    max(const vector<_T>& a, const _S& b)
    {
        return expr<ops::max_func<vector<_T> >,
                    vector<_T>, _S >(a, b);
    }

    // max(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<ops::max_func<vector<_T> >,
         expr<_OP<vector<_T> >, _L, _R>,
         _S >
    max(const expr<_OP<vector<_T> >, _L, _R>& a, const _S& b)
    {
        return expr<ops::max_func<vector<_T> >,
                    expr<_OP<vector<_T> >, _L, _R>,
                    _S >(a, b);
    }

    // max(V)
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R,
              class _S>
    inline
    expr<ops::max_func<vector<_T> >,
         _S,
         expr<_OP<vector<_T> >, _L, _R> >
    max(const _S& b, const expr<_OP<vector<_T> >, _L, _R>& a)
    {
        return expr<ops::max_func<vector<_T> >,
                    _S,
                    expr<_OP<vector<_T> >, _L, _R> >(a, b);
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
    expr<ops::neg<vector<_T> >, vector<_T>, void>
    operator-(const vector<_T>& v) {
        return expr<ops::neg<vector<_T> >, vector<_T>, void>(v);
    };
    // unary minus expr
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R>
    inline
    expr<ops::neg<vector<_T> >,
         expr<_OP<vector<_T> >, _L, _R>,
         void>
    operator-(const expr<_OP<vector<_T> >, _L, _R>& v) {
        return expr<ops::neg<vector<_T> >,
                    expr<_OP<vector<_T> >, _L, _R>,
                    void>(v);
    }

    // unary not V
    template <class _T>
    inline
    expr<ops::bit_not<vector<_T> >, vector<_T>, void>
    operator~(const vector<_T>& v) {
        return expr<ops::bit_not<vector<_T> >, vector<_T>, void>(v);
    };
    // unary not expr
    template <class _T,
              template <class _T1> class _OP,
              class _L, class _R>
    inline
    expr<ops::bit_not<vector<_T> >,
         expr<_OP<vector<_T> >, _L, _R>,
         void>
    operator~(const expr<_OP<vector<_T> >, _L, _R>& v) {
        return expr<ops::bit_not<vector<_T> >,
                    expr<_OP<vector<_T> >, _L, _R>,
                    void>(v);
    }


#define DEFINE_OCLVEC_OPERATOR(op, eq_op, op_name)                      \
    /* operator op(V, V) */                                             \
    template <class _T>                                                 \
    inline                                                              \
    expr<ops:: op_name<vector<_T> >, vector<_T>, vector<_T> >           \
    operator op (const vector<_T>& a, const vector<_T>& b) {            \
        return expr<ops:: op_name<vector<_T> >,                         \
                    vector<_T>, vector<_T> >(a,b);                      \
    }                                                                   \
    /* operator op(V, _T) */                                            \
    template <class _T, class _S>                                       \
    inline                                                              \
    expr<ops:: op_name<vector<_T> >, vector<_T>, _S>                    \
    operator op (const vector<_T>& a, const _S& b) {                    \
        return expr<ops:: op_name<vector<_T> >, vector<_T>, _S>(a,b);   \
    }                                                                   \
    /* operator op(_T, V) */                                            \
    template <class _T, class _S>                                       \
    inline                                                              \
    expr<ops:: op_name<vector<_T> >, _S, vector<_T> >                   \
    operator op (const _S& a, const vector<_T>& b) {                    \
        return expr<ops:: op_name<vector<_T> >, _S, vector<_T> >(a,b);  \
    }                                                                   \
    /* operator op(V, expr) */                                          \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<ops:: op_name<vector<_T> >,                                    \
         vector<_T>,                                                    \
         expr<_OP<vector<_T> >, _L, _R> >                               \
    operator op (const vector<_T>& a,                                   \
                 const expr<_OP<vector<_T> >, _L, _R>& b) {             \
        return expr<ops:: op_name<vector<_T> >,                         \
                    vector<_T>,                                         \
                    expr<_OP<vector<_T>>, _L, _R> >(a, b);              \
    }                                                                   \
    /* operator op(_S, expr) */                                         \
    template <class _T, class _S,                                       \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<ops:: op_name<vector<_T> >, _S,                                \
         expr<_OP<vector<_T> >, _L, _R> >                               \
    operator op (const _S& a,                                           \
                 const expr<_OP<vector<_T> >, _L, _R>& b) {             \
        return expr<ops:: op_name<vector<_T> >,                         \
                    _S, expr<_OP<vector<_T> >, _L, _R> >(a, b);         \
    }                                                                   \
    /* operator op(expr, V) */                                          \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<ops:: op_name<vector<_T>>,                                     \
         expr<_OP<vector<_T> >, _L, _R>, vector<_T> >                   \
    operator op (const expr<_OP<vector<_T> >, _L, _R>& a,               \
                 const vector<_T>& b) {                                 \
        return expr<ops:: op_name<vector<_T> >,                         \
                    expr<_OP<vector<_T> >, _L, _R>,                     \
                    vector<_T> >(a, b);                                 \
    }                                                                   \
    /* operator op(expr, _S) */                                         \
    template <class _T, class _S,                                       \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    expr<ops:: op_name<vector<_T> >,                                    \
         expr<_OP<vector<_T> >, _L, _R>, _S>                            \
    operator op (const expr<_OP<vector<_T> >,                           \
                 _L, _R>& a, const _S& b) {                             \
        return expr<ops:: op_name<vector<_T> >,                         \
                    expr<_OP<vector<_T> >, _L, _R>, _S>(a, b);          \
    }                                                                   \
    /* operator op(expr, expr)  */                                      \
    template <class _T,                                                 \
              template <class _V> class _OP1, class _L1, class _R1,     \
              template <class _V> class _OP2, class _L2, class _R2>     \
    inline                                                              \
    expr<ops:: op_name<vector<_T> >,                                    \
         expr<_OP1<vector<_T> >, _L1, _R1>,                             \
         expr<_OP2<vector<_T> >, _L2, _R2> >                            \
    operator op(const expr<_OP1<vector<_T> >, _L1, _R1>& a,             \
                const expr<_OP2<vector<_T> >, _L2, _R2>& b) {           \
        return expr<ops:: op_name<vector<_T> >,                         \
                    expr<_OP1<vector<_T> >, _L1, _R1>,                  \
                    expr<_OP2<vector<_T> >, _L2, _R2> > (a, b);         \
    }                                                                   \
    /* operator eq_op V */                                              \
    template <class _T>                                                 \
    inline                                                              \
    vector<_T>& operator eq_op(vector<_T>& a, const vector<_T>& r) {    \
        a = a op r;                                                     \
        return a;                                                       \
    }                                                                   \
    /* operator eq_op _T */                                             \
    template <class _T>                                                 \
    inline                                                              \
    vector<_T>& operator eq_op(vector<_T>& a, const _T& r) {            \
        a = a op r;                                                     \
        return a;                                                       \
    }                                                                   \
    /* operator eq_op expr */                                           \
    template <class _T,                                                 \
              template <class _V> class _OP, class _L, class _R>        \
    inline                                                              \
    vector<_T>&                                                         \
    operator eq_op(vector<_T>& a,                                       \
                   const expr<_OP<vector<_T> >, _L, _R>& r) {           \
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
    expr<ops:: op_name<typename vector<_T>::mask_type >,                \
         _T, vector<_T> >                                               \
    operator op(const _T& a, const vector<_T>& b) {                     \
        return expr<ops:: op_name <typename vector<_T>::mask_type>,     \
                    _T, vector<_T> >(a, b);                             \
    }                                                                   \
                                                                        \
    template <typename _T>                                              \
    expr<ops:: op_name<typename vector<_T>::mask_type >,                \
         vector<_T>, vector<_T> >                                       \
    operator op(const vector<_T>& a, const vector<_T>& b) {             \
        return expr<ops:: op_name <typename vector<_T>::mask_type>,     \
                    vector<_T>, vector<_T> >(a, b);                     \
    }                                                                   \
                                                                        \
    template <typename _T>                                              \
    expr<ops:: op_name<typename vector<_T>::mask_type >,                \
         vector<_T>, _T>                                                \
    operator op(const vector<_T>& a, const _T& b) {                     \
        return expr<ops:: op_name <typename vector<_T>::mask_type>,     \
                    vector<_T>, _T>(a, b);                              \
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
ocl::vector<_T>::vector(std::size_t n, const _T* p)
    : base_type{n*sizeof(_T)}
{
    copy_from_host(p);
}

template <class _T>
inline
ocl::vector<_T>::vector(std::size_t s, const _T& i)
    : base_type{s * sizeof(_T)}
{
    if (s) {
        execute(*this, i);
    }
}

template <class _T>
template <typename _U>
inline
ocl::vector<_T>::vector(std::size_t s, const _U& i)
    : base_type{s * sizeof(_T)}
{
    if (s) {
        execute(*this, i);
    }
}

template <class _T>
inline
ocl::vector<_T>::vector(const vector& r)
    : base_type(r)
{
}

template <class _T>
inline
ocl::vector<_T>::vector(vector&& r)
    : base_type(std::move(r))
{
}

template <class _T>
inline
ocl::vector<_T>::vector(const std::vector<_T>& r)
    : base_type{sizeof(_T) * r.size()}
{
    copy_from_host(&r[0]);
}

template <class _T>
inline
ocl::vector<_T>::vector(std::initializer_list<_T> l)
    : base_type{sizeof(_T) * l.size()}
{
    copy_from_host(l.begin());
}

template <class _T>
template <template <class _V> class _OP, class _L, class _R>
inline
ocl::
vector<_T>::vector(const expr<_OP<vector<_T> >, _L, _R>& r)
    : base_type{eval_size(r)*sizeof(_T)}
{
    if (buffer_size()) {
        execute(*this, r);
    }
}

template <class _T>
inline
ocl::vector<_T>&
ocl::vector<_T>::operator=(const vector& r)
{
    base_type::operator=(r);
    return *this;
}

template <class _T>
inline
ocl::vector<_T>&
ocl::vector<_T>::operator=(vector&& r)
{
    base_type::operator=(std::move(r));
    return *this;
}

template <class _T>
inline
ocl::vector<_T>::operator std::vector<_T> ()
    const
{
    std::size_t n(this->size());
    std::vector<_T> v(n);
    copy_to_host(&v[0]);
    return v;
}

template <class _T>
inline
std::size_t
ocl::vector<_T>::size() const
{
    return buffer_size()/sizeof(_T);
}

template <class _T>
inline
std::size_t ocl::eval_size(const vector<_T>& v)
{
    return v.size();
}

template <class _T>
inline
ocl::impl::be_data_ptr
ocl::backend_data(const vector<_T>& v)
{
    return v.backend_data();
}

template <class _T>
std::string
ocl::eval_args(const std::string& p, const vector<_T>& r, unsigned& arg_num,
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
    s << impl::type_2_name<_T>::v()
      << "* arg"  << arg_num;
    ++arg_num;
    return s.str();
}

template <class _T>
std::string
ocl::eval_vars(const vector<_T>& r, unsigned& arg_num, bool read)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(8) << impl::type_2_name<_T>::v()
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
std::string ocl::eval_results(vector<_T>& r,
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
ocl::bind_args(impl::kernel& k, vector<_T>& r, unsigned& arg_num)
{
    if (impl::be_data::instance()->debug() != 0) {
        std::cout << "binding lvec<"
                  << impl::type_2_name<_T>::v()
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
ocl::bind_args(impl::kernel& k, const vector<_T>& r, unsigned& arg_num)
{
    if (impl::be_data::instance()->debug() != 0) {
        std::cout << "binding const lvec<"
                  << impl::type_2_name<_T>::v()
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
