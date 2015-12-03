#if !defined (__OCL_VECTOR_H__)
#define __OCL_VECTOR_H__ 1

#include <ocl/config.h>
#include <ocl/expr_kernel.h>
#include <ocl/impl_type_2_name.h>
#include <initializer_list>

namespace ocl {

    // vector base class wrapping an opencl buffer and a
    // (shared) pointer to opencl backend data
    class vector_base {
        // shared pointer to the backend data
        std::shared_ptr<impl::be_data> _bed;
        // backend buffer object
        cl::Buffer _b;
    protected:
        // default constructor
        vector_base();
        // constructor, with size
        explicit vector_base(std::size_t s);
        // constructor, copies s bytes from src to
        // buffer
        vector_base(std::size_t s, const char* src);
        // swap two vector base objects
        vector_base& swap(vector_base& r);
    public:
        // return the size of the vector in bytes
        std::size_t buffer_size() const;
        // return the underlying opencl buffer
        const cl::Buffer& buf() const;
        // return the opencl backend information
        std::shared_ptr<impl::be_data>&
        backend_data();
        // return the opencl backend information
        const std::shared_ptr<impl::be_data>&
        backend_data() const;
    };

    // vector: representation of data on the acceleration device
    template <class _T>
    class vector : public vector_base {
        // count of elements
        std::size_t _size;
    public:
        using value_type = _T;
        // size of the vector
        std::size_t size() const;
        // default constructor.
        vector() : vector_base{} {}
        // constructor from memory buffer
        vector(std::size_t n, const _T* s);
        // constructor with size and initializer
        vector(std::size_t n, const _T& i);
        // constructor from initializer list
        vector(std::initializer_list<_T> l);
        // copy constructor
        vector(const vector& v);
        // construction from std::vector, forces move of data
        // from host to device
        vector(const std::vector<_T>& v);
        // assignment operator from vector
        vector& operator=(const vector& v);
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
        typedef const vector<_T>& type;
    };

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
    bind_args(cl::Kernel& k, vector<_T>& r,  unsigned& arg_num);
    // bind_args for const arguments
    template <class _T>
    void
    bind_args(cl::Kernel& k, const vector<_T>& r,  unsigned& arg_num);

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

            struct add{ const char* operator()() const { return "+"; }};
            struct sub{ const char* operator()() const { return "-"; }};
            struct mul{ const char* operator()() const { return "*"; }};
            struct div{ const char* operator()() const { return "/"; }};

            struct bit_and{ const char* operator()() const { return "&"; }};
            struct bit_or{ const char* operator()() const { return "|"; }};
            struct bit_xor{ const char* operator()() const { return "^"; }};

            struct shl{ const char* operator()() const { return "<<"; }};
            struct shr{ const char* operator()() const { return ">>"; }};
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


}


inline
ocl::vector_base::vector_base()
    : _bed{}, _b{}
{
}


inline
ocl::vector_base::vector_base(std::size_t s)
    : _bed{impl::be_data::instance()},
      _b{_bed->c(), CL_MEM_READ_WRITE, s}
{
}

inline
ocl::vector_base::vector_base(std::size_t s, const char* src)
    : _bed{impl::be_data::instance()},
      _b{_bed->c(), CL_MEM_READ_WRITE, s}
{
    impl::queue& q= _bed->q();
    q.enqueueWriteBuffer(_b,
                         true,
                         0, s,
                         src,
                         nullptr,
                         nullptr);
}

inline
ocl::vector_base&
ocl::vector_base::swap(vector_base& r)
{
    std::swap(_bed, r._bed);
    std::swap(_b, r._b);
    return *this;
}

inline
std::size_t
ocl::vector_base::buffer_size()
    const
{
    std::size_t r{_b() != nullptr ?
            _b.getInfo<CL_MEM_SIZE>(nullptr) : 0};
    return r;
}

inline
const cl::Buffer&
ocl::vector_base::buf()
    const
{
    return _b;
}

inline
std::shared_ptr<ocl::impl::be_data>&
ocl::vector_base::backend_data()
{
    return _bed;
}

inline
const std::shared_ptr<ocl::impl::be_data>&
ocl::vector_base::backend_data()
    const
{
    return _bed;
}

template <class _T>
inline
ocl::vector<_T>::vector(std::size_t n, const _T* p)
    : vector_base{n*sizeof(_T)}, _size{n}
{
    if (_size) {
        std::size_t s=_size*sizeof(_T);
        std::shared_ptr<impl::be_data>& bed=
            this->backend_data();
        impl::queue& q= bed->q();
        q.enqueueWriteBuffer(this->buf(),
                             true,
                             0, s,
                             p,
                             nullptr,
                             nullptr);
    }
}

template <class _T>
inline
ocl::vector<_T>::vector(std::size_t s, const _T& i)
    : vector_base{s * sizeof(_T)}, _size{s}
{
    if (_size) {
        execute(*this, i);
    }
}

template <class _T>
inline
ocl::vector<_T>::vector(const vector& r)
    : vector_base{r.size() * sizeof(_T)},
    _size{r.size()}
{
    if (_size) {
#pragma message("copy constructor via opencl")
        execute(*this, r);
    }
}

template <class _T>
inline
ocl::vector<_T>::vector(const std::vector<_T>& r)
    : vector_base{sizeof(_T) * r.size()}, _size{r.size()}
{
    if (_size) {
        std::size_t s=_size*sizeof(_T);
        impl::queue& q= backend_data()->q();
        q.enqueueWriteBuffer(this->buf(),
                             true,
                             0, s,
                             &r[0],
                             nullptr,
                             nullptr);
    }
}

template <class _T>
inline
ocl::vector<_T>::vector(std::initializer_list<_T> l)
    : vector_base{sizeof(_T) * l.size()}, _size{l.size()}
{
    if (_size) {
        std::size_t s=_size*sizeof(_T);
        impl::queue& q= backend_data()->q();
        q.enqueueWriteBuffer(this->buf(),
                             true,
                             0, s,
                             l.begin(),
                             nullptr,
                             nullptr);
    }
}

template <class _T>
template <template <class _V> class _OP, class _L, class _R>
inline
ocl::
vector<_T>::vector(const expr<_OP<vector<_T> >, _L, _R>& r)
    : vector_base{eval_size(r)*sizeof(_T)}, _size{eval_size(r)}
{
    if (_size) {
        execute(*this, r);
    }
}

template <class _T>
inline
ocl::vector<_T>&
ocl::vector<_T>::operator=(const vector& r)
{
    if (this != &r) {
        if (size() == r.size()) {
            execute(*this, r);
        } else {
            vector t(r);
            swap(t);
            std::swap(this->_size, t._size);
        }
    }
    return *this;
}

template <class _T>
inline
ocl::vector<_T>::operator std::vector<_T> ()
    const
{
    std::size_t n(this->size());
    std::vector<_T> v(n);
    if (n) {
        std::size_t s(n*sizeof(_T));
        std::shared_ptr<impl::be_data> bed(backend_data());
        impl::queue& q= bed->q();
        q.enqueueReadBuffer(this->buf(),
                            true,
                            0, s,
                            &v[0],
                            nullptr,
                            nullptr);
    }
    return v;
}

template <class _T>
inline
std::size_t
ocl::vector<_T>::size() const
{
    return _size;
}

template <class _T>
inline
std::size_t ocl::eval_size(const vector<_T>& v)
{
    return v.size();
}

template <class _T>
std::string ocl::eval_args(const std::string& p,
                           const vector<_T>& r,
                           unsigned& arg_num,
                           bool ro)
{
    std::ostringstream s;
    if (!p.empty()) {
        s << p << ",\n";
    }
    s << "\t__global " ;
    if (ro) {
        s<< "const ";
    }
    s << impl::type_2_name<_T>::v()
      << "* arg"  << arg_num;
    ++arg_num;
    return s.str();
}

template <class _T>
std::string ocl::eval_vars(const vector<_T>& r, unsigned& arg_num,
                           bool read)
{
    std::ostringstream s;
    s << '\t' << impl::type_2_name<_T>::v()
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
    std::ostringstream s;
    s << "\targ" << res_num << "[gid]="
      << " v" << res_num << ';';
    ++res_num;
    return s.str();
}


template <class _T>
void ocl::bind_args(cl::Kernel& k,
                    vector<_T>& r,
                    unsigned& arg_num)
{
    std::cout << "binding buffer to arg " << arg_num
              << std::endl;
    k.setArg(arg_num, r.buf());
    ++arg_num;
}

template <class _T>
void ocl::bind_args(cl::Kernel& k,
                    const vector<_T>& r,
                    unsigned& arg_num)
{
    std::cout << "binding constant buffer to arg " << arg_num
              << std::endl;
    k.setArg(arg_num, r.buf());
    ++arg_num;
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_VECTOR_H__
