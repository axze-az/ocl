#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <cmath>


namespace ocl {

        // eval_size: template helper function determining
        // the vector length of an expression
        template <class _T>
        std::size_t eval_size(const _T& t) {
                return 1;
        }

        // eval_size specialized for std::vector, returns
        // the size of the vector
        template <class _T>
        std::size_t eval_size(const std::vector<_T>& v) {
                return v.size();
        }

        // return an ascii name for r with "vx" with x
        // describing the argument number, arg_num is increased
        template <class _T>
        std::string eval_ops(const _T& r, unsigned& arg_num) {
                std::ostringstream s;
                s << "v" << arg_num;
                std::string a(s.str());
                ++arg_num;
                return a;
        }

        // generate: type tX = argX;
        // for temporary variables
        template <class _T>
        std::string eval_vars(const _T& r, unsigned& arg_num,
                              bool read) {
                std::ostringstream s;
                s << '\t' << impl::type_2_name<_T>::v()
                  << " v" << arg_num;
                if (read== true) {
                        s << " = arg"
                          << arg_num << ";";
                }
                std::string a(s.str());
                ++arg_num;
                return a;
        }

        // generate: type argX
        // for kernel arguments
        template <class _T>
        std::string eval_args(const std::string& p,
                              const _T& r,
                              unsigned& arg_num,
                              bool ro) {
                static_cast<void>(ro);
                std::ostringstream s;
                if (!p.empty()) {
                        s << p << ",\n";
                }
                s << "\t" ;
                s << impl::type_2_name<_T>::v()
                  << " arg"  << arg_num;
                ++arg_num;
                return s.str();
        }

        // bind an openCL kernel argument
        template <class _T>
        void bind_args(cl::Kernel& k,
                       const _T& r,
                       unsigned& arg_num)
        {
                std::cout << "binding to arg " << arg_num
                          << std::endl;
                k.setArg(arg_num, r);
                ++arg_num;
        }

        // default expression traits for simple/unspecialized
        // types
        template <class _T>
        struct expr_traits {
                typedef const _T type;
        };

        // the expression template
        template <class _OP, class _L, class _R>
        struct expr {
                typename expr_traits<_L>::type _l;
                typename expr_traits<_R>::type _r;
                constexpr expr(const _L& l, const _R& r) :
                        _l(l), _r(r) {}
        };

        // eval_size specialized for expr<>, returns
        // the maximum of all eval_size function applied
        // recursivly
        template <class _OP, class _L, class _R>
        std::size_t eval_size(const expr<_OP, _L, _R>& a)
        {
                std::size_t l=eval_size(a._l);
                std::size_t r=eval_size(a._r);
                return std::max(l, r);
        }

        template <class _OP, class _L, class _R>
        std::string
        eval_vars(const expr<_OP, _L, _R>& a, unsigned& arg_num,
                  bool read)
        {
                auto l=eval_vars(a._l, arg_num, read);
                auto r=eval_vars(a._r, arg_num, read);
                return std::string(l + '\n' + r);
        }

        template <class _OP, class _L, class _R>
        std::string
        eval_ops(const expr<_OP, _L, _R>& a, unsigned& arg_num)
        {
                auto l=eval_ops(a._l, arg_num);
                auto r=eval_ops(a._r, arg_num);
                std::string t(_OP::body(l, r));
                return std::string("(") + t + std::string(")");
        }

        template <class _OP, class _L, class _R>
        std::string
        eval_args(const std::string& p,
                  const expr<_OP, _L, _R>& r,
                  unsigned& arg_num,
                  bool ro)
        {
                std::string left(eval_args(p, r._l, arg_num, ro));
                return eval_args(left, r._r, arg_num, ro);
        }


        template <class _OP, class _L, class _R>
        void bind_args(cl::Kernel& k,
                       const expr<_OP, _L, _R>& r,
                       unsigned& arg_num)
        {
                bind_args(k, r._l, arg_num);
                bind_args(k, r._r, arg_num);
        }


        template <class _RES, class _EXPR>
        class expr_kernel {
        public:
                expr_kernel();
                void
                execute(_RES& res, const _EXPR& r, const void* addr)
                        const;
        private:
                impl::pgm_kernel_lock&
                get_kernel(_RES& res, const _EXPR& r, const void* addr)
                        const;

                impl::pgm_kernel_lock
                gen_kernel(_RES& res, const _EXPR& r, const void* addr)
                        const;
        };

        template <class _RES, class _EXPR>
        void execute(_RES& res, const _EXPR& r);

        class vector_base {
                // shared pointer to the backend data
                std::shared_ptr<impl::be_data> _bed;
                // backend buffer object
                cl::Buffer _b;
        public:
                vector_base() : m_bed(), _b() {}
                vector_base(std::size_t s) 
                        : _bed(impl::be_data::instance()),
                          _b(_bed->c(),
                             CL_MEM_READ_WRITE,
                             s) {
                }
                vector_base(std::size_t s, const char* src)
                        : _bed(impl::be_data::instance()),
                          _b(_bed->c(),
                             CL_MEM_READ_WRITE,
                             s) {
                        impl::queue& q= _bed->q();
                        q.enqueueWriteBuffer(_b,
                                             true,
                                             0, s,
                                             src,
                                             nullptr,
                                             nullptr);
                }
                std::size_t buffer_size() const {
                        std::size_t r(
                                _b.GetInfo<CL_MEM_SIZE>(nullptr));
                        return r;
                }
                const cl::Buffer& buf() const {
                        return _b;
                }
                std::shared_ptr<impl::be_data>& backend_data() {
                        return m_bed;
                }
        };

        // vector: representation of data on the acceleration device
        template <class _T>
        class vector {
                // shared pointer to the backend data
                std::shared_ptr<impl::be_data> m_bed;
                // count of elements
                std::size_t _size;
                // backend buffer object
                cl::Buffer _b;
        public:
                // size of the vector
                std::size_t size() const {
                        return _size;
                }
                // return the underlying opencl buffer
                const cl::Buffer& buf() const  {
                        return _b;
                }
                // return the opencl backend information
                std::shared_ptr<impl::be_data>&
                backend_data() {
                        return m_bed;
                }
                // default constructor.
                vector() : _size(0), _b() {}
                // constructor from memory buffer
                vector(std::size_t n, const _T* s);
                // constructor with size and initializer
                vector(std::size_t n, const _T& i);
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
                explicit vector(std::size_t n);
                template <template <class _V> class _OP,
                          class _L, class _R>
                vector(const expr<_OP<vector<_T> >, _L, _R>& r);
                // conversion operator to std::vector, forces move of
                // data to host
                operator std::vector<_T> () const;
        };

        template <class _T>
        struct expr_traits<vector<_T> > {
                typedef const vector<_T>& type;
        };

        template <class _T>
        std::size_t eval_size(const vector<_T>& v) {
                return v.size();
        }


        template <class _T>
        std::string eval_vars(const vector<_T>& r, unsigned& arg_num,
                              bool read) {
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
        std::string eval_args(const std::string& p,
                              const vector<_T>& r,
                              unsigned& arg_num,
                              bool ro) {
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
        std::string eval_results(vector<_T>& r,
                                 unsigned& res_num) {
                std::ostringstream s;
                s << "\targ" << res_num << "[gid]="
                  << " v" << res_num << ';';
                ++res_num;
                return s.str();
        }


        template <class _T>
        void bind_args(cl::Kernel& k,
                       const vector<_T>& r,
                       unsigned& arg_num)
        {
                std::cout << "binding buffer to arg " << arg_num
                          << std::endl;
                k.setArg(arg_num, r.buf());
                ++arg_num;
        }

        namespace ops {

                template <class _T>
                struct add {
                        static
                        std::string body(const std::string& l,
                                         const std::string& r) {
                                std::string res(l);
                                res += " + ";
                                res += r;
                                return res;
                        }
                };

                template <class _T>
                struct sub {
                        static
                        std::string body(const std::string& l,
                                         const std::string& r) {
                                std::string res(l);
                                res += " - ";
                                res += r;
                                return res;
                        }
                };

                template <class _T>
                struct mul {
                        static
                        std::string body(const std::string& l,
                                         const std::string& r) {
                                std::string res(l);
                                res += " * ";
                                res += r;
                                return res;
                        }
                };

                template <class _T>
                struct div {
                        static
                        std::string body(const std::string& l,
                                         const std::string& r) {
                                std::string res(l);
                                res += " / ";
                                res += r;
                                return res;
                        }
                };

        }

#define DEFINE_OCLVEC_FP_OPERATOR(vx, scalar, op, eq_op, op_name)       \
        /* operator op(V, V) */                                         \
        inline                                                          \
        expr<ops:: op_name<vx>, vx, vx>                                 \
        operator op (const vx& a, const vx& b) {                        \
                return expr<ops:: op_name<vx>, vx, vx>(a,b);            \
        }                                                               \
        /* operator op(V, scalar) */                                    \
        inline                                                          \
        expr<ops:: op_name<vx>, vx, scalar>                             \
        operator op (const vx& a, const scalar& b) {                    \
                return expr<ops:: op_name<vx>, vx, scalar>(a,b);        \
        }                                                               \
        /* operator op(scalar, V) */                                    \
        inline                                                          \
        expr<ops:: op_name<vx>, scalar, vx>                             \
        operator op (const scalar& a, const vx& b) {                    \
                return expr<ops:: op_name<vx>, scalar, vx>(a,b);        \
        }                                                               \
        /* operator op(V, expr) */                                      \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, vx, expr<_OP<vx>, _L, _R> >             \
        operator op (const vx& a, const expr<_OP<vx>, _L, _R>& b) {     \
                return expr<ops:: op_name<vx>,                          \
                            vx, expr<_OP<vx>, _L, _R> >(a, b);          \
        }                                                               \
        /* operator op(scalar, expr) */                                 \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, scalar, expr<_OP<vx>, _L, _R> >         \
        operator op (const scalar& a, const expr<_OP<vx>, _L, _R>& b) { \
                return expr<ops:: op_name<vx>,                          \
                            scalar, expr<_OP<vx>, _L, _R> >(a, b);      \
        }                                                               \
        /* operator op(expr, V) */                                      \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, expr<_OP<vx>, _L, _R>, vx>              \
        operator op (const expr<_OP<vx>, _L, _R>& a, const vx& b) {     \
                return expr<ops:: op_name<vx>,                          \
                            expr<_OP<vx>, _L, _R>, vx>(a, b);           \
        }                                                               \
        /* operator op(expr, scalar) */                                 \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, expr<_OP<vx>, _L, _R>, scalar>          \
        operator op (const expr<_OP<vx>, _L, _R>& a, const scalar& b) { \
                return expr<ops:: op_name<vx>,                          \
                            expr<_OP<vx>, _L, _R>, scalar>(a, b);       \
        }                                                               \
        /* operator op(expr, expr)  */                                  \
        template <template <class _V> class _OP1, class _L1, class _R1, \
                  template <class _V> class _OP2, class _L2, class _R2> \
        inline                                                          \
        expr<ops:: op_name<vx>,                                         \
             expr<_OP1<vx>, _L1, _R1>, expr<_OP2<vx>, _L2, _R2> >       \
        operator op(const expr<_OP1<vx>, _L1, _R1>& a,                  \
                    const expr<_OP2<vx>, _L2, _R2>& b) {                \
                return expr<ops:: op_name<vx>,                          \
                            expr<_OP1<vx>, _L1, _R1>,                   \
                            expr<_OP2<vx>, _L2, _R2> > (a, b);          \
        }                                                               \
        /* operator eq_op V */                                          \
        inline                                                          \
        vx& operator eq_op(vx& a, const vx& r) {                        \
                a = a op r;                                             \
                return a;                                               \
        }                                                               \
        /* operator eq_op scalar */                                     \
        inline                                                          \
        vx& operator eq_op(vx& a, const scalar& r) {                    \
                a = a op r;                                             \
                return a;                                               \
        }                                                               \
        /* operator eq_op expr */                                       \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        vx& operator eq_op(vx& a, const expr<_OP<vx>, _L, _R>& r) {     \
                a = a op r;                                             \
                return a;                                               \
        }


#define DEFINE_OCLVEC_FP_OPERATORS(vx, scalar)                   \
        DEFINE_OCLVEC_FP_OPERATOR(vx, scalar, +, +=, add)        \
        DEFINE_OCLVEC_FP_OPERATOR(vx, scalar, -, -=, sub)        \
        DEFINE_OCLVEC_FP_OPERATOR(vx, scalar, *, *=, mul)        \
        DEFINE_OCLVEC_FP_OPERATOR(vx, scalar, /, /=, div)

#define DEFINE_OCLSCALAR_FP_OPERATORS(vx)                  \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, +, +=, add)       \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, -, -=, sub)       \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, *, *=, mul)       \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, /, /=, div)


        DEFINE_OCLVEC_FP_OPERATORS(vector<float>, float);
        DEFINE_OCLVEC_FP_OPERATORS(vector<cftal::vec::v8f32>, cftal::vec::v8f32);
}


template <class _RES, class _SRC>
ocl::expr_kernel<_RES, _SRC>::expr_kernel()
{
}

template <class _RES, class _SRC>
void
ocl::expr_kernel<_RES, _SRC>::
execute(_RES& res, const _SRC& r, const void* cookie)
        const
{
        impl::pgm_kernel_lock& pk=get_kernel(res, r, cookie);

        std::unique_lock<impl::pgm_kernel_lock> _l(pk);
        // bind args
        unsigned arg_num{0};
        bind_args(pk._k, res, arg_num);
        bind_args(pk._k, r, arg_num);
        // execute the kernel
        std::size_t s(eval_size(res));

        impl::queue& q= impl::be_data::instance()->q();
        impl::device& d= impl::be_data::instance()->d();
        // execute
        std::cout << "executing kernel" << std::endl;
        std::size_t local_size(
                pk._k.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(d, nullptr));
        local_size = std::min(local_size, s);
        std::cout << "kernel: global size: " << s
                  << " local size: " << local_size << std::endl;

        q.enqueueNDRangeKernel(pk._k,
                               cl::NullRange,
                               cl::NDRange(s),
                               cl::NDRange(local_size),
                               nullptr);
        std::cout << "execution done" << std::endl;
        // q.flush();
}

template <class _RES, class _SRC>
ocl::impl::pgm_kernel_lock&
ocl::expr_kernel<_RES, _SRC>::
get_kernel(_RES& res, const _SRC& r, const void* cookie)
        const
{
        using namespace impl;
        impl::be_data_ptr& b= res.backend_data();

        std::unique_lock<be_data> _l(*b);

        be_data::iterator f(b->find(cookie));
        if (f == b->end()) {
                pgm_kernel_lock pkl(gen_kernel(res, r, cookie));
                std::pair<be_data::iterator, bool> ir(
                        b->insert(cookie, pkl));
                f = ir.first;
        } else {
                std::cout << "using cached kernel expr_kernel_" << cookie
                          << std::endl;
        }
        return f->second;
}

template <class _RES, class _SRC>
ocl::impl::pgm_kernel_lock
ocl::expr_kernel<_RES, _SRC>::
gen_kernel(_RES& res, const _SRC& r, const void* cookie)
        const
{
        std::ostringstream s;
        s << "expr_kernel_" << cookie;
        std::string k_name(s.str());
        s.str("");

        s << "__kernel void " << k_name
          << std::endl
          << "(\n";

        // argument generation
        unsigned arg_num{0};
        s << eval_args("", res, arg_num, false)
          << ','
          << std::endl;
        s << eval_args("", r, arg_num, true) << std::endl;

        s << ")" << std::endl;

        // begin body
        s << "{" << std::endl;

        // global id
        s << "\tsize_t gid = get_global_id(0);" << std::endl;

        // temporary variables
        unsigned var_num{1};
        s << eval_vars(r, var_num, true)
          << std::endl;

        // result variable
        unsigned res_num{0};
        s << eval_vars(res, res_num, false) << "= "
          << std::endl;
        // the operations
        unsigned body_num{1};
        s << "\t\t" << eval_ops(r, body_num) << ';'
          << std::endl;
        // write back
        res_num = 0;
        s << eval_results(res, res_num)
          << std::endl;

        // end body
        s << "}" << std::endl;

        std::cout << "--- source code ------------------\n";
        std::cout << s.str();

        std::string ss(s.str());
        cl::Program::Sources sv;
        sv.push_back(std::make_pair(ss.c_str(), ss.size()));

        using namespace impl;
        be_data_ptr& bd= res.backend_data();

        cl::Program pgm(bd->c(), sv);
        std::vector<cl::Device> vk(1, bd->d());
        pgm.build(vk);
        kernel k(pgm, k_name.c_str());

        std::cout << "-- compiled with success ---------\n";

        pgm_kernel_lock pkl(pgm, k);
        return pkl;
}


template <class _RES, class _EXPR>
void
ocl::execute(_RES& res, const _EXPR& r)
{
        auto pf=ocl::execute<_RES, _EXPR>;
        const void* pv=reinterpret_cast<const void*>(pf);
        expr_kernel<_RES, _EXPR> k;
        k.execute(res, r, pv);
}

template <class _T>
inline
ocl::vector<_T>::vector(std::size_t n, const _T* p)
        : _size(n), _b()
{
        if (_size) {
                std::size_t s=_size*sizeof(_T);
                impl::be_data* bed= impl::be_data::instance();
                _b = impl::buffer(bed->c(),
                                  CL_MEM_READ_WRITE,
                                  s);
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
        : m_bed(impl::be_data::instance()), _size(s), _b()
{
        if (_size) {
                _b = impl::buffer(m_bed->c(),
                                  CL_MEM_READ_WRITE,
                                  _size*sizeof(_T));
                execute(*this, i);
        }
}

template <class _T>
inline
ocl::vector<_T>::vector(const vector& r)
        : m_bed(r.m_bed) , _size(r._size), _b()
{
        if (_size) {
                _b = impl::buffer(m_bed->c(),
                                  CL_MEM_READ_WRITE,
                                  _size*sizeof(_T));
                execute(*this, r);
        }
}

template <class _T>
inline
ocl::vector<_T>::vector(const std::vector<_T>& r)
        : m_bed(impl::be_data::instance()), _size(r.size()), _b()
{
        if (_size) {
                std::size_t s=_size*sizeof(_T);
                _b = impl::buffer(m_bed->c(),
                                  CL_MEM_READ_WRITE,
                                  s);
                impl::queue& q= m_bed->q();
                q.enqueueWriteBuffer(this->buf(),
                                     true,
                                     0, s,
                                     &r[0],
                                     nullptr,
                                     nullptr);
        }
}

template <class _T>
template <template <class _V> class _OP, class _L, class _R>
inline
ocl::vector<_T>::vector(const expr<_OP<vector<_T> >, _L, _R>& r)
        : m_bed(impl::be_data::instance()), _size(eval_size(r)), _b()
{
        // typedef expr_kernel<_OP<vector<_T> >, _L, _R> kernel_t;
        // expr_t::_k.execute(*this, r);
        if (_size) {
                _b = impl::buffer(m_bed->c(),
                                  CL_MEM_READ_WRITE,
                                  _size*sizeof(_T));
                execute(*this, r);
        }
}

template <class _T>
inline
ocl::vector<_T>::operator std::vector<_T> ()
        const
{
        std::size_t n(this->size());
        std::size_t s(n*sizeof(_T));
        std::vector<_T> v(n);
        impl::queue& q= m_bed->q();
        q.enqueueReadBuffer(this->buf(),
                            true,
                            0, s,
                            &v[0],
                            nullptr,
                            nullptr);
        return v;
}


// using namespace ocl;

template <class _T>
_T
test_func(const _T& a, const _T& b)
{
        // return _T( (2.0 + a + b) / (a * b)  + (a + a * b ) - a);

        return _T((2.0 + a + b) / (a * b)  + (a + a * b ) - a) *
                ((6.0 + a + b) / (a * b)  + (a + a * b ) - a);
}

template <class _T>
_T
test_func(const _T& a, const _T& b, const _T& c)
{
        return _T((a+b *c) *c + 2.0f);
}

namespace {

        template <class _T>
        _T rel_error(const _T& a, const _T& b)
        {
                _T e((a -b ));
                e = e < _T(0) ? -e : e;
                _T m((a+b)*_T(0.5));
                if (m != _T(0)) {
                        e /= m;
                }
                return e;
        }

}


int main()
{
        try {

                using namespace ocl;

                using cftal::vec::v8f32;

                const unsigned SIZE=16384;
                const unsigned BEIGNET_MAX_BUFFER_SIZE=16384*4096;
                std::cout << "using buffers of "
                          << double(SIZE*sizeof(float))/(1024*1024)
                          << "MiB\n";
                float a(2.0f), b(3.0f);

                vector<float> v0(SIZE, a);
                // std::vector<float> vha(SIZE, a);
                vector<float> va(v0);
                std::vector<float> vhb(SIZE, 3.0f);
                vector<float> vb(vhb);
                vector<float> vc= test_func(va, vb);
                vector<float> vd= test_func(va, vb, vc);
                vector<float> vd2= test_func(va, vb, vc);

                float c= test_func(a, b);
                float d= test_func(a, b, c);

                std::vector<float> res(vd);

                vector<v8f32> vva(SIZE, a);
                vector<v8f32> vvb(SIZE, b);
                vector<v8f32> vvc(SIZE, c);
                vector<v8f32> vres(test_func(vva, vvb, vvc));

                if (SIZE <= 4096) {
                        for (std::size_t i=0; i< res.size(); ++i) {
                                std::cout << i << ' ' << res[i] << std::endl;
                        }
                } else {
                        for (std::size_t i=0; i< res.size(); ++i) {
                                float e=rel_error(res[i], d);
                                if (e > 1e-7) {
                                        std::ostringstream m;
                                        m << "res[" << i << " ]="
                                          << std::setprecision(12)
                                          << res[i] << " != " << d
                                          << " e= " << e;
                                        throw std::runtime_error(m.str());
                                }
                        }
                }

                std::cout << "scalar " << d << std::endl;

        }
        catch (const std::runtime_error& e) {
                std::cout << "caught exception: " << e.what()
                           << std::endl;
        }
        catch (const ocl::impl::error& e) {
                std::cout << "caught exception: " << e.what()
                          << '\n'
                          << ocl::impl::err2str(e)
                          << std::endl;
        }
#if 1
        std::vector<cl::Device> v(ocl::impl::devices());
        std::cout << v.size() << std::endl;
        for (std::size_t i = 0; i< v.size(); ++i) {
                std::cout << ocl::impl::device_info(v[i]);
        }
        ocl::impl::device dd(ocl::impl::default_device());
        std::cout << "selected device: \n";
        std::cout << ocl::impl::device_info(dd);

        try {
                const ocl::impl::device& bed =
                        ocl::impl::be_data::instance()->d();
                std::cout << "\nselected backend device: \n";
                std::cout << ocl::impl::device_info(bed);
        }
        catch (const ocl::impl::error& e) {
                std::cout << "caught exception: " << e.what()
                          << '\n'
                          << ocl::impl::err2str(e)
                          << std::endl;
        }
#endif
        return 0;
}
