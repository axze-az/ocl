#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>

#if !defined (CL_MEM_HOST_READ_ONLY)
#define CL_MEM_HOST_READ_ONLY CL_MEM_READ_ONLY
#endif

namespace ocl {

        namespace impl {
                class be_data {
                public:
                        be_data(const be_data&) = delete;
                        be_data& operator=(const be_data&) = delete;

                        void lock() {
                                _m.lock();
                        }

                        bool try_lock() {
                                return _m.try_lock();
                        }

                        void unlock() {
                                _m.unlock();
                        }

                        device& d() {
                                return _d;
                        }
                        queue& q() {
                                return _q;
                        }
                        context& c() {
                                return _c;
                        }

                        typedef std::pair<std::mutex, kernel> kernel_type;
                        typedef std::map<void*, kernel_type> kernel_map_type;
                        typedef kernel_map_type::iterator iterator_type;
                        typedef kernel_map_type::const_iterator 
                        const_iterator_type;

                        const_iterator_type
                        find(void* cookie)
                                const;

                        
                        
                        
                        static
                        be_data* instance();
                private:
                        std::mutex _m;
                        device _d;
                        context _c;
                        queue _q;
                        kernel_map_type _kmap;
                        
                        be_data();
                        static be_data* _instance;
                };

                context& be_context() {
                        return be_data::instance()->c();
                }
                
                device& be_device() {
                        return be_data::instance()->d();
                }

                queue& be_queue() {
                        return be_data::instance()->q();
                }

        }


        template <class _T>
        std::size_t eval_size(const _T& t) {
                return 1;
        }

        template <class _T>
        std::size_t eval_size(const std::vector<_T>& v) {
                return v.size();
        }


        template <class _T>
        std::string eval_ops(const _T& r, unsigned& arg_num) {
                std::ostringstream s;
                s << "v" << arg_num;
                std::string a(s.str());
                ++arg_num;
                return a;
        }

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

        template <class _T>
        struct expr_traits {
                typedef const _T type;
        };

        template <class _OP, class _L, class _R> 
        struct expr {
                typename expr_traits<_L>::type _l;
                typename expr_traits<_R>::type _r;
                constexpr expr(const _L& l, const _R& r) :
                        _l(l), _r(r) {}
        };

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
                std::pair<void*, impl::kernel>&
                get_kernel(_RES& res, const _EXPR& r, const void* addr)
                        const;

                impl::kernel 
                gen_kernel(_RES& res, const _EXPR& r, const void* addr)
                        const;
        };

        template <class _RES, class _EXPR>
        void execute(_RES& res, const _EXPR& r);


        
        template <class _T>
        class vec {
                std::size_t _size;
                cl::Buffer _b;
        public:
                std::size_t size() const {
                        return _size;
                }
                cl::Buffer buf() const  {
                        return _b; 
                }
                vec() : _size(0), _b() {}
                vec(std::size_t n, const _T* s);
                vec(std::size_t n, const _T& i);
                vec(const vec& v);
                vec(const std::vector<_T>& v);
                vec& operator=(const vec& v);
                vec& operator=(const _T& i);

                explicit vec(std::size_t n);
                template <template <class _V> class _OP, 
                          class _L, class _R>
                vec(const expr<_OP<vec<_T> >, _L, _R>& r);
                
                operator std::vector<_T> () const;
        };

        template <class _T>
        struct expr_traits<vec<_T> > {
                typedef const vec<_T>& type;
        };

        template <class _T>
        std::size_t eval_size(const vec<_T>& v) {
                return v.size();
        }

       
        template <class _T>
        std::string eval_vars(const vec<_T>& r, unsigned& arg_num, 
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
                              const vec<_T>& r,
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
        std::string eval_results(vec<_T>& r, 
                                 unsigned& res_num) {
                std::ostringstream s;
                s << "\targ" << res_num << "[gid]=" 
                  << " v" << res_num << ';';
                ++res_num;
                return s.str();
        }


        template <class _T>
        void bind_args(cl::Kernel& k, 
                       const vec<_T>& r, 
                       unsigned&  arg_num)
        {
                std::cout << "binding to arg " << arg_num 
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


        DEFINE_OCLVEC_FP_OPERATORS(vec<float>, float);

        
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
        ocl::impl::kernel k(gen_kernel(res, r, cookie));
        // bind args
        unsigned arg_num{0};
        bind_args(k, res, arg_num);
        bind_args(k, r, arg_num);
        // execute the kernel
        std::size_t s(eval_size(res));

        impl::queue& q= impl::be_data::instance()->q();
        // execute
        std::cout << "executing kernel" << std::endl;
        q.enqueueNDRangeKernel(k, 
                               cl::NullRange,
                               cl::NDRange(s),
                               cl::NullRange,
                               nullptr);
        std::cout << "excution done" << std::endl;
        q.flush();
}


template <class _RES, class _SRC>
ocl::impl::kernel
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

        cl::Program pgm(impl::be_data::instance()->c(),
                        sv);
        std::vector<impl::device> vk(1, impl::be_data::instance()->d());
        pgm.build(vk);
        impl::kernel k(pgm, k_name.c_str());

        std::cout << "-- compiled with success ---------\n";

        return k;
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
ocl::vec<_T>::vec(std::size_t n, const _T* p)
        : _size(n), _b()
{
        if (_size) {
                std::size_t s=_size*sizeof(_T);
                _b = impl::buffer(impl::be_context(),
                                  CL_MEM_READ_WRITE,
                                  s);
                impl::queue& q= impl::be_queue();
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
ocl::vec<_T>::vec(std::size_t s, const _T& i)
        : _size(s), _b()
{
        if (_size) {
                _b = impl::buffer(impl::be_context(),
                                  CL_MEM_READ_WRITE, 
                                  _size*sizeof(_T));
                execute(*this, i);
        }
}

template <class _T>
inline
ocl::vec<_T>::vec(const vec& r)
        : _size(r._size), _b()
{
        if (_size) {
                _b = impl::buffer(impl::be_context(),
                                  CL_MEM_READ_WRITE, 
                                  _size*sizeof(_T));
                execute(*this, r);
        }
}

template <class _T>
inline
ocl::vec<_T>::vec(const std::vector<_T>& r)
        : _size(r.size()), _b()
{
        if (_size) {
                std::size_t s=_size*sizeof(_T);
                _b = impl::buffer(impl::be_context(),
                                  CL_MEM_READ_WRITE,
                                  s);
                impl::queue& q= impl::be_queue();
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
ocl::vec<_T>::vec(const expr<_OP<vec<_T> >, _L, _R>& r)
        : _size(eval_size(r)), _b()
{
        // typedef expr_kernel<_OP<vec<_T> >, _L, _R> kernel_t;
        // expr_t::_k.execute(*this, r);
        if (_size) {
                _b = impl::buffer(impl::be_context(),
                                  CL_MEM_READ_WRITE, 
                                  _size*sizeof(_T));
                execute(*this, r);
        }
}

template <class _T>
inline
ocl::vec<_T>::operator std::vector<_T> ()
        const
{
        std::size_t n(this->size());
        std::size_t s(n*sizeof(_T));
        std::vector<_T> v(n);
        impl::queue& q= impl::be_queue();
        q.enqueueReadBuffer(this->buf(),
                            true,
                            0, s,
                            &v[0],
                            nullptr,
                            nullptr);
        return v;
}


ocl::impl::be_data*
ocl::impl::be_data::_instance= nullptr;

ocl::impl::be_data*
ocl::impl::be_data::instance()
 {
        if (_instance == nullptr) {
                _instance = new be_data();
        }
        return _instance;
}

#if 0                        
ocl::impl::be_data::be_data()
        : _d(default_device()), _c(_d), _q(_c, _d)
{
        // create context from device, command queue from context and
        // device
}
#else
ocl::impl::be_data::be_data()
        : _d(default_device()), _c(), _q()
{
        // create context from device, command queue from context and
        // device
        std::vector<cl::Device> vd;
        vd.push_back(_d);
        _c= cl::Context(vd);
        _q= cl::CommandQueue(_c, _d);
}
#endif

using namespace ocl;

template <class _T>
_T
test_func(const _T& a, const _T& b)
{
        return _T( (2.0f *a + b) / (a * b)  + (a + a * b ) - a);
}

template <class _T>
_T
test_func(const _T& a, const _T& b, const _T& c)
{
        return _T(a+b *c ) *c + 2.0f;
}


int main()
{
        try {
                const int SIZE=4096;
                float a(2.0f), b(3.0f);

                vec<float> v0(SIZE, a);
                vec<float> va(v0);
                std::vector<float> vhb(SIZE, 3.0f);
                vec<float> vb(vhb);
                vec<float> vc= test_func(va, vb);
                vec<float> vd= test_func(va, vb, vc);

                float c= test_func(a, b);
                float d= test_func(a, b, c);
                
                std::vector<float> res(vd);

                for (std::size_t i=0; i< res.size(); ++i) {
                        std::cout << i << ' ' << res[i] << std::endl;
                }

                std::cout << "scalar " << d << std::endl;

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
