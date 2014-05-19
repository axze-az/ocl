#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>

namespace ocl {

        namespace impl {
                class be_data {
                public:
                        device& d() {
                                return _d;
                        }
                        queue& q() {
                                return _q;
                        }
                        context& c() {
                                return _c;
                        }
                        
                        typedef std::map<void*, kernel> kernel_map_type;
                        typedef std::map<void*, kernel>::iterator 
                        iterator_type;

                        static
                        be_data* instance();
                private:
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
                return t;
        }

        template <class _T>
        std::size_t eval_size(const std::vector<_T>& v) {
                return v.size();
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
        std::size_t eval_size(const expr<_OP, _L, _R>& a) {
                std::size_t l=eval_size(a._l);
                std::size_t r=eval_size(a._r);
                return std::max(l, r);
        }
        
        template <class _OP, class _L, class _R>
        std::string 
        eval_vars(const expr<_OP, _L, _R>& a, unsigned& arg_num,
                  bool read) {
                auto l=eval_vars(a._l, arg_num, read);
                auto r=eval_vars(a._r, arg_num, read);
                return std::string(l + '\n' + r);
        }

        template <class _OP, class _L, class _R>
        std::string 
        eval_ops(const expr<_OP, _L, _R>& a, unsigned& arg_num) {
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
                  bool ro) {
                std::string left(eval_args(p, r._l, arg_num, ro));
                return eval_args(left, r._r, arg_num, ro);
        }

        template <class _OP, class _L, class _R>
        void bind_args(cl::Kernel& k, 
                       const expr<_OP, _L, _R>& r,
                       unsigned& arg_num) {
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
                vec& operator=(const vec& v);
                vec& operator=(const _T& i);

                explicit vec(std::size_t n);
                template <template <class _V> class _OP, 
                          class _L, class _R>
                vec(const expr<_OP<vec<_T> >, _L, _R>& r);
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
                k.setArg(arg_num, r.buffer());
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

#define DEFINE_OCLVEC_FP_OPERATOR(vx, op, eq_op, op_name)               \
        /* operator op(V, V) */                                         \
        inline                                                          \
        expr<ops:: op_name<vx>, vx, vx>                                 \
        operator op (const vx& a, const vx& b) {                        \
                return expr<ops:: op_name<vx>, vx, vx>(a,b);            \
        }                                                               \
        /* operator op(V, expr) */                                      \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, vx, expr<_OP<vx>, _L, _R> >             \
        operator op (const vx& a, const expr<_OP<vx>, _L, _R>& b) {     \
                return expr<ops:: op_name<vx>,                          \
                            vx, expr<_OP<vx>, _L, _R> >(a, b);          \
        }                                                               \
        /* operator op(expr, V) */                                      \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, expr<_OP<vx>, _L, _R>, vx>              \
        operator op (const expr<_OP<vx>, _L, _R>& a, const vx& b) {     \
                return expr<ops:: op_name<vx>,                          \
                            expr<_OP<vx>, _L, _R>, vx>(a, b);           \
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
        /* operator eq_op expr */                                       \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        vx& operator eq_op(vx& a, const expr<_OP<vx>, _L, _R>& r) {     \
                a = a op r;                                             \
                return a;                                               \
        }


#define DEFINE_OCLVEC_FP_OPERATORS(vx)                  \
        DEFINE_OCLVEC_FP_OPERATOR(vx, +, +=, add)       \
        DEFINE_OCLVEC_FP_OPERATOR(vx, -, -=, sub)       \
        DEFINE_OCLVEC_FP_OPERATOR(vx, *, *=, mul)       \
        DEFINE_OCLVEC_FP_OPERATOR(vx, /, /=, div) 

#define DEFINE_OCLSCALAR_FP_OPERATORS(vx)                  \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, +, +=, add)       \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, -, -=, sub)       \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, *, *=, mul)       \
        DEFINE_OCLSCALAR_FP_OPERATOR(vx, /, /=, div) 


        DEFINE_OCLVEC_FP_OPERATORS(vec<float>);
        
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
        // unsigned arg_num{0};
        // bind_args(k, res, arg_num);
        // bind_args(k, r, arg_num);
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

        cl::Program pgm(impl::be_data::instance()->c(),
                        s.str(),
                        false);
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
ocl::vec<_T>::vec(std::size_t s, const _T& i)
        : _size(s), _b()
{
        if (_size) {
                _b = impl::buffer(impl::be_context(),
                                  CL_MEM_HOST_READ_ONLY, 
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
                                  CL_MEM_HOST_READ_ONLY, 
                                  _size*sizeof(_T));
                execute(*this, r);
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
                                  CL_MEM_HOST_READ_ONLY, 
                                  _size*sizeof(_T));
                execute(*this, r);
        }
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
                        
ocl::impl::be_data::be_data()
        : _d(default_device()), _c(_d), _q(_c, _d)
{
        // create context from device, command queue from context and
        // device
}

using namespace ocl;

vec<float>
test_func(const vec<float>& a, const vec<float>& b)
{
        // vec<float> n(2.0f);
        return vec<float>( (a + b) / (a * b)  + (a + a * b ) - a);
}

vec<float>
test_func(const vec<float>& a, const vec<float>& b,
          const vec<float>& c)
{
        return (a+b *c ) *c;
}


int main()
{
        try {
                vec<float> v0(1024, 2.0f);
                vec<float> a(v0), b(1024, 3.0f);
                vec<float> c= test_func(a, b);
                vec<float> d= test_func(a, b, c);
                
                static_cast<void>(&d);
        }
        catch (const ocl::impl::error& e) {
                std::cout << "caught exception: " << e.what()
                          << '\n'
                          << ocl::impl::err2str(e)
                          << std::endl;
        }
#if 0
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
