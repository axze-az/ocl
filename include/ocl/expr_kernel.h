#if !defined (__OCL_EXPR_KERNEL_H__)
#define __OCL_EXPR_KERNEL_H__ 1

#include <ocl/config.h>
#include <ocl/expr.h>
#include <thread>
#include <chrono>

namespace ocl {

    namespace impl {
        // inserts defines/pragmas into the source code
        void
        insert_headers(std::ostream& s);
        // throws an exception because of missing
        // backend data
        __attribute__((__noreturn__))
        void
        missing_backend_data();

    }

    // Opencl kernel for expressions
    template <class _RES, class _EXPR>
    class expr_kernel {
    public:
        expr_kernel() = default;
        static
        void
        execute(_RES& res, const _EXPR& r, const void* addr);
    private:
        // get the backend data pointer
        static
        impl::be_data_ptr
        backend_ptr(const _RES& res, const _EXPR& r);
        // returns a cached kernel for (res, r) or calls gen_kernel
        // and inserts the new kernel into the cache
        static
        impl::pgm_kernel_lock&
        get_kernel(_RES& res, const _EXPR& r, const void* addr,
                   impl::be_data_ptr& b);
        // generate a new kernel for (res, r)
        static
        impl::pgm_kernel_lock
        gen_kernel(_RES& res, const _EXPR& r, const void* addr,
                   impl::be_data_ptr& b);
    };

    // generate and execute an opencl kernel for an
    // expression
    template <class _RES, class _EXPR>
    void execute(_RES& res, const _EXPR& r);
}

template <class _RES, class _SRC>
ocl::impl::be_data_ptr
ocl::expr_kernel<_RES, _SRC>::
backend_ptr(const _RES& res, const _SRC& r)
{
    impl::be_data_ptr b=backend_data(res);
    if (b == nullptr) {
        b=backend_data(r);
        if (b == nullptr) {
            impl::missing_backend_data();
        }
    }
    return b;
}

template <class _RES, class _SRC>
void
ocl::expr_kernel<_RES, _SRC>::
execute(_RES& res, const _SRC& r, const void* cookie)
{
    impl::event ev;
    impl::be_data_ptr b=backend_ptr(res, r);
    impl::pgm_kernel_lock& pk=get_kernel(res, r, cookie, b);
    {
        std::unique_lock<impl::pgm_kernel_lock> _l(pk);
        // bind args
        std::size_t s(eval_size(res));
        const auto& sc=s;
        unsigned arg_num{0};
        bind_args(pk._k, sc, arg_num);
        bind_args(pk._k, res, arg_num);
        bind_args(pk._k, r, arg_num);
        // execute the kernel
        ev=b->enqueue_kernel(pk, s);
    }
    // otherwise we leak memory
    ev.wait();
}

template <class _RES, class _SRC>
ocl::impl::pgm_kernel_lock&
ocl::expr_kernel<_RES, _SRC>::
get_kernel(_RES& res, const _SRC& r, const void* cookie,
           impl::be_data_ptr& b)
{
    using namespace impl;
    std::unique_lock<be_data> _l(*b);
    be_data::iterator f(b->find(cookie));
    if (f == b->end()) {
        pgm_kernel_lock pkl(gen_kernel(res, r, cookie, b));
        std::pair<be_data::iterator, bool> ir(
            b->insert(cookie, pkl));
        f = ir.first;
    } else {
        if (b->debug() != 0) {
            std::cout << "using cached kernel expr_kernel_" << cookie
                      << std::endl;
        }
    }
    return f->second;
}

template <class _RES, class _SRC>
ocl::impl::pgm_kernel_lock
ocl::expr_kernel<_RES, _SRC>::
gen_kernel(_RES& res, const _SRC& r, const void* cookie,
           impl::be_data_ptr& b)
{
    const char nl='\n';
    std::ostringstream s;
    s << "k_" << cookie;
    std::string k_name(s.str());
    s.str("");
    impl::insert_headers(s);
    s << "__kernel void " << k_name
      << "\n(\n";
    // element count:
    std::string element_count=spaces(4) +
        impl::type_2_name<unsigned long>::v() + " n";
    // argument generation
    unsigned arg_num{0};
    s << eval_args(element_count, res, arg_num, false)
      << ",\n";
    s << eval_args("", r, arg_num, true);
    s << "\n)\n";
    // begin body
    s << "{\n";

    // global id
    s << spaces(4) << "ulong gid = get_global_id(0);\n";
    s << spaces(4) << "if (gid < n) {\n";
    // temporary variables
    unsigned var_num{1};
    s << eval_vars(r, var_num, true)
      << nl;

    // result variable
    unsigned res_num{0};
    s << eval_vars(res, res_num, false) << "= ";
    // the operations
    unsigned body_num{1};
    s << eval_ops(r, body_num) << ";\n";
    // write back
    res_num = 0;
    s << eval_results(res, res_num)
      << nl;
    // end if
    s << spaces(4) << "}\n";
    // end body
    s << "}\n";
    using namespace impl;
    std::string ss(s.str());
    if (b->debug() != 0) {
        std::cout << "--- source code ------------------\n";
        std::cout << ss;
    }
    program pgm=program::create_with_source(ss, b->c());
    try {
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=clc++");
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=CL1.1");
        pgm.build("-cl-std=CL1.1");
    }
    catch (const error& e) {
        std::cerr << "error info: " << e.what() << '\n';
        std::cerr << pgm.build_log() << std::endl;
        throw;
    }
    kernel k(pgm, k_name);
    if (b->debug() != 0) {
        std::cout << "-- compiled with success ---------\n";
    }

    pgm_kernel_lock pkl(pgm, k);
    return pkl;
}

template <class _RES, class _EXPR>
void
ocl::execute(_RES& res, const _EXPR& r)
{
    // auto pf=execute<_RES, _EXPR>;
    void (*pf)(_RES&, const _EXPR&) = execute<_RES, _EXPR>  ;
    const void* pv=reinterpret_cast<const void*>(pf);
    expr_kernel<_RES, _EXPR>::execute(res, r, pv);
}

// Local variables:
// mode: c++
// end:
#endif
