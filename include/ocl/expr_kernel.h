#if !defined (__OCL_EXPR_KERNEL_H__)
#define __OCL_EXPR_KERNEL_H__ 1

#include <ocl/config.h>
#include <ocl/expr.h>
#include <thread>
#include <chrono>

#define USE_ARG_BUFFER 1

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
        be::data_ptr
        backend_ptr(const _RES& res, const _EXPR& r);
        // returns a cached kernel for (res, r) or calls gen_kernel
        // and inserts the new kernel into the cache
        static
        be::pgm_kernel_lock&
        get_kernel(_RES& res, const _EXPR& r, const void* addr,
                   be::data_ptr& b);
        // generate a new kernel for (res, r)
        static
        be::pgm_kernel_lock
        gen_kernel(_RES& res, const _EXPR& r, const void* addr,
                   be::data_ptr& b);
    };

    // generate and execute an opencl kernel for an
    // expression
    template <class _RES, class _EXPR>
    void execute(_RES& res, const _EXPR& r);
}

template <class _RES, class _SRC>
ocl::be::data_ptr
ocl::expr_kernel<_RES, _SRC>::
backend_ptr(const _RES& res, const _SRC& r)
{
    be::data_ptr b=backend_data(res);
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
    be::event ev;
    be::data_ptr b=backend_ptr(res, r);
    be::pgm_kernel_lock& pk=get_kernel(res, r, cookie, b);
#if USE_ARG_BUFFER > 0
    be::argument_buffer ab;
    std::size_t s(eval_size(res));
    // size argument first
    bind_non_buffer_args(s, ab);
    // rest of arguments later
    bind_non_buffer_args(r, ab);
    ab.pad_to_multiple_of<128>();
    auto& dcq=b->dcq();
    auto& c=dcq.c();
    be::buffer dev_ab(c, ab.size());
    auto& q=dcq.q();
    auto& wl=dcq.wl();
    auto& dcq_mtx=dcq.mtx();
    be::event cpy_ev;
    {
        std::unique_lock<be::mutex> _lq(dcq_mtx);
        cpy_ev=q.enqueue_write_buffer_async(dev_ab, 0,
                                            ab.size(),
                                            ab.data());
        wl.insert(cpy_ev);
    }
    {
        std::unique_lock<be::pgm_kernel_lock> _lk(pk);
        unsigned buf_num=0;
        bind_buffer_args(res, buf_num, pk._k);
        bind_buffer_args(r, buf_num, pk._k);
        pk._k.set_arg(buf_num++, dev_ab);
        if (b->debug() != 0) {
            std::cout << "binding argument buffer of size " << ab.size()
                      << '\n';
        }
        pk._k.set_arg(buf_num, ab.size(), nullptr);
        if (b->debug() != 0) {
            std::cout << "binding local buffer of size " << ab.size()
                      << '\n';
        }
        {
            std::unique_lock<be::mutex> _lq(dcq_mtx);
            ev=b->enqueue_kernel(pk, s);
        }
    }
    cpy_ev.wait();
#else
    {
        std::unique_lock<be::pgm_kernel_lock> _l(pk);
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
#endif
    // otherwise we leak memory
    ev.wait();
}

template <class _RES, class _SRC>
ocl::be::pgm_kernel_lock&
ocl::expr_kernel<_RES, _SRC>::
get_kernel(_RES& res, const _SRC& r, const void* cookie,
           be::data_ptr& b)
{
    auto& kc=b->kcache();
    std::unique_lock<be::mutex> _l(kc.mtx());
    be::kernel_cache::iterator f(kc.find(cookie));
    if (f == kc.end()) {
        be::pgm_kernel_lock pkl(gen_kernel(res, r, cookie, b));
        std::pair<be::kernel_cache::iterator, bool> ir(
            kc.insert(cookie, pkl));
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
ocl::be::pgm_kernel_lock
ocl::expr_kernel<_RES, _SRC>::
gen_kernel(_RES& res, const _SRC& r, const void* cookie,
           be::data_ptr& b)
{
    std::ostringstream s;
    s << "k_" << cookie;
    std::string k_name(s.str());
    s.str("");
    s << "arg_" << cookie;
    std::string k_arg_name(s.str());
    s.str("");
    impl::insert_headers(s);

#if USE_ARG_BUFFER > 0
    unsigned decl_nb_args(0);
    // argument structure with the scalar arguments
    s << "struct " << k_arg_name << " {\n"
         "    ulong _n __attribute__((aligned("
         << alignof(int64_t) <<")));\n"
      << decl_non_buffer_args(r, decl_nb_args) << "};\n\n";
    // kernel argument
    unsigned buf_args(0);
    s << "__kernel void " << k_name
      << "\n(\n";
    s << decl_buffer_args(res, buf_args, false);
    s << decl_buffer_args(r, buf_args, true);
    s << spaces(4) << "__global const struct " << k_arg_name << "* pg,\n";
    s << spaces(4) << "__local const struct " << k_arg_name << "* pa\n)\n";
    s << "{\n"
         "    /* copy arguments into pa */\n"
         "    uint lid = get_local_id(0);\n"
         "    uint lsz = get_local_size(0);\n"
         "    __global const uint* ps=(__global const uint*)pg;\n"
         "    __local uint* pd=(__local uint*)pa;\n"
         "    const int arg_size=sizeof(*pa);\n"
         "    const int cpy_size=(arg_size+3) >> 2;\n"
         "    for (uint i=0; i<cpy_size; i+= lsz) {\n"
         "        uint idx=i*lsz + lid;\n"
         "        uint tidx= idx < cpy_size ? idx : cpy_size-1;\n"
         "        pd[tidx]=ps[tidx];\n"
         "    }\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    /* start execution */\n"
         "    ulong gid = get_global_id(0);\n"
         "    if (gid < pa->_n) {\n";
    var_counters c{1};
    s << fetch_args(r, c);
    var_counters cr{0};
    s << store_result(res, cr);
    s << eval_ops(r, cr._var_num) << ";\n";
    s << "    }\n"
         "}\n\n";
#else
    const char nl='\n';
    // the real kernel follows now
    s << "__kernel void " << k_name
      << "\n(\n";
    // element count:
    std::string element_count=spaces(4) +
        be::type_2_name<unsigned long>::v() + " n";
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
#endif
    using namespace impl;
    std::string ss(s.str());
    if (b->debug() != 0) {
        std::cout << "--- source code ------------------\n";
        std::cout << ss;
    }
    be::program pgm=be::program::create_with_source(ss, b->dcq().c());
    try {
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=clc++");
        // pgm=program::build_with_source(ss, d->c(), "-cl-std=CL1.1");
        pgm.build("-cl-std=CL1.1 -cl-mad-enable");
    }
    catch (const be::error& e) {
        std::cerr << "error info: " << e.what() << '\n';
        std::cerr << pgm.build_log() << std::endl;
        throw;
    }
    be::kernel k(pgm, k_name);
    if (b->debug() != 0) {
        std::cout << "-- compiled with success ---------\n";
    }

    be::pgm_kernel_lock pkl(pgm, k);
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
