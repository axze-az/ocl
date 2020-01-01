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
        execute(_RES& res, const _EXPR& r,
                be::data_ptr b, size_t s,
                const void* addr);
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
                   be::data_ptr& b, size_t lmem_size=0);
        // generate a new kernel for (res, r)
        static
        be::pgm_kernel_lock
        gen_kernel(_RES& res, const _EXPR& r, const void* addr,
                   be::data_ptr& b, size_t lmem_size=0);
    };

    // generate and execute an opencl kernel for an
    // expression
    template <class _RES, class _EXPR>
    void execute(_RES& res, const _EXPR& r, be::data_ptr b, size_t s);
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
execute(_RES& res, const _SRC& r,
        be::data_ptr b, size_t s,
        const void* cookie)
{
    be::event ev;
#if USE_ARG_BUFFER > 0
    be::argument_buffer ab;
    // size argument first
    bind_non_buffer_args(s, ab);
    // rest of arguments later
    bind_non_buffer_args(r, ab);
    ab.pad_to_max_alignment();
    size_t ab_size=ab.size();
    auto& dcq=b->dcq();
    auto& c=dcq.c();
    be::buffer dev_ab(c, ab_size, be::buffer::read_only);
    auto& q=dcq.q();
    auto& wl=dcq.wl();
    auto& dcq_mtx=dcq.mtx();
    auto& d=dcq.d();
    be::event cpy_ev;
    {
        std::unique_lock<be::mutex> _lq(dcq_mtx);
        cpy_ev=q.enqueue_write_buffer_async(dev_ab, 0,
                                            ab_size,
                                            ab.data());
        q.flush();
        // insert the event before we queue the kernel
        // into the wait list
    }
    size_t lmem_size=be::request_local_mem(d, ab_size);
    be::pgm_kernel_lock& pk=get_kernel(res, r, cookie, b, lmem_size);
    be::kexec_1d_info ki(d, pk._k, s);
    {
        std::unique_lock<be::pgm_kernel_lock> _lk(pk);
        unsigned buf_num=0;
        bind_buffer_args(res, buf_num, pk._k);
        bind_buffer_args(r, buf_num, pk._k);
        pk._k.set_arg(buf_num++, dev_ab);
        if (b->debug() != 0) {
            std::string kn=pk._k.name();
            std::ostringstream st;
            st << std::this_thread::get_id() << ": "
               << kn << ": binding argument buffer of size "
               << ab_size << '\n';
            be::data::debug_print(st.str());
        }
        {
            std::unique_lock<be::mutex> _lq(dcq_mtx);
            // wait also for the argument buffer
            wl.insert(cpy_ev);
            ev=b->enqueue_1d_kernel(pk._k, ki);
        }
    }
    cpy_ev.wait();
#else
    be::pgm_kernel_lock& pk=get_kernel(res, r, cookie, b);
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
           be::data_ptr& b, size_t lmem_size)
{
    auto& kc=b->kcache();
    std::unique_lock<be::mutex> _l(kc.mtx());
    be::kernel_cache::iterator f(kc.find(cookie));
    if (f == kc.end()) {
        be::pgm_kernel_lock pkl(gen_kernel(res, r, cookie, b, lmem_size));
        std::pair<be::kernel_cache::iterator, bool> ir(
            kc.insert(cookie, pkl));
        f = ir.first;
    } else {
        if (b->debug() != 0) {
            std::string kn=f->second._k.name();
            std::ostringstream s;
            s << std::this_thread::get_id() << ": "
              << kn << ": using cached kernel " << cookie
              << '\n';
            be::data::debug_print(s.str());
        }
    }
    return f->second;
}

template <class _RES, class _SRC>
ocl::be::pgm_kernel_lock
ocl::expr_kernel<_RES, _SRC>::
gen_kernel(_RES& res, const _SRC& r, const void* cookie,
           be::data_ptr& b, size_t lmem_size)
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
    const char nl='\n';
    // the real kernel follows now
    s << "void " << k_name << "_func"
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
    s << "}\n\n";

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
    if (lmem_size == 0) {
        s << spaces(4) << "__global const struct "
          << k_arg_name << "* pa\n)\n";
        s << "{\n";
    } else {
        s << spaces(4) << "__global const struct "
        << k_arg_name << "* pg\n)\n";
        const size_t uints_to_cpy=((lmem_size + 3) >>2);
        s << "{\n"
             "    /* copy arguments into __args: */\n"
             "    __local union {\n";
        s << "       struct " << k_arg_name << " _a;\n"
             "       uint _u[" << uints_to_cpy << "];\n"
             "    } __args;\n"
             "    {\n"
             "        uint lid= get_local_id(0);\n"
             "        uint lsz= get_local_size(0);\n"
             "        __global const uint* ps= (__global const uint*)pg;\n"
             "        __local uint* pd= __args._u;\n"
             "        const int cpy_size= " << uints_to_cpy << ";\n"
             "        __attribute__((opencl_unroll_hint()))\n"
             "        for (uint i= 0; i < cpy_size; i+= lsz) {\n"
             "            uint idx= i + lid;\n"
             // "            uint tidx= idx < cpy_size ? idx : 0;\n"
             // "            pd[tidx]= ps[tidx];\n"
             "            if (idx < cpy_size) {\n"
             "                pd[idx] = ps[idx];\n"
             "            }\n"
             "        }\n"
             "        barrier(CLK_LOCAL_MEM_FENCE);\n"
             "    }\n"
             "    __local const struct "
          << k_arg_name << "* pa= &__args._a;\n";
    }
#if 1
    var_counters c{0};
    s << "    " << k_name << "_func("
      << "pa->_n, "
      << concat_args(res, c) << ", "
      << concat_args(r, c) << ");\n";
#else
    s << "    /* start execution */\n"
         "    ulong gid = get_global_id(0);\n"
         "    if (gid < pa->_n) {\n";
    var_counters c{1};
    s << fetch_args(r, c);
    var_counters cr{0};
    s << store_result(res, cr);
    s << eval_ops(r, cr._var_num) << ";\n";
    s << "    }\n";
#endif
    s << "}\n";

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
        std::ostringstream st;
        st << std::this_thread::get_id() << ": "
           << k_name << ": --- source code ------------------\n"
           << ss;
        be::data::debug_print(st.str());
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
        std::ostringstream st;
        st << std::this_thread::get_id() << ": "
           << k_name << ": --- compiled with success --------\n";
        be::data::debug_print(st.str());
    }
    be::pgm_kernel_lock pkl(pgm, k);
    return pkl;
}

template <class _RES, class _EXPR>
void
ocl::execute(_RES& res, const _EXPR& r, be::data_ptr b, size_t s)
{
    // auto pf=execute<_RES, _EXPR>;
    void (*pf)(_RES&, const _EXPR&, be::data_ptr, size_t) =
        execute<_RES, _EXPR>;
    const void* pv=reinterpret_cast<const void*>(pf);
    expr_kernel<_RES, _EXPR>::execute(res, r, b, s, pv);
}

// Local variables:
// mode: c++
// end:
#endif
