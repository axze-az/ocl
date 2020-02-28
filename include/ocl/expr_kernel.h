#if !defined (__OCL_EXPR_KERNEL_H__)
#define __OCL_EXPR_KERNEL_H__ 1

#include <ocl/config.h>
#include <ocl/expr.h>
#include <ocl/expr_custom.h>
#include <ocl/dvec_base.h>
#include <experimental/memory>
#include <thread>
#include <chrono>

namespace ocl {

    namespace impl {

        // inserts defines/pragmas into the source code
        void
        insert_headers(std::ostream& s, size_t lmem_size=0);

        be::kernel_handle
        compile(const std::string& s, const std::string& k_name,
                const be::data_ptr& b);

        // throws an exception because of missing
        // backend data
        __attribute__((__noreturn__))
        void
        missing_backend_data();

        // generate the second part of the key of a kernel
        template <typename _T>
        std::string
        gen_key(const _T& s);

        // generate the second part of the key of a kernel
        template <typename _OP, typename _L, typename _R>
        std::string
        gen_key(const expr<_OP, _L, _R>& e);

        // generate the second part of the key of a kernel
        template <typename _OP, typename _L>
        std::string
        gen_key(const expr<_OP, _L, void>& e);

        // generate the second part of the key of a kernel
        std::string
        gen_key(const cf_body& r);

        // generate the second part of the key of a kernel
        std::string
        gen_key(const ck_body& r);

        void
        print_arg_buffer_info(be::kernel& k, size_t ab_size);

        // execute the kernel corresponding to the expression
        // res, r
        template <class _RES, class _EXPR>
        void
        execute(_RES& res, const _EXPR& r,
                const be::data_ptr& b, size_t s,
                const void* addr);

        void
        print_cached_kernel_info(be::kernel_cache::iterator f,
                                 const be::kernel_key& kk);

        // returns a cached kernel for (res, r) or calls gen_kernel
        // and inserts the new kernel into the cache
        template <class _RES, class _EXPR>
        be::kernel_handle
        get_kernel(_RES& res, const _EXPR& r, const void* addr,
                   const be::data_ptr& b, size_t lmem_size=0);

        // result of gen_kernel_src
        class ksrc_info {
            std::string _kname;
            std::string _s;
            bool _custom;
        public:
            ksrc_info(const std::string& kname, const std::string& s,
                      bool custom_k);
            ksrc_info(const ksrc_info& r);
            ksrc_info(ksrc_info&& r);
            ksrc_info& operator=(const ksrc_info& r);
            ksrc_info& operator=(ksrc_info&& r);
            ~ksrc_info();
            const std::string& name() const { return _kname; }
            const std::string& source() const { return _s; }
            const bool& custom() const { return _custom; }
        };

        // generates kernel source without headers
        template <class _RES, class _EXPR>
        ksrc_info
        gen_kernel_src(_RES& res, const _EXPR& r, const void* addr);

        // generates custom kernel source without headers
        template <class _RES, typename _OP, typename _R>
        ksrc_info
        gen_kernel_src(_RES& res,
                       const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r,
                       const void* addr);

        // generate a new kernel for (res, r)
        template <class _RES, class _EXPR>
        be::kernel_handle
        gen_kernel(_RES& res, const _EXPR& r, const void* addr,
                   const be::data_ptr& b, size_t lmem_size=0);
    }

    // generate and execute an opencl kernel for an
    // expression
    template <class _RES, class _EXPR>
    void
    execute(_RES& res, const _EXPR& r, const be::data_ptr& b, size_t s);

    // generate and execute a custom kernel for r
    template <typename _OP, typename _R>
    void
    execute_custom(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r,
                   size_t s, const be::data_ptr& b);

    // generate and execute a custom kernel for r
    template <class _RES, typename _OP, typename _R>
    void
    execute_custom(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r,
                   size_t s);
}

template <typename _T>
std::string
ocl::impl::
gen_key(const _T& r)
{
    static_cast<void>(r);
    return std::string();
}

template <typename _OP, typename _L, typename _R>
std::string
ocl::impl::
gen_key(const expr<_OP, _L, _R>& r)
{
    std::string sl=gen_key(r._l);
    std::string sr=gen_key(r._r);
    if (sl.empty())
        return sr;
    if (sr.empty())
        return sl;
    return sl + sr;
}

template <typename _OP, typename _L>
std::string
ocl::impl::
gen_key(const expr<_OP, _L, void>& r)
{
    return gen_key(r._l);
}

inline
std::string
ocl::impl::
gen_key(const cf_body& r)
{
    return r.body();
}

inline
std::string
ocl::impl::
gen_key(const ck_body& r)
{
    return r.body();
}

template <class _RES, class _SRC>
void
ocl::impl::
execute(_RES& res, const _SRC& r,
        const be::data_ptr& b, size_t s,
        const void* cookie)
{
    be::event ev;
    be::argument_buffer ab;
    // size argument first
    bind_non_buffer_args(s, ab);
    // rest of arguments later
    bind_non_buffer_args(r, ab);
    ab.pad_to_max_alignment();
    size_t ab_size=ab.size();
    auto& dcq=b->dcq();
    auto& c=dcq.c();
    be::buffer dev_ab(c, ab_size,
                      be::buffer::read_only|be::buffer::copy_host_ptr,
                      ab.data());
    auto& dcq_mtx=dcq.mtx();
    auto& d=dcq.d();
    size_t lmem_size=be::request_local_mem(d, ab_size);
    be::kernel_handle pk=get_kernel(res, r, cookie, b, lmem_size);
    be::kexec_1d_info ki(d, pk.k(), s);
    {
        be::scoped_lock _lk(pk.mtx());
        unsigned buf_num=0;
        bind_buffer_args(res, buf_num, pk.k(), ki._local_size);
        bind_buffer_args(r, buf_num, pk.k(), ki._local_size);
        pk.k().set_arg(buf_num++, dev_ab);
        if (b->debug() != 0) {
            print_arg_buffer_info(pk.k(), ab_size);
        }
        {
            be::scoped_lock _lq(dcq_mtx);
            ev=b->enqueue_1d_kernel(pk.k(), ki);
        }
    }
    // otherwise we leak memory
    ev.wait();
}

template <class _RES, class _SRC>
ocl::be::kernel_handle
ocl::impl::
get_kernel(_RES& res, const _SRC& r, const void* cookie,
           const be::data_ptr& b, size_t lmem_size)
{
    auto& kc=b->kcache();
    std::string kl=gen_key(r);
    be::kernel_key kk(cookie, kl);
    be::scoped_lock _l(kc.mtx());
    be::kernel_cache::iterator f(kc.find(kk));
    if (f == kc.end()) {
        be::kernel_handle pkl(gen_kernel(res, r, cookie, b, lmem_size));
        std::pair<be::kernel_cache::iterator, bool> ir(
            kc.insert(kk, pkl));
        f = ir.first;
    } else {
        if (b->debug() != 0) {
            print_cached_kernel_info(f, kk);
        }
    }
    return f->second;
}

template <class _RES, class _SRC>
ocl::impl::ksrc_info
ocl::impl::
gen_kernel_src(_RES& res, const _SRC& r, const void* cookie)
{
    std::ostringstream s;
    s << "k_" << cookie;
    std::string k_name(s.str());
    s.str("");
    be::kernel_functions fnames;
    s << def_custom_func(fnames, r);

    // the real kernel follows now
    s << "inline void " << k_name;
    s << "\n(\n";
    const char nl='\n';
    // argument generation
    unsigned arg_num{0};
    // element count is the first argument
    s << "    "  << be::type_2_name<unsigned long>::v() << " n,\n"
      << eval_args(res, arg_num, false)
      << ",\n"
      << eval_args(r, arg_num, true)
      << "\n)\n"
      // begin body
         "{\n"
      // global id
         "    ulong gid = get_global_id(0);\n"
         "    if (gid < n) {\n";
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
    s << "    }\n";
    // end body
    s << "}\n\n";

    return ksrc_info(k_name, s.str(), false);
}

template <class _RES, typename _OP, typename _R>
ocl::impl::ksrc_info
ocl::impl::
gen_kernel_src(_RES& res,
               const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r,
               const void* addr)
{
    static_cast<void>(addr);
    static_cast<void>(res);
    // the real kernel follows now
    std::string s = r._l.body() + '\n';
    return ksrc_info(r._l.name(), s, true);
}

template <class _RES, class _SRC>
ocl::be::kernel_handle
ocl::impl::
gen_kernel(_RES& res, const _SRC& r, const void* cookie,
           const be::data_ptr& b, size_t lmem_size)
{
    ksrc_info ksi=gen_kernel_src(res, r, cookie);
    std::string k_name = ksi.name();
    std::ostringstream s;
    s << cookie;
    const std::string s_cookie=s.str();
    s.str("");
    impl::insert_headers(s, lmem_size);
    s << ksi.source();
    std::string k_arg_name("arg_" + s_cookie);
    std::string k_func_name(k_name);
    k_name=std::string("e_") + s_cookie;
    const bool const_buffer_args=ksi.custom() == false ? true : false;
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
    s << decl_buffer_args(r, buf_args, const_buffer_args);
    if (lmem_size == 0) {
        s << "    __global const struct "
          << k_arg_name << "* pa\n)\n";
        s << "{\n";
    } else {
        s << "    __global const struct "
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
             "            if (idx < cpy_size) {\n"
             "                pd[idx] = ps[idx];\n"
             "            }\n"
             "        }\n"
             "        barrier(CLK_LOCAL_MEM_FENCE);\n"
             "    }\n"
             "    __local const struct "
          << k_arg_name << "* pa= &__args._a;\n";
    }
    var_counters c{0};
    s << "    " << k_func_name << "("
      << "pa->_n, ";
    // custom kernels may have no result:
    std::string ra=concat_args(res, c);
    if (!ra.empty()) {
        s << ra << ", ";
    }
    s << concat_args(r, c) << ");\n"
         "}\n";
    std::string ss=s.str();
    return compile(ss, k_name, b);
}

template <class _RES, class _EXPR>
void
ocl::
execute(_RES& res, const _EXPR& r, const be::data_ptr& b, size_t s)
{
    // auto pf=execute<_RES, _EXPR>;
    void (*pf)(_RES&, const _EXPR&, const be::data_ptr&, size_t) =
        execute<_RES, _EXPR>;
    const void* pv=reinterpret_cast<const void*>(pf);
    impl::execute(res, r, b, s, pv);
}

template <typename _OP, typename _R>
void
ocl::
execute_custom(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r,
               size_t s, const be::data_ptr& b)
{
    struct tag {};
    impl::ignored_arg<tag> v;
    execute(v, r, b, s);
}

template <class _RES, typename _OP, typename _R>
void
ocl::
execute_custom(const expr<dop::custom_k<_OP>, impl::ck_body, _R>& r,
               size_t s)
{
    execute_custom(r, s, be::data::instance());
}

// Local variables:
// mode: c++
// end:
#endif
