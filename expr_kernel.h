#if !defined (__OCL_EXPR_KERNEL_H__)
#define __OCL_EXPR_KERNEL_H__ 1

#include <ocl/config.h>
#include <ocl/expr.h>
#include <thread>
#include <chrono>

namespace ocl {

    // Opencl kernel for expressions
    template <class _RES, class _EXPR>
    class expr_kernel {
    public:
        expr_kernel() = default;
        void
        execute(_RES& res, const _EXPR& r, const void* addr)
            const;
    private:
        // returns a cached kernel for (res, r) or calls gen_kernel
        // and inserts the new kernel into the cache
        impl::pgm_kernel_lock&
        get_kernel(_RES& res, const _EXPR& r, const void* addr)
            const;
        // generate a new kernel for (res, r)
        impl::pgm_kernel_lock
        gen_kernel(_RES& res, const _EXPR& r, const void* addr)
            const;
    };

    // generate and execute an opencl kernel for an
    // expression
    template <class _RES, class _EXPR>
    void execute(_RES& res, const _EXPR& r);
}

template <class _RES, class _SRC>
void
ocl::expr_kernel<_RES, _SRC>::
execute(_RES& res, const _SRC& r, const void* cookie)
    const
{
    impl::pgm_kernel_lock& pk=get_kernel(res, r, cookie);
    // execute the kernel
    impl::be_data_ptr& b= res.backend_data();
    impl::queue& q= b->q();
    impl::device& d= b->d();

    std::unique_lock<impl::pgm_kernel_lock> _l(pk);
    // bind args
    std::size_t s(eval_size(res));
    const auto& sc=s;
    unsigned arg_num{0};
    bind_args(pk._k, sc, arg_num);
    bind_args(pk._k, res, arg_num);
    bind_args(pk._k, r, arg_num);
    if (b->debug() != 0) {
        std::cout << "executing kernel" << std::endl;
    }
    std::size_t local_size(
        pk._k.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(d, nullptr));
    std::size_t gs= ((s+local_size-1)/local_size)*local_size;
    if (b->debug() != 0) {
        std::cout << "kernel: size: " << s
                  << " global size: " << gs
                  << " local size: " << local_size
                  << std::endl;
    }
    auto& evs=b->evs();
    const std::vector<impl::event>* pev= evs.empty() ? nullptr : &evs;
    impl::event ev;
    q.enqueueNDRangeKernel(pk._k,
                           cl::NullRange,
                           cl::NDRange(gs),
                           cl::NullRange, //cl::NDRange(local_size),
                           pev,
                           &ev);
    evs.clear();
    evs.emplace_back(ev);
    if (b->debug() != 0) {
        std::cout << "execution done" << std::endl;
    }
    q.flush();
    // std::this_thread::sleep_for(std::chrono::seconds(1));
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
gen_kernel(_RES& res, const _SRC& r, const void* cookie)
    const
{
    std::ostringstream s;
    s << "expr_kernel_" << cookie;
    std::string k_name(s.str());
    s.str("");
    s << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    s << "__kernel void " << k_name
      << std::endl
      << "(\n";
    // buffer length:
    std::string buf_len=spaces(4) +
        impl::type_2_name<unsigned long>::v() + " n";
    // argument generation
    unsigned arg_num{0};
    s << eval_args(buf_len, res, arg_num, false)
      << ','
      << std::endl;
    s << eval_args("", r, arg_num, true) << std::endl;

    s << ")" << std::endl;

    // begin body
    s << "{" << std::endl;

    // global id
    s << spaces(4) << "ulong gid = get_global_id(0);" << std::endl;
    s << spaces(4) << "if (gid < n) { " << std::endl;
    // temporary variables
    unsigned var_num{1};
    s << eval_vars(r, var_num, true)
      << std::endl;

    // result variable
    unsigned res_num{0};
    s << eval_vars(res, res_num, false) << "= ";
    // the operations
    unsigned body_num{1};
    s << eval_ops(r, body_num) << ';'
      << std::endl;
    // write back
    res_num = 0;
    s << eval_results(res, res_num)
      << std::endl;
    // end if
    s << spaces(4) << "}" << std::endl;
    // end body
    s << "}" << std::endl;
    using namespace impl;
    be_data_ptr& bd= res.backend_data();
    if (bd->debug() != 0) {
        std::cout << "--- source code ------------------\n";
        std::cout << s.str();
    }
    std::string ss(s.str());
    cl::Program::Sources sv;
    sv.push_back(ss);

    cl::Program pgm(bd->c(), sv);
    std::vector<cl::Device> vk(1, bd->d());

    try {
        // pgm.build(vk , "-cl-std=clc++");
        pgm.build(vk , "-cl-std=CL1.1");
    }
    catch (const cl::Error& e) {
        std::string op(pgm.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(
                           bd->d(), nullptr));
        std::cerr << "build options: " << op << '\n';
        std::string em(pgm.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                           bd->d(), nullptr));
        std::cerr << "error info: " << em << '\n';
        throw;
    }
    kernel k(pgm, k_name.c_str());
    if (bd->debug() != 0) {
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
    expr_kernel<_RES, _EXPR> k;
    k.execute(res, r, pv);
}

// Local variables:
// mode: c++
// end:
#endif
