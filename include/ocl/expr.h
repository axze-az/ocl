#if !defined (__OCL_EXPR_H__)
#define __OCL_EXPR_H__ 1

#include <ocl/config.h>
#include <ocl/impl_be_data.h>
#include <ocl/impl_type_2_name.h>
#include <iostream>
#include <sstream>

namespace ocl {

    struct spaces : public std::string {
        spaces(unsigned int n) : std::string(n, ' ') {}
    };
    // return back end data:
    template <class _T>
    impl::be_data_ptr backend_data(const _T& t);
    // Eval_size
    template <class _T>
    std::size_t eval_size(const _T& t);
    // eval_args, returns the opencl source code
    // for all arguments of generated from r,
    // increments arg_num.
    template <class _T>
    std::string
    eval_args(const std::string& p, const _T& r,
              unsigned& arg_num, bool ro);
    // eval_vars
    template <class _T>
    std::string
    eval_vars(const _T& r, unsigned& arg_num, bool read);
    // eval_ops
    template <class _T>
    std::string eval_ops(const _T& r, unsigned& arg_num);

    // eval_results, unimplemented
    template <class _T>
    std::string eval_results(_T& r, unsigned& res_num);

    // bind_args for const arguments
    template <class _T>
    void bind_args(impl::kernel& k, const _T& r,  unsigned& arg_num);

    // bind_args for const arguments
    template <class _T>
    void bind_args(impl::kernel& k, _T& r,  unsigned& arg_num);

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
        constexpr expr(const _L& l, const _R& r);
    };

    template <class _OP, class _L>
    struct expr<_OP, _L, void> {
        typename expr_traits<_L>::type _l;
        constexpr expr(const _L& l) : _l{l} {};
    };

    // backend_data specialized for expr
    template <class _OP, class _L, class _R>
    impl::be_data_ptr backend_data(const expr<_OP, _L, _R>& a);
    // and for unary expressions
    template <class _OP, class _L>
    impl::be_data_ptr backend_data(const expr<_OP, _L, void>& a);

    // eval_size specialized for expr<>
    template <class _OP, class _L, class _R>
    std::size_t eval_size(const expr<_OP, _L, _R>& a);

    template <class _OP, class _L>
    std::size_t eval_size(const expr<_OP, _L, void>& a);

    // eval_args specialized for expr<>
    template <class _OP, class _L, class _R>
    std::string
    eval_args(const std::string& p,
              const expr<_OP, _L, _R>& r,
              unsigned& arg_num,
              bool ro);

    template <class _OP, class _L>
    std::string
    eval_args(const std::string& p,
              const expr<_OP, _L, void>& r,
              unsigned& arg_num,
              bool ro);

    // eval_vars specialized for expr<>
    template <class _OP, class _L, class _R>
    std::string
    eval_vars(const expr<_OP, _L, _R>& a, unsigned& arg_num,
              bool read);

    template <class _OP, class _L>
    std::string
    eval_vars(const expr<_OP, _L, void>& a, unsigned& arg_num,
              bool read);

    // eval_vars specialized for expr<>
    template <class _OP, class _L, class _R>
    std::string
    eval_ops(const expr<_OP, _L, _R>& a, unsigned& arg_num);

    template <class _OP, class _L>
    std::string
    eval_ops(const expr<_OP, _L, void>& a, unsigned& arg_num);

    // bind_args specialized for expr<>
    template <class _OP, class _L, class _R>
    void bind_args(impl::kernel& k,
                   const expr<_OP, _L, _R>& r,
                   unsigned& arg_num);

    template <class _OP, class _L>
    void bind_args(impl::kernel& k,
                   const expr<_OP, _L, void>& r,
                   unsigned& arg_num);

}

template <class _T>
inline
ocl::impl::be_data_ptr
ocl::backend_data(const _T& t)
{
    return nullptr;
}

template <class _T>
inline
std::size_t
ocl::eval_size(const _T& t)
{
    static_cast<void>(t);
    return 1;
}

template <class _T>
std::string ocl::eval_args(const std::string& p,
                           const _T& r,
                           unsigned& arg_num,
                           bool ro)
{
    static_cast<void>(ro);
    std::ostringstream s;
    if (!p.empty()) {
        s << p << ",\n";
    }
    s << spaces(4) ;
    s << impl::type_2_name<_T>::v()
      << " arg"  << arg_num;
    ++arg_num;
    return s.str();
}

template <class _T>
std::string ocl::eval_vars(const _T& r, unsigned& arg_num,
                           bool read)
{
    std::ostringstream s;
    s << spaces(8) << impl::type_2_name<_T>::v()
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
std::string ocl::eval_ops(const _T& r, unsigned& arg_num)
{
    std::ostringstream s;
    s << "v" << arg_num;
    std::string a(s.str());
    ++arg_num;
    return a;
}

template <class _T>
void
ocl::bind_args(impl::kernel& k, _T& r, unsigned& arg_num)
{
    if (impl::be_data::instance()->debug() != 0) {
        std::cout << "binding nonconst to arg " << arg_num
                  << std::endl;
    }
    k.set_arg(arg_num, sizeof(_T), &r);
    ++arg_num;
}

template <class _T>
void
ocl::bind_args(impl::kernel& k, const _T& r, unsigned& arg_num)
{
    if (impl::be_data::instance()->debug() != 0) {
        std::cout << "binding const to arg " << arg_num
                  << std::endl;
    }
    k.set_arg(arg_num, sizeof(_T), &r);
    ++arg_num;
}

template <class _OP, class _L, class _R>
inline
constexpr
ocl::expr<_OP, _L, _R>::expr(const _L& l, const _R& r)
    :  _l(l), _r(r)
{
}

template <class _OP, class _L, class _R>
ocl::impl::be_data_ptr
ocl::backend_data(const expr<_OP, _L, _R>& a)
{
    impl::be_data_ptr pl=backend_data(a._l);
    if (pl != nullptr) {
        return pl;
    }
    return backend_data(a._r);
}

template <class _OP, class _L>
ocl::impl::be_data_ptr
ocl::backend_data(const expr<_OP, _L, void>& a)
{
    return backend_data(a._l);
}

template <class _OP, class _L, class _R>
std::size_t ocl::eval_size(const expr<_OP, _L, _R>& a)
{
    std::size_t l=eval_size(a._l);
    std::size_t r=eval_size(a._r);
    return std::max(l, r);
}

template <class _OP, class _L>
std::size_t ocl::eval_size(const expr<_OP, _L, void>& a)
{
    std::size_t l=eval_size(a._l);
    return l;
}

template <class _OP, class _L, class _R>
std::string ocl::eval_args(const std::string& p,
                           const expr<_OP, _L, _R>& r,
                           unsigned& arg_num,
                           bool ro)
{
    std::string left(eval_args(p, r._l, arg_num, ro));
    return eval_args(left, r._r, arg_num, ro);
}

template <class _OP, class _L>
std::string ocl::eval_args(const std::string& p,
                           const expr<_OP, _L, void>& r,
                           unsigned& arg_num,
                           bool ro)
{
    std::string left(eval_args(p, r._l, arg_num, ro));
    return left;
}


template <class _OP, class _L, class _R>
std::string ocl::eval_vars(const expr<_OP, _L, _R>& a, unsigned& arg_num,
                           bool read)
{
    std::string l=eval_vars(a._l, arg_num, read);
    std::string r=eval_vars(a._r, arg_num, read);
    return std::string(l + "\n"+  r);
}

template <class _OP, class _L>
std::string ocl::eval_vars(const expr<_OP, _L, void>& a, unsigned& arg_num,
                           bool read)
{
    auto l=eval_vars(a._l, arg_num, read);
    return std::string(l + "\n");
}


template <class _OP, class _L, class _R>
std::string ocl::eval_ops(const expr<_OP, _L, _R>& a, unsigned& arg_num)
{
    auto l=eval_ops(a._l, arg_num);
    auto r=eval_ops(a._r, arg_num);
    std::string t(_OP::body(l, r));
    return std::string("(") + t + std::string(")");
}

template <class _OP, class _L>
std::string ocl::eval_ops(const expr<_OP, _L, void>& a, unsigned& arg_num)
{
    auto l=eval_ops(a._l, arg_num);
    std::string t(_OP::body(l));
    return std::string("(") + t + std::string(")");
}


template <class _OP, class _L, class _R>
void ocl::bind_args(impl::kernel& k, const expr<_OP, _L, _R>& r,
                    unsigned& arg_num)
{
    bind_args(k, r._l, arg_num);
    bind_args(k, r._r, arg_num);
}

template <class _OP, class _L>
void ocl::bind_args(impl::kernel& k, const expr<_OP, _L, void>& r,
                    unsigned& arg_num)
{
    bind_args(k, r._l, arg_num);
}

// Local variables:
// mode: c++
// end:
#endif
