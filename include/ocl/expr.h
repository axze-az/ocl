#if !defined (__OCL_EXPR_H__)
#define __OCL_EXPR_H__ 1

#include <ocl/config.h>
#include <ocl/be/data.h>
#include <ocl/be/type_2_name.h>
#include <iostream>
#include <sstream>

namespace ocl {

    struct spaces : public std::string {
        spaces(unsigned int n) : std::string(n, ' ') {}
    };
    // return back end data:
    template <class _T>
    be::data_ptr backend_data(const _T& t);
    // evaluate the size of an expression
    template <class _T>
    std::size_t eval_size(const _T& t);

    // generate the declarations for the members of the structure
    // with the scalar arguments, must increment arg_num if something
    // was generated
    template <typename _T>
    std::string
    decl_non_buffer_args(const _T& p, unsigned& arg_num);

    // declare buffer arguments, must increment arg_num if something
    // generated
    template <typename _T>
    std::string
    decl_buffer_args(const _T& p, unsigned& arg_num, bool read_only);

    struct var_counters {
        unsigned _var_num;
        unsigned _buf_num;
        unsigned _scalar_num;
        var_counters() = default;
        var_counters(unsigned buf_offs)
            : _var_num(buf_offs), _buf_num(buf_offs), _scalar_num(0) {};
    };
    template <typename _T>
    std::string
    fetch_args(const _T& r, var_counters& c);

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
    void bind_args(be::kernel& k, const _T& r,  unsigned& arg_num);

    // bind_args for const arguments
    template <class _T>
    void bind_args(be::kernel& k, _T& r,  unsigned& arg_num);

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
    be::data_ptr backend_data(const expr<_OP, _L, _R>& a);
    // and for unary expressions
    template <class _OP, class _L>
    be::data_ptr backend_data(const expr<_OP, _L, void>& a);

    // eval_size specialized for expr<>
    template <class _OP, class _L, class _R>
    std::size_t eval_size(const expr<_OP, _L, _R>& a);

    template <class _OP, class _L>
    std::size_t eval_size(const expr<_OP, _L, void>& a);

    // decl_non_buffer_args specialized for expressions
    template <class _OP, class _L, class _R>
    std::string
    decl_non_buffer_args(const expr<_OP, _L, _R>& r,
                         unsigned& arg_num);

    template <class _OP, class _L>
    std::string
    decl_non_buffer_args(const expr<_OP, _L, void>& r,
                         unsigned& arg_num);

    // decl_non_buffer_args specialized for expressions
    template <class _OP, class _L, class _R>
    std::string
    decl_buffer_args(const expr<_OP, _L, _R>& r,
                     unsigned& arg_num, bool read_only);

    template <class _OP, class _L>
    std::string
    decl_buffer_args(const expr<_OP, _L, void>& r,
                     unsigned& arg_num, bool read_only);

    template <class _OP, class _L, class _R>
    std::string
    fetch_args(const expr<_OP, _L, _R>& e, var_counters& c);

    template <class _OP, class _L>
    std::string
    fetch_args(const expr<_OP, _L, void>& e, var_counters& c);

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
    void bind_args(be::kernel& k,
                   const expr<_OP, _L, _R>& r,
                   unsigned& arg_num);

    template <class _OP, class _L>
    void bind_args(be::kernel& k,
                   const expr<_OP, _L, void>& r,
                   unsigned& arg_num);

}

template <class _T>
inline
ocl::be::data_ptr
ocl::backend_data(const _T& t)
{
    static_cast<void>(t);
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
std::string
ocl::decl_non_buffer_args(const _T& r, unsigned& arg_num)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(4) << be::type_2_name<_T>::v()
      << " _a" << arg_num
      << " __attribute__((aligned(" << alignof(_T) << ")));\n";
    ++arg_num;
    return s.str();
}

template <class _T>
std::string
ocl::decl_buffer_args(const _T& r, unsigned& arg_num, bool read_only)
{
    static_cast<void>(r);
    static_cast<void>(arg_num);
    static_cast<void>(read_only);
    return std::string();
}

template <class _T>
std::string
ocl::fetch_args(const _T& r, var_counters& c)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(8) << "const " << be::type_2_name<_T>::v()
      << " v" << c._var_num
      << " = pa->_a" << c._scalar_num << ";\n";
    ++c._var_num;
    ++c._scalar_num;
    return s.str();
}

template <class _T>
std::string ocl::eval_args(const std::string& p,
                           const _T& r,
                           unsigned& arg_num,
                           bool ro)
{
    static_cast<void>(r);
    static_cast<void>(ro);
    std::ostringstream s;
    if (!p.empty()) {
        s << p << ",\n";
    }
    s << spaces(4) ;
    s << be::type_2_name<_T>::v()
      << " arg"  << arg_num;
    ++arg_num;
    return s.str();
}

template <class _T>
std::string ocl::eval_vars(const _T& r, unsigned& arg_num,
                           bool read)
{
    static_cast<void>(r);
    std::ostringstream s;
    s << spaces(8) << be::type_2_name<_T>::v()
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
    static_cast<void>(r);
    std::ostringstream s;
    s << "v" << arg_num;
    std::string a(s.str());
    ++arg_num;
    return a;
}

template <class _T>
void
ocl::bind_args(be::kernel& k, _T& r, unsigned& arg_num)
{
    if (be::data::instance()->debug() != 0) {
        std::cout << "binding "
                  << be::type_2_name<_T>::v()
                  << " to arg " << arg_num
                  << std::endl;
    }
    k.set_arg(arg_num, sizeof(_T), &r);
    ++arg_num;
}

template <class _T>
void
ocl::bind_args(be::kernel& k, const _T& r, unsigned& arg_num)
{
    if (be::data::instance()->debug() != 0) {
        std::cout << "binding const "
                  << be::type_2_name<_T>::v()
                  << " to arg " << arg_num
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
ocl::be::data_ptr
ocl::backend_data(const expr<_OP, _L, _R>& a)
{
    be::data_ptr pl=backend_data(a._l);
    if (pl != nullptr) {
        return pl;
    }
    return backend_data(a._r);
}

template <class _OP, class _L>
ocl::be::data_ptr
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
std::string
ocl::
decl_non_buffer_args(const expr<_OP, _L, _R>& e, unsigned& arg_num)
{
    std::string l=decl_non_buffer_args(e._l, arg_num);
    std::string r=decl_non_buffer_args(e._r, arg_num);
    return l+r;
}

template <class _OP, class _L>
std::string
ocl::
decl_non_buffer_args(const expr<_OP, _L, void>& e, unsigned& arg_num)
{
    return decl_non_buffer_args(e._l, arg_num);
}

template <class _OP, class _L, class _R>
std::string
ocl::
decl_buffer_args(const expr<_OP, _L, _R>& e,
                 unsigned& arg_num, bool read_only)
{
    std::string l=decl_buffer_args(e._l, arg_num, read_only);
    std::string r=decl_buffer_args(e._r, arg_num, read_only);
    return l+r;
}

template <class _OP, class _L>
std::string
ocl::
decl_buffer_args(const expr<_OP, _L, void>& e,
                 unsigned& arg_num, bool read_only)
{
    return decl_buffer_args(e._l, arg_num, read_only);
}

template <class _OP, class _L>
std::string
ocl::
fetch_args(const expr<_OP, _L, void>& e, var_counters& c)
{
    return fetch_args(e._l, c);
}

template <class _OP, class _L, class _R>
std::string
ocl::
fetch_args(const expr<_OP, _L, _R>& e, var_counters& c)
{
    std::string l=fetch_args(e._l, c);
    std::string r=fetch_args(e._r, c);
    return l+r;
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
void ocl::bind_args(be::kernel& k, const expr<_OP, _L, _R>& r,
                    unsigned& arg_num)
{
    bind_args(k, r._l, arg_num);
    bind_args(k, r._r, arg_num);
}

template <class _OP, class _L>
void ocl::bind_args(be::kernel& k, const expr<_OP, _L, void>& r,
                    unsigned& arg_num)
{
    bind_args(k, r._l, arg_num);
}

// Local variables:
// mode: c++
// end:
#endif
