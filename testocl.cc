#include "ocl.h"
#include <iostream>
#include <sstream>
#include <atomic>
#include <CL/opencl.h>
#include <CL/cl.hpp>

namespace ocl {


	template <class _T>
	struct expr_traits {
		typedef const _T type;
	};

	template <class _OP, class _L, class _R>
	struct expr_kernel;

	template <class _OP, class _L, class _R> 
	struct expr {
		typename expr_traits<_L>::type _l;
		typename expr_traits<_R>::type _r;
		constexpr expr(const _L& l, const _R& r) :
			_l(l), _r(r) {}
		static expr_kernel<_OP, _L, _R> _k;
	};

	template <class _OP, class _L, class _R>
	std::string 
	eval_vars(const expr<_OP, _L, _R>& a, unsigned& arg_num) {
		auto l=eval_vars(a._l, arg_num);
		auto r=eval_vars(a._r, arg_num);
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
		  unsigned& arg_num) {
		std::string left(eval_args(p, r._l, arg_num));
		return eval_args(left, r._r, arg_num);
	}

	template <class _OP, class _L, class _R>
	void bind_args(const expr<_OP, _L, _R>& r) {
	}
	
	template <class _OP, class _L, class _R> 
	expr_kernel<_OP, _L, _R>
	expr<_OP, _L, _R>::_k;
	
	template <class _OP, class _L, class _R>
	struct expr_kernel {
		expr_kernel();
		void execute(const expr<_OP, _L, _R>& r);
	};

	template <class _T>
	class vec {
	public:
		vec() {}
		vec(std::size_t n, const _T* s);
		vec(std::size_t n, const _T& i);
		vec(std::size_t n);
                template <template <class _V> class _OP, 
			  class _L, class _R>
		vec(const expr<_OP<vec<_T> >, _L, _R>& r);
	};

	template <class _T>
	struct expr_traits<vec<_T> > {
		typedef const vec<_T> type;
	};

	template <class _T>
	std::string eval_ops(const vec<_T>& r, unsigned& arg_num) {
		std::ostringstream s;
		s << "v" << arg_num;
		std::string a(s.str());
		++arg_num;
		return a;
	}

	template <class _T>
	std::string eval_vars(const vec<_T>& r, unsigned& arg_num) {
		std::ostringstream s;
		s << '\t' << impl::type_2_name<_T>::v() 
		  << " v" << arg_num 
		  << " = arg" 
		  << arg_num << "[tid];";
		std::string a(s.str());
		++arg_num;
		return a;
	}

	template <class _T>
	std::string eval_args(const std::string& p, 
			      const vec<_T>& r,
			      unsigned& arg_num) {
		std::ostringstream s;
		if (!p.empty()) {
			s << p << ",\n";
		}
		s << "\t__global const " << impl::type_2_name<_T>::v() 
		  << "* arg"  << arg_num;
		++arg_num;
		return s.str();
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

#define DEFINE_OCLVEC_FP_OPERATOR(vx, op, eq_op, op_name)		\
        /* operator op(V, V) */                                         \
        inline                                                          \
        expr<ops:: op_name<vx>, vx, vx>                                 \
        operator op (const vx& a, const vx& b) {			\
                return expr<ops:: op_name<vx>, vx, vx>(a,b);		\
        }                                                               \
        /* operator op(V, expr) */                                      \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, vx, expr<_OP<vx>, _L, _R> >             \
        operator op (const vx& a, const expr<_OP<vx>, _L, _R>& b) {	\
                return expr<ops:: op_name<vx>,				\
                            vx, expr<_OP<vx>, _L, _R> >(a, b);          \
	}								\
        /* operator op(expr, V) */                                      \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        expr<ops:: op_name<vx>, expr<_OP<vx>, _L, _R>, vx>              \
        operator op (const expr<_OP<vx>, _L, _R>& a, const vx& b) {	\
                return expr<ops:: op_name<vx>,				\
                            expr<_OP<vx>, _L, _R>, vx>(a, b);           \
	}								\
	/* operator op(expr, expr)  */					\
	template <template <class _V> class _OP1, class _L1, class _R1,	\
		  template <class _V> class _OP2, class _L2, class _R2>	\
	inline								\
	expr<ops:: op_name<vx>,						\
	     expr<_OP1<vx>, _L1, _R1>, expr<_OP2<vx>, _L2, _R2> >	\
	operator op(const expr<_OP1<vx>, _L1, _R1>& a,			\
		    const expr<_OP2<vx>, _L2, _R2>& b) {		\
		return expr<ops:: op_name<vx>,				\
			    expr<_OP1<vx>, _L1, _R1>,			\
			    expr<_OP2<vx>, _L2, _R2> > (a, b);		\
	}								\
        /* operator eq_op V */                                          \
        inline                                                          \
        vx& operator eq_op(vx& a, const vx& r) {			\
                a = a op r;                                             \
                return a;                                               \
        }                                                               \
        /* operator eq_op expr */                                       \
        template <template <class _V> class _OP, class _L, class _R>    \
        inline                                                          \
        vx& operator eq_op(vx& a, const expr<_OP<vx>, _L, _R>& r) {	\
                a = a op r;                                             \
                return a;                                               \
        }

#define DEFINE_OCLVEC_FP_OPERATORS(vx)			\
	DEFINE_OCLVEC_FP_OPERATOR(vx, +, +=, add)	\
	DEFINE_OCLVEC_FP_OPERATOR(vx, -, -=, sub)	\
	DEFINE_OCLVEC_FP_OPERATOR(vx, *, *=, mul)	\
	DEFINE_OCLVEC_FP_OPERATOR(vx, /, /=, div) 


	DEFINE_OCLVEC_FP_OPERATORS(vec<float>);
}

 
template <class _OP, class _L, class _R>
ocl::expr_kernel<_OP, _L, _R>::expr_kernel()
{
}

template <class _OP, class _L, class _R>
void 
ocl::expr_kernel<_OP, _L, _R>::
execute(const expr<_OP, _L, _R>& r)
{
	unsigned arg_num{0};
	std::cout << eval_args("", r, arg_num) << std::endl;
	unsigned var_num{0    };
	std::cout << eval_vars(r, var_num) << std::endl;
	unsigned body_num{0};
	std::cout << eval_ops(r, body_num) << std::endl;
}


template <class _T>
template <template <class _V> class _OP, class _L, class _R>
inline
ocl::vec<_T>::vec(const expr<_OP<vec<_T> >, _L, _R>& r)
{
	typedef expr<_OP<vec<_T> >, _L, _R> expr_t;
	// typedef expr_kernel<_OP<vec<_T> >, _L, _R> kernel_t;
	expr_t::_k.execute(r);
}

using namespace ocl;

vec<float>
test_func(const vec<float>& a, const vec<float>& b)
{
	return vec<float>( (a + b) / (a * b)  + (a + a * b ));
}


int main()
{
	vec<float> a, b;
	vec<float> c= test_func(a, b);

	static_cast<void>(&c);
	return 0;
}
