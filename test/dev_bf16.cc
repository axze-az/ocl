//
// Copyright (C) 2010-2026 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#include <cftal/vec.h>
#include <ocl/test/ops.h>
#include <sstream>

namespace cftal {
    // we have the same template in ocl and cftal
    // template <class _T> size_t eval_size(const _T&)
    inline
    std::size_t
    eval_size(bf16_t s) {
        return 1;
    }

    std::string_view
    def_custom_func(ocl::be::kernel_functions& fnames,
                    const bf16_t& v);
}

namespace ocl {

    using cftal::bf16_t;
    using cftal::operator ""_bf16;

    namespace be {

        template <>
        struct type_2_name<bf16_t> {
            static
            constexpr
            std::string_view v() {
                // use ushort here instead of the bf16_t typedef because
                // otherwise even the vector copy and assignment functions
                // require the bf16_t typedef in the kernel sources
                return "bf16_t";
            }
        };
    }

    namespace impl {

        // just in case someone changes the default mask_type
        template <>
        struct dvec_select_mask_value<bf16_t> {
            using type = bf16_t;
        };
    }

    namespace dop {

        // conversion operations for bf16_t
        struct bf16_base {

            struct emit_type_def {
                static
                std::string_view
                name();

                static
                std::string_view
                body();
            };

            struct bf16_to_f32 {
                static
                std::string_view
                name();

                static
                std::string_view
                body();
            };

            struct f32_to_bf16 {
                static
                std::string_view
                name();

                static
                std::string_view
                body();
            };

            struct _neg {
                static
                std::string_view
                name();

                static
                std::string_view
                body();

                static
                std::string
                add_func(be::kernel_functions& fnames);
            };

            struct _abs {
                static
                std::string_view
                name();

                static
                std::string_view
                body();

                static
                std::string
                add_func(be::kernel_functions& fnames);
            };

            struct _isinf {
                static
                std::string_view
                name();

                static
                std::string_view
                body();

                static
                std::string
                add_func(be::kernel_functions& fnames);
            };

            struct _isnan {
                static
                std::string_view
                name();

                static
                std::string_view
                body();

                static
                std::string
                add_func(be::kernel_functions& fnames);
            };

            struct _or {
                static
                std::string_view
                name();

                static
                std::string_view
                body();

                static
                std::string
                add_func(be::kernel_functions& fnames);
            };

            struct _and {
                static
                std::string_view
                name();

                static
                std::string_view
                body();

                static
                std::string
                add_func(be::kernel_functions& fnames);
            };

            struct _xor {
                static
                std::string_view
                name();

                static
                std::string_view
                body();

                static
                std::string
                add_func(be::kernel_functions& fnames);
            };

            static
            std::string
            add_conversions(be::kernel_functions& fnames);

            static
            std::string
            unary_function(const std::string& l,
                           const std::string_view& op,
			   bool op_is_operator);

            static
            std::string
            binary_function(const std::string& l, const std::string& r,
                            const std::string_view& op,
                            bool op_is_operator);

            static
            std::string
            cmp_operator(const std::string& l, const std::string& r,
                         const std::string_view& op);
        };

        template <>
        struct neg<dvec<bf16_t> > : public bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct add<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct sub<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct mul<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct div<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct lt<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct le<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct eq<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct ne<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct ge<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct gt<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct bit_or<dvec<bf16_t> > : public bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct bit_and<dvec<bf16_t> > : public bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };

        template <>
        struct bit_xor<dvec<bf16_t> > : public bf16_base {
            static
            std::string
            body(const std::string& l, const std::string& r);
        };


        template <>
        struct abs_f<dvec<bf16_t> > : public bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct rint_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct isinf_f<dvec<bf16_t> > : public bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct isnan_f<dvec<bf16_t> > : public bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct sqrt_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct rsqrt_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct exp_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct expm1_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct exp2_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct exp10_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct log_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct log1p_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct log2_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };	

        template <>
        struct log10_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };	
	
        template <>
        struct sinh_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct cosh_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct tanh_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct convert_rte<bf16_t, bf16_t> {
            static
            const std::string&
            body(const std::string& l) {
                return l;
            }
        };

        template <>
        struct convert_rte<float, bf16_t> {
            static
            std::string
            body(const std::string& l) {
                std::string r=std::string(bf16_base::bf16_to_f32::name())
                    + '(' + l + ')';
                return r;
            }
        };

        template <>
        struct convert_rte<bf16_t, float> {
            static
            std::string
            body(const std::string& l) {
                std::string r=std::string(bf16_base::f32_to_bf16::name())
                    + '(' + l + ')';
                return r;
            }
        };


        template <typename _S>
        struct convert_rte<bf16_t, _S> {
            static
            std::string
            body(const std::string& l) {
                std::string r0=convert_rte<float, _S>::body(l);
                std::string r1=convert_rte<bf16_t, float>::body(r0);
                return r1;
            }
        };

        template <typename _D>
        struct convert_rte<_D, bf16_t> {
            static
            std::string
            body(const std::string& l) {
                std::string r0=convert_rte<float, bf16_t>::body(l);
                std::string r1=convert_rte<_D, float>::body(r0);
                return r1;
            }
        };

    }

    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const dvec<bf16_t>& v);

    template <template <class _DVEC> class _OP,
              typename _L, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<_OP<dvec<bf16_t> >, _L, _R>& e );

    template <template <class _DVEC> class _OP,
              typename _L>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<_OP<dvec<bf16_t> >, _L, void>& e );

    template <typename _L, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::bit_or<dvec<bf16_t> >, _L, _R>& e );

    template <typename _L, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::bit_and<dvec<bf16_t> >, _L, _R>& e );

    template <typename _L, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::bit_xor<dvec<bf16_t> >, _L, _R>& e );

    template <typename _L>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::neg<dvec<bf16_t> >, _L, void>& e );

    template <typename _L>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::abs_f<dvec<bf16_t> >, _L, void>& e );

    template <typename _L>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::isinf_f<dvec<bf16_t> >, _L, void>& e );

    template <typename _L>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<dop::isnan_f<dvec<bf16_t> >, _L, void>& e );

    bool
    all_of(const dvec<bf16_t>& v);

    bool
    none_of(const dvec<bf16_t>& v);

    bool
    any_of(const dvec<bf16_t>& v);


    dvec<bf16_t>
    uniform_float_random_vector(rand48& rnd,
                                bf16_t min_val, bf16_t max_val);

    dvec<bf16_t>
    uniform_float_random_vector(rand& rnd,
                                bf16_t min_val, bf16_t max_val);

    namespace test {
        bool
        dvec_bf16();
    }
}

std::string_view
ocl::dop::bf16_base::emit_type_def::
name()
{
    return "__BF16_T_DEFINED__";
}

std::string_view
ocl::dop::bf16_base::emit_type_def::
body()
{
    return
        "#if !defined (__BF16_T_DEFINED__)\n"
        "#define __BF16_T_DEFINED__ 1\n"
        "struct bf16_s {\n"
        "    short _v;\n"
        "};\n"
        "typedef struct bf16_s bf16_t;\n"
        "#endif\n\n";
}

std::string_view
ocl::dop::bf16_base::bf16_to_f32::
name()
{
    return "__bf16_to_f32";
}

std::string_view
ocl::dop::bf16_base::bf16_to_f32::
body()
{
    return
        "inline\n"
        "float __bf16_to_f32(bf16_t s)\n"
        "{\n"
        "    unsigned int us=s._v;\n"
        "    us <<=16;\n"
        "    float r= as_float(us);\n"
        "    return r;\n"
        "}\n\n";
}

std::string_view
ocl::dop::bf16_base::f32_to_bf16::
name()
{
    return "__f32_to_bf16";
}

std::string_view
ocl::dop::bf16_base::f32_to_bf16::
body()
{
    return
        "inline\n"
        "bf16_t __f32_to_bf16(float ff)\n"
        "{\n"
        "    int f=as_int(ff);\n"
        "    int af=f & 0x7fffffff;\n"
        "    int sf=f & 0x80000000;\n"
        "    int r_nan = af;\n"
        "    const int rnd_bias = 0x7fff;\n"
        "    const int rnd_bias_p1 = 0x8000;\n"
        "    // force round nearest even if bit 16 is set\n"
        "    int r_def= (af & 0x00010000) ? af + rnd_bias_p1 : af + rnd_bias;\n"
        "    // subnormal result:\n"
        "    int r_sn = 0;\n"
        "    // select subnormal normal\n"
        "    int r_def_sn = (af < 0x00800000) ? r_sn : r_def;\n"
        "    // select nan or subnormal normal\n"
        "    int r = (af > 0x7f800000) ? r_nan : r_def_sn;\n"
        "    r |= sf;\n"
        "    r >>= 16;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n\n";
}

std::string_view
ocl::dop::bf16_base::_neg::
name()
{
    return "__bf16_neg";
}

std::string_view
ocl::dop::bf16_base::_neg::
body()
{
    return
        "inline\n"
        "bf16_t __bf16_neg(bf16_t a) {\n"
        "    short r= a._v ^ 0x8000;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n";
}

std::string
ocl::dop::bf16_base::_neg::
add_func(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=name();
    if (fnames.insert(fn1) == true) {
        s += body();
    }
    return s;
}

std::string_view
ocl::dop::bf16_base::_abs::
name()
{
    return "__bf16_abs";
}

std::string_view
ocl::dop::bf16_base::_abs::
body()
{
    return
        "inline\n"
        "bf16_t __bf16_abs(bf16_t a) {\n"
        "    short r= a._v & 0x7fff;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n\n";
}

std::string
ocl::dop::bf16_base::_abs::
add_func(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=name();
    if (fnames.insert(fn1) == true) {
        s += body();
    }
    return s;
}

std::string_view
ocl::dop::bf16_base::_isinf::
name()
{
    return "__bf16_isinf";
}

std::string_view
ocl::dop::bf16_base::_isinf::
body()
{
    return
        "inline\n"
        "bf16_t __bf16_isinf(bf16_t a) {\n"
        "    short aa= a._v & 0x7fff;\n"
        "    short r= aa == 0x7f80 ? ~0 : 0;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n\n";
}

std::string
ocl::dop::bf16_base::_isinf::
add_func(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=name();
    if (fnames.insert(fn1) == true) {
        s += body();
    }
    return s;
}

std::string_view
ocl::dop::bf16_base::_isnan::
name()
{
    return "__bf16_isnan";
}

std::string_view
ocl::dop::bf16_base::_isnan::
body()
{
    return
        "inline\n"
        "bf16_t __bf16_isnan(bf16_t a) {\n"
        "    short aa= a._v & 0x7fff;\n"
        "    short r= aa > 0x7f80 ? ~0 : 0;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n\n";
}

std::string
ocl::dop::bf16_base::_isnan::
add_func(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=name();
    if (fnames.insert(fn1) == true) {
        s += body();
    }
    return s;
}

std::string_view
ocl::dop::bf16_base::_or::
name()
{
    return "__bf16_or";
}

std::string_view
ocl::dop::bf16_base::_or::
body()
{
    return
        "inline\n"
        "bf16_t __bf16_or(bf16_t a, bf16_t b) {\n"
        "    short r= a._v | b._v;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n\n";
}

std::string
ocl::dop::bf16_base::_or::
add_func(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=name();
    if (fnames.insert(fn1) == true) {
        s += body();
    }
    return s;
}

std::string_view
ocl::dop::bf16_base::_and::
name()
{
    return "__bf16_and";
}

std::string_view
ocl::dop::bf16_base::_and::
body()
{
    return
        "inline\n"
        "bf16_t __bf16_and(bf16_t a, bf16_t b) {\n"
        "    short r= a._v & b._v;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n\n";
}

std::string
ocl::dop::bf16_base::_and::
add_func(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=name();
    if (fnames.insert(fn1) == true) {
        s += body();
    }
    return s;
}

std::string_view
ocl::dop::bf16_base::_xor::
name()
{
    return "__bf16_xor";
}

std::string_view
ocl::dop::bf16_base::_xor::
body()
{
    return
        "inline\n"
        "bf16_t __bf16_xor(bf16_t a, bf16_t b) {\n"
        "    short r= a._v ^ b._v;\n"
        "    bf16_t rr;\n"
        "    rr._v=r;\n"
        "    return rr;\n"
        "}\n\n";
}

std::string
ocl::dop::bf16_base::_xor::
add_func(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=name();
    if (fnames.insert(fn1) == true) {
        s += body();
    }
    return s;
}


std::string
ocl::dop::bf16_base::
add_conversions(be::kernel_functions& fnames)
{
    const auto fn0=emit_type_def::name();
    std::string s;
    if (fnames.insert(fn0) == true) {
        s = emit_type_def::body();;
    }
    const auto fn1=bf16_to_f32::name();
    if (fnames.insert(fn1) == true) {
        s += bf16_base::bf16_to_f32::body();
    }
    const auto fn2=f32_to_bf16::name();
    if (fnames.insert(fn2) == true) {
        s += bf16_base::f32_to_bf16::body();
    }
    return s;
}

std::string
ocl::dop::bf16_base::
unary_function(const std::string& l,
	       const std::string_view& op,
	       bool op_is_operator)
{
    std::ostringstream s;
    if (op_is_operator) {
	s << f32_to_bf16::name() << '('
	  << op << bf16_to_f32::name() << '('
	  << l
	  << "))";
    } else {
	s << f32_to_bf16::name() << '('
	  << op << '(' << bf16_to_f32::name() << '('
	  << l
	  << ")))";
    }
    return s.str();
}

std::string
ocl::dop::bf16_base::
binary_function(const std::string& l, const std::string& r,
                const std::string_view& op,
                bool op_is_operator)
{
    std::ostringstream s;
    if (op_is_operator) {
        s << f32_to_bf16::name() << '('
          << bf16_to_f32::name() << '('
          << l
          << ')'
          << op
          << bf16_to_f32::name() << '('
          << r
          << "))";
    } else {
        s << f32_to_bf16::name() << '('
          << op << '('
          << bf16_to_f32::name() << '('
          << l
          << "), "
          << bf16_to_f32::name() << '('
          << r
          << ")))";
    }
    return s.str();
}

std::string
ocl::dop::bf16_base::
cmp_operator(const std::string& l, const std::string& r,
             const std::string_view& op)
{
    std::ostringstream s;
    s << '('
      << bf16_to_f32::name() << '('
      << l        << ')'
      << op
      << bf16_to_f32::name() << '('
      << r
      << ")) ? (bf16_t){._v=~0} : (bf16_t){._v=0}";
    return s.str();
}

std::string
ocl::dop::neg<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string r{_neg::name()};
    r+= '(' + l + ')';
    return r;
}

std::string
ocl::dop::add<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return binary_function(l, r, names::add()(), true);
}

std::string
ocl::dop::sub<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return binary_function(l, r, names::sub()(), true);
}

std::string
ocl::dop::mul<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return binary_function(l, r, names::mul()(), true);
}

std::string
ocl::dop::div<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return binary_function(l, r, names::div()(), true);
}

std::string
ocl::dop::lt<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return cmp_operator(l, r, names::lt()());
}

std::string
ocl::dop::le<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return cmp_operator(l, r, names::le()());
}

std::string
ocl::dop::eq<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return cmp_operator(l, r, names::eq()());
}

std::string
ocl::dop::ne<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return cmp_operator(l, r, names::ne()());
}

std::string
ocl::dop::ge<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return cmp_operator(l, r, names::ge()());
}

std::string
ocl::dop::gt<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    return cmp_operator(l, r, names::gt()());
}

std::string
ocl::dop::bit_or<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    std::string rr{_or::name()};
    rr+= '(' + l + ", "+ r + ')';
    return rr;
}

std::string
ocl::dop::bit_and<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    std::string rr{_and::name()};
    rr+= '(' + l + ", "+ r + ')';
    return rr;
}

std::string
ocl::dop::bit_xor<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l, const std::string& r)
{
    std::string rr{_xor::name()};
    rr+= '(' + l + ", "+ r + ')';
    return rr;
}

std::string
ocl::dop::abs_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string r{_abs::name()};
    r+='(' + l + ')';
    return r;
}

std::string
ocl::dop::rint_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_rint()(), false);
}

std::string
ocl::dop::isinf_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string r{_isinf::name()};
    r+='(' + l + ')';
    return r;
}

std::string
ocl::dop::isnan_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string r{_isnan::name()};
    r+='(' + l + ')';
    return r;
}

std::string
ocl::dop::sqrt_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_sqrt()(), false);
}

std::string
ocl::dop::rsqrt_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_rsqrt()(), false);
}

std::string
ocl::dop::exp_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_exp()(), false);
}

std::string
ocl::dop::expm1_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_expm1()(), false);
}

std::string
ocl::dop::exp2_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_exp2()(), false);
}

std::string
ocl::dop::exp10_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_exp10()(), false);
}

std::string
ocl::dop::log_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_log()(), false);
}

std::string
ocl::dop::log1p_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_log1p()(), false);
}

std::string
ocl::dop::log2_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_log2()(), false);
}

std::string
ocl::dop::log10_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_log10()(), false);
}

std::string
ocl::dop::sinh_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_sinh()(), false);
}

std::string
ocl::dop::cosh_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_cosh()(), false);
}

std::string
ocl::dop::tanh_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_tanh()(), false);
}

std::string_view
cftal::
def_custom_func(ocl::be::kernel_functions& fnames,
                const bf16_t& v)
{
    static_cast<void>(v);
    const auto fn1=ocl::dop::bf16_base::emit_type_def::name();
    std::string_view s;
    if (fnames.insert(fn1) == true) {
        s = ocl::dop::bf16_base::emit_type_def::body();;
    }
    return s;
}

std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const dvec<bf16_t>& v)
{
    static_cast<void>(v);
    return dop::bf16_base::add_conversions(fnames);
}

template <template <class _DVEC> class _OP, typename _L, typename _R>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<_OP<dvec<bf16_t> >, _L, _R>& e )
{
    std::string r(def_custom_func(fnames, e._l));
    r += def_custom_func(fnames, e._r);
    r += dop::bf16_base::add_conversions(fnames);
    return r;
}

template <template <class _DVEC> class _OP, typename _L>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<_OP<dvec<bf16_t> >, _L, void>& e )
{
    std::string r(def_custom_func(fnames, e._l));
    r += dop::bf16_base::add_conversions(fnames);
    return r;
}

template <typename _L, typename _R>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::bit_or<dvec<bf16_t> >, _L, _R>& e )
{
    std::string r(def_custom_func(fnames, e._l));
    r += def_custom_func(fnames, e._r);
    r += dop::bit_or<dvec<bf16_t> >::_or::add_func(fnames);
    return r;
}

template <typename _L, typename _R>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::bit_and<dvec<bf16_t> >, _L, _R>& e )
{
    std::string r(def_custom_func(fnames, e._l));
    r += def_custom_func(fnames, e._r);
    r += dop::bit_and<dvec<bf16_t> >::_and::add_func(fnames);
    return r;
}

template <typename _L, typename _R>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::bit_xor<dvec<bf16_t> >, _L, _R>& e )
{
    std::string r(def_custom_func(fnames, e._l));
    r += def_custom_func(fnames, e._r);
    r += dop::bit_xor<dvec<bf16_t> >::_xor::add_func(fnames);
    return r;
}

template <typename _L>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::neg<dvec<bf16_t> >, _L, void>& e )
{
    std::string r(dop::neg<dvec<bf16_t> >::_neg::add_func(fnames));
    r += def_custom_func(fnames, e._l);
    return r;
}

template <typename _L>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::abs_f<dvec<bf16_t> >, _L, void>& e )
{
    std::string r(dop::abs_f<dvec<bf16_t> >::_abs::add_func(fnames));
    r += def_custom_func(fnames, e._l);
    return r;
}

template <typename _L>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::isinf_f<dvec<bf16_t> >, _L, void>& e )
{
    std::string r(dop::isinf_f<dvec<bf16_t> >::_isinf::add_func(fnames));
    r += def_custom_func(fnames, e._l);
    return r;
}

template <typename _L>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<dop::isnan_f<dvec<bf16_t> >, _L, void>& e )
{
    std::string r(dop::isnan_f<dvec<bf16_t> >::_isnan::add_func(fnames));
    r += def_custom_func(fnames, e._l);
    return r;
}

bool
ocl::
all_of(const dvec<bf16_t>& v)
{
    const dvec<int16_t>& t=reinterpret_cast<const dvec<int16_t>&>(v);
    return all_of(t);
}

bool
ocl::
none_of(const dvec<bf16_t>& v)
{
    const dvec<int16_t>& t=reinterpret_cast<const dvec<int16_t>&>(v);
    return none_of(t);
}

bool
ocl::
any_of(const dvec<bf16_t>& v)
{
    const dvec<int16_t>& t=reinterpret_cast<const dvec<int16_t>&>(v);
    return any_of(t);
}

ocl::dvec<ocl::bf16_t>
ocl::
uniform_float_random_vector(rand& rnd, bf16_t min_val, bf16_t max_val)
{
    const auto f_min_val=static_cast<float>(min_val);
    const auto f_max_val=static_cast<float>(max_val);
    dvec<float> t=uniform_float_random_vector(rnd, f_min_val, f_max_val);
    dvec<bf16_t> r=cvt<dvec<bf16_t>>(t);
    return r;
}

ocl::dvec<ocl::bf16_t>
ocl::
uniform_float_random_vector(rand48& rnd, bf16_t min_val, bf16_t max_val)
{
    const auto f_min_val=static_cast<float>(min_val);
    const auto f_max_val=static_cast<float>(max_val);
    dvec<float> t=uniform_float_random_vector(rnd, f_min_val, f_max_val);
    dvec<bf16_t> r=cvt<dvec<bf16_t>>(t);
    return r;
}


bool
ocl::test::dvec_bf16()
{
    int r=3;
    try {
        const size_t N=1024*1024;
        dvec<bf16_t> v(0.0_bf16, N);
        dvec<bf16_t> s=v+v;
        dvec<bf16_t> d=v-v;
        dvec<bf16_t> p=v*v;
        dvec<bf16_t> q=v/v;
        dvec<bf16_t> m=(s-d)*p/q;

        dvec<bf16_t> a_m=abs(m);
        dvec<bf16_t> a_n=-m;
        dvec<bf16_t>::mask_type a_i=isinf(a_n);
        dvec<bf16_t>::mask_type a_nan=isnan(a_n);

        dvec<bf16_t>::mask_type c_lt=s < v;
        dvec<bf16_t>::mask_type c_le=s <= v;
        dvec<bf16_t>::mask_type c_eq=s == v;
        dvec<bf16_t>::mask_type c_ne=s != v;
        dvec<bf16_t>::mask_type c_ge=s >= v;
        dvec<bf16_t>::mask_type c_gt=s > v;


        dvec<float> mf=cvt<dvec<float>>(m);
        dvec<bf16_t> m2=cvt<dvec<bf16_t>>(mf);

        r=0;
    }
    catch (const std::exception& ex)  {
        std::cerr << "caught exception:\n"
                  << ex.what() << '\n';
    }
    catch (...) {
        std::cerr << "unspecified exception type\n";
    }
    if (r) {
        return r;
    }
#if 1
    r = 3;
    try {
        using namespace cftal;
        using namespace ocl;
        using namespace ocl::test;

        using rtype = bf16_t;
        constexpr const std::size_t NMAX=8*16384;
        std::cout << "testing buffers with up to "
                  << NMAX-1 << " elements\n.";
        for (std::size_t i=4; i<NMAX; ++i) {
            if ((i & 0x7f) == 0x7f || i==1) {
                std::cout << "using buffers with "
                          <<  i
                          << " elements (" << i*sizeof(rtype)
                          << " bytes)\r" << std::flush;
            }
            ops<rtype> t(i);
            if (t.perform() == false) {
                std::cout << "\ntest for vector length " << i << " failed\n";
                std::exit(3);
            }
        }
        std::cout << "\ntest passed\n";
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: " << e.what()
                  << std::endl;
    }
#endif
    return r;
}



int main()
{
    bool r=ocl::test::dvec_bf16();
    return r==true ? 0 : 1;
}
