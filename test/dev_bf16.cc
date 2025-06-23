//
// Copyright (C) 2010-2025 Axel Zeuner
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
#include "ocl/ocl.h"
#include <cftal/vec.h>
#include <ocl/test/ops.h>
#include <sstream>

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
                return "ushort";
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
                body();
            };

            struct bf16_to_f32 {
                static
                std::string_view
                name();

                static
                std::string
                body();
            };

            struct f32_to_bf16 {
                static
                std::string_view
                name();

                static
                std::string
                body();
            };

            static
            std::string
            unary_function(const std::string& l,
                           const std::string_view& op);

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
        struct neg<dvec<bf16_t> > : private bf16_base {
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
        struct abs_f<dvec<bf16_t> > : private bf16_base {
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
        struct isinf_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

        template <>
        struct isnan_f<dvec<bf16_t> > : private bf16_base {
            static
            std::string
            body(const std::string& l);
        };

    }

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

    dvec<bf16_t>
    uniform_float_random_vector(rand48& rnd,
                                bf16_t min_val, bf16_t max_val);


    namespace test {
        bool
        dvec_bf16();
    }
}

std::string_view
ocl::dop::bf16_base::emit_type_def::
body()
{
    return
        "#if !defined (__BF16_T_DEFINED__)\n"
        "#define __BF16_T_DEFINED__ 1\n"
        "typedef ushort bf16_t;\n"
        "#endif\n\n";
}

std::string_view
ocl::dop::bf16_base::bf16_to_f32::
name()
{
    return "__bf16_to_f32";
}

std::string
ocl::dop::bf16_base::bf16_to_f32::
body()
{
    std::ostringstream s;
    s << emit_type_def::body()
      << "inline\n"
         "float " << name() << "(bf16_t s)\n"
         "{\n"
         "    unsigned int us=s;\n"
         "    us <<=16;\n"
         "    float r= as_float(us);\n"
         "    return r;\n"
         "}\n\n";
    return s.str();
}

std::string_view
ocl::dop::bf16_base::f32_to_bf16::
name()
{
    return "__f32_to_bf16";
}

std::string
ocl::dop::bf16_base::f32_to_bf16::
body()
{
    std::ostringstream s;
    s << emit_type_def::body()
      << "inline\n"
         "bf16_t " << name() << "(float ff)\n"
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
         "    return r;\n"
         "}\n\n";
    return s.str();
}

std::string
ocl::dop::bf16_base::
unary_function(const std::string& l, const std::string_view& op)
{
    std::ostringstream s;
    s << f32_to_bf16::name() << '('
      << op << bf16_to_f32::name() << '('
      << l
      << "))";
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
    s << bf16_to_f32::name() << '('
      << l        << ')'
      << op
      << bf16_to_f32::name() << '('
      << r
      << ')';
    return s.str();
}

std::string
ocl::dop::neg<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string r='(' + l + " ^ 0x8000)";
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
ocl::dop::abs_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string r='(' + l + " & 0x7fff)";
    return r;
}

std::string
ocl::dop::rint_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    return unary_function(l, names::f_rint()());
}

std::string
ocl::dop::isinf_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string abs_l=abs_f<dvec<bf16_t> >::body(l);
    std::string s='(' + abs_l + " == 0x7f80)";
    return s;
}

std::string
ocl::dop::isnan_f<ocl::dvec<ocl::bf16_t> >::
body(const std::string& l)
{
    std::string abs_l=abs_f<dvec<bf16_t> >::body(l);
    std::string s='(' + abs_l + " > 0x7f80)";
    return s;
}

template <template <class _DVEC> class _OP, typename _L, typename _R>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<_OP<dvec<bf16_t> >, _L, _R>& e )
{
    static_cast<void>(e);
    const auto fn1=dop::bf16_base::bf16_to_f32::name();
    std::string s;
    if (fnames.insert(fn1) == true) {
        s = dop::bf16_base::bf16_to_f32::body() + '\n';
    }
    const auto fn2=dop::bf16_base::f32_to_bf16::name();
    if (fnames.insert(fn2) == true) {
        s += dop::bf16_base::f32_to_bf16::body() + '\n';
    }
    return s;
}

template <template <class _DVEC> class _OP, typename _L>
std::string
ocl::
def_custom_func(be::kernel_functions& fnames,
                const expr<_OP<dvec<bf16_t> >, _L, void>& e )
{
#if 1
    return def_custom_func<_OP, _L, void>(fnames, e);
#else
    static_cast<void>(e);
    const auto fn1=dop::bf16_base::bf16_to_f32::name();
    std::string s;
    if (fnames.insert(fn1) == true) {
        s = dop::bf16_base::bf16_to_f32::body() + '\n';
    }
    const auto fn2=dop::bf16_base::f32_to_bf16::name();
    if (fnames.insert(fn2) == true) {
        s += dop::bf16_base::f32_to_bf16::body() + '\n';
    }
    return s;
#endif
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
        dvec<bf16_t>::mask_type a_i=isinf(m);
        dvec<bf16_t>::mask_type a_nan=isnan(m);

        dvec<bf16_t>::mask_type c_lt=s < v;
        dvec<bf16_t>::mask_type c_le=s <= v;
        dvec<bf16_t>::mask_type c_eq=s == v;
        dvec<bf16_t>::mask_type c_ne=s != v;
        dvec<bf16_t>::mask_type c_ge=s >= v;
        dvec<bf16_t>::mask_type c_gt=s > v;
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
#if 0
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
