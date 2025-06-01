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
#include <sstream>

namespace ocl {

    namespace impl {


    }

    using cftal::bf16_t;
    using cftal::operator ""_bf16;

    namespace be {

        template <>
        struct type_2_name<bf16_t> {
            static
            constexpr
            std::string_view v() {
                return "ushort";
            }
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
        };
    }

    template <template <class _DVEC> class _OP,
              typename _L, typename _R>
    std::string
    def_custom_func(be::kernel_functions& fnames,
                    const expr<_OP<dvec<bf16_t> >, _L, _R>& e );

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

bool
ocl::test::dvec_bf16()
{
    bool r=false;
    try {
        const size_t N=1024*1024;
        dvec<bf16_t> v(0.0_bf16, N);
        dvec<bf16_t> s=v+v;
        r=true;
    }
    catch (const std::exception& ex)  {
        std::cerr << "caught exception:\n"
                  << ex.what() << '\n';
    }
    catch (...) {
        std::cerr << "unspecified exception type\n";
        r=false;
    }
    return r;
}



int main()
{
    bool r=ocl::test::dvec_bf16();
    return r==true ? 0 : 1;
}
