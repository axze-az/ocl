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

    namespace test {

    }
}

std::string_view
ocl::dop::bf16_base::emit_type_def::
body()
{
    return
        "#if !defined (__BF16_T_DEFINED__)\n"
        "#define __BF16_T_DEFINED__ 1\n"
        "typedef short int bf16_t;\n"
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
         "float " << name() << "(bf16_t s)"
         "{\n"
         "    unsigned int us=s;\n"
         "    us <<=16;"
         "    float r= as_float(us);"
         "    return r;"
         "}\n";
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
         "bf16_t " << name() << "(float ff)"
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
         "}\n";
    return s.str();
}

int main()
{
    return 0;
}
