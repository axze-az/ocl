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
#include "ocl/dvec.h"

namespace ocl {
    namespace test {

        bool
        test_xxx_of();

        bool
        expect(const std::string& msg, bool expect, bool res);
    }

}

bool
ocl::test::expect(const std::string& msg, bool expect, bool res)
{
    bool r=expect==res;
    if (r==false) {
        std::cout << msg << " failed\n";
    }
    return r;
}



bool
ocl::test::test_xxx_of()
{
    bool r=true;
    for (int j=0; j<1024; j+=16 ) {
        for (int i=4; i<128; ++i) {
            size_t s=j*1024 + i;
            if ((i & 0xf)==0) {
                std::cout << "testing size " << s << "\r" << std::flush;
            }
            std::vector h0(s, 2.4f);
            dvec<float> v0(h0);
            dvec<float>::mask_type t00= v0==0.0f;
            r &= expect("all_of(v0==0.0f)=false  ", false, all_of(t00));
            r &= expect("any_of(v0==0.0f)=false  ", false, any_of(t00));
            r &= expect("none_of(v0==0.0f)=true  ", true, none_of(t00));
            dvec<float>::mask_type t01= v0==2.4f;
            r &= expect("all_of(v0==2.4f)=true   ", true, all_of(t01));
            r &= expect("any_of(v0==2.4f)=true   ", true, any_of(t01));
            r &= expect("none_of(v0==2.4f)=false ", false, none_of(t01));
            h0.back() = 0.0f;
            dvec<float> v1(h0);
            dvec<float>::mask_type t10= v1==0.0f;
            r &= expect("all_of(v1==0.0f)=false  ", false, all_of(t10));
            r &= expect("any_of(v1==0.0f)=true   ", true, any_of(t10));
            r &= expect("none_of(v1==0.0f)=false ", false, none_of(t10));
            dvec<float>::mask_type t11= v1==2.4f;
            r &= expect("all_of(v0==2.4f)=false   ", false, all_of(t11));
            r &= expect("any_of(v0==2.4f)=true    ", true, any_of(t11));
            r &= expect("none_of(v0==2.4f)=false  ", false, none_of(t11));
        }
    }
    std::cout << '\n';
    return r;
}

int main()
{
    bool r=ocl::test::test_xxx_of();
    return r==true ? 0 : 1;
}
