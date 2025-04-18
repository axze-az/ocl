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
#include <cftal/math/func_constants.h>
#include <cftal/math/func_constants_f32.h>
#include <cftal/math/func_constants_f64.h>
#include <cftal/as.h>
#include <vector>

namespace ocl { namespace test {
        template <typename _T, typename _I>
        void subnormals(const std::string& tname, _I inc);
        void subnormals_f32();
        void subnormals_f64();
    }
}

template <typename _T, typename _I>
void
ocl::test::subnormals(const std::string& tname, _I inc)
{
    using fc_t=cftal::math::func_constants<_T>;
    const _T msub=fc_t::max_denormal();
    const _I maxi=cftal::as<_I>(msub);
    std::vector<_T> vh;
    for (_I i=0; i<=maxi; i+= inc) {
        _T f=cftal::as<_T>(i);
        vh.push_back(f);
    }
    std::cout << "created a dvec<" << tname
              << "> with " << vh.size() << " elements\n";
    dvec<_T> v0(vh);
    dvec<_T> v1=(v0*_T(2.0))*_T(0.5);
    std::vector<_T> vr(v1);
    size_t cnt=0;
    for (size_t i=0; i< vr.size(); ++i) {
        if (vr[i] != _T(0.0))
            ++cnt;
    }
    std::cout << cnt << " elements after operations are not equal zero"
              << std::endl;
    if (cnt == 0) {
        std::cout << tname << ": no subnormal support\n";
    }
    if (cnt == vr.size()-1) {
        std::cout << tname << ": subnormal support\n";
    }
}

void
ocl::test::subnormals_f32()
{
    subnormals<float, uint32_t>("float", 1);
}

void
ocl::test::subnormals_f64()
{
    subnormals<double, uint64_t>("double", 0x20000000);
}

int main()
{
    ocl::test::subnormals_f32();
    ocl::test::subnormals_f64();
    return 0;
}
