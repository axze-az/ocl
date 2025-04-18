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
#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <cmath>

namespace ocl {


}

// using namespace ocl;

template <class _T>
_T
test_func(const _T& a, const _T& b)
{
    // return _T( (2.0 + a + b) / (a * b)  + (a + a * b ) - a);

    return _T((2.0f + a + b) / (a * b)  + (a + a * b ) - a) *
        ((6.0f + a + b) / (a * b)  + (a + a * b ) - a);
}

template <class _T>
_T
test_func(const _T& a, const _T& b, const _T& c)
{
    return (a+b *c) *c + 2.0f;
}

namespace {

    template <class _T>
    _T rel_error(const _T& a, const _T& b)
    {
        _T e((a -b ));
        e = e < _T(0) ? -e : e;
        _T m((a+b)*_T(0.5));
        if (m != _T(0)) {
            e /= m;
        }
        return e;
    }

}


int main()
{
    try {

        using namespace ocl;

        // using cftal::v8f32;

        // const unsigned BEIGNET_MAX_BUFFER_SIZE=16384*4096;
        // const unsigned GALLIUM_MAX_BUFFER_SIZE=2048*4096;
        const unsigned SIZE=2048*4096;
        std::cout << "using buffers of "
                  << double(SIZE*sizeof(float))/(1024*1024)
                  << "MiB\n";
        float a(2.0f), b(3.0f);

        dvec<float> v0(SIZE, a);
        // std::vector<float> vha(SIZE, a);
        dvec<float> va(v0);
        std::vector<float> vhb(SIZE, 3.0f);
        dvec<float> vb(vhb);
        dvec<float> vc= test_func(va, vb);
        dvec<float> vd= test_func(va, vb, vc);
        dvec<float> vd2= test_func(va, vb, vc);

        float c= test_func(a, b);
        float d= test_func(a, b, c);

        std::vector<float> res(vd);
#if 0
        dvec<cftal::v8f32> vva(SIZE/8, a);
        dvec<cftal::v8f32> vvb(SIZE/8, b);
        dvec<cftal::v8f32> vvc(SIZE/8, c);
        dvec<cftal::v8f32> vres(test_func(vva, vvb, vvc));
#endif
        if (SIZE <= 4096) {
            for (std::size_t i=0; i< res.size(); ++i) {
                std::cout << i << ' ' << res[i] << std::endl;
            }
        } else {
            for (std::size_t i=0; i< res.size(); ++i) {
                float e=rel_error(res[i], d);
                if (e > 1e-7) {
                    std::ostringstream m;
                    m << "res[" << i << " ]="
                      << std::setprecision(12)
                      << res[i] << " != " << d
                      << " e= " << e;
                    throw std::runtime_error(m.str());
                }
            }
        }

        std::cout << "scalar " << d << std::endl;

    }
    catch (const ocl::impl::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << ocl::impl::err2str(e)
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: " << e.what()
                  << std::endl;
    }
    return 0;
}
