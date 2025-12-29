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
#include <ocl/ocl.h>
#include <ocl/random.h>
#include <ocl/test/tools.h>
#include <ocl/test/ops.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>


int main()
{
    int r=3;
    using namespace cftal;
    using namespace ocl;
    using namespace ocl::test;

    using rtype = float;
    try {
        const size_t N=1024*1024;
        dvec<rtype> v(rtype(0), N);
        dvec<rtype> s=v+v;
        dvec<rtype> d=v-v;
        dvec<rtype> p=v*v;
        dvec<rtype> q=v/v;
        dvec<rtype> m=(s-d)*p/q;

        dvec<rtype> a_m=abs(m);

        dvec<rtype>::mask_type c_lt=s < v;
        dvec<rtype>::mask_type c_le=s <= v;
        dvec<rtype>::mask_type c_eq=s == v;
        dvec<rtype>::mask_type c_ne=s != v;
        dvec<rtype>::mask_type c_ge=s >= v;
        dvec<rtype>::mask_type c_gt=s > v;

        dvec<rtype> a_n=-m;
        dvec<rtype>::mask_type a_i=isinf(a_n);
        dvec<rtype>::mask_type a_nan=isnan(a_n);



        dvec<rtype> mf=cvt<dvec<rtype>>(m);
        dvec<rtype> m2=cvt<dvec<rtype>>(mf);
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
    try {
	constexpr const std::size_t NMAX=16*16384;
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
        r = 0;
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
