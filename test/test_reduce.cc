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
#include <ocl/ocl.h>
#include <ocl/test/tools.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <cmath>

int main()
{
    try {

        using namespace ocl;
        using cftal::v8f32;

        const unsigned SIZE=128*1024*1024;
        std::cout << "using buffers of "
                  << double(SIZE*sizeof(float))/(1024*1024)
                  << "MiB\n";
        float a(2.0f), b(3.0f);

        dvec<float> va(a, SIZE);
        dvec<float> vb(b, SIZE);
        typename dvec<float>::mask_type tgt= va == vb;

        // va != vb
        // all_of == false
        bool ao = all_of(tgt);
        // none_of == true
        bool no = none_of(tgt);
        // any_of == false
        bool so = any_of(tgt);
        if (ao != false || no != true || so != false) {
            throw std::runtime_error("xxx_of failed.");
        }
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "caught exception: " << e.what()
                  << std::endl;
    }
    return 0;
}
