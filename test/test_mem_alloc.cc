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
#include <cftal/vec.h>
#include <ocl/ocl.h>
#include <iostream>
#include <iomanip>


int main()
{
    try {
        using namespace ocl;

        using ftype = float;

        constexpr const std::size_t elem_count=(64*1024*1024ULL)/sizeof(ftype);
        std::vector<dvec<ftype> > vv;

        size_t dmem=be::data::instance()->dcq().d().global_memory_size();
        double dmem_mb=static_cast<double>(dmem)/(1024.0*1024.0);
        std::cout << "available device memory: " << dmem_mb << " MB\n";

        constexpr const std::size_t max_count=
            (2*1024*1024ULL*1024ULL)/sizeof(ftype)/elem_count;
        std::cout << std::fixed << std::setprecision(2);
        for (std::size_t i=0; i<max_count; ++i) {
            double mb=(double(vv.size())*elem_count*sizeof(ftype))/
                double(1024*1024);
            std::cout << "total allocated memory: "
                      << std::setw(7) << mb << " MB\n";
            auto init_val=static_cast<ftype>(i+1);
            dvec<ftype> vi(init_val, elem_count);
            vv.emplace_back(std::move(vi));
            for (size_t j=0; j<vv.size(); ++j) {
                vv[j] += init_val;
            }
        }
        std::cout << "test passed\n";
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: ocl::be::error: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: runtime error: " << e.what()
                  << std::endl;
    }
    return 0;
}
