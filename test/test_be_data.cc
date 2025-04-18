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
#include "ocl/be/devices.h"
#include "ocl/be/data.h"

int main()
{
    int r;
    try {

        std::vector<ocl::be::device> v(ocl::be::devices());
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "count of devices: " << v.size() << std::endl;
        for (std::size_t i = 0; i< v.size(); ++i) {
            std::cout << std::string(60, '-') << std::endl;
            std::cout << ocl::be::device_info(v[i]);
        }
        ocl::be::device dd(ocl::be::default_device());
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "selected device: \n";
        std::cout << ocl::be::device_info(dd);

        const ocl::be::device& bed =
            ocl::be::data::instance()->dcq().d();
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "selected backend device: \n";
        std::cout << ocl::be::device_info(bed);
        r = 0;
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
        r = 3;
    }
    return r;
}
