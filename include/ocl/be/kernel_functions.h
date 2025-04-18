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
#if !defined (__OCL_BE_KERNEL_FUNCTIONS_H__)
#define __OCL_BE_KERNEL_FUNCTIONS_H__ 1

#include <ocl/config.h>
#include <string>
#include <set>

namespace ocl {
    namespace be {

        // set of strings for custom functions in kernels
        class kernel_functions {
            std::set<std::string> _s;
        public:
            kernel_functions();
            kernel_functions(const kernel_functions& r);
            kernel_functions(kernel_functions&& r);
            kernel_functions& operator=(const kernel_functions& r);
            kernel_functions& operator=(kernel_functions&& r);
            ~kernel_functions();
            bool
            insert(const std::string& fn);
            void
            clear();
        };

    }
}

// local variables:
// mode: c++
// end:
#endif

