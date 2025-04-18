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
#if !defined (__OCL_BE_KERNEL_KEY_H__)
#define __OCL_BE_KERNEL_KEY_H__ 1

#include <ocl/config.h>
#include <string>
#include <iosfwd>
#include <cstdint>

namespace ocl {

    namespace be {

        // kernel_key: unique identification of a kernel
        class kernel_key {
            const void* _p;
            std::string _s;
        public:
            kernel_key(const void* p, const std::string& s);
            kernel_key(const kernel_key& r);
            kernel_key(kernel_key&& r);
            kernel_key& operator=(const kernel_key& r);
            kernel_key& operator=(kernel_key&& r);
            ~kernel_key();
            intptr_t h() const { return intptr_t(_p); }
            const std::string& l() const { return _s; }
        };

        bool operator<(const kernel_key& a, const kernel_key& b);
        bool operator<=(const kernel_key& a, const kernel_key& b);
        bool operator==(const kernel_key& a, const kernel_key& b);
        bool operator!=(const kernel_key& a, const kernel_key& b);
        bool operator>=(const kernel_key& a, const kernel_key& b);
        bool operator>(const kernel_key& a, const kernel_key& b);

        struct print_kernel_key {
            const kernel_key& _k;
            print_kernel_key(const kernel_key& k) : _k(k) {}
        };

        std::ostream&
        operator<<(std::ostream& s, const print_kernel_key& kk);
    }

}

// Local variables:
// mode: c++
// end:
#endif
