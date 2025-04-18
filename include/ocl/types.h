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
#if !defined (__OCL_TYPES_H__)
#define __OCL_TYPES_H__ 1

#include <ocl/config.h>
#include <cstdint>

namespace ocl {

    using std::uint8_t;
    using std::uint16_t;
    using std::uint32_t;
    using std::uint64_t;

    using std::int8_t;
    using std::int16_t;
    using std::int32_t;
    using std::int64_t;

    using f32_t = float;
    using f64_t = double;

    using std::size_t;
    using std::ptrdiff_t;
    using ssize_t = ptrdiff_t;
}
// Local variables:
// mode: c++
// end:
#endif
