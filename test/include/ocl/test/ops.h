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
#if !defined (__OCL_TEST_OPS_H__)
#define __OCL_TEST_OPS_H__ 1

#include <ocl/config.h>
#include <ocl/test/ops_base.h>


namespace ocl {

    namespace test {

        template <typename _T>
        class ops : public ops_base<_T> {
            using b_t = ops_base<_T>;
            using b_t::_res;
            using b_t::_a0;
            using b_t::_a1;
            using b_t::_h_res;
            using b_t::_h_a0;
            using b_t::_h_a1;
            using b_t::check_res;
        public:
            ops(size_t n) : ops_base<_T>(n, _T(-256.0), _T(256.0)) {}
            bool perform();
        };

    }
}

template <typename _T>
bool
ocl::test::ops<_T>::perform()
{
    bool rc=true;
    // assignment:
    _res = _a0;
    _h_res = _h_a0;
    rc &= check_res("assignment v = v");
    // addition
    _res = _a0 + _a1;
    _h_res = _h_a0 + _h_a1;
    rc &= check_res("addition v v");
    // subtraction
    _res = _a0 - _a1;
    _h_res = _h_a0 - _h_a1;
    rc &= check_res("subtraction v v");
    // multiplication
    _res = _a0 * _a1;
    _h_res = _h_a0 * _h_a1;
    rc &= check_res("multiplication v v");
    // division
    _res = _a0 / _a1;
    _h_res = _h_a0 / _h_a1;
    rc &= check_res("division v v");
    // neg
    _res = -_a0;
    _h_res = -_h_a0;
    rc &= check_res("negation v");
    // abs
    _res = abs(_a0);
    _h_res = abs(_h_a0);
    rc &= check_res("abs");
#if 0
    // sqrt
    _res = sqrt(_a0);
    _h_res = sqrt(_h_a0);
    rc &= check_res("sqrt");
    // link against cftal.so is missing yet
    _res = rsqrt(_a0);
    _h_res = rsqrt(_h_a0);
    rc &= check_res("rsqrt");
#endif
    return rc;
}

#endif
