#include <cftal/vec.h>
#include <ocl/ocl.h>
#include <ocl/random.h>
#include <ocl/test/tools.h>
#include <ocl/test/ops_base.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>

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
    // sqrt
    _res = sqrt(_a0);
    _h_res = sqrt(_h_a0);
    rc &= check_res("sqrt");
#if 0
    // link against cftal.so is missing yet
    _res = rsqrt(_a0);
    _h_res = rsqrt(_h_a0);
    rc &= check_res("rsqrt");
#endif
    return rc;
}


int main()
{
    try {
        using namespace cftal;
        using namespace ocl;
        using namespace ocl::test;

        using rtype = float;
        constexpr const std::size_t NMAX=16*16384;
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
    return 0;
}
