#if !defined (__OCL_TEST_OPS_BASE_H__)
#define __OCL_TEST_OPS_BASE_H__ 1

#include <ocl/config.h>
#include <ocl/dvec.h>
#include <ocl/random.h>
#include <cftal/lvec.h>
#include <random>

#define USE_DEVICE_COMPARE 1

namespace ocl {

    namespace test {

        using cftal::lvec;

        template <typename _T>
        class ops_base {
        protected:
            // result device buffer
            dvec<_T> _res;
            // arg0 device buffer
            dvec<_T> _a0;
            // arg1 device buffer
            dvec<_T> _a1;
            // host results on the device
            dvec<_T> _h_res_d;
            // comparison between _h_res_d and _h_res on device
            typename dvec<_T>::mask_type _cmp_res;
            // comparison between _h_res_d and _h_res on host
            lvec<typename dvec<_T>::mask_type::value_type> _h_cmp_res;
            // device results on the host
            lvec<_T> _h_d_res;
            // result host buffer
            lvec<_T> _h_res;
            // arg0 host buffer
            lvec<_T> _h_a0;
            // arg1 host buffer
            lvec<_T> _h_a1;
            ops_base(size_t n, const _T& min_val, const _T& max_val);
            bool check_res(const std::string& msg);
            bool check_res(const std::string& msg,
                           const _T& max_rel_tol);
        };

        extern
        template class ops_base<float>;
        extern
        template class ops_base<double>;
    }
}

template <typename _T>
ocl::test::ops_base<_T>::
ops_base(size_t n, const _T& min_val, const _T& max_val)
    : _res(_T(0), n), _a0(_res), _a1(_res), _h_res_d(_res),
      _cmp_res(typename dvec<_T>::mask_value_type(0), n),
      _h_cmp_res(typename dvec<_T>::mask_value_type(0), n),
      _h_d_res(_T(0), n),
      _h_res(_h_d_res), _h_a0(_h_d_res), _h_a1(_h_d_res)
{
#if 1
    rand48 rnd(n);
    rnd.seed_times_global_id(n);
    _a0 = uniform_float_random_vector(rnd, min_val, max_val);
    _a1 = uniform_float_random_vector(rnd, min_val, max_val);
    _a0.copy_to_host(&_h_a0[0]);
    _a1.copy_to_host(&_h_a1[0]);
#else
    std::mt19937_64 rnd;
    rnd.seed(n);
    std::uniform_real_distribution<_T> distrib(_T(-2.0), _T(2.0));
    for (std::size_t i=0; i<n; ++i) {
        _h_a0[i]=distrib(rnd);
        _h_a1[i]=distrib(rnd);
    }
    _a0.copy_from_host(&_h_a0[0]);
    _a1.copy_from_host(&_h_a1[0]);
#endif
}

template <typename _T>
bool
ocl::test::ops_base<_T>::check_res(const std::string& msg)
{
#if USE_DEVICE_COMPARE>0
    // copy host results to device
    _h_res_d.copy_from_host(&_h_res[0]);
    // compare on device and make the result buffer compatible with
    // the results of lvec/vec comparisons
    _cmp_res = select(((_res == _h_res_d) |
                       ((_res != _res) & (_h_res_d != _h_res_d))),
                      -1, 0);
    bool res=all_of(_cmp_res);
    if (res==false) {
        _res.copy_to_host(&_h_d_res[0]);
        // copy back comparison result
        _cmp_res.copy_to_host(&_h_cmp_res[0]);
        // dump(_h_cmp_res, "cmp res:");
        std::cout << "\nFAILED: " << msg <<std::endl;
        std::cout << std::scientific
                  << std::setprecision(10);
        // std::cout << std::hexfloat;
        for (std::size_t i=0; i<_h_cmp_res.size(); ++i) {
            if (_h_cmp_res[i] != 0)
                continue;
            _T _hr=_h_res[i];
            _T _dr=_h_d_res[i];
            _T _f0=_h_a0[i];
            _T _f1=_h_a1[i];
            _T _rd= (_hr - _dr)/(_T(0.5)*(_hr + _dr));
            _T max_rd=0x1.0p-24*8;
            std::cout << "vector entry " << i << ' '
                      << _f0 << ' ' << _f1 << ' ' << _hr << ' ' << _dr
                      << ' ' << _rd << ' ' << max_rd << '\n';
        }
        // dump(_h_a0, "_a0");
        // dump(_h_a1, "_a1");
        // dump(_h_d_res, "device result: ");
        // dump(_h_res, "host result: ");
        std::cout << "FAILED: " << msg <<std::endl;
        std::cout << std::scientific
                  << std::setprecision(10);
    }
#else
    _res.copy_to_host(&_h_d_res[0]);
    // check for equality or both nan
    typename lvec<_T>::mask_type cv = (_h_d_res == _h_res)
        | ((_h_d_res != _h_d_res) & (_h_res != _h_res));
    bool res=all_of(cv);
    if (res==false) {
        std::cout << "\nFAILED: " << msg <<std::endl;
        std::cout << std::scientific
                  << std::setprecision(10);
        std::cout << std::hexfloat;
        for (std::size_t i=0; i<cv.size(); ++i) {
            if (cv[i] != 0)
                continue;
            _T _hr=_h_res[i];
            _T _dr=_h_d_res[i];
            _T _f0=_h_a0[i];
            _T _f1=_h_a1[i];
            std::cout << "vector entry " << i << ' '
                      << _f0 << ' ' << _f1 << ' ' << _hr << ' ' << _dr
                      << ' ' << _hr - _dr << '\n';
        }
        // dump(_h_a0, "_a0");
        // dump(_h_a1, "_a1");
        // dump(_h_d_res, "device result: ");
        // dump(_h_res, "host result: ");
        std::cout << "FAILED: " << msg <<std::endl;
        std::cout << std::scientific
                  << std::setprecision(10);
    }
#endif
    return res;
}

template <typename _T>
bool
ocl::test::ops_base<_T>::
check_res(const std::string& msg, const _T& max_rel_tol)
{
#if USE_DEVICE_COMPARE>0
    // copy host results to device
    _h_res_d.copy_from_host(&_h_res[0]);
    // compare on device and make the result buffer compatible with
    // the results of lvec/vec comparisons
    dvec<_T> t=abs((_res - _h_res_d)/(_T(0.5)*(_res + _h_res_d)));
    _cmp_res = select((((_res != _res) & (_h_res_d != _h_res_d)) |
                       (_res == _h_res_d)|
                       (t < max_rel_tol)),
                      -1, 0);
#if 0
    // compare on device and make the result buffer compatible with
    // the results of lvec/vec comparisons
    _cmp_res = select(((_res == _h_res_d) |
                       ((_res != _res) & (_h_res_d != _h_res_d))),
                      -1, 0);
#endif
    bool res=all_of(_cmp_res);
    if (res==false) {
        _res.copy_to_host(&_h_d_res[0]);
        // copy back comparison result
        _cmp_res.copy_to_host(&_h_cmp_res[0]);
        // dump(_h_cmp_res, "cmp res:");
        std::cout << "\nFAILED: " << msg <<std::endl;
        std::cout << std::scientific
                  << std::setprecision(10);
        // std::cout << std::hexfloat;
        for (std::size_t i=0; i<_h_cmp_res.size(); ++i) {
            if (_h_cmp_res[i] != 0)
                continue;
            _T _hr=_h_res[i];
            _T _dr=_h_d_res[i];
            _T _f0=_h_a0[i];
            _T _f1=_h_a1[i];
            _T _rd= (_hr - _dr)/(_T(0.5)*(_hr + _dr));
            _T max_rd=0x1.0p-24*8;
            std::cout << "vector entry " << i << ' '
                      << _f0 << ' ' << _f1 << ' ' << _hr << ' ' << _dr
                      << ' ' << _rd << ' ' << max_rd << '\n';
        }
        // dump(_h_a0, "_a0");
        // dump(_h_a1, "_a1");
        // dump(_h_d_res, "device result: ");
        // dump(_h_res, "host result: ");
        std::cout << "FAILED: " << msg <<std::endl;
        std::cout << std::scientific
                  << std::setprecision(10);
    }
#else
#error "unimplemented"
#endif
    return res;
}
// local variables:
// mode: c++
// end:
#endif
