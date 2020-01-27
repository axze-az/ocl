#include <cftal/vec.h>
#include <ocl/ocl.h>
#include <ocl/random.h>
#include <ocl/test/tools.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <cmath>
#include <random>

#define USE_DEVICE_COMPARE 1

namespace ocl {

    namespace test {

        template <typename _T>
        class ops {
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
        public:
            ops(size_t n);
            bool perform();
        private:
            bool check_res(const std::string& msg);
        };

        template <typename _T>
        bool test_ops(dvec<_T>& res,
                      const dvec<_T>& s0, const dvec<_T>& s1);

    }
}

template <typename _T>
ocl::test::ops<_T>::ops(size_t n)
    : _res(n), _a0(n), _a1(n),
      _h_res_d(n),
      _cmp_res(n),
      _h_cmp_res(n),
      _h_d_res(n),
      _h_res(n), _h_a0(n), _h_a1(n)
{
#if 1
    const _T min_val=-2.0;
    const _T max_val=2.0;
    rand rnd(n);
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
ocl::test::ops<_T>::check_res(const std::string& msg)
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
        std::cout << std::hexfloat;
        for (std::size_t i=0; i<_h_cmp_res.size(); ++i) {
            if (_h_cmp_res[i] != 0)
                continue;
            _T _hr=_h_res[i];
            _T _dr=_h_d_res[i];
            _T _f0=_h_a0[i];
            _T _f1=_h_a1[i];
            std::cout << _f0 << ' ' << _f1 << ' ' << _hr << ' ' << _dr
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
            std::cout << _f0 << ' ' << _f1 << ' ' << _hr << ' ' << _dr
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
#if 0
    static const char* fname="__divide";
    static const char* fbody=
        "float __divide(float a, float b)\n"
        "{\n"
        // "    float q0 = a/b;\n"
        // "    float q1 = fma(-q0, b, a)/b;\n"
        // "    return q0 + q1;\n"
        "    float xn=1.0f/b;\n"
        "    xn = fma(xn, fma(xn, -b, 1.0f), xn);\n"
        "    float yn= a*xn;\n"
        "    yn= fma(xn, fma(yn, -b, a), yn);\n"
<        "    return yn;\n"
        "}\n";
    _res=custom_func<float>(fname, fbody, _a0, _a1);
#else
    _res = _a0 / _a1;
#endif
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
#if 0
    // x_{n+1} = 1/2(x_n + a/x_n)
    //         = 0.5 x_n + 0.5*a/x_n
    //         = 1.0 x_n - 0.5 x_n + 0.5*a/x_n
    //         = 1.0 x_n + (0.5*a/x_n - 0.5 x_n)
    //         = 1.0 x_n + 0.5 * (a/x_n - x_n)
    //         = a*r_n + 0.5 * (a/(a*r_n) - a*r_rn)
    //         = a*r_n + 0.5 * (1/(r_n) - a*r_rn)
    static const char* fname="__sqrt";
    static const char* fbody=
        "float __rcp(float b)\n"
        "{\n"
        "    float xn=1.0f/b;\n"
        "    xn = fma(xn, fma(xn, -b, 1.0f), xn);\n"
        "    // xn = fma(xn, fma(xn, -b, 1.0f), xn);\n"
        "    return xn;\n"
        "}\n"
        "\n"
        "float __divide(float a, float b)\n"
        "{\n"
        // "    float q0 = a/b;\n"
        // "    float q1 = fma(-q0, b, a)/b;\n"
        // "    return q0 + q1;\n"
        "    float xn=1.0f/b;\n"
        "    xn = fma(xn, fma(xn, -b, 1.0f), xn);\n"
        "    float yn= a*xn;\n"
        "    yn= fma(xn, fma(yn, -b, a), yn);\n"
        "    return yn;\n"
        "}\n"
        "\n"
        "float __sqrt(float a)\n"
        "{\n"
#if 1
        "    float r=rsqrt(a);\n"
        "    float rah, ral;\n"
        "    rah=a*r;\n"
        "    ral=fma(-a, r, rah);\n"
        "    float th= fma(r, rah, -1.0f);\n"
        "    th=fma(r, ral, th);\n"
        "    r= fma(-0.5f*r*a, th, r*a);\n"
        "    return r;\n"
#else
        "    float r=rsqrt(a);\n"
        "    float rah, ral;\n"
        "    rah=a*r;\n"
        "    ral=fma(-a, r, rah);\n"
        "    float th= fma(r, rah, -1.0f);\n"
        "    th=fma(r, ral, th);\n"
        "    r= fma(-0.5f*r, th, r);\n"
        "    return a*r;\n"
#endif
        "}\n";
    _res=custom_func<float>(fname, fbody, _a0);
#else
    _res = sqrt(_a0);
#endif
    _h_res = sqrt(_h_a0);
    rc &= check_res("sqrt");
    // _res = rsqrt(_a0);
    // _res = rsqrt(_h_a0);
    // rc &= check_res("rsqrt");
    return rc;
}

// using namespace ocl;
template <class _T>
_T
test_func(const _T& a, const _T& b)
{
    // return _T( (2.0 + a + b) / (a * b)  + (a + a * b ) - a);
    return 42.0f + (a+b)/(a-b) - 6.0f + (a+b)*(a-b) + 2.0f;
    // return _T((2.0f + a + b) / (a * b)  + (a + a * b ) - a) *
    //    ((6.0f + a + b) / (a * b)  + (a + a * b ) - a);
}

template <class _T>
_T
test_func(const _T& a, const _T& b, const _T& c)
{
    return _T((a+b *c) *c + 2.0f);
}

namespace {

    template <class _T>
    _T rel_error(const _T& a, const _T& b)
    {
        _T e((a -b ));
        e = e < _T(0) ? -e : e;
        _T m((a+b)*_T(0.5));
        if (m != _T(0)) {
            e /= m;
        }
        return e;
    }
}

int main()
{
    try {
        using namespace cftal;
        using namespace ocl;
        using namespace ocl::test;

        using rtype = float;
        // using itype = int64_t;
        // using v8fXX = cftal::vec<ftype, 8>;
        for (std::size_t i=4; i<16* 16384; ++i) {
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
#if 0
        using ftype = float;
        // const unsigned BEIGNET_MAX_BUFFER_SIZE=16384*4096;
        // const unsigned GALLIUM_MAX_BUFFER_SIZE=2048*4096;
        rtype a(rtype(2.0)), b(rtype(3.0));
        rtype c= test_func(a, b);
        for (std::size_t i=4; i<5; ++i) {
            if ((i & 0x7f) == 0x7f || i==1) {
                std::cout << "using buffers of "
                          <<  i*sizeof(rtype)
                          << " bytes\r" << std::flush;
            }
            dvec<ftype> va(i, a);
            dvec<ftype> vb(i, b);
            dvec<ftype> vc=test_func(va, vb);
            std::vector<ftype> vhc(vc);
            for (std::size_t j=0; j<i; ++j) {
                if (vhc[j] != c) {
                    std::cout << "error for elem "
                              << j << " of dvec<> of size" << i
                              << std::endl;
                    std::exit(3);
                }
            }
        }
#endif
        std::cout << "\ntest passed\n";
#if 0
        std::vector<ftype> vhb(SIZE, ftype(3.0));
        dvec<ftype> vb(vhb);
        dvec<ftype> vc= test_func(va, test_func(va, vb));
        dvec<ftype> vd(test_func(va, vb, vc));
        dvec<ftype> vd2= test_func(va, vb, vc);

        dvec<int32_t> tgt= vc < vd;

        ftype c= test_func(a, test_func(a, b));
        ftype d= test_func(a, b, c);

        std::vector<ftype> res(vd);

        dvec<v8fXX> vva(SIZE/8, a);
        dvec<v8fXX> vvb(SIZE/8, b);
        dvec<v8fXX> vvc(SIZE/8, c);
        dvec<v8fXX> vres(test_func(vva, vvb, vvc));

        if (SIZE <= 4096) {
            for (std::size_t i=0; i< res.size(); ++i) {
                std::cout << i << ' ' << res[i] << std::endl;
            }
        } else {
            for (std::size_t i=0; i< res.size(); ++i) {
                ftype e=rel_error(res[i], d);
                if (e > 1e-7) {
                    std::ostringstream m;
                    m << "res[" << i << " ]="
                      << std::setprecision(12)
                      << res[i] << " != " << d
                      << " e= " << e;
                    throw std::runtime_error(m.str());
                }
            }
        }
        std::cout << "scalar " << d << std::endl;

        dvec<ftype> cvt_dst = -vd2;
        dvec<ftype> abs_dst = abs(cvt_dst);
        dvec<itype> iv= ~(cvt_to<dvec<itype> >(cvt_dst)*2);
        dvec<ftype> ivf= as<dvec<ftype> >(iv);

        impl::be_data::instance()->clear();
#endif
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
