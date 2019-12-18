#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <cmath>

namespace ocl {

#if 0
    template <class _T>
    std::size_t eval_size(const std::vector<_T>& v) {
        return v.size();
    }
#endif
}

// using namespace ocl;
template <class _T>
_T
test_func(const _T& a, const _T& b)
{
    // return _T( (2.0 + a + b) / (a * b)  + (a + a * b ) - a);

    return _T((2.0f + a + b) / (a * b)  + (a + a * b ) - a) *
        ((6.0f + a + b) / (a * b)  + (a + a * b ) - a);
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

        using namespace ocl;

        using ftype = double;
        using itype = int64_t;
        using v8fXX = cftal::vec<ftype, 8>;

        // const unsigned BEIGNET_MAX_BUFFER_SIZE=16384*4096;
        const unsigned GALLIUM_MAX_BUFFER_SIZE=2048*4096;
        const unsigned SIZE=GALLIUM_MAX_BUFFER_SIZE;
        std::cout << "using buffers of "
                  << double(SIZE*sizeof(ftype))/(1024*1024)
                  << "MiB\n";
        ftype a(2.0f), b(3.0f);

        vector<ftype> v0(SIZE, a);
        // std::vector<ftype> vha(SIZE, a);
        vector<ftype> va(v0);
        std::vector<ftype> vhb(SIZE, 3.0f);
        vector<ftype> vb(vhb);
        vector<ftype> vc= test_func(va, test_func(va, vb));
        vector<ftype> vd(test_func(va, vb, vc));
        vector<ftype> vd2= test_func(va, vb, vc);

        vector<int32_t> tgt= vc < vd;

        ftype c= test_func(a, test_func(a, b));
        ftype d= test_func(a, b, c);

        std::vector<ftype> res(vd);

        vector<v8fXX> vva(SIZE/8, a);
        vector<v8fXX> vvb(SIZE/8, b);
        vector<v8fXX> vvc(SIZE/8, c);
        vector<v8fXX> vres(test_func(vva, vvb, vvc));

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

        vector<ftype> cvt_dst = -vd2;
        vector<ftype> abs_dst = abs(cvt_dst);
        vector<itype> iv= ~(cvt_to<vector<itype> >(cvt_dst)*2);
        vector<ftype> ivf= as<vector<ftype> >(iv);

        impl::be_data::instance()->clear();
    }
    catch (const ocl::impl::error& e) {
        std::cout << "caught exception: " << e.what()
                  << '\n'
                  << ocl::impl::err2str(e)
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: " << e.what()
                  << std::endl;
    }
    return 0;
}
