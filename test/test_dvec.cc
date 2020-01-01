#include <cftal/vec.h>
#include <ocl/ocl.h>
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
    namespace test {
        template <typename _V0, typename _V1>
        struct dump {
            const _V0& _v0;
            const _V1& _v1;
            dump(const _V0& v0, const _V1& v1) : _v0(v0), _v1(v1) {}
        };
        template <typename _V0, typename _V1>
        dump<_V0, _V1>
        dump_from(const _V0& v0, const _V1& v1) {
            return dump<_V0, _V1>(v0, v1);
        }

        template <typename _V0, typename _V1>
        std::ostream&
        operator<<(std::ostream& s, const dump<_V0, _V1>& d) {
            using e0_t = typename _V0::value_type;
            using e1_t = typename _V1::value_type;
            std::vector<e0_t> v0(d._v0);
            std::vector<e1_t> v1(d._v1);
            size_t nm=std::min(v0.size(), v1.size());
            for (size_t i=0; i<nm; ++i) {
                s << v0[i] << ' ' << v1[i] << '\n';
            }
            return s;
        }
    }
}

// using namespace ocl;
template <class _T>
_T
test_func(const _T& a, const _T& b)
{
    // return _T( (2.0 + a + b) / (a * b)  + (a + a * b ) - a);
    return 42.0 + (a+b)/(a-b) - 6.0 + (a+b)*(a-b) + 2.0;
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
        using ftype = float;
        // using itype = int64_t;
        // using v8fXX = cftal::vec<ftype, 8>;

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
