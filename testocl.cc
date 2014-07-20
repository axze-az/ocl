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

        template <class _T>
        std::size_t eval_size(const std::vector<_T>& v) {
                return v.size();
        }


}




// using namespace ocl;

template <class _T>
_T
test_func(const _T& a, const _T& b)
{
        // return _T( (2.0 + a + b) / (a * b)  + (a + a * b ) - a);

        return _T((2.0 + a + b) / (a * b)  + (a + a * b ) - a) *
                ((6.0 + a + b) / (a * b)  + (a + a * b ) - a);
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

                using cftal::vec::v8f32;

                const unsigned BEIGNET_MAX_BUFFER_SIZE=16384*4096;
                const unsigned GALLIUM_MAX_BUFFER_SIZE=2048*4096;
                const unsigned SIZE=1024*4096;
                std::cout << "using buffers of "
                          << double(SIZE*sizeof(float))/(1024*1024)
                          << "MiB\n";
                float a(2.0f), b(3.0f);

                vector<float> v0(SIZE, a);
                // std::vector<float> vha(SIZE, a);
                vector<float> va(v0);
                std::vector<float> vhb(SIZE, 3.0f);
                vector<float> vb(vhb);
                vector<float> vc= test_func(va, vb);
                vector<float> vd= test_func(va, vb, vc);
                vector<float> vd2= test_func(va, vb, vc);

                float c= test_func(a, b);
                float d= test_func(a, b, c);

                std::vector<float> res(vd);

                vector<v8f32> vva(SIZE/8, a);
                vector<v8f32> vvb(SIZE/8, b);
                vector<v8f32> vvc(SIZE/8, c);
                vector<v8f32> vres(test_func(vva, vvb, vvc));

                if (SIZE <= 4096) {
                        for (std::size_t i=0; i< res.size(); ++i) {
                                std::cout << i << ' ' << res[i] << std::endl;
                        }
                } else {
                        for (std::size_t i=0; i< res.size(); ++i) {
                                float e=rel_error(res[i], d);
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

        }
        catch (const std::runtime_error& e) {
                std::cout << "caught exception: " << e.what()
                          << std::endl;
        }
        catch (const ocl::impl::error& e) {
                std::cout << "caught exception: " << e.what()
                          << '\n'
                          << ocl::impl::err2str(e)
                          << std::endl;
        }
#if 1
        std::vector<cl::Device> v(ocl::impl::devices());
        std::cout << v.size() << std::endl;
        for (std::size_t i = 0; i< v.size(); ++i) {
                std::cout << ocl::impl::device_info(v[i]);
        }
        ocl::impl::device dd(ocl::impl::default_device());
        std::cout << "selected device: \n";
        std::cout << ocl::impl::device_info(dd);

        try {
                const ocl::impl::device& bed =
                        ocl::impl::be_data::instance()->d();
                std::cout << "\nselected backend device: \n";
                std::cout << ocl::impl::device_info(bed);
        }
        catch (const ocl::impl::error& e) {
                std::cout << "caught exception: " << e.what()
                          << '\n'
                          << ocl::impl::err2str(e)
                          << std::endl;
        }
#endif
        return 0;
}
