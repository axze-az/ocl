#include "ocl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <atomic>
#include <mutex>
#include <memory> // for shared_ptr
#include <random>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iterator>


namespace ocl {

    // template <class _T>
    class mrand48 {
        vector<std::uint64_t> _state;

        static const std::uint64_t A;
        static const std::uint32_t C;
        static const std::uint64_t M;
        static const std::uint64_t MM;
        static const float REC;
    public:
        inline
        vector<std::uint32_t>
        next() {
            vector<std::uint64_t> nv= ((_state * A + C) & MM);
            // _state = (_state * A + C) & MM;
            _state= nv;
            nv = nv / (1<<16);
            return cvt_to<vector<std::uint32_t> >(nv);
        }

        void 
        seed(const vector<uint64_t>& gid){
            // m_state = (gid * 65536) | 0x330E;
            _state = (7*gid) ^ 0x330E;
        }

        vector<float>
        nextf() {
            vector<float> r=max(
                min(cvt_to<vector<float> >(next()) * REC, 1.0f), 0.0f);
            return r;
        }
    };


    //
    template <typename _T, std::size_t _N>
    class rnd_distribution {
        std::uint32_t _val[_N];
        _T _min;
        _T _max;
        _T _rec_interval;
    public:
        rnd_distribution(const _T& min_val, const _T& max_val)
            : _min{min_val}, _max{max_val},
              _rec_interval{_T(1)/(_max - _min)} {}
        void insert(const _T& v);
        using const_iterator = const std::uint32_t*;
        const _T& min_val() const { return _min; }
        const _T& max_val() const { return _max; }
        const_iterator begin() const { return std::begin(_val); }
        const_iterator end() const { return std::end(_val); }
    };

    template <typename _T, std::size_t _N>
    std::ostream& operator<<(std::ostream& s,
                             const rnd_distribution<_T, _N>& d);

}

const std::uint64_t ocl::mrand48::A=0x5DEECE66Dul;
const std::uint32_t ocl::mrand48::C=0xBL;
const std::uint64_t ocl::mrand48::M=(1ULL<<48);
const std::uint64_t ocl::mrand48::MM=(M-1);
const float ocl::mrand48::REC= 1.0f/uint32_t(-1);


template <typename _T, std::size_t _N>
void
ocl::rnd_distribution<_T, _N>::insert(const _T& v)
{
    if (v < _min) {
        std::ostringstream e;
        e << "invalid entry " << v
          << " lt " << _min;
        throw std::runtime_error(e.str());
    }
#if 0
    if (v > _max) {
        std::ostringstream e;
        e << "invalid entry " << v
          << " gt " << _max;
        throw std::runtime_error(e.str());
    }
#endif
    _T offset = (v - _min) * _rec_interval * (_N);
    std::uint32_t o= offset;
    _val[o] +=1;
}

template <typename _T, std::size_t _N>
std::ostream& ocl::operator<<(std::ostream& s,
                              const rnd_distribution<_T, _N>& d)
{
    const _T& min_v= d.min_val();
    const _T& max_v= d.max_val();
    const _T interval= max_v - min_v;
    std::size_t i=0;
    for (const std::uint32_t* b= std::begin(d), *e=std::end(d);
         b != e; ++b, ++i) {
        _T v= min_v + (i * interval)/_N;
        s << std::setw(8) << v << ' '
          << std::setw(8) << *b
          << std::endl;
    }
    return s;
}


int main()
{
    try {

        const int _N=1000000;
        const float _R=1./_N;
        std::uniform_int_distribution<> dx(0, _N+1);
        std::mt19937 rnd;

        ocl::rnd_distribution<float, 20> dst(0, 1.0);
        for (long int i=0; i<10000000; ++i) {
            float r= _R * dx(rnd);
            dst.insert(r);
        }
#if 0        
        using namespace ocl;

        const unsigned _N = 1024;
        
        std::vector<std::uint64_t> gid(_N, 0ull);
        for (std::size_t i=0; i<gid.size(); ++i)
            gid[i] = i;
        vector<std::uint64_t> dg=gid;
        ocl::mrand48 t;
        t.seed(dg);

        ocl::rnd_distribution<float, 20> dst(0, 1.0);

        for (int i=0; i<50; ++i) {
            vector<float> f= t.nextf();
            std::cout << "iteration " << i << std::endl;
            std::vector<float> fh(f);
            for (std::size_t j=0; j<fh.size(); ++j)
                dst.insert(fh[j]);
            if ((i & (32-1)) == (32-1)) {
                for (std::size_t j=0; j<fh.size(); ++j) {
                    std::cout << std::setw(2) << j
                              << ": " << fh[j] << std::endl;
                }
            }
        }
#endif
        std::cout << dst << std::endl;
        
    }
    catch (const ocl::impl::error& e) {
        std::cout << "caught ocl::impl::error: " << e.what()
                  << '\n'
                  << ocl::impl::err2str(e)
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught runtime_error: " << e.what()
                  << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "caught std::exception: " << e.what()
                  << std::endl;
    }
    return 0;
}
