#include <ocl/ocl.h>
#include <ocl/random.h>
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

#if 0
    // POSIX random
    static unsigned long next = 1;

    /* RAND_MAX assumed to be 32767 */
    int myrand(void)
    {
        next = next * 1103515245 + 12345;
        return((unsigned)(next/65536) % 32768);
    }

    void mysrand(unsigned int seed)
    {
        next = seed;
    }
#endif
#if 0
    class srand {
        dvec<std::uint32_t> _next;
    public:
        srand() : _next() {}
        srand(const dvec<uint32_t>& gid) : _next{gid} {}
        void
        seed(const dvec<uint64_t>& gid){
            _next = cvt_to<dvec<uint32_t> >(gid);
        }
        void
        seed(const dvec<uint32_t>& gid){
            _next = gid;
        }
        // dvec<uint32_t>
        auto
        next() {
            _next = (_next * 1103515245 + 12345);
            return (_next>>16) & 0x7fff;
        }
        auto
        nextf() {
            return (cvt_to<dvec<float>>(next()) * (1.0f/32768.f));
        }
    };
#endif

    // template <class _T>
    class rand48 {
        dvec<std::uint64_t> _state;

        static const std::uint64_t A;
        static const std::uint64_t C;
        static const std::uint64_t M;
        static const std::uint64_t MM;
        static const float REC;

        inline
        void
        next() {
            _state= ((_state * A + C) & MM);
        }
    public:
        // returns non negative numbers between 0 and 2^31
        inline
        dvec<std::int32_t>
        lrand48() {
            next();
            return cvt<dvec<std::int32_t> >(_state) & (0x7fffffff);
        }

        // returns non negative numbers between -2^31 and 2^31
        inline
        dvec<std::int32_t>
        mrand48() {
            next();
            return cvt<dvec<std::int32_t> >(_state);
        }

        // returns floating point variables in interval [0, 1.0)
        inline
        dvec<float>
        drand48() {
            next();
            return cvt<dvec<float> >(_state);
        }

        void
        seed(const dvec<uint64_t>& gid){
            _state = ((gid * 65536) | 0x330E) & MM;
        }

        dvec<float>
        nextf() {
            dvec<float> t=cvt<dvec<float> >(drand48()) * REC;
            dvec<float> r=max(min(t, 1.0f), 0.0f);
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
              _rec_interval{_T(1)/(_max - _min)} {
            std::fill(std::begin(_val), std::end(_val), 0u);
        }
        void insert(const _T& v);
        using const_iterator = const std::uint32_t*;
        const _T& min_val() const { return _min; }
        const _T& max_val() const { return _max; }
        const_iterator begin() const { return std::cbegin(_val); }
        const_iterator end() const { return std::cend(_val); }
    };

    template <typename _T, std::size_t _N>
    std::ostream& operator<<(std::ostream& s,
                             const rnd_distribution<_T, _N>& d);

}

const std::uint64_t ocl::rand48::A=0x5DEECE66Dul;
const std::uint64_t ocl::rand48::C=0xBL;
const std::uint64_t ocl::rand48::M=(1ULL<<48);
const std::uint64_t ocl::rand48::MM=(M-1);
const float ocl::rand48::REC= 1.0f/uint32_t(-1);


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

    std::uint64_t n=0;
    for (auto b= std::cbegin(d), e=std::cend(d); b!=e; ++b) {
        n+= *b;
    }
    const double rn=1.0/n;

    std::size_t i=0;
    for (auto b= std::cbegin(d), e=std::cend(d); b != e; ++b, ++i) {
        _T v= min_v + (i * interval)/_N;
        double rt= *b * rn;
        s << std::setprecision(6) << std::fixed;
        s << std::setw(8) << v << ' '
          << std::setw(8) << *b
          << ' ' << rt
          << std::endl;
    }
    return s;
}


int main()
{
    try {

        // const int _N=1000000;
        const unsigned _N = 256*1024;
#if 0
        const float _R=1.f/_N;
        std::uniform_int_distribution<> dx(0, _N+1);
        std::mt19937 rnd;


        ocl::rnd_distribution<float, 20> dst(0, 1.0);
        for (long int i=0; i<10000000; ++i) {
            float r= _R * dx(rnd);
            dst.insert(r);
        }
#else
        using namespace ocl;

        std::vector<std::uint64_t> gid(_N, 0ull);
        for (std::size_t i=0; i<gid.size(); ++i)
            gid[i] = i;
        dvec<std::uint64_t> dg=gid;
        //ocl::rand48 t;
        ocl::srand t;
        t.seed(dg);

        ocl::rnd_distribution<float, 25> dst(0, 1.0);
        dvec<float> f;

        for (int i=0; i<20000; ++i) {
            f=t.nextf();
            // if ((i & 0xff)==0xff) {
            //    std::cout << "iteration " << i << std::endl;
            // }
            std::vector<float> fh(f);
#if 1
            for (std::size_t j=0; j<fh.size(); ++j) {
                try {
                    dst.insert(fh[j]);
                }
                catch (...) {
                    for (std::size_t j=0; j<fh.size(); ++j) {
                        std::cout << std::setw(2) << j
                                  << ": " << fh[j] << std::endl;
                    }
                    throw;
                }

            }
#endif

#if 1
            if ((i & (256-1)) == (256-1)) {
#if 1
                std::cout << '.' << std::flush;
#else
                for (std::size_t j=0; j<fh.size(); ++j) {
                    std::cout << std::setw(2) << j
                              << ": " << fh[j] << std::endl;
                }
#endif
            }
#endif
        }
#endif

        std::cout << std::endl << dst << std::endl;
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught ocl::impl::error: " << e.what()
                  << '\n'
                  << e.error_string()
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
