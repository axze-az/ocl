#include <ocl/ocl.h>
#include <ocl/random.h>
#include <ocl/test/tools.h>
#include <cftal/lvec.h>
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

    class rnd_histogram {
        float _min;
        float _max;
        float _rec_interval;
        std::uint32_t _n;
        dvec<std::uint32_t> _val;
    public:
        rnd_histogram(const float& min_val, const float& max_val,
                      std::uint32_t n,
                      be::data_ptr p=be::data::instance())
            : _min(min_val), _max(max_val),
              _rec_interval(float(1)/(_max - _min)),
              _n(n),
              _val(p, _n+2, 0) {}
        void
        insert(const dvec<float>& v);
        std::vector<std::uint32_t>
        values() const { return std::vector<std::uint32_t>(_val); }
        const float& min_val() const { return _min; }
        const float& max_val() const { return _max; }
        const std::uint32_t& n() const { return _n; }
    };

    std::ostream& operator<<(std::ostream& s,
                             const rnd_histogram& d);
}

void
ocl::rnd_histogram::insert(const dvec<float>& v)
{
    const char* kname="update_histogram_float";
    const char* ksrc=
        "void\n"
        "update_histogram_float(ulong n,\n"
        "                       __global uint* h,\n"
        "                       int entries,\n"
        "                       float min_val,\n"
        "                       float max_val,\n"
        "                       float rec_interval,\n"
        "                       __global const float* s)\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        float v=s[gid];\n"
        "        float offset = (v - min_val) * rec_interval * entries;\n"
        "        uint o= offset;\n"
        "        if (v > max_val) {\n"
        "            o=entries + 1;\n"
        "        }\n"
        "        if (v < min_val) {\n"
        "            o=entries;\n"
        "        }\n"
        "        atomic_add(h+o, 1);\n"
        "    }\n"
        "}\n";
    auto ck=custom_kernel<int>(kname, ksrc,
                               _val,
                               n(),
                               min_val(), max_val(), _rec_interval,
                               v);
    execute_custom(ck, v.size(), _val.backend_data());
}

std::ostream&
ocl::operator<<(std::ostream& s, const rnd_histogram& d)
{
    const float& min_v= d.min_val();
    const float& max_v= d.max_val();
    const float interval= max_v - min_v;

    auto dd=d.values();
    std::size_t _N=d.n();
    std::uint64_t n=0;
    for (auto b= std::cbegin(dd), e=std::cend(dd); b!=e; ++b) {
        n+= *b;
    }
    const double rn=1.0/n;

    for (std::size_t i=0; i<_N; ++i) {
        float v= min_v + (i * interval)/_N;
        double rt= dd[i] * rn;
        s << std::setprecision(6) << std::fixed;
        s << std::setw(8) << v << ' '
          << std::setw(8) << dd[i]
          << ' ' << rt
          << std::endl;
    }
    std::cout << "below: " << dd[_N] << std::endl;
    std::cout << "above: " << dd[_N+1] << std::endl;
    return s;
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
#if 1
    if (v > _max) {
        std::ostringstream e;
        e << "invalid entry " << v
          << " gt " << _max;
        std::cout << e.str() << std::endl;
        // throw std::runtime_error(e.str());
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
        const unsigned _N = 32*1024*1024;
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
#if 0
        std::vector<std::uint64_t> gid(_N, 0ull);
        for (std::size_t i=0; i<gid.size(); ++i)
            gid[i] = i;
        dvec<std::uint64_t> dg=gid;
        // ocl::rand48 t;
        ocl::srand t;
        t.seed(dg);
#else
        ocl::rand t(_N);
#endif
        // ocl::rnd_distribution<float, 40> dst(0, 1.0);
        ocl::rnd_histogram hdst(0, 1.0f, 40);
        dvec<float> f;
        cftal::lvec<float > fh(_N);
        for (int k=0; k<72; ++k) {
            for (int i=0; i<8; ++i) {
                f=t.nextf();
                hdst.insert(f);
#if 0
                f.copy_to_host(&fh[0]);
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
                // test::dump(f, "result vec");
            }
            std::cout << '.' << std::flush;
        }
#endif

        // std::cout << std::endl << dst << std::endl;
        std::cout << std::endl << hdst << std::endl;
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
