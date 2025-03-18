#include <ocl/ocl.h>
#include <ocl/random.h>
#include <ocl/test/tools.h>
#include <cftal/vsvec.h>
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
        // values low
        dvec<std::uint32_t> _val;
        // values high
        dvec<std::uint32_t> _val_hi;
    public:
        rnd_histogram(const float& min_val, const float& max_val,
                      std::uint32_t n,
                      be::data_ptr p=be::data::instance())
            : _min(min_val), _max(max_val),
              _rec_interval(float(1)/(_max - _min)),
              _n(n),
              _val(p, 0u, n+2),
              _val_hi(p, 0u, n+2) {
        }
        void
        insert(const dvec<float>& v);
        std::vector<std::uint64_t>
        values() const;
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
        "                       __global uint* h_hi,\n"
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
        "        o = (v > max_val) ? entries + 1 : o;\n"
        "        o = (v < min_val) ? entries : o;\n"
        "        uint old=atomic_add((h+o), 1);\n"
        "        uint inc_hi= select(0, 1, old == ~0);\n"
        "        atomic_add((h_hi+o), inc_hi);\n"
        "    }\n"
        "}\n";
    auto ck=custom_kernel<int>(kname, ksrc,
                               _val,
                               _val_hi,
                               n(),
                               min_val(), max_val(), _rec_interval,
                               v);
    execute_custom(ck, v.size(), _val.backend_data());
}

std::vector<std::uint64_t>
ocl::rnd_histogram::values()
    const
{
    dvec<uint64_t> r= ((cvt<dvec<uint64_t> >(_val_hi)) << 32)
        +(cvt<dvec<uint64_t> >(_val));
    return std::vector<uint64_t>(r);
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
          << std::setw(10) << dd[i]
          << std::setprecision(10)
          << ' ' << rt
          << std::endl;
    }
    std::cout << "below: " << dd[_N] << std::endl;
    std::cout << "above: " << dd[_N+1] << std::endl;
    return s;
}

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
    if (v > _max) {
        std::ostringstream e;
        e << "invalid entry " << v
          << " gt " << _max;
        std::cout << e.str() << std::endl;
        // throw std::runtime_error(e.str());
    }
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
#define __USE_HOST 0
    try {

        // const int _N=1000000;
        const unsigned _N = 8*1024*1024;
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
#if 1
        ocl::rand48 t(_N);
#else
        ocl::rand t(_N);
#endif
        constexpr const int _M=10;
#if __USE_HOST>0
        ocl::rnd_distribution<float, _M> dst(0, 1.0);
#else
        ocl::rnd_histogram hdst(0.0f, 1.0f, _M);
#endif
        dvec<float> f;
        cftal::vsvec<float > fh(0.0f, _N);
        for (int l=0; l<4; ++l) {
            for (int k=0; k<72; ++k) {
                for (int i=0; i<16; ++i) {
                    f=t.nextf();
#if __USE_HOST==0
                    hdst.insert(f);
#else
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
            std::cout << '\n';
        }
#endif

#if __USE_HOST>0
        std::cout << dst << std::endl;
#else
        std::cout << hdst << std::endl;
#endif
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught ocl::be::error: " << e.what()
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
