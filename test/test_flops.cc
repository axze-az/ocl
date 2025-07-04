//
// Copyright (C) 2010-2025 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#include <cftal/vec.h>
#include <ocl/ocl.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

#define TEST_PEAK_SP_ONLY 0

namespace ocl {

    namespace impl {
        __cf_body
        gen_peak_flops(const std::string_view& tname,
                       size_t n,
                       bool use_fma, bool use_mad,
                       const std::string_view& literal_suffix);

        template <typename _T>
        _T
        peak_flops(_T x, size_t n);
    }

    template <typename _T>
    auto
    peak_flops(const dvec<_T>& x, size_t n);

    namespace test {
        template <typename _T>
        float horner_gflops(be::data_ptr bedp);

        template <typename _T>
        float peak_gflops(be::data_ptr bedp);


        void
        test_gflops(int argc, char** argv);


    }
}

ocl::impl::__cf_body
ocl::impl::
gen_peak_flops(const std::string_view& tname,
               size_t n,
               bool use_fma, bool use_mad,
               const std::string_view& literal_suffix)
{
    std::ostringstream s;
    s << "peak_f_" << n << '_' << tname;
    if (use_fma) {
        s << "_fma";
    } else if (use_mad) {
        s << "_mad";
    }
    const std::string hname=s.str();
    s.str("");
    s << tname << " " << hname << "("
      << tname << " x)\n"
        "{\n";
    s << "    " << tname << " c=0x1.0p-124" << literal_suffix << ";\n"
      << "    const " << tname << " fac=-3.0" << literal_suffix << ";\n"
      << "    " << tname << " r=c;\n";
    for (size_t i=0; i<n; ++i) {
        if (use_fma) {
            s << "    r=fma(x, r, c);\n";
            if (i < n-1)
                s << "    c=fma(fac, c, c);\n";
        } else if (use_mad) {
            s << "    r=mad(x, r, c);\n";
            if (i < n-1)
                s << "    c=mad(fac, c, c);\n";
        } else {
            s << "    r=x*r+c;\n";
            if (i < n-1)
                s << "    c=fac*c+c;\n";
        }
    }
    s << "    return r;\n"
         "}\n";
    return impl::__cf_body(hname, s.str());
}

template <typename _T>
_T
ocl::impl::peak_flops(_T x, size_t n)
{
    const _T fac=_T(-3.0);
    _T c=_T(0x1.0p-124);
    _T r=c;
    for (size_t i=0; i<n; ++i) {
        r = x*r + c;
        if (i < n-1)
            c = fac* c + c;
    }
    return r;
}

template <typename _T>
auto
ocl::peak_flops(const dvec<_T>& x, size_t n)
{
    const auto tname= be::type_2_name<_T>::v();
    auto use_fma_mad=impl::horner_use_fma_mad(x);
    std::string_view lit_suffix="";
    if (tname=="float")
        lit_suffix="f";
    auto hb=impl::gen_peak_flops(tname, n,
                                 use_fma_mad.first, use_fma_mad.second,
                                 lit_suffix);
    return custom_func<_T>(hb.name(), hb.body(), x);
}

template <typename _T>
float
ocl::test::
horner_gflops(be::data_ptr bedp)
{
    constexpr const size_t COEFF_COUNT=256+1;
    std::cout << "testing\n"  << bedp->dcq().d().name() << '\n'
              << be::type_2_name<_T>::v() << " gflops using a polynomial with "
              << COEFF_COUNT << " coefficients\n";
    _T coeffs[COEFF_COUNT];
    _T ci=_T(1);
    for (size_t i=0; i<COEFF_COUNT; ++i) {
        coeffs[i]= (i & 1)==1 ? -ci : ci;
        ci *= _T(0.875);
    }
    constexpr const size_t elem_count=(128*1024*1024ULL)/sizeof(_T);
    constexpr const size_t _N=48;
    constexpr const size_t _WARMUP=4;
    float gflops=0.0f;
    std::cout << std::fixed << std::setprecision(1);
    try {
        for (size_t i=0; i<_N+_WARMUP; ++i) {
            dvec<_T> v_src(bedp, _T(0.25), elem_count);
            dvec<_T> v_dst(v_src.backend_data(), elem_count);
            auto start = std::chrono::steady_clock::now();
            v_dst=horner(v_src, coeffs);
            auto end = std::chrono::steady_clock::now();
            auto ns_elapsed=(end - start).count();
            _T r=hsum(v_dst);
            if (r==0) {
                std::cerr << "probably something wrong here\n";
            }
            if (i >= _WARMUP) {
                float gflops_i=
                    (elem_count*(COEFF_COUNT-1)*2)/float(ns_elapsed);
                std::cout << std::setw(12) << gflops_i;
                if (((i-_WARMUP) % 6)== 5) {
                    std::cout << '\n';
                } else {
                    std::cout << ' ' << std::flush;
                }
                gflops += gflops_i;
            }
        }
        if ((_N % 6)!= 0) {
            std::cout << '\n';
        }
        gflops *= 1.0f/float(_N);
        std::cout << "mean: " << gflops << std::endl;
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: ocl::be::error: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: runtime error: " << e.what()
                  << std::endl;
    }
    return gflops;
}

template <typename _T>
float
ocl::test::
peak_gflops(be::data_ptr bedp)
{
    constexpr const size_t COUNT=128;
    std::cout << "testing\n"  << bedp->dcq().d().name() << '\n'
              << be::type_2_name<_T>::v() << " gflops using a polynomial with "
              << COUNT << " calculated coefficients\n";
    constexpr const size_t elem_count=(128*1024*1024ULL)/sizeof(_T);
    constexpr const size_t _N=48;
    constexpr const size_t _WARMUP=4;
    float gflops=0.0f;
    std::cout << std::fixed << std::setprecision(1);
    try {
        const _T x=_T(0.25);
#if 0
        _T res=impl::peak_flops(x, COUNT);
        std::cout << "res= "
                  // << std::scientific << std::setprecision(18)
                  << res << std::endl;
#endif
        for (size_t i=0; i<_N+_WARMUP; ++i) {
            dvec<_T> v_src(bedp, x, elem_count);
            dvec<_T> v_dst(v_src.backend_data(), elem_count);
            auto start = std::chrono::steady_clock::now();
            v_dst=peak_flops(v_src, COUNT);
            auto end = std::chrono::steady_clock::now();
            auto ns_elapsed=(end - start).count();
            _T r=hsum(v_dst);
            if (r==0) {
                std::cerr << "probably something wrong here\n";
            }
            if (i >= _WARMUP) {
                float gflops_i=
                    (elem_count*(COUNT*4-2))/float(ns_elapsed);
                std::cout << std::setw(12) << gflops_i;
                if (((i-_WARMUP) % 6)==5) {
                    std::cout << '\n';
                } else {
                    std::cout << ' '  << std::flush;
                }
                gflops += gflops_i;
            }
        }
        if ((_N % 6)!= 0) {
            std::cout << '\n';
        }
        gflops *= 1.0f/float(_N);
        std::cout << "mean: " << gflops << std::endl;
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: ocl::be::error: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: runtime error: " << e.what()
                  << std::endl;
    }
    return gflops;
}


void
ocl::test::test_gflops(int argc, char** argv)
{
    int device=-1;
    using std::string_view;
    int err=0;
    for (int i=1; i<argc; ++i) {
        string_view argi(argv[i]);
        string_view::size_type eq_pos=argi.find('=');
        string_view ai(argi.substr(0, eq_pos));
        if (ai == "--device") {
            if (argi.length()>eq_pos) {
                string_view pl=argi.substr(eq_pos+1);
                std::istringstream is(std::string(pl.data(), pl.length()));
                int32_t d=0;
                is >> d;
                if (is.fail() || !is.eof()) {
                    if (pl.size()==0) {
                        std::cerr << "device number is missing\n";
                    } else {
                        std::cerr << "invalid device number " << pl << '\n';
                    }
                    ++err;
                } else {
                    device=d;
                }
            } else {
                std::cerr << "device number is missing\n";
                ++err;
            }
        } else {
            std::cerr << "invalid argument " << argi << '\n';
            ++err;
        }
    }
    if (err) {
        std::cout << "usage: "  << argv[0]  << " [--device=X]\n";
        return;
    }
    try {
        std::vector<be::device> v(be::devices());
        if (device >= int(v.size())) {
            std::cerr << "device number " << device << "is undefined:\n";
            for (size_t i=0; i<v.size(); ++i) {
                std::cerr << i << ": " << v[i].name() << '\n';
            }
            return;
        }
        if (device > -1) {
            auto bedp=be::data::create(v[device]);
#if TEST_PEAK_SP_ONLY==0
            test::horner_gflops<float>(bedp);
            test::horner_gflops<double>(bedp);
#endif
            test::peak_gflops<float>(bedp);
#if TEST_PEAK_SP_ONLY==0
            test::peak_gflops<double>(bedp);
#endif
        }  else {
            for (auto& d : v) {
                auto bedp=be::data::create(d);
#if TEST_PEAK_SP_ONLY==0
                test::horner_gflops<float>(bedp);
                test::horner_gflops<double>(bedp);
#endif
                test::peak_gflops<float>(bedp);
#if TEST_PEAK_SP_ONLY==0
                test::peak_gflops<double>(bedp);
#endif
            }
        }
    }
    catch (const be::error& e) {
        std::cout << "caught exception: be::error: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: runtime error: " << e.what()
                  << std::endl;
    }
}

#if 1

int main(int argc, char** argv)
{
    ocl::test::test_gflops(argc, argv);
    return 0;
}

#else

int main()
{
    // [-0.292893230915069580078125, 0.4142135679721832275390625] : | p - f | <= 2^-31.90625
    // coefficients for log generated by sollya
    // x^1 : +0x8p-3f
    constexpr
    const float log_c1=+1.0000000000e+00f;
    // x^2 : -0x8p-4f
    constexpr
    const float log_c2=-5.0000000000e-01f;
    // x^3 : +0xa.aaaa3p-5f
    constexpr
    const float log_c3=+3.3333310485e-01f;
    // x^4 : -0x8.00002p-5f
    constexpr
    const float log_c4=-2.5000005960e-01f;
    // x^5 : +0xc.cd2a4p-6f
    constexpr
    const float log_c5=+2.0002228022e-01f;
    // x^6 : -0xa.aaebep-6f
    constexpr
    const float log_c6=-1.6668221354e-01f;
    // x^7 : +0x9.1974bp-6f
    constexpr
    const float log_c7=+1.4217869937e-01f;
    // x^8 : -0xf.dfab7p-7f
    constexpr
    const float log_c8=-1.2401335686e-01f;
    // x^9 : +0xf.39c7cp-7f
    constexpr
    const float log_c9=+1.1895081401e-01f;
    // x^10 : -0xe.fa013p-7f
    constexpr
    const float log_c10=-1.1700453609e-01f;
    // x^11 : +0x8.a773bp-7f
    constexpr
    const float log_c11=+6.7610226572e-02f;
    static_assert(log_c1==1.0f, "constraint violated");
    static_assert(log_c2==-0.5f, "constraint violated");
    constexpr
    static const float coeffs[]={
        log_c11, log_c10,
        log_c9, log_c8, log_c7, log_c6,
        log_c5, log_c4, log_c3, log_c2, log_c1
    };

    try {
        using namespace ocl;

        using ftype = float;

        constexpr const std::size_t elem_count=(64*1024*1024ULL);

        for (size_t i=0; i<32; ++i) {
            dvec<ftype> v_src(0.5f, elem_count);
            dvec<ftype> v_dst(0.0f, elem_count);
            auto start = std::chrono::steady_clock::now();
            v_dst=horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+  // 10
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+  // 20
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs)+
                horner(v_src, coeffs); // 30
            auto end = std::chrono::steady_clock::now();
            auto ns_elapsed=(end - start).count();
            // std::cout << ns_elapsed << std::endl;
            double gflops=(elem_count *(30*20+29))/double(ns_elapsed);
            std::cout << gflops << std::endl;
        }
        std::cout << "test passed\n";
    }
    catch (const ocl::be::error& e) {
        std::cout << "caught exception: ocl::be::error: " << e.what()
                  << '\n'
                  << e.error_string()
                  << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cout << "caught exception: runtime error: " << e.what()
                  << std::endl;
    }
    return 0;
}
#endif
