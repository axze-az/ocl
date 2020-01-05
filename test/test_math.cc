#include <cftal/vec_cvt.h>
#include <cftal/math/elem_func.h>
#include <cftal/math/elem_func_core_f32.h>
#include <cftal/d_real.h>
#include "ocl/dvec.h"
// #include <vexcl/vexcl.hpp>
#include <chrono>
#include <thread>


/*
You can dump the list of kernels and the LLVM IR when a program runs by
doing the following:
CLOVER_DEBUG_FILE=clover_dump CLOVER_DEBUG=clc,llvm,asm
PATH_TO_YOUR_TEST_PROGRAM
--> use native instead of asm

That'll generate a set of files called clover_dump.cl, clover_dump.ll,
clover_dump.asm with:
a) The CL source that the program tried to compile
b) The LLVM IR for the CL source.
c) The generated machine code for the LLVM IR on your card.

If the CL source is missing built-in function implementations, libclc (
libclc.llvm.org) will gladly accept patches to implement them.

If you want to just do a test-compile of the CL source, I use the following
command (with libclc/llvm/mesa all in /usr/local/):
clang -S -emit-llvm -o $1.ll -include /usr/local/include/clc/clc.h
-I/usr/local/include/ -Dcl_clang_storage_class_specifiers -target amdgcn--
-mcpu=pitcairn -c $1
*/

namespace cftal {

    template <>
    struct d_real_traits<ocl::dvec<float> > {
        constexpr static const bool fma=true;
        using cmp_result_type = typename ocl::dvec<int32_t>;
        using int_type = ocl::dvec<int32_t>;

        static
        bool any_of_v(const cmp_result_type& b) {
            return true;
        }

        static
        bool all_of_v(const cmp_result_type& b) {
            return false;
        }

        static
        bool none_of_v(const cmp_result_type& b) {
            return false;
        }

        static
        ocl::dvec<float>
        sel (const cmp_result_type& s,
             const ocl::dvec<float>& on_true,
             const ocl::dvec<float>& on_false) {
            // return select(s, on_true, on_false);
            return on_true;
        }

        static
        void
        split(const ocl::dvec<float> & a,
              ocl::dvec<float>& h,
              ocl::dvec<float>& l) {
#if 0
            const int32_t msk=
                const_u32<0xfffff000U>::v.s32();
            using vi_type = ocl::dvec<int32_t>;
            vi_type& hi=reinterpret_cast<vi_type&>(h);
            const vi_type& ai=reinterpret_cast<const vi_type&>(a);
            hi = ai & msk;
            l= a - h;
#else
            const int32_t msk=
                const_u32<0xfffff000U>::v.s32();
            using vi_type = ocl::dvec<int32_t>;
            using vf_type = ocl::dvec<float>;
            using ocl::as;
            h=as<vf_type>(as<vi_type>(a) & msk);
            l=a - h;
#endif
        }

        constexpr
        static
        float
        scale_div_threshold() {
            // -126 + 24
            return 0x1.p-102f;
        }

    };
}

namespace ocl {

    template <typename _X, typename _C1, typename _C0>
    auto
    horner(const _X& x, const _C1& c1, const _C0& c0);

    template <typename _X,
              typename _CN, typename _CNM1, typename ... _CS>
    auto
    horner(const _X& x, const _CN& cn,
           const _CNM1& cnm1, _CS... cs);

    namespace impl {

        template <typename _T, typename _C, size_t _N, size_t _I>
        struct unroll_horner {
            template <typename _E>
            static
            auto
            v(_T x, _E e, const _C* p) {
                using v_t = unroll_horner<_T, _C, _N, _I-1>;
                return v_t::v(x, x*e + p[_N-_I], p);
            }
        };

        template <typename _T, typename _C, size_t _N>
        struct unroll_horner<_T, _C, _N, 1> {
            template <typename _E>
            static
            auto
            v(_T x, _E e, const _C* p) {
                return x*e + p[_N-1];
            }
        };

    }

    template <typename _T, typename _C, size_t _N>
    auto
    horner(const dvec<_T>& x, const _C(&ci)[_N]);

    namespace test {

        const int VEC_SIZE=1;

        const int ELEMENTS=((4*1024*1024)/VEC_SIZE)-1;
        void
        test_add12cond(const dvec<float>& x);

        void
        test_mul12(const dvec<float>& x);

        void
        test_horner(const dvec<float>& x);
    }
};

template <typename _T, typename _C, size_t _N>
auto
ocl::horner(const dvec<_T>& x, const _C(&a)[_N])
{
#if 1
    std::ostringstream hs;
    hs << "horner" << _N; 
    const std::string hname=hs.str();
    hs.str("");
    const std::string tname= be::type_2_name<_T>::v();
    const std::string cname= be::type_2_name<_C>::v();
    hs << tname << " " << hname << "( " << tname
       << " v0, __arg_local const " << cname << "* c)\n"
        "{\n"
        "    "<< tname << " r=c[0];\n";
    for (size_t i=1; i<_N; ++i) {
        hs << "    r=v0*r+c["<< i<<"];\n";
    }
    hs << "    return r;\n"
          "}\n";
    const std::string hbody=hs.str();
    return custom_func<_T>(hname, hbody, x, a); 
#else
    static_assert(_N > 0, "invalid call to horner(x, array)");
    const _C* pa=a;
    using _T_t= const dvec<_T>&;
    return impl::unroll_horner<_T_t, _C, _N, _N-1>::v(x, a[0], pa);
#endif
}

template <typename _X, typename _C1, typename _C0>
auto
ocl::horner(const _X& x, const _C1& c1, const _C0& c0)
{
    return x*c1 + c0;
}

template <typename _X,
          typename _CN, typename _CNM1, typename ... _CS>
auto
ocl::horner(const _X& x, const _CN& cn,
            const _CNM1& cnm1, _CS... cs)
{
    auto t = horner(x, cn, cnm1);
    auto r = horner(x, t, cs...);
    return r;
}

void
ocl::test::test_add12cond(const dvec<float>& x)
{
    using vf_type = dvec<float>;

    vf_type a=x, b=x;
    using d_ops=cftal::d_real_ops<vf_type, false>;
    vf_type h, l;
    d_ops::add12cond(h, l, a, b);
    vf_type hh, ll;
    d_ops::add22cond(hh, ll, h, l, h, l);
    // std::cout << __PRETTY_FUNCTION__ << std::endl;
}

void
ocl::test::test_mul12(const dvec<float>& x)
{
    using vf_type = dvec<float>;
    vf_type a=x, b=x;
    using d_ops=cftal::d_real_ops<vf_type, false>;
    vf_type h, l;
    d_ops::mul12(h, l, a, b);
    vf_type hh, ll;
    d_ops::sqr22(hh, ll, h, l);
    d_ops::div22(hh, ll, hh, ll, h, l);
    // std::cout << __PRETTY_FUNCTION__ << std::endl;
    // std::dvec<float> hh(h), hl(l);
}

void
ocl::test::test_horner(const dvec<float>& x)
{
    using vf_type = dvec<float>;
    // using vi_type = dvec<int32_t>;
    constexpr
    const float log_c1=+1.0000000000000000000000e+00;
    // x^2 : -0x8p-4
    constexpr
    const float log_c2=-5.0000000000000000000000e-01;
    // x^3 : +0xa.aaaaaaaaaaac8p-5
    constexpr
    const float log_c3=+3.3333333333333353687422e-01;
    // x^4 : -0x8.0000000000208p-5
    constexpr
    const float log_c4=-2.5000000000000360822483e-01;
    // x^5 : +0xc.ccccccccc6e9p-6
    constexpr
    const float log_c5=+1.9999999999991630028617e-01;
    // x^6 : -0xa.aaaaaaaa58168p-6
    constexpr
    const float log_c6=-1.6666666666549315167778e-01;
    // x^7 : +0x9.24924927ac58p-6
    constexpr
    const float log_c7=+1.4285714286842710052383e-01;
    // x^8 : -0x8.00000027e75c8p-6
    constexpr
    const float log_c8=-1.2500000014516901569728e-01;
    // x^9 : +0xe.38e38cbfa4d38p-7
    constexpr
    const float log_c9=+1.1111111042490638689539e-01;
    // x^10 : -0xc.ccccb8d164208p-7
    constexpr
    const float log_c10=-9.9999990695125454132075e-02;
    // x^11 : +0xb.a2e8e4465066p-7
    constexpr
    const float log_c11=+9.0909110510099810920082e-02;
    // x^12 : -0xa.aaad942807438p-7
    constexpr
    const float log_c12=-8.3333680479023994336352e-02;
    // x^13 : +0x9.d89c440648528p-7
    constexpr
    const float log_c13=+7.6922925200565275827280e-02;
    // x^14 : -0x9.24504f5c6c73p-7
    constexpr
    const float log_c14=-7.1420706511023362983437e-02;
    // x^15 : +0x8.88565bbd4181p-7
    constexpr
    const float log_c15=+6.6660685343332942709438e-02;
    // x^16 : -0x8.0381a20dc6aap-7
    constexpr
    const float log_c16=-6.2607006194914049945766e-02;
    // x^17 : +0xf.1b610aa965e5p-8
    constexpr
    const float log_c17=+5.9011521437603756123913e-02;
    // x^18 : -0xe.04b526d45cb08p-8
    constexpr
    const float log_c18=-5.4759332637660980414029e-02;
    // x^19 : +0xc.e282859d9531p-8
    constexpr
    const float log_c19=+5.0331266041742109274004e-02;
    // x^20 : -0xd.55ffa43077a8p-8
    constexpr
    const float log_c20=-5.2093484483036633925224e-02;
    // x^21 : +0xf.7e2701c5769a8p-8
    constexpr
    const float log_c21=+6.0518682415443704469826e-02;
    // x^22 : -0xd.b9eebf859befp-8
    constexpr
    const float log_c22=-5.3618356474188763605149e-02;
    // x^23 : +0xb.508a0d3fd5d08p-9
    constexpr
    const float log_c23=+2.2098840825417579575296e-02;

    static_assert(log_c1 == 1.0, "constraint violated");
    static const float ci[]={
        log_c23, log_c22, log_c21, log_c20, log_c19,
        log_c18, log_c17, log_c16, log_c15, log_c14,
        log_c13, log_c12, log_c11, log_c10, log_c9,
        log_c8,  log_c7,  log_c6,  log_c5,  log_c4,
        log_c3,  log_c2,  log_c1
    };
    for (int i=0; i<512; ++i) {
#if 0
        vf_type y0=horner(x,
                          log_c23, log_c22, log_c21, log_c20, log_c19,
                          log_c18, log_c17, log_c16, log_c15, log_c14,
                          log_c13, log_c12, log_c11, log_c10, log_c9,
                          log_c8,  log_c7,  log_c6,  log_c5,  log_c4,
                          log_c3,  log_c2,  log_c1);
#endif
        vf_type y1=horner(x, ci);
        // vi_type eq=y0 == y1;
        // vi_type eq0=y0 == 1.0f;
    }
    // std::cout << __PRETTY_FUNCTION__ << std::endl;
    // ocl::impl::be_data::instance()->q().finish();
}

int main()
{
    try {
        const int count=1024*2;
#if 0
        vex::Context ctx( vex::Filter::GPU);
        std::vector<float> xh(ocl::test::ELEMENTS, 1.1235f);
        ocl::dvec<float> x(ctx, ocl::test::ELEMENTS);
        vex::copy(xh, x);
#endif
        ocl::dvec<float> x(ocl::test::ELEMENTS, 1.1235f);
        // ocl::test::test_add12cond();
        for (int i=0; i<count; ++i) {
            // ocl::test::test_mul12(x);
            // ocl::test::test_add12cond(x);
            ocl::test::test_horner(x);
            if ((i & 3)==3) {
                std::cout << '.' << std::flush;
            }
            // std::chrono::seconds s4(1);
            // std::this_thread::sleep_for(s4);
        }
        // ocl::test::test_horner();
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

