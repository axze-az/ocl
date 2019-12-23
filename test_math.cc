#include <cftal/vec_cvt.h>
#include <cftal/math/elem_func.h>
#include <cftal/math/elem_func_core_f32.h>
#include <cftal/d_real.h>
#include <ocl/ocl.h>
#include <ocl/vector.h>
#include <chrono>
#include <thread>


namespace cftal {

    template <>
    struct d_real_traits<ocl::vector<float> > {
        constexpr static const bool fma=true;
        using cmp_result_type = typename ocl::vector<int32_t>;
        using int_type = ocl::vector<int32_t>;

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
        ocl::vector<float>
        sel (const cmp_result_type& s,
             const ocl::vector<float>& on_true,
             const ocl::vector<float>& on_false) {
            // return select(s, on_true, on_false);
            return on_true;
        }

        static
        void
        split(const ocl::vector<float> & a,
              ocl::vector<float>& h,
              ocl::vector<float>& l) {
#if 0
            const int32_t msk=
                const_u32<0xfffff000U>::v.s32();
            using vi_type = ocl::vector<int32_t>;
            using ocl::as;
            vi_type& hi=reinterpret_cast<vi_type&>(h);
            const vi_type& ai=reinterpret_cast<const vi_type&>(a);
            hi = ai & msk;
            l= a - h;
#else
            const int32_t msk=
                const_u32<0xfffff000U>::v.s32();
            using vi_type = ocl::vector<int32_t>;
            using vf_type = ocl::vector<float>;
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
    horner(const vector<_T>& x, const _C(&ci)[_N]);

    namespace test {
        const int ELEMENTS=4*1024*1024-1;
        void
        test_add12cond(const vector<float>& x);

        void
        test_mul12(const vector<float>& x);

        void
        test_horner(const vector<float>& x);
    }
};

template <typename _T, typename _C, size_t _N>
auto
ocl::horner(const vector<_T>& x, const _C(&a)[_N])
{
    static_assert(_N > 0, "invalid call to horner(x, array)");
    const _C* pa=a;
    using _T_t= const vector<_T>&;
    return impl::unroll_horner<_T_t, _C, _N, _N-1>::v(x, a[0], pa);
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
ocl::test::test_add12cond(const vector<float>& x)
{
    using vf_type = vector<float>;

    vf_type a=x, b=x;
    using d_ops=cftal::d_real_ops<vf_type, false>;
    vf_type h, l;
    d_ops::add12cond(h, l, a, b);
    vf_type hh, ll;
    d_ops::add22cond(hh, ll, h, l, h, l);
    // std::cout << __PRETTY_FUNCTION__ << std::endl;
}

void
ocl::test::test_mul12(const vector<float>& x)
{
    using vf_type = vector<float>;
    vf_type a=x, b=x;
    using d_ops=cftal::d_real_ops<vf_type, false>;
    vf_type h, l;
    d_ops::mul12(h, l, a, b);
    vf_type hh, ll;
    d_ops::sqr22(hh, ll, h, l);
    d_ops::div22(hh, ll, hh, ll, h, l);
    // std::cout << __PRETTY_FUNCTION__ << std::endl;
    // std::vector<float> hh(h), hl(l);
}

void
ocl::test::test_horner(const vector<float>& x)
{
    using vf_type = vector<float>;
    using vi_type = vector<int32_t>;
    static const float ci[]={
        1.0f, 2.0/10.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };
    vf_type y0=horner(x, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    vf_type y1=horner(x, ci);
    vi_type eq=y0 == y1;
    vi_type eq0=y0 == 1.0f;
    // std::cout << __PRETTY_FUNCTION__ << std::endl;
}

int main()
{
    try {
        const int count=1;
        ocl::vector<float> x(ocl::test::ELEMENTS, 1.1235f);
        // ocl::test::test_add12cond();
        for (int i=0; i<count; ++i) {
            ocl::test::test_mul12(x);
            ocl::test::test_add12cond(x);
            ocl::test::test_horner(x);
            if ((i & 3)==3) {
                std::cout << '.' << std::flush;
            }
            // std::chrono::seconds s4(1);
            // std::this_thread::sleep_for(s4);
        }
        // ocl::test::test_horner();
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

