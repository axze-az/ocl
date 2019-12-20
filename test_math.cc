#include <cftal/vec_cvt.h>
#include <cftal/math/elem_func.h>
#include <cftal/math/elem_func_core_f32.h>
#include <cftal/d_real.h>
#include <ocl/ocl.h>
#include <ocl/vector.h>


namespace cftal {

    template <>
    struct d_real_traits<ocl::vector<float> > {
        constexpr static const bool fma=true;
        using cmp_result_type = typename ocl::vector<int32_t>;
        using int_type = ocl::vector<int32_t>;

        static
        bool any(const cmp_result_type& b) {
            return true;
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
            const int32_t msk=
                const_u32<0xfffff000U>::v.s32();
            using vi_type = ocl::vector<int32_t>;
            using vf_type = ocl::vector<float>;
            vi_type ai=ocl::as<vi_type>(a);
            vf_type th=ocl::as<vf_type>(ai & msk);
            vf_type tl=a - th;
            h = th;
            l = tl;
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
    _X
    horner(const _X& x, const _C1& c1, const _C0& c0);

    template <typename _T, typename _C, size_t _N>
    vector<_T>
    horner(const vector<_T>& x, const _C(&ci)[_N]);


    namespace test {
        const int ELEMENTS=4096*1024;
        void
        test_add12cond();

        void
        test_mul12();

        void
        test_horner();
    }
};

template <typename _T, typename _C, size_t _N>
ocl::vector<_T>
ocl::horner(const vector<_T>& x, const _C(&a)[_N])
{
    const _C* pa=a;
    vector<_T> r(x.size(), pa[0]);
    static_assert(_N > 0, "invalid call to horner(x, array)");
#pragma GCC unroll 256
#pragma clang loop unroll(full)
    for (std::size_t i=1; i<_N; ++i) {
        r= horner(x, r, pa[i]);
    }
    return r;
}

template <typename _X, typename _C1, typename _C0>
_X
ocl::horner(const _X& x, const _C1& c1, const _C0& c0)
{
    return x*c1 + c0;
}

void
ocl::test::test_add12cond()
{
    using vf_type = vector<float>;
    vf_type a(ELEMENTS, 1.0f), b(ELEMENTS, 2.0f);
    using d_ops=cftal::d_real_ops<vf_type, false>;
    vf_type h, l;
    d_ops::add12cond(h, l, a, b);
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}

void
ocl::test::test_mul12()
{
    using vf_type = vector<float>;
    vf_type a(ELEMENTS, 1.0f), b(ELEMENTS, 2.0f);
    using d_ops=cftal::d_real_ops<vf_type, false>;
    vf_type h, l;
    d_ops::mul12(h, l, a, b);
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}

void
ocl::test::test_horner()
{
    using vf_type = vector<float>;
    vf_type x(ELEMENTS, 2.0f);
    static const float ci[]={
        1.0f, 2.0f, 3.0f
    };
    vf_type y=horner(x, ci);
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}

int main()
{
    try {
        ocl::test::test_add12cond();
        ocl::test::test_mul12();
        ocl::test::test_horner();
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

