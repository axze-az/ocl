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
#include <cftal/vec_cvt.h>
#include <cftal/math/elem_func.h>
#include <cftal/math/elem_func_core_f32.h>
#include <cftal/d_real.h>
#include <ocl/test/ops_base.h>
#include "ocl/dvec.h"
#include <chrono>
#include <thread>
#include <limits>

#define DEBUG_MATH 0

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

    namespace test {

        template <typename _T>
        class test_functions : public ops_base<_T> {
            using b_t = ops_base<_T>;
            using b_t::_res;
            using b_t::_a0;
            using b_t::_a1;
            using b_t::_h_res;
            using b_t::_h_a0;
            using b_t::_h_a1;
            using b_t::check_res;
        public:
            test_functions(size_t n)
                : b_t(n, _T(1.0), _T(1.0)) {
            }
            bool perform();
        };

        const int VEC_SIZE=1;
        const int ELEMENTS=((4*1024*1024)/VEC_SIZE)-1;

        float
        test_add12cond(const dvec<float>& x);

        float
        test_mul12(const dvec<float>& x);

    }
};

template <typename _T>
bool
ocl::test::test_functions<_T>::perform()
{
    bool rc=true;

    // hsum
    _T r_hsum=hsum(_h_a0);
    _T d_hsum=hsum(_a0);

#if DEBUG_MATH > 0
    std::cout << "size:" << _h_a0.size()
              << ' '<< r_hsum << ' '  << d_hsum << '\n';
#endif
    _T delta_hsum=r_hsum - d_hsum;
    _T rel_delta_hsum=delta_hsum/((r_hsum+d_hsum)*_T(0.5));

    _T max_rel_err_hsum=
        _a0.size() * std::numeric_limits<_T>::epsilon() * _T(1.0);
    using std::abs;
    if (abs(rel_delta_hsum) > max_rel_err_hsum) {
        std::cout << std::setprecision(19) << std::scientific;
        std::cout << "hsum elements: " << _a0.size()
                  << " max_rel_err: " << max_rel_err_hsum << '\n';
        std::cout << "delta: " << delta_hsum
                  << "\nrel_delta: "  << rel_delta_hsum << '\n';
        rc = false;
    }

    // dot_product
    _T r_dot_product=dot_product(_h_a0, _h_a1);
    _T d_dot_product=dot_product(_a0, _a1);

#if DEBUG_MATH > 0
    std::cout << "size:" << _h_a0.size()
               << ' ' << r_dot_product << ' '  << d_dot_product << '\n';
#endif
    _T delta_dot_product=r_dot_product - d_dot_product;
    _T rel_delta_dot_product=
        delta_dot_product/((r_dot_product+d_dot_product)*_T(0.5));
    _T max_rel_err_dot_product=
        _a0.size() * std::numeric_limits<_T>::epsilon() * _T(2.0);
    if (abs(rel_delta_dot_product) > max_rel_err_dot_product) {
        std::cout << std::setprecision(19) << std::scientific;
        std::cout << "dot_product elements: " << _a0.size()
                  << " max_rel_err: " << max_rel_err_dot_product << '\n';
        std::cout << "delta: " << delta_dot_product
                  << "\nrel_delta: "  << rel_delta_dot_product << '\n';
        rc = false;
    }

    return rc;
}

float
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
    vf_type t=hh+ll;
    float r=hsum(t);
    return r;
}

float
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
    vf_type t=hh+ll;
    float r=hsum(t);
    return r;
}

int main()
{
    try {
        using namespace cftal;
        using namespace ocl;
        using namespace ocl::test;

        const size_t max_buffer_size=512*1024*1024;
        for (std::size_t i=4; i<max_buffer_size/sizeof(float); i = (i*5)>>2) {
            if (1) {
                std::cout << "using buffers with "
                          <<  i
                          << " elements (" << i*sizeof(float)
                          << " bytes)"
#if DEBUG_MATH>0
                          << '\n'
#else
                          << '\r'
#endif
                          << std::flush;
            }
            test_functions<float> t(i);
            if (t.perform() == false) {
                std::cout << "\ntest for vector length " << i << " failed\n";
                std::exit(3);
            }
        }
        std::cout << "\nfloat test passed\n";
        for (std::size_t i=4; i<max_buffer_size/sizeof(double); i= (i*5)>>2) {
            if (1) {
                std::cout << "using buffers with "
                          <<  i
                          << " elements (" << i*sizeof(double)
                          << " bytes)"
#if DEBUG_MATH>0
                          << '\n'
#else
                          << '\r'
#endif
                          << std::flush;
            }
            test_functions<double> t(i);
            if (t.perform() == false) {
                std::cout << "\ntest for vector length " << i << " failed\n";
                std::exit(3);
            }
        }
        std::cout << "\ndouble test passed\n";
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

