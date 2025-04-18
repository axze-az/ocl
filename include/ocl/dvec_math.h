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
#if !defined (__OCL_DVEC_MATH_H__)
#define __OCL_DVEC_MATH_H__ 1

#include <ocl/config.h>
#include <ocl/dvec_func.h>

namespace ocl {

    namespace impl {
        __cf_body
        gen_horner(const std::string_view& tname,
                   const std::string_view& coeff_tname,
                   size_t n, bool use_fma, bool use_mad);
        __cf_body
        gen_horner2(const std::string_view& tname,
                    const std::string_view& coeff_tname,
                    size_t n,
                    bool x2_internal, bool use_fma, bool use_mad);
        __cf_body
        gen_horner4(const std::string_view& tname,
                    const std::string_view& coeff_tname,
                    size_t _N,
                    bool x2x4_internal, bool use_fma, bool use_mad);

        template <typename _T>
        std::pair<bool, bool>
        horner_use_fma_mad(const dvec<_T>& x);
    }

    template <typename _X, typename _C1, typename _C0>
    auto
    horner(const _X& x, const _C1& c1, const _C0& c0);

    template <typename _X,
              typename _CN, typename _CNM1, typename ... _CS>
    auto
    horner(const _X& x, const _CN& cn,
           const _CNM1& cnm1, _CS... cs);

    template <typename _T, typename _C, size_t _N>
    auto
    horner(const dvec<_T>& x, const _C(&ci)[_N]);

    template <typename _T, typename _C, size_t _N>
    auto
    horner2(const dvec<_T>& x, const dvec<_T>& x2, const _C(&ci)[_N]);

    template <typename _T, typename _C, size_t _N>
    auto
    horner2(const dvec<_T>& x, const _C(&ci)[_N]);

    template <typename _T, typename _C, size_t _N>
    auto
    horner4(const dvec<_T>& x, const dvec<_T>& x2, const dvec<_T>& x4,
            const _C(&ci)[_N]);

    template <typename _T, typename _C, size_t _N>
    auto
    horner4(const dvec<_T>& x,
            const _C(&ci)[_N]);
}

template <typename _T>
std::pair<bool, bool>
ocl::impl::horner_use_fma_mad(const dvec<_T>& x)
{
    bool use_fma=std::is_same_v<_T, double>;
    bool use_mad=std::is_floating_point_v<_T>;
    if (use_mad) {
        if (std::is_same_v<float, _T> &&
            x.backend_data()->supports(be::data::query_bool::fp32_fma)) {
            use_fma = true;
            use_mad = false;
        }
    }
    return std::make_pair(use_fma, use_mad);
}

template <typename _T, typename _C, size_t _N>
auto
ocl::horner(const dvec<_T>& x, const _C(&a)[_N])
{
    const auto tname= be::type_2_name<_T>::v();
    const auto cname= be::type_2_name<_C>::v();
    auto use_fma_mad=impl::horner_use_fma_mad(x);
    auto hb=impl::gen_horner(tname, cname, _N,
                             use_fma_mad.first, use_fma_mad.second);
    return custom_func<_T>(hb.name(), hb.body(), x, a);
}

template <typename _T, typename _C, size_t _N>
auto
ocl::horner2(const dvec<_T>& x, const dvec<_T>& x2,
             const _C(&a)[_N])
{
    static_assert(_N > 1, "invalid call to horner2(x, x2, array)");
    const auto tname= be::type_2_name<_T>::v();
    const auto cname= be::type_2_name<_C>::v();
    auto use_fma_mad=impl::horner_use_fma_mad(x);
    auto hb=impl::gen_horner2(tname, cname, _N,
                              false, use_fma_mad.first, use_fma_mad.second);
    return custom_func<_T>(hb.name(), hb.body(), x, x2, a);
}

template <typename _T, typename _C, size_t _N>
auto
ocl::horner2(const dvec<_T>& x,
             const _C(&a)[_N])
{
    static_assert(_N > 1, "invalid call to horner2(x, array)");
    const auto tname= be::type_2_name<_T>::v();
    const auto cname= be::type_2_name<_C>::v();
    auto use_fma_mad=impl::horner_use_fma_mad(x);
    auto hb=impl::gen_horner2(tname, cname, _N,
                              true, use_fma_mad.first, use_fma_mad.second);
    return custom_func<_T>(hb.name(), hb.body(), x, a);
}

template <typename _T, typename _C, size_t _N>
auto
ocl::horner4(const dvec<_T>& x, const dvec<_T>& x2, const dvec<_T>& x4,
             const _C(&a)[_N])
{
    static_assert(_N > 3, "invalid call to horner4(x, x2, x4, array)");
    const auto tname= be::type_2_name<_T>::v();
    const auto cname= be::type_2_name<_C>::v();
    auto use_fma_mad=impl::horner_use_fma_mad(x);
    auto hb=impl::gen_horner4(tname, cname, _N,
                              false, use_fma_mad.first, use_fma_mad.second);
   return custom_func<_T>(hb.name(), hb.body(), x, x2, x4, a);
}

template <typename _T, typename _C, size_t _N>
auto
ocl::horner4(const dvec<_T>& x,
             const _C(&a)[_N])
{
    static_assert(_N > 3, "invalid call to horner4(x, array)");
    const auto tname= be::type_2_name<_T>::v();
    const auto cname= be::type_2_name<_C>::v();
    auto use_fma_mad=impl::horner_use_fma_mad(x);
    auto hb=impl::gen_horner4(tname, cname, _N,
                              true, use_fma_mad.first, use_fma_mad.second);
   return custom_func<_T>(hb.name(), hb.body(), x, a);
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

// local variables:
// mode: c++
// end:
#endif
