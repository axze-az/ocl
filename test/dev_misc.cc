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
#include "ocl/ocl.h"
#include "ocl/test/tools.h"
#include <cftal/math/func_traits_f32_s32.h>
#include <cftal/math/func_traits_f64_s32.h>
#include <cftal/math/elem_func_core_f64.h>
#include <cftal/d_real.h>

namespace ocl {

    namespace impl {
        // worker function for evem_elements(v)
        __ck_body
        even_elements(const std::string_view& tname);
    }

    // returns only the even elements of s
    template <typename _T>
    dvec<_T>
    even_elements(const dvec<_T>& s);

    namespace impl {
        // worker function for odd_elements(v)
        __ck_body
        odd_elements(const std::string_view& tname);
    }

    // returns only the odd elements of s
    template <typename _T>
    dvec<_T>
    odd_elements(const dvec<_T>& s);

    namespace impl {
        // worker function for combine_even_odd(e, o)
        __ck_body
        combine_even_odd(const std::string_view& tname);
    }

    // return the interleaved elemnent of e and o
    template <typename _T>
    dvec<_T>
    combine_even_odd(const dvec<_T>& e, const dvec<_T>& o);

    namespace impl {
        // worker function for select_even_odd
        __ck_body
        select_even_odd(const std::string_view& tname);
    }

    template <typename _T>
    dvec<_T>
    select_even_odd(const dvec<_T>& e, const dvec<_T>& o);


    namespace impl {
        // worker function for copy_even_to_odd(v)
        __ck_body
        copy_even_to_odd(const std::string_view& tname);
    }

    // copy the even elements of s to the odd elements
    template <typename _T>
    dvec<_T>
    copy_even_to_odd(const dvec<_T>& s);

    namespace impl {
        // worker function for copy_odd_to_even(v)
        __ck_body
        copy_odd_to_even(const std::string_view& tname);
    }

    // copy the odd elements of s to the even elements
    template <typename _T>
    dvec<_T>
    copy_odd_to_even(const dvec<_T>& s);


    namespace impl {
        // worker function for permute(i, v)
        __ck_body
        permute(const std::string_view& tname,
                const std::string_view& iname);
    }

    // permute the vector using idx
    template <typename _T, typename _I>
    dvec<_T>
    permute(const dvec<_I>& i, const dvec<_T>& s);

    namespace impl {
        // worker function for permute(i, v, v)
        __ck_body
        permute2(const std::string_view& tname,
                 const std::string_view& iname);
    }

    template <typename _T, typename _I>
    dvec<_T>
    permute(const dvec<_I>& i, const dvec<_T>& s1, const dvec<_T>& s2);
}

namespace cftal {

    template <>
    struct d_real_traits<ocl::dvec<double> > {
        using cmp_result_type = typename ocl::dvec<double>::mask_type;
        using int_type = ocl::dvec<int32_t>;

        static constexpr bool fma = false;

        static
        bool any_of_v(const cmp_result_type& b) {
            return any_of(b);
        }

        static
        bool all_of_v(const cmp_result_type& b) {
            return all_of(b);
        }

        static
        bool none_of_v(const cmp_result_type& b) {
            return none_of(b);
        }

        static
        ocl::dvec<double>
        sel (const cmp_result_type& s,
             const ocl::dvec<double>& on_true,
             const ocl::dvec<double>& on_false) {
            return select(s, on_true, on_false);
        }

        static
        void
        split(const ocl::dvec<double>& a,
              ocl::dvec<double>& h,
              ocl::dvec<double>& l) {
            const double msk=
                const_u64<0xf8000000U, 0xffffffffU>::v.f64();
            const ocl::dvec<double> th = a & msk;
            const ocl::dvec<double> tl = a - th;
            h = th;
            l = tl;
        }

        constexpr
        static
        double
        scale_div_threshold() {
            // -1022 + 53
            return 0x1.p-969;
        }
    };

    namespace math {

        struct dvec_func_traits_f64
            : public func_traits<double, int32_t> {
            using vf_type = ocl::dvec<double>;
            using vmf_type = typename ocl::dvec<double>::mask_type;
            using vi_type = ocl::dvec<int32_t>;
            using vmi_type = typename ocl::dvec<int32_t>::mask_type;
            using vu_type = ocl::dvec<uint32_t>;
            // integer vector with the same length as vf_type
            using vi2_type = ocl::dvec<int32_t>;
            using vmi2_type = typename ocl::dvec<int32_t>::mask_type;

            using vli_type = ocl::dvec<int64_t>;
            using vmli_type= typename ocl::dvec<int64_t>::mask_type;

            using vdf_type = d_real<vf_type>;

            static
            constexpr std::size_t NVF() {
                return 0;
            }

            static
            constexpr std::size_t NVI() {
                return 0;
            }


            static
            vmf_type
            vmi_to_vmf(const vmi_type& mi) {
                return ocl::cvt<vmf_type>(mi);
            }

            static
            vmi_type
            vmf_to_vmi(const vmf_type& mf) {
                return ocl::cvt<vmi_type>(mf);
            }

            static
            vmi2_type
            vmf_to_vmi2(const vmf_type& mf) {
                return ocl::as<vmi2_type>(mf);
            };

            static
            vmf_type
            vmi2_to_vmf(const vmi2_type& mf) {
                return ocl::as<vmf_type>(mf);
            };

            static
            bool any_of_v(const vmf_type& b) {
                return any_of(b);
            }

            static
            bool all_of_v(const vmf_type& b) {
                return all_of(b);
            }

            static
            bool none_of_v(const vmf_type& b) {
                return none_of(b);
            }

            static
            bool any_of_v(const vmi_type& b) {
                return any_of(b);
            }

            static
            bool all_of_v(const vmi_type& b) {
                return all_of(b);
            }

            static
            bool none_of_v(const vmi_type& b) {
                return none_of(b);
            }

#if 0
            static
            bool any_of_v(const vmi2_type& b) {
                return any_of(b);
            }

            static
            bool all_of_v(const vmi2_type& b) {
                return all_of(b);
            }

            static
            bool none_of_v(const vmi2_type& b) {
                return none_of(b);
            }
#endif
            static
            vi_type sel(const vmi_type& msk,
                        const vi_type& t, const vi_type& f) {
                return select(msk, t, f);
            }

            static
            vi_type sel_val_or_zero(const vmi_type& msk,
                                    const vi_type& t) {
                return select(msk, t, 0);
            }

            static
            vi_type sel_val_or_zero(const vmi_type& msk,
                                    const int32_t& t) {
                return select(msk, t, 0);
            }

            static
            vi_type sel_zero_or_val(const vmi_type& msk,
                                    const vi_type& f) {
                return select(msk, 0, f);
            }

            static
            vf_type sel(const vmf_type& msk,
                        const vf_type& t, const vf_type& f) {
                return select(msk, t, f);
            }

            static
            vf_type sel(const vmf_type& msk,
                        const double& t, const vf_type& f) {
                return select(msk, t, f);
            }

            static
            vf_type sel(const vmf_type& msk,
                        const vf_type& t, const double& f) {
                return select(msk, t, f);
            }


            static
            vf_type sel_val_or_zero(const vmf_type& msk,
                                    const vf_type& t) {
                return select(msk, t, 0.0);
            }

            static
            vf_type sel_zero_or_val(const vmf_type& msk,
                                    const vf_type& f) {
                return select(msk, 0.0, f);
            }

#if 0
            static
            vi2_type sel(const vmi2_type& msk,
                         const vi2_type& t, const vi2_type& f) {
                return select(msk, t, f);
            }

            static
            vi2_type sel_val_or_zero(const vmi2_type& msk,
                                     const vi2_type& t) {
                return select_val_or_zero(msk, t);
            }
            static
            vi2_type sel_zero_or_val(const vmi2_type& msk,
                                     const vi2_type& f) {
                return select_zero_or_val(msk, f);
            }

#endif
            static
            vli_type sel(const vmli_type& msk,
                         const vli_type& t, const vli_type& f) {
                return select(msk, t, f);
            }

            static
            vf_type insert_exp_vi2(const vi2_type& e) {
                vi2_type ep(e << 20);
                vf_type r= ocl::as<vf_type>(ep);
                r &= exp_f64_msk::v.f64();
                return r;
            }

            static
            vi2_type vi_to_vi2(const vi_type& r) {
                vi2_type t=combine_even_odd(r, r);
                return t;
            }

            static
            vi_type vi2_odd_to_vi(const vi2_type& r) {
                vi_type t=odd_elements(r);
                return t;
            }

            static
            vi_type vi2_even_to_vi(const vi2_type& r) {
                vi_type t=even_elements(r);
                return t;
            }

            static
            vf_type insert_exp(const vi_type& e) {
                vi_type ep(e << 20);
                vi_type zero(e.size(), 0);
                vi2_type ir(combine_even_odd(zero, ep));
                vf_type r= ocl::as<vf_type>(ir);
                // r &= vf_type(exp_f64_msk::v.f64());
                return r;
            }

            static
            vi_type extract_exp(const vf_type& d) {
                vf_type m(d & exp_f64_msk::v.f64());
                vi2_type di= ocl::as<vi2_type>(m);
                vi_type r= odd_elements(di);
                r >>= 20;
                return r;
            }

            static
            vi_type extract_high_word(const vf_type& d) {
                vi2_type di=ocl::as<vi2_type>(d);
                return odd_elements(di);
            }

            static
            vi_type extract_low_word(const vf_type& d) {
                vi2_type di=ocl::as<vi2_type>(d);
                return even_elements(di);
            }

            static
            void
            extract_words(vi_type& low_word, vi_type& high_word,
                          const vf_type& d) {
                vi2_type di=ocl::as<vi2_type>(d);
                low_word=even_elements(di);
                high_word=odd_elements(di);
            }

            static
            void
            extract_words_vi2(vi2_type& low_word, vi2_type& high_word,
                              const vf_type& x) {
                vi2_type di=ocl::as<vi2_type>(x);
                low_word = di;
                high_word = di;
            }

            static
            vf_type
            combine_words(const vi_type& l, const vi_type& h) {
                vi2_type vi= combine_even_odd(l, h);
                vf_type r= ocl::as<vf_type>(vi);
                return r;
            }

            static
            vf_type
            combine_words_vi2(const vi2_type& l, const vi2_type& h) {
                vi2_type t= select_even_odd(l, h);
                vf_type r=ocl::as<vf_type>(t);
                return r;
            }

            static
            vf_type clear_low_word(const vf_type& d) {
                const uint64_t mu=0xffffffff00000000ULL;
                const bytes8 mf(mu);
                return d & mf.f64();
            }

            static
            vli_type as_vli(const vf_type& d) {
                vli_type r=ocl::as<vli_type>(d);
                return r;
            }

            static
            vf_type as_vf(const vli_type& l) {
                vf_type r=ocl::as<vf_type>(l);
                return r;
            }

            static
            vf_type cvt_i_to_f(const vi_type& i) {
                return ocl::cvt<vf_type>(i);
            }

            static
            vi_type cvt_f_to_i(const vf_type& f) {
                return ocl::cvt<vi_type>(f);
            }

            static
            vi2_type cvt_f_to_i2(const vf_type& f) {
#if 1
                vi_type t=ocl::cvt<vi_type>(f);
                vi2_type r=combine_even_odd(t, t);
#else
                // the number is 2^52+2^51
                vf_type fr=f + 0x1.8p52;
                vi2_type r=as<vi2_type>(fr);
                r=copy_even_to_odd(r);
#endif
                return r;
            }

            // including rounding towards zero
            static
            vi_type cvt_rz_f_to_i(const vf_type& f) {
                return ocl::cvt_rz<vi_type>(f);
            }
        };

        template <>
        struct func_traits<ocl::dvec<double>, ocl::dvec<int32_t> >
            : public dvec_func_traits_f64 {
        };

    }
}

namespace ocl {

    namespace test {
        void
        elements();

        dvec<double>
        device_func(const dvec<double>& x);

        void
        functions();
    }
}

/// Implementation section

void
ocl::test::elements()
{
    const size_t N=13;
    static const int tbl[N]={
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    };
    dvec<int> v_all(tbl, std::size(tbl));
    dump(v_all, "v_all");
    dvec<int> v_even=even_elements(v_all);
    dump(v_even, "v_even");
    dvec<int> v_odd=odd_elements(v_all);
    dump(v_odd, "v_odd");
    dvec<int> v_comb=combine_even_odd(v_even, v_odd);
    dump(v_comb, "v_comb");

    dvec<int> v_cp_even=copy_even_to_odd(v_all);
    dump(v_cp_even, "v_cp_even");
    dvec<int> v_cp_odd=copy_odd_to_even(v_all);
    dump(v_cp_odd, "v_cp_odd");

    static const float values[N]= {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f, 12.0f
    };
    std::cout << std::fixed << std::setprecision(4);
    dvec<float> v_perm0(values, std::size(values));
    dump(v_perm0, "v_perm1");
    dvec<float> v_perm1_r=permute(v_all, v_perm0);
    dump(v_perm1_r, "v_perm1_r");
    dvec<float> v_perm1=2.0*v_perm0;

    static const int idx2[N]={
        0, 1+N, 2, 3+N, 4, 5+N, 6, 7+N, 8+N, 9, 10+N, -11, 12+N
    };
    dvec<int> v_idx2(idx2, std::size(idx2));
    dump(v_idx2, "v_idx2");
    dvec<float> v_perm2_r=permute(v_idx2, v_perm0, v_perm1);
    dump(v_perm2_r, "v_perm2_r");

    dvec<int> v_all2=v_all + v_all;
    dvec<int> v_sel_even_odd=select_even_odd(v_all, v_all2);
    dump(v_sel_even_odd, "v_sel_even_odd");
}

ocl::dvec<double>
ocl::test::device_func(const dvec<double>& x)
{
#if 0
    using traits_t=cftal::math::func_traits<ocl::dvec<double>,
                                            ocl::dvec<int32_t> >;
    using func_t=cftal::math::elem_func<double, traits_t>;
    return func_t::cbrt(x);
#else
    return x;
#endif
}


int main()
{
    ocl::test::elements();
    return 0;
}
