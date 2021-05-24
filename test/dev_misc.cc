#include "ocl/ocl.h"
#include "ocl/test/tools.h"
#include <cftal/math/func_traits_f32_s32.h>
#include <cftal/math/func_traits_f64_s32.h>
#include <cftal/d_real.h>

namespace cftal {
    
    template <>
    struct d_real_traits<ocl::dvec<double> > {
        using cmp_result_type = typename ocl::dvec<double>::mask_type;
        using int_type = ocl::dvec<int32_t>;

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
        split(const ocl::dvec<double> & a,
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

#if 0
            static
            vmi2_type
            vmf_to_vmi2(const vmf_type& mf) {
                return 
                    cvt_mask<typename vmi2_type::value_type, 2 * _N,
                             typename vmf_type::value_type, _N>::v(mf);
            };

            static
            vmf_type
            vmi2_to_vmf(const vmi2_type& mf) {
                return
                    cvt_mask<typename vmf_type::value_type, _N,
                             typename vmi2_type::value_type, 2*_N>::v(mf);
            };
#endif            

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
            vf_type sel_val_or_zero(const vmf_type& msk,
                                    const vf_type& t) {
                return select(msk, t, 0);
            }

            static
            vf_type sel_zero_or_val(const vmf_type& msk,
                                    const vf_type& f) {
                return select(msk, 0, f);
            }

            static
            vli_type sel(const vmli_type& msk,
                         const vli_type& t, const vli_type& f) {
                return select(msk, t, f);
            }

            static
            vf_type insert_exp(const vi2_type& e) {
                vi2_type ep(e << 20);
                vf_type r= ocl::as<vf_type>(ep);
                r &= vf_type(exp_f64_msk::v.f64());
                return r;
            }

#if 0            
            static
            vi2_type vi_to_vi2(const vi_type& r) {
                vi2_type t=ocl::combine_even_odd(r, r);
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
                vec<int32_t, _N*2> ir(combine_zeroeven_odd(ep));
                vf_type r= as<vf_type>(ir);
                // r &= vf_type(exp_f64_msk::v.f64());
                return r;
            }

            static
            vi_type extract_exp(const vf_type& d) {
                const vf_type msk(exp_f64_msk::v.f64());
                vf_type m(d & msk);
                vec<int32_t, _N*2> di= as<vec<int32_t, _N*2> >(m);
                vi_type r= odd_elements(di);
                r >>= 20;
                return r;
            }

            static
            vi_type extract_high_word(const vf_type& d) {
                vec<int32_t, _N*2> di=as<vec<int32_t, _N*2> >(d);
                return odd_elements(di);
            }

            static
            vi_type extract_low_word(const vf_type& d) {
                vec<int32_t, _N*2> di=as<vec<int32_t, _N*2> >(d);
                return even_elements(di);
            }

            static
            void
            extract_words(vi_type& low_word, vi_type& high_word,
                          const vf_type& d) {
                vec<int32_t, _N*2> di=as<vec<int32_t, _N*2> >(d);
                low_word=even_elements(di);
                high_word=odd_elements(di);
            }
#endif
            static
            void
            extract_words(vi2_type& low_word, vi2_type& high_word,
                          const vf_type& x) {
                vi2_type di=ocl::as<vi2_type>(x);
                low_word = di;
                high_word = di;
            }

#if 0
            static
            vf_type
            combine_words(const vi_type& l, const vi_type& h) {
                ocl::dvec<int32_t> vi= combine_even_odd(l, h);
                vf_type r= ocl::as<vf_type>(vi);
                return r;
            }

            static
            vf_type
            combine_words(const vi2_type& l, const vi2_type& h) {
                vi2_type t= select_even_odd(l, h);
                vf_type r=ocl::as<vf_type>(t);
                return r;
            }
#endif
            static
            vf_type clear_low_word(const vf_type& d) {
                const uint64_t mu=0xffffffff00000000ULL;
                const bytes8 mf(mu);
                const vf_type m(mf.f64());
                return d & m;
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

#if 0            
            static
            vi2_type cvt_f_to_i2(const vf_type& f) {
#if 0
                vi_type t=ocl::cvt<vi_type>(f);
                vi2_type r=combine_even_odd(t, t);
#else
                // the number is 2^52+2^51
                vf_type fr=f + 0x1.8p52;
                vi2_type r=ocl::as<vi2_type>(fr);
                r=copy_even_to_odd(r);
#endif
                return r;
            }
#endif
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

    namespace impl {
        __ck_body
        even_elements(const std::string_view& tname);
    }
    
    // returns only the even elements of s
    template <typename _T>
    dvec<_T>
    even_elements(const dvec<_T>& s);
    
    namespace impl {
        __ck_body
        odd_elements(const std::string_view& tname);
    }
    
    // returns only the odd elements of s
    template <typename _T>
    dvec<_T>
    odd_elements(const dvec<_T>& s);
        
    // return the interleaved elemnent of e and o
    template <typename _T>
    dvec<_T>
    combine_even_odd(const dvec<_T>& e, const dvec<_T>& o);
    
    
    // copy the even elements of s to the odd elements
    template <typename _T>
    dvec<_T>
    copy_even_to_odd(const dvec<_T>& s);
    
    // copy the odd elements of s to the even elements
    template <typename _T>
    dvec<_T>
    copy_odd_to_even(const dvec<_T>& s);      
    
    namespace test {
        void
        elements();
    }
}

ocl::impl::__ck_body
ocl::impl::even_elements(const std::string_view& tname)
{
    std::ostringstream s;
    s << "even_elements_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
         "ulong n,\n"
         "__global " << tname << "* res,\n"
         "__global const " << tname << "* src\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        ulong sgid=2*gid;\n"
        "        res[gid] = src[sgid];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);    
}

template <typename _T>
ocl::dvec<_T>
ocl::even_elements(const dvec<_T>& s)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::even_elements(tname);
    size_t n=(s.size() +1) >> 1;
    dvec<_T> r=custom_kernel_with_size<_T>(ckb.name(), ckb.body(), n, s);
    return r;
}

ocl::impl::__ck_body
ocl::impl::odd_elements(const std::string_view& tname)
{
    std::ostringstream s;
    s << "odd_elements_"  << tname;
    const std::string kname = s.str();
    s.str("");
    s << "void " << kname << "(\n"
         "ulong n,\n"
         "__global " << tname << "* res,\n"
         "__global const " << tname << "* src\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        ulong sgid=2*gid+1;\n"
        "        res[gid] = src[sgid];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);    
}

template <typename _T>
ocl::dvec<_T>
ocl::odd_elements(const dvec<_T>& s)
{
    const auto tname=be::type_2_name<_T>::v();
    impl::__ck_body ckb=impl::odd_elements(tname);
    size_t n=s.size() >> 1;
    dvec<_T> r=custom_kernel_with_size<_T>(ckb.name(), ckb.body(), n, s);
    return r;
}

void
ocl::test::elements()
{
    static const int tbl[]={
        0, 1, 0, 1, 0, 1, 0, 1, 0
    };
    dvec<int> v_all(std::size(tbl), tbl);
    dump(v_all, "v_all");
    dvec<int> v_even=even_elements(v_all);
    dump(v_even, "v_even");
    dvec<int> v_odd=odd_elements(v_all);
    dump(v_odd, "v_odd");
}


int main()
{
    ocl::test::elements();
    return 0;
}
