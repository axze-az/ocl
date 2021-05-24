#if !defined (__OCL_TEST_TOOLS_H__)
#define __OCL_TEST_TOOLS_H__ 1

#include <ocl/config.h>
#include <ocl/dvec.h>
#include <cftal/lvec.h>

namespace ocl {

    namespace test {

        using cftal::lvec;

        template <typename _T>
        dvec<_T>
        make_vec(std::size_t s, const _T& v) {
            return dvec<_T>(s, v);
        }
        
        template <typename _T>
        void
        dump(const dvec<_T>& v, const std::string& pfx="") {
            std::vector<_T> vh(v);
            std::cout << &v << ' ' << pfx << " (" << vh.size()
                      << " elements)\n";
            for (std::size_t i=0; i<vh.size(); ++i) {
                std::cout << vh[i];
                if ((i&7)==7) {
                    std::cout << '\n';
                } else if (i+1 < vh.size()){
                    std::cout << ", ";
                }
            }
            if ((vh.size() & 7) != 0) {
                std::cout << '\n';
            }
        }

        template <typename _T>
        void
        dump(const cftal::lvec<_T>& vh, const std::string& pfx="") {
            std::cout << &vh << ' ' << pfx << " (" << vh.size()
                      << " elements)\n";
            for (std::size_t i=0; i<vh.size(); ++i) {
                std::cout << vh[i];
                if ((i&7)==7) {
                    std::cout << '\n';
                } else if (i+1 < vh.size()){
                    std::cout << ", ";
                }
            }
            if ((vh.size() & 7) != 0) {
                std::cout << '\n';
            }
        }
        
        // create a dvec<_T> from an lvec
        template <typename _T>
        dvec<_T>
        make_dvec(const lvec<_T>& s) {
            dvec<_T> r(s.size(), &s[0]);
            return r;
        }

        template <typename _T>
        lvec<_T>
        make_lvec(const dvec<_T>& d) {
            lvec<_T> dh(d.size());
            d.copy_to_host(&dh[0]);
            return dh;
        }
        
        // compare the results of a calculation between device and host
        template <typename _T>
        bool
        compare_d_h(const dvec<_T>& d, const lvec<_T> h) {
            lvec<_T> dh=make_lvec(d);
            typename lvec<_T>::mask_type cv=dh == h;
            return all_of(cv);
        }
    }
}

// local variables:
// mode: c++
// end:
#endif
