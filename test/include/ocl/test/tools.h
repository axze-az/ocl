#if !defined (__OCL_TEST_TOOLS_H__)
#define __OCL_TEST_TOOLS_H__ 1

#include <ocl/config.h>
#include <ocl/dvec.h>
#include <cftal/vsvec.h>

namespace ocl {

    namespace test {

        using cftal::vsvec;

        template <typename _T>
        dvec<_T>
        make_vec(std::size_t s, const _T& v) {
            return dvec<_T>(s, v);
        }

        template <typename _C>
        std::ostream&
        dump(std::ostream& s, const _C& vh, const std::string& pfx="") {
            using std::size;
            using std::cbegin;
            using std::cend;
            std::size_t n=size(vh);
            s << &vh << ' ' << pfx << " (" << size(vh)
              << " elements)\n";
            std::size_t i=0;
            for (auto b=cbegin(vh), e=cend(vh); b!=e; ++b, ++i) {
                s << *b;
                if ((i&7)==7) {
                    s << '\n';
                } else if (i+1 < n){
                    s << ", ";
                }
            }
            if ((n & 7) != 0) {
                s << '\n';
            }
            return s;
        }

        template <typename _T>
        std::ostream&
        dump(std::ostream& s, const dvec<_T>& v, const std::string& pfx="") {
            std::vector<_T> vh(v);
            return dump(s, vh, pfx);
        }

        template <typename _C>
        void
        dump(const _C& vh, const std::string& pfx="") {
            dump(std::cout, vh, pfx);
        }

#if 0
        template <typename _T>
        void
        dump(const cftal::vsvec<_T>& vh, const std::string& pfx="") {
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
#endif

        // create a dvec<_T> from an vsvec
        template <typename _T>
        dvec<_T>
        make_dvec(const vsvec<_T>& s) {
            dvec<_T> r(s.size(), &s[0]);
            return r;
        }

        template <typename _T>
        vsvec<_T>
        make_vsvec(const dvec<_T>& d) {
            vsvec<_T> dh(d.size());
            d.copy_to_host(&dh[0]);
            return dh;
        }

        // compare the results of a calculation between device and host
        template <typename _T>
        bool
        compare_d_h_on_host(const dvec<_T>& d, const vsvec<_T> h) {
            vsvec<_T> dh=make_vsvec(d);
            typename vsvec<_T>::mask_type cv=dh == h;
            return all_of(cv);
        }

        template <typename _T>
        bool
        compare_d_h_on_device(const dvec<_T>& d, const vsvec<_T> h) {
            dvec<_T> hd=make_dvvec(d);
            typename vsvec<_T>::mask_type cv=d == hd;
            return all_of(cv);
        }
    }
}

// local variables:
// mode: c++
// end:
#endif
