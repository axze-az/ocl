#if !defined (__OCL_TEST_TOOLS_H__)
#define __OCL_TEST_TOOLS_H__ 1

#include <ocl/config.h>
#include <ocl/dvec.h>

namespace ocl {

    namespace test {
        template <typename _T>
        dvec<_T>
        make_vec(std::size_t s, const _T& v) {
            return dvec<_T>(s, v);
        }

        template <typename _T>
        void
        dump(const dvec<_T>& v, const std::string& pfx="") {
            std::cout << &v << ' ' << pfx << '\n';
            std::vector<_T> vh(v);
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

    }
}

// local variables:
// mode: c++
// end:
#endif
