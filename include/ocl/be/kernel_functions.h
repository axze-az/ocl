#if !defined (__OCL_BE_KERNEL_FUNCTIONS_H__)
#define __OCL_BE_KERNEL_FUNCTIONS_H__ 1

#include <ocl/config.h>
#include <string>
#include <set>

namespace ocl {
    namespace be {

        // set of strings for custom functions in kernels
        class kernel_functions {
            std::set<std::string> _s;
        public:
            kernel_functions();
            kernel_functions(const kernel_functions& r);
            kernel_functions(kernel_functions&& r);
            kernel_functions& operator=(const kernel_functions& r);
            kernel_functions& operator=(kernel_functions&& r);
            ~kernel_functions();
            bool
            insert(const std::string& fn);
            void
            clear();
        };

    }
}

// local variables:
// mode: c++
// end:
#endif

