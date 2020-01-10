#if !defined (__OCL_BE_KERNEL_KEY_H__)
#define __OCL_BE_KERNEL_KEY_H__ 1

#include <ocl/config.h>
#include <string>
#include <iosfwd>

namespace ocl {

    namespace be {

        // kernel_key: unique identification of a kernel
        class kernel_key {
            const void* _p;
            std::string _s;
        public:
            kernel_key(const void* p, const std::string& s);
            kernel_key(const kernel_key& r);
            kernel_key(kernel_key&& r);
            kernel_key& operator=(const kernel_key& r);
            kernel_key& operator=(kernel_key&& r);
            ~kernel_key();
            intptr_t h() const { return intptr_t(_p); }
            const std::string& l() const { return _s; }
        };

        bool operator<(const kernel_key& a, const kernel_key& b);
        bool operator<=(const kernel_key& a, const kernel_key& b);
        bool operator==(const kernel_key& a, const kernel_key& b);
        bool operator!=(const kernel_key& a, const kernel_key& b);
        bool operator>=(const kernel_key& a, const kernel_key& b);
        bool operator>(const kernel_key& a, const kernel_key& b);

        struct print_kernel_key {
            const kernel_key& _k;
            print_kernel_key(const kernel_key& k) : _k(k) {}
        };
        
        std::ostream&
        operator<<(std::ostream& s, const print_kernel_key& kk);
    }
    
}

// Local variables:
// mode: c++
// end:
#endif
