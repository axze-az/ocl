#if !defined (__OCL_BE_ARGUMENT_BUFFER_H__)
#define __OCL_BE_ARGUMENT_BUFFER_H__ 1

#include <ocl/config.h>
#include <vector>

namespace ocl {
    namespace be {

        class argument_buffer {
            std::vector<char> _v;
        public:
            argument_buffer() : _v() {
                _v.reserve(4096);
            };
            // allow access to the stored data
            const char* data() const { return _v.data(); }
            // amount of data
            size_t size() const { return _v.size(); }
            // clear the buffer
            void clear() { _v.clear(); }
            // insert an argument into the buffer
            template <typename _T>
            void
            insert(const _T& t) {
                constexpr size_t st=sizeof(_T);
                constexpr size_t at=alignof(_T);
                constexpr size_t atm1=at-1;
                static_assert((at & atm1) == 0,
                              "type with non power of 2 alignment?");
                const size_t s=_v.size();
                // how many bytes are used from the last alignment?
                const size_t m=s&atm1;
                // const size_t pad = m ? at - m : 0;
                const size_t pad= (at - m) & atm1;
                const size_t ns=s+pad;
                const size_t nn=ns+st;
                _v.resize(nn, char(0xff));
                char* cd=_v.data() + ns;
                _T* d=reinterpret_cast<_T*>(cd);
                *d = t;
            }
        };
    }
}

// Local variables:
// mode: c++
// end:
#endif
