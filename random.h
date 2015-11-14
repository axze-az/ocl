#if !defined (__OCL_RANDOM_H__)
#define __OCL_RANDOM_H__ 1

#include <ocl/config.h>
#include <ocl/vector.h>

namespace ocl {

    class srand {
        vector<std::uint32_t> _next;
    public:

        constexpr static const std::uint32_t max_val = 0x7fff;
        constexpr static const float _R= (1.0f/32768.f);
        
        srand() : _next() {}
        srand(const vector<uint32_t>& gid) : _next{gid} {}
        void 
        seed(const vector<uint64_t>& gid){
            _next = cvt_to<vector<uint32_t> >(gid);
        }
        void 
        seed(const vector<uint32_t>& gid){
            _next = gid;
        }
        // vector<uint32_t>
        auto
        next() {
            _next = (_next * 1103515245 + 12345);
            return (_next>>16) & max_val;
        }
        auto
        nextf() {
            return (cvt_to<vector<float>>(next()) * _R);
        }
    };
    
}


// Local variables:
// mode: c++
// end:
#endif // __OCL_RANDOM_H__ 
