#if !defined (__OCL_RANDOM_H__)
#define __OCL_RANDOM_H__ 1

#include <ocl/config.h>
#include <ocl/dvec.h>

namespace ocl {

    class srand {
        dvec<std::uint32_t> _next;
    public:

        constexpr static const std::uint32_t max_val() { return 0x7fff; }
        constexpr static const float _R() { return 1.0f/32768.f; }
        
        srand() : _next() {}
        srand(const dvec<uint32_t>& gid) : _next{gid} {}
        void 
        seed(const dvec<uint64_t>& gid){
            _next = cvt<dvec<uint32_t> >(gid);
        }
        void 
        seed(const dvec<uint32_t>& gid){
            _next = gid;
        }
        // dvec<uint32_t>
        auto
        next() {
            _next = (_next * 1103515245 + 12345);
            return (_next>>17) /* & max_val()*/ ;
        }
        auto
        nextf() {
            return (cvt<dvec<float> >(next()) * _R());
        }
    };

    class rand {
        dvec<std::uint32_t> _next;

        static
        dvec<std::uint32_t>
        vgid(std::size_t s, be::data_ptr p);
    public:
        constexpr static const std::uint32_t max_val() {
            return 0x7fffffff;
        }
        rand(std::size_t s);
        rand(std::size_t s, be::data_ptr p);
        // set a new seed and size
        void
        seed(const dvec<uint32_t>& v) {
            _next = v;
        }
        std::size_t size() const { return _next.size(); }
        // return a vector with random contents
        dvec<std::int32_t>
        next();
        // return a float vector with random contents
        // in [0, 1.0f)
        dvec<float>
        nextf();
    };
    
}


// Local variables:
// mode: c++
// end:
#endif // __OCL_RANDOM_H__ 
