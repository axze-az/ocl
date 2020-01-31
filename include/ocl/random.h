#if !defined (__OCL_RANDOM_H__)
#define __OCL_RANDOM_H__ 1

#include <ocl/config.h>
#include <ocl/dvec.h>

namespace ocl {

#if 0
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
#endif

    class random_base {
    protected:
        static
        dvec<std::uint32_t>
        fill_with_global_id(std::size_t s, be::data_ptr p);
    };
    
    class rand : private random_base {
        dvec<std::uint32_t> _next;
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

        void
        seed_times_global_id(uint32_t n);

        std::size_t size() const { return _next.size(); }
        // return a vector with random contents
        dvec<std::int32_t>
        next();
        // return a float vector with random contents
        // in [0, 1.0f)
        dvec<float>
        nextf();
    };


    class rand48 : private random_base {
        dvec<std::uint64_t> _state;
        static const std::uint64_t A;
        static const std::uint64_t C;
        static const std::uint64_t M;
        static const std::uint64_t MM;
        void
        next();
    public:
        rand48(size_t s, be::data_ptr p= be::data::instance());
        // returns non negative numbers between 0 and 2^31
        dvec<std::int32_t>
        lrand48();
        // returns numbers between -2^31 and 2^31
        dvec<std::int32_t>
        mrand48();
        // returns floating point values in interval [0, 1.0)
        dvec<float>
        drand48();
        void
        seed(const dvec<uint64_t>& gid);
        dvec<float>
        nextf();
    };
    
    dvec<float>
    uniform_float_random_vector(rand& rnd,
                                float min_val, float max_val);
}


// Local variables:
// mode: c++
// end:
#endif // __OCL_RANDOM_H__
