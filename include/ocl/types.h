#if !defined (__OCL_TYPES_H__)
#define __OCL_TYPES_H__ 1

#include <ocl/config.h>
#include <cstdint>

namespace ocl {

    using std::uint8_t;
    using std::uint16_t;
    using std::uint32_t;
    using std::uint64_t;

    using std::int8_t;
    using std::int16_t;
    using std::int32_t;
    using std::int64_t;

    using f32_t = float;
    using f64_t = double;

    using std::size_t;
    using std::ptrdiff_t;
    using ssize_t = ptrdiff_t;
}
// Local variables:
// mode: c++
// end:
#endif
