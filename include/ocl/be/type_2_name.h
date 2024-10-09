#if !defined (__OCL_BE_TYPE_2_NAME_H__)
#define __OCL_BE_TYPE_2_NAME_H__ 1

#include <ocl/config.h>
#include <ocl/types.h>
#include <cftal/vec.h>
#include <string_view>
#include <typeinfo>
#include <ocl/be/types.h>

namespace ocl {

    namespace be {

        std::string
        demangle(const char* mangled_name);

        template <typename _T>
        struct type_2_name {
            static
            std::string v() {
                // static_assert(0, "specialization required");
                return demangle(typeid(_T).name());
            }
        };

#define TYPE_2_NAME(type, name) \
    template <>      \
    struct type_2_name<type> { \
        static constexpr const char* v() { return #name ; } \
    }

        TYPE_2_NAME(cl_ulong, ulong);
        TYPE_2_NAME(cl_ulong2, ulong2);
        TYPE_2_NAME(cl_ulong4, ulong4);
        TYPE_2_NAME(cl_ulong8, ulong8);
        TYPE_2_NAME(cl_ulong16, ulong16);

        TYPE_2_NAME(cl_long, ulong);
        TYPE_2_NAME(cl_long2, long2);
        TYPE_2_NAME(cl_long4, long4);
        TYPE_2_NAME(cl_long8, long8);
        TYPE_2_NAME(cl_long16, long16);

        TYPE_2_NAME(cl_uint, uint);
        TYPE_2_NAME(cl_uint2, uint2);
        TYPE_2_NAME(cl_uint4, uint4);
        TYPE_2_NAME(cl_uint8, uint8);
        TYPE_2_NAME(cl_uint16, uint16);

        TYPE_2_NAME(cl_int, int);
        TYPE_2_NAME(cl_int2, int2);
        TYPE_2_NAME(cl_int4, int4);
        TYPE_2_NAME(cl_int8, int8);
        TYPE_2_NAME(cl_int16, int16);

        TYPE_2_NAME(cl_ushort, ushort);
        TYPE_2_NAME(cl_ushort2, ushort2);
        TYPE_2_NAME(cl_ushort4, ushort4);
        TYPE_2_NAME(cl_ushort8, ushort8);
        TYPE_2_NAME(cl_ushort16, ushort16);

        TYPE_2_NAME(cl_short, short);
        TYPE_2_NAME(cl_short2, short2);
        TYPE_2_NAME(cl_short4, short4);
        TYPE_2_NAME(cl_short8, short8);
        TYPE_2_NAME(cl_short16, short16);

        TYPE_2_NAME(cl_char, char);
        TYPE_2_NAME(cl_char2, char2);
        TYPE_2_NAME(cl_char4, char4);
        TYPE_2_NAME(cl_char8, char8);
        TYPE_2_NAME(cl_char16, char16);

        TYPE_2_NAME(cl_uchar, uchar);
        TYPE_2_NAME(cl_uchar2, uchar2);
        TYPE_2_NAME(cl_uchar4, uchar4);
        TYPE_2_NAME(cl_uchar8, uchar8);
        TYPE_2_NAME(cl_uchar16, uchar16);

        template <>
        struct type_2_name<char> {
            static
            constexpr const char* v() {
                return "char";
            }
        };

        TYPE_2_NAME(cl_float, float);
        TYPE_2_NAME(cl_float2, float2);
        TYPE_2_NAME(cl_float4, float4);
        TYPE_2_NAME(cl_float8, float8);
        TYPE_2_NAME(cl_float16, float16);

        TYPE_2_NAME(cl_double, double);
        TYPE_2_NAME(cl_double2, double2);
        TYPE_2_NAME(cl_double4, double4);
        TYPE_2_NAME(cl_double8, double8);
        TYPE_2_NAME(cl_double16, double16);

#if 0
        TYPE_2_NAME(cl_half, half);
        TYPE_2_NAME(cl_half2, half2);
        TYPE_2_NAME(cl_half4, half4);
        TYPE_2_NAME(cl_half8, half8);
        TYPE_2_NAME(cl_half16, half16);
#endif

        std::string
        type_2_name_vec_t(const char* tname, size_t n);
        std::string
        type_2_name_vec_t(const std::string_view& tname, size_t n);

        template <typename _T, std::size_t _N>
        struct type_2_name<cftal::vec<_T, _N> > {
            static_assert(_N <= 16, "invalid vector size for OpenCL");
            static
            std::string v() {
                return type_2_name_vec_t(type_2_name<_T>::v(), _N);
            }
        };

    }
}

// Local variables:
// mode: c++
// end:
#endif
