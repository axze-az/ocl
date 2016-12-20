#if !defined (__OCL_IMPL_TYPE_2_NAME_H__)
#define __OCL_IMPL_TYPE_2_NAME_H__ 1

#include <ocl/config.h>
#include <cftal/vec.h>
#include <sstream>

namespace ocl {

    namespace impl {

        template <typename _T>
        struct type_2_name {
            static
            constexpr const char* v() {
                // static_assert(0, "specialization required");
                return "missing type_2_name";
            }
        };

        template <>
        struct type_2_name<std::int16_t> {
            static
            constexpr const char* v() {
                return "short";
            }
        };

        template <>
        struct type_2_name<std::uint16_t> {
            static
            constexpr const char* v() {
                return "ushort";
            }
        };

        template <>
        struct type_2_name<std::int32_t> {
            static
            constexpr const char* v() {
                return "int";
            }
        };

        template <>
        struct type_2_name<std::uint32_t> {
            static
            constexpr const char* v() {
                return "uint";
            }
        };

        template <>
        struct type_2_name<std::int64_t> {
            static
            constexpr const char* v() {
                return "long";
            }
        };

        template <>
        struct type_2_name<std::uint64_t> {
            static
            constexpr const char* v() {
                return "ulong";
            }
        };

        template <>
        struct type_2_name<float> {
            static
            constexpr const char* v() {
                return "float";
            }
        };

        template <>
        struct type_2_name<double> {
            static
            constexpr const char* v() {
                return "double";
            }
        };

        template <typename _T, std::size_t _N>
        struct type_2_name<cftal::vec<_T, _N> > {
            static_assert(_N <= 16, "invalid vector size for OpenCL");
            static
            const std::string v() {
                std::string t(type_2_name<_T>::v());
                std::ostringstream s;
                s << _N;
                return  t+s.str();
            }
        };
        
#if 1
        template <>
        struct type_2_name<cftal::v4f32> {
            static
            constexpr const char* v() {
                return "float4";
            }
        };

        template <>
        struct type_2_name<cftal::v8f32> {
            static
            constexpr const char* v() {
                return "float8";
            }
        };
#endif
    }
}

// Local variables:
// mode: c++
// end:
#endif
