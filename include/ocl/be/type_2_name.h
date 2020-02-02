#if !defined (__OCL_BE_TYPE_2_NAME_H__)
#define __OCL_BE_TYPE_2_NAME_H__ 1

#include <ocl/config.h>
#include <ocl/types.h>
#include <cftal/vec.h>
#include <string_view>
#include <typeinfo>

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

        template <>
        struct type_2_name<char> {
            static
            constexpr const char* v() {
                return "char";
            }
        };

        template <>
        struct type_2_name<signed char> {
            static
            constexpr const char* v() {
                return "char";
            }
        };

        template <>
        struct type_2_name<unsigned char> {
            static
            constexpr const char* v() {
                return "uchar";
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

        std::string
        type_2_name_vec_t(const char* tname, size_t n);
        std::string
        type_2_name_vec_t(const std::string_view& tname, size_t n);

        template <typename _T, std::size_t _N>
        struct type_2_name<cftal::vec<_T, _N> > {
            static_assert(_N <= 16, "invalid vector size for OpenCL");
            static
            std::string v() {
                return type_2_name(type_2_name<_T>::v(), _N);
            }
        };

    }
}

// Local variables:
// mode: c++
// end:
#endif
