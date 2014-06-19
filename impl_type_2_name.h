#if !defined (__OCL_TYPE_2_NAME_H__)
#define __OCL_TYPE_2_NAME_H__ 1

#include <ocl/config.h>
#include <cftal/vec.h>

namespace ocl {

        namespace impl {

                template <typename _T>
                struct type_2_name {
                        static
                        constexpr const char* v();
                };

                template <>
                struct type_2_name<std::int16_t> {
                        static
                        constexpr const char* v() {
                                return "short int";
                        }
                };

                template <>
                struct type_2_name<std::uint16_t> {
                        static
                        constexpr const char* v() {
                                return "unsigned short int";
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
                                return "unsigned int";
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
                                return "unsigned long";
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

                template <>
                struct type_2_name<cftal::vec::v4f32> {
                        static 
                        constexpr const char* v() {
                                return "float4";
                        }
                };

                template <>
                struct type_2_name<cftal::vec::v8f32> {
                        static 
                        constexpr const char* v() {
                                return "float8";
                        }
                };

        }
}

// Local variables:
// mode: c++
// end:
#endif
