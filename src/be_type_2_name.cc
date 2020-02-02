#include "ocl/be/type_2_name.h"
#include <cxxabi.h>
#include <memory>
#include <sstream>

std::string
ocl::be::demangle( const char* mangled_name )
{
    std::size_t len = 0 ;
    int status = 0 ;
    std::unique_ptr< char, decltype(&std::free) > ptr(
        __cxxabiv1::__cxa_demangle( mangled_name, nullptr, &len, &status ),
        &std::free ) ;
    return ptr.get() ;
}

std::string
ocl::be::
type_2_name_vec_t(const char* tname, size_t n)
{
    return type_2_name_vec_t(std::string_view(tname), n);
}

std::string
ocl::be::
type_2_name_vec_t(const std::string_view& tname, size_t n)
{
    std::ostringstream s;
    s << tname << n;
    return s.str();
}
