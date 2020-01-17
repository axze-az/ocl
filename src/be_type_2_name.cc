#include "ocl/be/type_2_name.h"
#include <cxxabi.h>
#include <memory>

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

