#include "ocl/be/kernel_functions.h"

ocl::be::kernel_functions::kernel_functions()
    : _s()
{
}

ocl::be::kernel_functions::kernel_functions(const kernel_functions& r)
    : _s(r._s)
{
}

ocl::be::kernel_functions::kernel_functions(kernel_functions&& r)
    : _s(std::move(r._s))
{
}

ocl::be::kernel_functions&
ocl::be::kernel_functions::operator=(const kernel_functions& r)
{
    if (&r != this) {
        _s = r._s;
    }
    return *this;
}

ocl::be::kernel_functions&
ocl::be::kernel_functions::operator=(kernel_functions&& r)
{
    _s = std::move(r._s);
    return *this;
}

ocl::be::kernel_functions::~kernel_functions()
{
}

bool
ocl::be::kernel_functions::insert(const std::string& fn)
{
    return _s.insert(fn).second;
}

