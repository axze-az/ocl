#include "ocl/be/kernel_cache.h"

ocl::be::scoped_lock::scoped_lock(mutex& m)
    : _mtx(m)
{
    _mtx.lock();
}

ocl::be::scoped_lock::~scoped_lock()
{
    _mtx.unlock();
}

ocl::be::kernel_cache::kernel_cache()
    : _kmap(), _mtx()
{
}

ocl::be::kernel_cache::iterator
ocl::be::kernel_cache::
find(const kernel_key& cookie)
{
    return _kmap.find(cookie);
}

void
ocl::be::kernel_cache::
erase(iterator f)
{
    _kmap.erase(f);
}

std::pair<ocl::be::kernel_cache::iterator, bool>
ocl::be::kernel_cache::
insert(const kernel_key& cookie, const kernel_handle& v)
{
    return _kmap.insert(std::make_pair(cookie, v));
}

void
ocl::be::kernel_cache::clear()
{
    _kmap.clear();
}

std::size_t
ocl::be::kernel_cache::size() const
{
    return _kmap.size();
}

