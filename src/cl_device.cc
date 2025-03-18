#include "ocl/be/types.h"
#include <algorithm>

ocl::cl::
device::device(cl_device_id id, bool retain)
    : _id(id)
{
#if CL_TARGET_OPENCL_VERSION>=120
    if (_id && retain && is_subdevice()) {
        auto cr=clRetainDevice(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
#else
    static_cast<void>(retain);
#endif
}

ocl::cl::
device::device(const device& r)
    : _id(r._id)
{
#if CL_TARGET_OPENCL_VERSION>=120
    if (_id && is_subdevice()) {
        auto cr=clRetainDevice(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
#endif
}

ocl::cl::
device::device(device&& r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::device&
ocl::cl::device::
operator=(const device& r)
{
    if (this != &r) {
#if CL_TARGET_OPENCL_VERSION>=120
        if (_id && is_subdevice()) {
            auto cr=clReleaseDevice(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
#endif
        _id = r._id;
#if CL_TARGET_OPENCL_VERSION>=120
        if (_id && is_subdevice()) {
            auto cr=clRetainDevice(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
#endif
    }
    return *this;
}

ocl::cl::device&
ocl::cl::
device::operator=(device&& r)
{
#if CL_TARGET_OPENCL_VERSION>=120
    if (_id && is_subdevice()) {
        auto cr=clReleaseDevice(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
#endif
    _id = r._id;
    r._id =0;
    return *this;
}

ocl::cl::
device::~device()
{
#if CL_TARGET_OPENCL_VERSION>=120
    if (_id && is_subdevice()) {
        auto cr=clReleaseDevice(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
#endif
}

void
ocl::cl::device::
info(cl_device_info id, size_t res_size, void* res, size_t* ret_res)
    const
{
    auto cr=clGetDeviceInfo(_id, id, res_size, res, ret_res);
    error::throw_on(cr, __FILE__, __LINE__);
}

std::string
ocl::cl::device::
info(cl_device_info id)
    const
{
    size_t ret_res;
    info(id, 0, nullptr, &ret_res);
    // std::vector<char> s(ret_res);
    char* s=static_cast<char*>(alloca(ret_res));
    info(id, ret_res, &s[0], nullptr);
    return std::string(&s[0], ret_res-1);
}

bool
ocl::cl::
device::is_subdevice()
    const
{
#if CL_TARGET_OPENCL_VERSION >= 120
    cl_device_id parent_id;
    auto cr=clGetDeviceInfo(_id, CL_DEVICE_PARENT_DEVICE,
                            sizeof(cl_device_id), &parent_id, nullptr);
    if (cr == CL_SUCCESS)
        return parent_id != 0;
#endif
    return false;
}

cl_device_type
ocl::cl::device::
type() const
{
    return get_info<cl_device_type>(CL_DEVICE_TYPE);
}

std::string
ocl::cl::device::
name() const
{
    return info(CL_DEVICE_NAME);
}

std::string
ocl::cl::device::
vendor() const
{
    return info(CL_DEVICE_VENDOR);
}

std::string ocl::cl::device::
driver_version() const
{
    return info(CL_DRIVER_VERSION);
}

std::vector<std::string>
ocl::cl::device::
extensions() const
{
    std::string s=info(CL_DEVICE_EXTENSIONS);
    std::vector<std::string> r;
    size_t l=s.length();
    for (size_t i=0; i<l; ) {
        size_t n=s.find_first_of("\t ", i);
        if (n == std::string::npos)
            n=l;
        std::string ei=s.substr(i, n-i);
        r.emplace_back(ei);
        i=n+1;
    }
    return r;
}

bool
ocl::cl::device::
supports_extension(const std::string& name) const
{
    auto v=extensions();
    auto e=std::cend(v);
    return std::find(std::cbegin(v), e, name) != e;
}

uint64_t
ocl::cl::device::
global_memory_size() const
{
    return get_info<uint64_t>(CL_DEVICE_GLOBAL_MEM_SIZE);
}

uint64_t
ocl::cl::device::
local_memory_size() const
{
    return get_info<uint64_t>(CL_DEVICE_LOCAL_MEM_SIZE);
}

uint32_t
ocl::cl::device::
address_bits() const
{
    return get_info<uint32_t>(CL_DEVICE_ADDRESS_BITS);
}

uint32_t
ocl::cl::device::
compute_units() const
{
    return get_info<uint32_t>(CL_DEVICE_MAX_COMPUTE_UNITS);
}

uint32_t ocl::cl::device::
max_work_group_size() const
{
    return get_info<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

uint32_t
ocl::cl::device::
max_work_iterm_dimensions() const
{
    return get_info<uint32_t>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
}

ocl::cl::platform
ocl::cl::device::platform() const
{
    cl_platform_id id=get_info<cl_platform_id>(CL_DEVICE_PLATFORM);
    return ocl::cl::platform(id);
}

