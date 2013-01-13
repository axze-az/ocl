#include "ocl/ocl.h"

std::vector<ocl::ll::platform> 
ocl::ll::platform::get()
{
	cl_uint n;
        cl_int err(clGetPlatformIDs(0, nullptr, &n));
	check_err(err, "ocl::ll::platform::get clGetPlatformIDs");
	std::vector<cl_platform_id> ids(n);
        err = clGetPlatformIDs(n, &ids[0], nullptr);
	check_err(err, "ocl::ll::platform::get clGetPlatformIDs");
	std::vector<platform> r(ids.begin(), ids.end());
	return r;
}

std::vector<ocl::ll::device>
ocl::ll::platform::devices(cl_device_type t)
{
	cl_uint n;
        cl_int err(clGetDeviceIDs((*this)(), t, 0, nullptr, &n));
	check_err(err, "ocl::ll::platform::devices clGetPlatformIDs");
	std::vector<cl_device_id> ids(n);
	err = clGetDeviceIDs((*this)(), t, n, &ids[0], nullptr);
	check_err(err, "ocl::ll::platform::devices clGetPlatformIDs");
	std::vector<device> r(ids.begin(), ids.end());
	return r;
}
