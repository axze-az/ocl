#include "ocl/ocl.h"

ocl::ll::context::
context(const std::vector<device>& devs,
	cl_context_properties* props,
	void (CL_CALLBACK * notifyFptr)(const char *,
					const void *,
					std::size_t,
					void*),
	void* data)
	: base_type(nullptr)
{
        cl_int err;
	cl_context i(clCreateContext(
			     props, 
			     static_cast<cl_uint>(devs.size()),
			     &(devs.front()()),
			     notifyFptr, data, &err));
	check_err(err);
	h(i);
}

ocl::ll::context::
context(const device& dev,
	cl_context_properties* props,
	void (CL_CALLBACK * notifyFptr)(const char *,
					const void *,
					std::size_t,
					void*),
	void* data)
	: base_type(nullptr)
{
        cl_int err;
	cl_context i(clCreateContext(
			     props, 
			     1,
			     &dev(),
			     notifyFptr, data, &err));
	check_err(err);
	h(i);
}
