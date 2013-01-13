#include "ocl/ocl.h"

ocl::ll::buffer::buffer(const context& ctx,
			cl_mem_flags flags,
			std::size_t size,
			void* host_ptr)
	: base_type()
{
        cl_int err;
        cl_mem m(clCreateBuffer(ctx(), flags, size, host_ptr, &err));
	check_err(err);
	h(m);
}


