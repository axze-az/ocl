#include "ocl/ocl.h"

ocl::ll::buffer::buffer(const context& ctx,
			cl_mem_flags flags,
			std::size_t size,
			void* host_ptr)
{
        cl_int error;
	cl_mem m(
		clCreateBuffer(ctx(), flags, size, 
			       host_ptr, &error));
	check_err(error);
	h(m);
}



