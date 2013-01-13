#include "ocl/ocl.h"

ocl::ll::queue::queue(const context& ctx,
		      const device& dev,
		      cl_command_queue_properties props)
	: base_type(nullptr)
{
        cl_int err;
	cl_command_queue q(
		clCreateCommandQueue(ctx(), dev(), props, &err));
	check_err(err, "ocl::ll::queue::queue clCreateCommandQueue");
	h(q);
}

