#include "ocl/be/types.h"

cl_event
ocl::cl::user_event::create(const context& ctx)
{
    cl_int err;
    cl_event ev=clCreateUserEvent(ctx(), &err);
    error::throw_on(err, __FILE__, __LINE__);
    return ev;
}

ocl::cl::user_event::user_event(const context& ctx)
    : event(create(ctx))
{
}

void
ocl::cl::user_event::set_status(cl_int exec_status)
{
    cl_int err=clSetUserEventStatus((*this)(), exec_status);
    error::throw_on(err, __FILE__, __LINE__);
}
