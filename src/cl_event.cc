#include "ocl/be/types.h"

ocl::cl::event::
event(const event &r)
    : _id(r._id)
{
    if (_id){
        auto cr=clRetainEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::event::
event(event &&r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::event&
ocl::cl::event::
operator=(const event &r)
{
    if(this != &r){
        if (_id){
            auto cr=clReleaseEvent(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
        _id = r._id;
        if (_id){
            auto cr=clRetainEvent(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
    }
    return *this;
}

ocl::cl::event&
ocl::cl::event::
operator=(event&& r)
{
    if(_id){
        auto cr=clReleaseEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
    _id = r._id;
    r._id = 0;
    return *this;
}

ocl::cl::event::
~event()
{
    if (_id){
        auto cr=clReleaseEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::event::
event(cl_event c, bool retain)
    : _id(c)
{
    if (_id && retain) {
        auto cr=clRetainEvent(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

void
ocl::cl::event::
wait()
{
    cl_int ret = clWaitForEvents(1, &_id);
    error::throw_on(ret, __FILE__, __LINE__);
}

