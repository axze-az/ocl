#include "ocl/be/types.h"

ocl::cl::mem_object::
mem_object(const mem_object &r)
    : _id(r._id)
{
    if (_id){
        auto cr=clRetainMemObject(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::mem_object::
mem_object(mem_object &&r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::mem_object&
ocl::cl::mem_object::
operator=(const mem_object &r)
{
    if(this != &r){
        if (_id){
            auto cr=clReleaseMemObject(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
        _id = r._id;
        if (_id){
            auto cr=clRetainMemObject(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
    }
    return *this;
}

ocl::cl::mem_object&
ocl::cl::mem_object::
operator=(mem_object&& r)
{
    if(_id){
        auto cr=clReleaseMemObject(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
    _id = r._id;
    r._id = 0;
    return *this;
}

ocl::cl::mem_object::
~mem_object()
{
    if (_id){
        auto cr=clReleaseMemObject(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::mem_object::
mem_object(cl_mem c, bool retain)
    : _id(c)
{
    if (_id && retain) {
        auto cr=clRetainMemObject(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

void
ocl::cl::mem_object::
info(cl_mem_info c, size_t s, void* res, size_t* rs)
    const
{
    int err=clGetMemObjectInfo(_id, c, s, res, rs);
    error::throw_on(err, __FILE__, __LINE__);
}
