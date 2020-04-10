#include "ocl/be/types.h"

ocl::cl::context::
context(const context &r)
    : _id(r._id)
{
    if (_id){
        auto cr=clRetainContext(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::context::
context(context &&r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::context&
ocl::cl::context::
operator=(const context &r)
{
    if(this != &r){
        if (_id){
            auto cr=clReleaseContext(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
        _id = r._id;
        if (_id){
            auto cr=clRetainContext(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
    }
    return *this;
}

ocl::cl::context&
ocl::cl::context::
operator=(context&& r)
{
    if(_id){
        auto cr=clReleaseContext(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
    _id = r._id;
    r._id = 0;
    return *this;
}

ocl::cl::context::
~context()
{
    if (_id){
        auto cr=clReleaseContext(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::context::
context(cl_context c, bool retain)
    : _id(c)
{
    if (_id && retain) {
        auto cr=clRetainContext(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::context::
context(const device& d, const cl_context_properties* p)
{
    cl_device_id device_id = d();
    cl_int err = 0;
    _id = clCreateContext(p, 1, &device_id, 0, 0, &err);
    if (_id == 0) {
        error::throw_on(err, __FILE__, __LINE__);
    }
}

ocl::cl::context::
context(const std::vector<device>& vd, const cl_context_properties* p)
{
    cl_int err = 0;
    _id = clCreateContext(p, vd.size(), 
                          reinterpret_cast<const cl_device_id*>(&vd[0]), 
                          0, 0, &err);
    if (_id == 0) {
        error::throw_on(err, __FILE__, __LINE__);
    }
}

void
ocl::cl::context::
info(cl_context_info id, size_t res_size, void* res, size_t* ret_res)
    const
{
    auto err=clGetContextInfo(_id, id, res_size, res, ret_res);
    error::throw_on(err, __FILE__, __LINE__);
}

ocl::cl::device
ocl::cl::context::
get_device() 
    const
{
    uint32_t n;
    info(CL_CONTEXT_NUM_DEVICES, sizeof(n), &n, nullptr);
    if (n==0)
        return device();
    cl_device_id* vi=
        static_cast<cl_device_id*>(alloca(n*sizeof(cl_device_id)));
    info(CL_CONTEXT_DEVICES, n*sizeof(cl_device_id), vi, nullptr);
    return device(vi[0]);
}

std::vector<ocl::cl::device>
ocl::cl::context::
get_devices() 
    const
{
    uint32_t n;
    info(CL_CONTEXT_NUM_DEVICES, sizeof(n), &n, nullptr);
    std::vector<device> vd;
    if (n) {
        cl_device_id* vi=
            static_cast<cl_device_id*>(alloca(n*sizeof(cl_device_id)));
        info(CL_CONTEXT_DEVICES, n*sizeof(cl_device_id), vi, nullptr);
        for (size_t i=0; i<n; ++i) {
            vd.emplace_back(vi[i]);
        }
    }
    return vd;
}

