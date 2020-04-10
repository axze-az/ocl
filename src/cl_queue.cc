#include "ocl/be/types.h"

ocl::cl::queue::
queue(const queue &r)
    : _id(r._id)
{
    if (_id){
        auto cr=clRetainCommandQueue(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::queue::
queue(queue &&r)
    : _id(r._id)
{
    r._id = 0;
}

ocl::cl::queue&
ocl::cl::queue::
operator=(const queue &r)
{
    if(this != &r){
        if (_id){
            auto cr=clReleaseCommandQueue(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
        _id = r._id;
        if (_id){
            auto cr=clRetainCommandQueue(_id);
            error::throw_on(cr, __FILE__, __LINE__);
        }
    }
    return *this;
}

ocl::cl::queue&
ocl::cl::queue::
operator=(queue&& r)
{
    if(_id){
        auto cr=clReleaseCommandQueue(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
    _id = r._id;
    r._id = 0;
    return *this;
}

ocl::cl::queue::
~queue()
{
    if (_id){
        auto cr=clReleaseCommandQueue(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::queue::
queue(cl_command_queue k, bool retain)
    : _id(k)
{
    if (_id && retain) {
        auto cr=clRetainCommandQueue(_id);
        error::throw_on(cr, __FILE__, __LINE__);
    }
}

ocl::cl::queue::
queue(const context& c, const device& d,
      cl_command_queue_properties p)
{
    cl_int err = 0;
    _id = clCreateCommandQueue(c(), d(), p, &err);
    if (!_id) {
        error::throw_on(err, __FILE__, __LINE__);
    }
}

ocl::cl::event
ocl::cl::queue::
enqueue_copy_buffer(const buffer& src_buffer,
                    const buffer& dst_buffer,
                    size_t src_offset,
                    size_t dst_offset,
                    size_t size,
                    const wait_list& evs)
{
    event e;
    cl_int err=clEnqueueCopyBuffer(_id,
                                   src_buffer(), dst_buffer(),
                                   src_offset, dst_offset,
                                   size,
                                   evs.size(),
                                   evs.get_event_ptr(),
                                   &e());
    error::throw_on(err, __FILE__, __LINE__);
    return e;
}

ocl::cl::event
ocl::cl::queue::
enqueue_read_buffer(const buffer &buffer,
                    size_t offset,
                    size_t size,
                    void *host_ptr,
                    const wait_list& evs)
{
    event e;
    cl_int err=clEnqueueReadBuffer(_id, buffer(), CL_TRUE,
                                   offset, size, host_ptr,
                                   evs.size(), evs.get_event_ptr(),
                                   &e());
    error::throw_on(err, __FILE__, __LINE__);
    return e;
}

ocl::cl::event
ocl::cl::queue::
enqueue_read_buffer_async(const buffer &buffer,
                          size_t offset,
                          size_t size,
                          void *host_ptr,
                          const wait_list& evs)
{
    event e;
    cl_int err=clEnqueueReadBuffer(_id, buffer(), CL_FALSE,
                                   offset, size, host_ptr,
                                   evs.size(), evs.get_event_ptr(),
                                   &e());
    error::throw_on(err, __FILE__, __LINE__);
    return e;
}

ocl::cl::event
ocl::cl::queue::
enqueue_write_buffer(const buffer &buffer,
                     size_t offset,
                     size_t size,
                     const void *host_ptr,
                     const wait_list& evs)
{
    event e;
    cl_int err = clEnqueueWriteBuffer(_id, buffer(), CL_TRUE,
                                      offset, size, host_ptr,
                                      evs.size(), evs.get_event_ptr(),
                                      &e());
    error::throw_on(err, __FILE__, __LINE__);
    return e;
}

ocl::cl::event
ocl::cl::queue::
enqueue_write_buffer_async(const buffer &buffer,
                           size_t offset,
                           size_t size,
                           const void *host_ptr,
                           const wait_list& evs)
{
    event e;
    cl_int err = clEnqueueWriteBuffer(_id, buffer(), CL_FALSE,
                                      offset, size, host_ptr,
                                      evs.size(), evs.get_event_ptr(),
                                      &e());
    error::throw_on(err, __FILE__, __LINE__);
    return e;
}

ocl::cl::event
ocl::cl::queue::
enqueue_nd_range_kernel(const kernel &kernel,
                        size_t work_dim,
                        const size_t* global_work_offset,
                        const size_t* global_work_size,
                        const size_t* local_work_size,
                        const wait_list& evs)
{
    event e;
    cl_int err = clEnqueueNDRangeKernel(_id, kernel(),
                                        static_cast<cl_uint>(work_dim),
                                        global_work_offset,
                                        global_work_size,
                                        local_work_size,
                                        evs.size(), evs.get_event_ptr(),
                                        &e());
    error::throw_on(err, __FILE__, __LINE__);
    return e;
}

void
ocl::cl::queue::
flush()
{
    cl_int err=clFlush(_id);
    error::throw_on(err, __FILE__, __LINE__);
}

void
ocl::cl::queue::
finish()
{
    cl_int err=clFinish(_id);
    error::throw_on(err, __FILE__, __LINE__);
}
