#if !defined (__DEVICE_VECTOR_H__)
#define __DEVICE_VECTOR_H__ 1

#include <ocl/config.h>
#include <ocl/impl_be_data.h>

namespace ocl {

        namespace device {

                class vector_base {
                        // shared pointer to the backend data
                        std::shared_ptr<impl::be_data> _bed;
                        // backend buffer object
                        cl::Buffer _b;
                public:
                        // default constructor
                        vector_base();
                        // constructor, with size
                        vector_base(std::size_t s);
                        // constructor, copies s bytes from src to
                        // buffer
                        vector_base(std::size_t s, const char* src);
                        // return the size of the vector in bytes
                        std::size_t buffer_size() const;
                        // return the underlying opencl buffer
                        const cl::Buffer& buf() const;
                        // return the opencl backend information
                        std::shared_ptr<impl::be_data>& 
                        backend_data();
                        // return the opencl backend information
                        const std::shared_ptr<impl::be_data>&
                        backend_data() const;
                };
        }
}


inline
ocl::device::vector_base::vector_base() 
        : _bed(), _b() 
{
}


inline
ocl::device::vector_base::vector_base(std::size_t s)
        : _bed(impl::be_data::instance()),
          _b(_bed->c(), CL_MEM_READ_WRITE, s) 
{
}

inline
ocl::device::vector_base::vector_base(std::size_t s, const char* src)
        : _bed(impl::be_data::instance()),
          _b(_bed->c(), CL_MEM_READ_WRITE, s) 
{
        impl::queue& q= _bed->q();
        q.enqueueWriteBuffer(_b,
                             true,
                             0, s,
                             src,
                             nullptr,
                             nullptr);
}

inline
std::size_t 
ocl::device::vector_base::buffer_size() 
        const 
{
        std::size_t r(_b() != nullptr ?
                      _b.getInfo<CL_MEM_SIZE>(nullptr) : 0);
        return r;
}

inline
const cl::Buffer& 
ocl::device::vector_base::buf() 
        const 
{
        return _b;
}

inline 
std::shared_ptr<ocl::impl::be_data>& 
ocl::device::vector_base::backend_data() 
{
        return _bed;
}

inline
const std::shared_ptr<ocl::impl::be_data>&
ocl::device::vector_base::backend_data() 
        const 
{
        return _bed;
}

// Local variables:
// mode: c++
// end:
#endif // __DEVICE_VECTOR_H__
