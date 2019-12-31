#if !defined (__OCL_DVEC_BASE_H__)
#define __OCL_DVEC_BASE_H__ 1

#include <ocl/config.h>
#include <ocl/be/data.h>

namespace ocl {

    // dvec base class wrapping an opencl buffer and a
    // (shared) pointer to opencl backend data
    class dvec_base {
        // shared pointer to the backend data
        be::data_ptr _bed;
        // backend buffer object
        be::buffer _b;
    protected:
        // destructor
        ~dvec_base();
        // default constructor
        dvec_base();
        // constructor, with size
        explicit dvec_base(std::size_t s);
        // constructor, with size and source
        dvec_base(std::size_t s, const void* p);
        // constructor with backend data ptr
        dvec_base(be::data_ptr pbe, std::size_t s);
        // constructor with backend data ptr, size and source
        dvec_base(be::data_ptr pbe, std::size_t s, const void* p);
        // copy constructor
        dvec_base(const dvec_base& r);
        // move constructor
        dvec_base(dvec_base&& r);
        // assignment operator
        dvec_base& operator=(const dvec_base& r);
        // move assignment operator
        dvec_base& operator=(dvec_base&& r);
        // swap two dvec base objects
        dvec_base& swap(dvec_base& r);
        // fill pattern p with pattern length into this (OPENCL 1.2)
        // void fill_on_device(const void* p, size_t ps);
        // device device copy
        void copy_on_device(const dvec_base& r);
        // host device copy
        void copy_from_host(const void* src);
        // device host copy
        void copy_to_host(void* dst)
            const;
    public:
        // return the size of the dvec in bytes
        std::size_t buffer_size() const;
        // return the underlying opencl buffer
        const be::buffer& buf() const;
        // return the opencl backend information
        be::data_ptr
        backend_data();
        // return the opencl backend information
        const be::data_ptr
        backend_data() const;
    };
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_DVEC_BASE_H__
