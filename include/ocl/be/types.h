#if !defined (__OCL_BE_TYPES_H__)
#define __OCL_BE_TYPES_H__ 1

#include <ocl/config.h>
#include <ocl/types.h>

#define USE_BOOST_COMPUTE 1

#if USE_BOOST_COMPUTE>0
#define CL_TARGET_OPENCL_VERSION 120
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL 1
#define BOOST_COMPUTE_THREAD_SAFE 1
#define BOOST_DISABLE_ASSERTS 1
#include <boost/compute/core.hpp>
#endif

#include <CL/cl.h>
#include <stdexcept>
#include <vector>

namespace ocl {
    namespace cl {

        class error;
        class device;
        class context;
        class event;
        class wait_list;
        class mem_obj;
        class buffer;
        class queue;
        class program;
        class kernel;

        class error : public std::runtime_error {
            cl_int _code;
            using base_type = std::runtime_error;
            static
            std::string
            to_string(cl_int code);
            static
            std::string
            to_string(cl_int code, const char* file, unsigned line);
        public:
            error(cl_int code);
            error(cl_int code, const char* file, unsigned line);
            error(const error& r);
            error(error&& r);
            error& operator=(const error& r);
            error& operator=(error&& r);
            virtual ~error();
            // check code and throw if required
            static
            void
            throw_on(cl_int code);
            // check code and throw if required
            static
            void
            throw_on(cl_int code, const char* file, unsigned line);
        };

        class device {
            cl_device_id _id;
        public:
            enum type {
                cpu = CL_DEVICE_TYPE_CPU,
                gpu = CL_DEVICE_TYPE_GPU,
                accelerator = CL_DEVICE_TYPE_ACCELERATOR
            };
            device();
            device(cl_device_id, bool retain=true);
            device(const device& r);
            device(device&& r);
            device& operator=(const device& r);
            device& operator=(device&& r);
            ~device();
            cl_device_id& operator()() { return _id; }
            const cl_device_id& operator()() const { return _id;}
            bool
            is_subdevice() const;
            
            void
            get_info(cl_device_info id, size_t res_size,
                     void* res, size_t* ret_res)
                const;

            std::string
            get_info_string(cl_device_info id)
                const;

            template <typename _T>
            _T get_info(cl_device_info id)
                const {
                _T res;
                get_info(id, sizeof(res), &res, nullptr);
                return res;
            }

            std::string name() const;
            std::string vendor() const;
            std::string driver_version() const;
            std::vector<std::string>
            extensions() const;
            uint64_t global_memory_size() const;
            uint64_t local_memory_size() const;
            uint32_t address_bits() const;
            uint32_t compute_units() const;
            uint32_t max_work_group_size() const;
            uint32_t max_work_iterm_dimensions() const;
        };

        class context {
            cl_context _id;
        public:
            context();
            context(cl_context, bool retain=true);
            context(const context& r);
            context(context&& r);
            context& operator=(const context& r);
            context& operator=(context&& r);
            ~context();
            cl_context& operator()() { return _id; }
            const cl_context& operator()() const { return _id;}
        };

    }
}

namespace ocl {
    namespace be {
#if USE_BOOST_COMPUTE>0
        namespace bc= boost::compute;
#else
        namespace bc= cl;
#endif
        using error = bc::opencl_error;
        using program = bc::program;
        using context = bc::context;
        using device = bc::device;
        using queue = bc::command_queue;
        using buffer = bc::buffer;
        using kernel = bc::kernel;
        using event = bc::event;
        using wait_list = bc::wait_list;
    }
}

// local variables:
// mode: c++
// end:
#endif
