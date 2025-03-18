#if !defined (__OCL_BE_TYPES_H__)
#define __OCL_BE_TYPES_H__ 1

#include <ocl/config.h>
#include <ocl/types.h>

#define USE_BOOST_COMPUTE 0

#if USE_BOOST_COMPUTE>0
#define CL_TARGET_OPENCL_VERSION 120
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL 1
#define BOOST_COMPUTE_THREAD_SAFE 1
#define BOOST_DISABLE_ASSERTS 1
#include <boost/compute/core.hpp>
#else
#define CL_TARGET_OPENCL_VERSION 120
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
        class platform;

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
            std::string error_string() const;
            // check code and throw if required
            static
            void
            _throw_on(cl_int code);
            // check code and throw if required
            static
            void
            _throw_on(cl_int code, const char* file, unsigned line);

            static
            void
            throw_on(cl_int code) {
                if (code != CL_SUCCESS)
                    _throw_on(code);
            }

            static
            void
            throw_on(cl_int code, const char* file, unsigned line) {
                if (code != CL_SUCCESS)
                    _throw_on(code, file, line);
            }
        };

        class device {
            cl_device_id _id;
            std::string
            info(cl_device_info id)
                const;
        public:
            enum type {
                cpu = CL_DEVICE_TYPE_CPU,
                gpu = CL_DEVICE_TYPE_GPU,
                accelerator = CL_DEVICE_TYPE_ACCELERATOR
            };
            device() : _id(0) {}
            explicit device(cl_device_id, bool retain=true);
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
            info(cl_device_info id, size_t res_size,
                 void* res, size_t* ret_res)
                const;

            template <typename _T>
            _T get_info(cl_device_info id)
                const {
                _T res;
                info(id, sizeof(res), &res, nullptr);
                return res;
            }

            cl_device_type type() const;
            std::string name() const;
            std::string vendor() const;
            std::string driver_version() const;
            std::vector<std::string>
            extensions() const;
            bool
            supports_extension(const std::string& name) const;
            uint64_t global_memory_size() const;
            uint64_t local_memory_size() const;
            uint32_t address_bits() const;
            uint32_t compute_units() const;
            uint32_t max_work_group_size() const;
            uint32_t max_work_iterm_dimensions() const;
            ocl::cl::platform platform() const;
        };

        class context {
            cl_context _id;
        public:
            context() : _id(0) {}
            context(const context& r);
            context(context&& r);
            context& operator=(const context& r);
            context& operator=(context&& r);
            ~context();
            cl_context& operator()() { return _id; }
            const cl_context& operator()() const { return _id;}

            explicit
            context(cl_context, bool retain=true);
            explicit
            context(const device& d,
                    const cl_context_properties* p=nullptr);

            context(const std::vector<device>& vd,
                    const cl_context_properties* p=nullptr);

            void
            info(cl_context_info id, size_t res_size,
                 void* res, size_t* ret_res)
                const;

            template <typename _T>
            _T get_info(cl_context_info id)
                const {
                _T res;
                info(id, sizeof(res), &res, nullptr);
                return res;
            }

            device
            get_device()
                const;

            std::vector<device>
            get_devices()
                const;

        };

        class mem_object {
            cl_mem _id;
        public:
            /// Flags for the creation of memory objects.
            enum mem_flags {
                read_write = CL_MEM_READ_WRITE,
                read_only = CL_MEM_READ_ONLY,
                write_only = CL_MEM_WRITE_ONLY,
                use_host_ptr = CL_MEM_USE_HOST_PTR,
                alloc_host_ptr = CL_MEM_ALLOC_HOST_PTR,
                copy_host_ptr = CL_MEM_COPY_HOST_PTR
#if CL_TARGET_OPENCL_VERSION>=120
                ,
                host_write_only = CL_MEM_HOST_WRITE_ONLY,
                host_read_only = CL_MEM_HOST_READ_ONLY,
                host_no_access = CL_MEM_HOST_NO_ACCESS
#endif
            };

            enum address_space {
                global_memory,
                local_memory,
                private_memory,
                constant_memory
            };
        protected:
            mem_object() : _id(0) {}
            mem_object(const mem_object& r);
            mem_object(mem_object&& r);
            mem_object& operator=(const mem_object& r);
            mem_object& operator=(mem_object&& r);
            ~mem_object();
        public:
            cl_mem& operator()() { return _id; }
            const cl_mem& operator()() const { return _id;}
            const cl_mem& get() const { return _id; }
            explicit
            mem_object(cl_mem c, bool retain=true);
            void
            info(cl_mem_info i, size_t s, void* res, size_t* rs)
                const;
        };

        class buffer : public mem_object {
        public:
            buffer() : mem_object() {}
            buffer(const buffer& r) : mem_object(r) {}
            buffer(buffer&& r) : mem_object(std::move(r)) {}
            buffer& operator=(const buffer& r) {
                mem_object::operator=(r);
                return *this;
            }
            buffer& operator=(buffer&& r) {
                mem_object::operator=(std::move(r));
                return *this;
            }
            ~buffer() {}
            explicit
            buffer(cl_mem c, bool retain=true) : mem_object(c, retain) {}
            buffer(const context& context, size_t size,
                   cl_mem_flags flags = read_write,
                   void *host_ptr = nullptr);
            size_t size() const;
        };

        class event {
            cl_event _id;
        public:
            enum execution_status {
                complete = CL_COMPLETE,
                running = CL_RUNNING,
                submitted = CL_SUBMITTED,
                queued = CL_QUEUED
            };
            enum command_type {
                ndrange_kernel = CL_COMMAND_NDRANGE_KERNEL,
                task = CL_COMMAND_TASK,
                native_kernel = CL_COMMAND_NATIVE_KERNEL,
                read_buffer = CL_COMMAND_READ_BUFFER,
                write_buffer = CL_COMMAND_WRITE_BUFFER,
                copy_buffer = CL_COMMAND_COPY_BUFFER,
                read_image = CL_COMMAND_READ_IMAGE,
                write_image = CL_COMMAND_WRITE_IMAGE,
                copy_image = CL_COMMAND_COPY_IMAGE,
                copy_image_to_buffer = CL_COMMAND_COPY_IMAGE_TO_BUFFER,
                copy_buffer_to_image = CL_COMMAND_COPY_BUFFER_TO_IMAGE,
                map_buffer = CL_COMMAND_MAP_BUFFER,
                map_image = CL_COMMAND_MAP_IMAGE,
                unmap_mem_object = CL_COMMAND_UNMAP_MEM_OBJECT,
                marker = CL_COMMAND_MARKER,
                aquire_gl_objects = CL_COMMAND_ACQUIRE_GL_OBJECTS,
                release_gl_object = CL_COMMAND_RELEASE_GL_OBJECTS
#if CL_TARGET_OPENCL_VERSION>=110
                ,
                read_buffer_rect = CL_COMMAND_READ_BUFFER_RECT,
                write_buffer_rect = CL_COMMAND_WRITE_BUFFER_RECT,
                copy_buffer_rect = CL_COMMAND_COPY_BUFFER_RECT
#endif
            };
            enum profiling_info {
                profiling_command_queued = CL_PROFILING_COMMAND_QUEUED,
                profiling_command_submit = CL_PROFILING_COMMAND_SUBMIT,
                profiling_command_start = CL_PROFILING_COMMAND_START,
                profiling_command_end = CL_PROFILING_COMMAND_END
            };

            event() : _id(0) {}
            event(const event& r);
            event(event&& r);
            event& operator=(const event& r);
            event& operator=(event&& r);
            ~event();
            cl_event& operator()() { return _id; }
            const cl_event& operator()() const { return _id;}
            explicit
            event(cl_event c, bool retain=true);
            void wait();
        };

        class user_event : public event {
            static
            cl_event create(const context& ctx);
        public:
            explicit user_event(const context& ctx);
            void set_status(cl_int exec_status);
        };

        class wait_list {
            std::vector<event> _ve;
        public:
            using iterator=std::vector<event>::iterator;
            using const_iterator=std::vector<event>::const_iterator;
            wait_list();
            explicit
            wait_list(const event &event);
            wait_list(const wait_list &other);
            wait_list(std::initializer_list<event> events);
            wait_list& operator=(const wait_list &other);
            wait_list(wait_list&& other);
            wait_list& operator=(wait_list&& other);
            ~wait_list();
            bool empty() const;
            size_t size() const;
            void clear();
            const cl_event* get_event_ptr() const;
            void reserve(size_t new_capacity);
            void insert(const event &event);
            void wait() const;
            const event& operator[](size_t pos) const;
            event& operator[](size_t pos);
            iterator begin();
            const_iterator begin() const;
            const_iterator cbegin() const;
            iterator end();
            const_iterator end() const;
            const_iterator cend() const;
        };

        class kernel {
            cl_kernel _id;
        public:
            kernel() : _id(0) {}
            kernel(const kernel &r);
            kernel& operator=(const kernel &r);
            kernel(kernel&& r);
            kernel& operator=(kernel&& r);
            ~kernel();
            explicit
            kernel(cl_kernel k, bool retain=true);
            cl_kernel& operator()() { return _id; }
            const cl_kernel& operator()() const { return _id;}

            kernel(const program& pgm, const std::string& kname);

            std::string name() const;

            void
            info(cl_kernel_info i, size_t s, void* res, size_t* rs)
                const;
            void
            work_group_info(const device& d,
                            cl_kernel_work_group_info i, size_t s,
                            void* res, size_t* rs)
                const;
            template <typename _T>
            _T
            get_work_group_info(const device& d, cl_kernel_work_group_info i)
                const {
                _T r;
                work_group_info(d, i, sizeof(_T), &r, nullptr);
                return r;
            }
            void
            set_arg(size_t index, size_t size, const void* value);

            template <class _T>
            void
            set_arg(size_t index, const _T& value) {
                set_arg(index, sizeof(_T), &value);
            }

            void
            set_arg(size_t index, const mem_object& mem) {
                set_arg(index, sizeof(cl_mem), &mem());
            }
        };

        class program {
            cl_program _id;
        public:
            program() : _id(0) {}
            program(const program &r);
            program& operator=(const program &r);
            program(program&& r);
            program& operator=(program&& r);
            ~program();
            explicit
            program(cl_program p, bool retain=true);
            cl_program& operator()() { return _id; }
            const cl_program& operator()() const { return _id;}

            void
            info(cl_program_info i, size_t ps, void* p, size_t* rps)
                const;

            std::vector<device>
            get_devices()
                const;

            void
            build_info(const device& d, cl_program_build_info i,
                       size_t ps, void* p, size_t* rps)
                const;

            std::string build_log() const;

            void
            build(const std::string& options = std::string());

            static
            program
            create_with_source(const std::string &source,
                               const context& context);
        };

        class queue {
            cl_command_queue _id;
        public:
            queue() : _id(0) {}
            queue(const queue &r);
            queue& operator=(const queue &r);
            queue(queue&& r);
            queue& operator=(queue&& r);
            ~queue();
            explicit
            queue(cl_command_queue p, bool retain=true);
            cl_command_queue& operator()() { return _id; }
            const cl_command_queue& operator()() const { return _id;}

            enum properties {
                enable_profiling =
                CL_QUEUE_PROFILING_ENABLE,
                enable_out_of_order_execution =
                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
#if CL_TARGET_OPENCL_VERSION >= 200
                ,
                on_device = CL_QUEUE_ON_DEVICE,
                on_device_default = CL_QUEUE_ON_DEVICE_DEFAULT
#endif
            };

            enum map_flags {
                map_read = CL_MAP_READ,
                map_write = CL_MAP_WRITE
#if CL_TARGET_OPENCL_VERSION >= 200
                ,
                map_write_invalidate_region = CL_MAP_WRITE_INVALIDATE_REGION
#endif
            };

#if CL_TARGET_OPENCL_VERSION>=120
            enum mem_migration_flags {
                migrate_to_host = CL_MIGRATE_MEM_OBJECT_HOST,
                migrate_content_undefined =
                CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
            };
#endif
            queue(const context& c, const device& d,
                  cl_command_queue_properties p=0);

            event
            enqueue_copy_buffer(const buffer &src_buffer,
                                const buffer &dst_buffer,
                                size_t src_offset,
                                size_t dst_offset,
                                size_t size,
                                const wait_list&events = wait_list());
            event
            enqueue_read_buffer(const buffer &buffer,
                                size_t offset,
                                size_t size,
                                void *host_ptr,
                                const wait_list &events = wait_list());
            event
            enqueue_read_buffer_async(const buffer &buffer,
                                      size_t offset,
                                      size_t size,
                                      void *host_ptr,
                                      const wait_list &events = wait_list());
            event
            enqueue_write_buffer(const buffer &buffer,
                                 size_t offset,
                                 size_t size,
                                 const void *host_ptr,
                                 const wait_list &events = wait_list());
            event
            enqueue_write_buffer_async(const buffer &buffer,
                                       size_t offset,
                                       size_t size,
                                       const void *host_ptr,
                                       const wait_list &events = wait_list());
            event
            enqueue_nd_range_kernel(const kernel &kernel,
                                    size_t work_dim,
                                    const size_t* global_work_offset,
                                    const size_t* global_work_size,
                                    const size_t* local_work_size,
                                    const wait_list& events = wait_list());
            event
            enqueue_1d_range_kernel(const kernel &kernel,
                                    size_t global_work_offset,
                                    size_t global_work_size,
                                    size_t local_work_size,
                                    const wait_list& events = wait_list()) {
                return enqueue_nd_range_kernel(
                    kernel, 1,
                    &global_work_offset,
                    &global_work_size,
                    local_work_size ? &local_work_size : nullptr,
                    events);
            }
            void flush();
            void finish();
        };

        class platform {
            cl_platform_id _id;
            std::string
            info(cl_platform_info i)
                const;
        public:
            explicit
            platform (cl_platform_id id) : _id(id) {}
            cl_platform_id& operator()() { return _id; }
            const cl_platform_id& operator()() const { return _id;}

            void
            info(cl_platform_info i, size_t s, void* p, size_t* rps)
                const;

            std::string name() const { return info(CL_PLATFORM_NAME); }

            std::string
            vendor() const { return info(CL_PLATFORM_VENDOR); }

            std::string
            profile() const { return info(CL_PLATFORM_PROFILE); }

            std::string
            version() const { return info(CL_PLATFORM_VERSION); }

            size_t
            device_count(cl_device_type type)
                const;

            std::vector<device>
            devices(cl_device_type = CL_DEVICE_TYPE_ALL)
                const;
        };

        class system {
        public:
            static
            std::vector<platform>
            platforms();

            static
            std::vector<device>
            devices();
        };
    }
}

namespace ocl {
    namespace be {
#if USE_BOOST_COMPUTE>0
        namespace bc= boost::compute;
        using error = bc::opencl_error;
        using queue = bc::command_queue;
#else
        namespace bc= cl;
        using error = bc::error;
        using queue = bc::queue;
#endif
        using program = bc::program;
        using platform = bc::platform;
        using context = bc::context;
        using device = bc::device;
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
