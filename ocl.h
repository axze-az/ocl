#if !defined (__OCL_H__)
#define __OCL_H__ 1

#include <ocl/config.h>
// #include <CL/cl.h>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

// #include <CL/cl.hpp>

namespace ocl {


        namespace impl {
                
                typedef cl::Error error;

                const char* err2str(const error& e);
                const char* err2str(int e);

                typedef std::vector<cl_context_properties> ctx_prop_vec_type;

                typedef cl::Context context;
                typedef cl::Device device;
                typedef cl::CommandQueue queue;
                typedef cl::Buffer buffer;
                typedef cl::Kernel kernel;

                struct data {
                        context _ctx;
                        device _dev;
                        data(const context& c, const device& d) : _ctx(c), _dev(d) {}
                };


                struct device_type {
                        enum type {
                                cpu = CL_DEVICE_TYPE_CPU,
                                gpu = CL_DEVICE_TYPE_GPU,
                                accel = CL_DEVICE_TYPE_ACCELERATOR,
#if defined (CL_DEVICE_TYPE_CUSTOM)
                                custom = CL_DEVICE_TYPE_CUSTOM,
#endif
                                all = CL_DEVICE_TYPE_ALL
                        };  
                };
                
                struct device_info {
                        device _d;
                        device_info(const device& d) : _d(d) {}
                };
                
                std::ostream& operator <<(std::ostream& s,
                                          const device_info& d);
                

                std::vector<device>
                filter_devices(const std::vector<device>& devs,
                               device_type::type t );

                // filter all gpu devices
                std::vector<device>
                gpu_devices(const std::vector<device>& devs);

                // filter all cpu devices
                std::vector<device>
                cpu_devices(const std::vector<device>& devs);

                // get all devices from all contexts
                std::vector<device>
                devices();
                // get all gpu devices from all contexts
                std::vector<device>
                gpu_devices();
                // get all cpu devices from all contexts
                std::vector<device>
                cpu_devices();

                // return the device with the maximum product
                // of units and frequency
                device
                device_with_max_freq_x_units(const std::vector<device>& v);
                
                // get the most? powerful gpu device 
                device
                default_gpu_device();
                // get a cpu device
                device
                default_cpu_device();
                // return a default device.
                device
                default_device();
                
        }


        namespace impl {

                template <typename _T>
                struct type_2_name {
                };

                template <>
                struct type_2_name<float> {
                        static 
                        constexpr const char* v() {
                                return "float";
                        }
                };

                template <>
                struct type_2_name<int> {
                        static
                        constexpr const char* v() {
                                return "int32_t";
                        }
                };

                template <>
                struct type_2_name<unsigned> {
                        static
                        constexpr const char* v() {
                                return "uint32_t";
                        }
                };

                template <>
                struct type_2_name<std::int64_t> {
                        static
                        constexpr const char* v() {
                                return "int64_t";
                        }
                };

                template <>
                struct type_2_name<std::uint64_t> {
                        static
                        constexpr const char* v() {
                                return "uint64_t";
                        }
                };
                
        }

  

        namespace ll {
                
                inline void 
                check_err(cl_uint rc, const std::string& msg="") {
                        if (rc != CL_SUCCESS) 
                                throw impl::error(rc, msg.c_str());
                }

                template <class _T>
                struct ref_cnt;

                template <>
                struct ref_cnt<cl_platform_id> {
                        static cl_int inc(cl_platform_id ) {
                                return CL_SUCCESS;
                        }
                        static cl_int dec(cl_platform_id ) {
                                return CL_SUCCESS;
                        }
                };

                template <>
                struct ref_cnt<cl_device_id> {
                        static cl_int inc(cl_device_id i) {
                                return 0; // clRetainDevice(i);
                        }
                        static cl_int dec(cl_device_id i) {
                                return 0; // clReleaseDevice(i);
                        }
                };

                template <>
                struct ref_cnt<cl_context> {
                        static cl_int inc(cl_context i) {
                                return clRetainContext(i);
                        }
                        static cl_int dec(cl_context i) {
                                return clReleaseContext(i);
                        }
                };

                template <>
                struct ref_cnt<cl_mem> {
                        static cl_int inc(cl_mem i) {
                                return clRetainMemObject(i);
                        }
                        static cl_int dec(cl_mem i) {
                                return clReleaseMemObject(i);
                        }
                };

                template <>
                struct ref_cnt<cl_command_queue> {
                        static cl_int inc(cl_command_queue i) { 
                                return clRetainCommandQueue(i); 
                        }
                        static cl_int dec(cl_command_queue i) { 
                                return clReleaseCommandQueue(i); 
                        }
                };

                template <>
                struct ref_cnt<cl_program> {
                        static cl_int inc(cl_program program) { 
                                return clRetainProgram(program); 
                        }
                        static cl_int dec(cl_program program) { 
                                return clReleaseProgram(program); 
                        }
                };

                template <>
                struct ref_cnt<cl_kernel> {
                        static cl_int inc(cl_kernel kernel) { 
                                return clRetainKernel(kernel); 
                        }
                        static cl_int dec(cl_kernel kernel) { 
                                return clReleaseKernel(kernel); 
                        }
                };
                

                template <class _T>
                class handle : protected ref_cnt<_T> {
                        _T _h;
                        typedef ref_cnt<_T> base_type;
                public:
                        handle(_T v) : _h(v) {}
                        handle(const handle& r) : _h(r._h) {
                                if (_h != 0)
                                        check_err(base_type::inc(_h));
                        }
                        handle(handle&& r) : _h(std::move(r._h)) {
                                r._h = 0;
                        }
                        handle& operator=(_T v) {
                                if (v != _h) {
                                        if (_h)
                                                check_err(base_type::dec(_h));
                                        _h = v;
                                }
                                return *this;
                        }
                        handle& operator=(const handle& r) {
                                if (&r != this) {
                                        if (_h)
                                                check_err(base_type::dec(_h));
                                        _h= r._h;
                                        if (_h)
                                                check_err(base_type::inc(_h));
                                }
                                return *this;
                        }
                        handle& operator=(handle&& r) {
                                std::swap(_h, r._h);
                                return *this;
                        }
                        ~handle() {
                                if (_h)
                                        check_err(base_type::dec(_h));
                        }
                        const _T& operator()() const { return _h; }
                        const _T& operator()() { return _h; }
                protected:
                        _T h() const { return _h; }
                        void h( _T v) { _h = v; }
                };

                class device : public handle<cl_device_id> {
                        typedef handle<cl_device_id> base_type;
                public:
                        device() : base_type(nullptr) {}
                        device(cl_device_id i) : base_type(i) {}
                };
                
                class platform : public handle<cl_platform_id> {
                        typedef handle<cl_platform_id> base_type;
                public:
                        platform() : base_type(nullptr) {}
                        platform(cl_platform_id i) : base_type(i) {}
                        static std::vector<platform> get();
                        std::vector<device> devices(cl_device_type t);
                        cl_int unload_compiler() {
                                return 0; //return clUnloadPlatformCompiler(h());
                        }
                };

                inline
                std::vector<platform> get_platforms() {
                        return platform::get();
                }
                
                class context : public handle<cl_context> {
                        typedef handle<cl_context> base_type;
                public:
                        context() : base_type(nullptr) {}
                        context(cl_context i) : base_type(i) {}
                        context(const std::vector<device>& devs,
                                cl_context_properties* props = nullptr,
                                void (CL_CALLBACK * notifyFptr)(
                                        const char *,
                                        const void *,
                                        std::size_t,
                                        void*) = nullptr,
                                void* data = nullptr);
                        context(const device& dev,
                                cl_context_properties* props = nullptr,
                                void (CL_CALLBACK * notifyFptr)(
                                        const char *,
                                        const void *,
                                        std::size_t,
                                        void*) = nullptr,
                                void* data = nullptr);
                };

                class memory : public handle<cl_mem> {
                        typedef handle<cl_mem> base_type;
                public:
                        memory() : base_type(nullptr) {}
                        memory(cl_mem m) : base_type(m) {}
                };

                class buffer : public memory {
                        typedef memory base_type;
                public:
                        buffer() : base_type() {}
                        buffer(cl_mem m) : base_type(m) {};
                        buffer(const context& ctx,
                               cl_mem_flags flags,
                               std::size_t size,
                               void* host_ptr = nullptr);
                };

                class program : public handle<cl_program> {
                        typedef handle<cl_program> base_type;
                public:
                        program() : base_type(nullptr) {}
                        program(cl_program p) : base_type(p) {}
                        program(const context& ctx,
                                const std::string& src,
                                bool build = false);
                        program(const context& ctx,
                                const std::vector<std::string>& src,
                                bool build = false);
                };

                class event {
                };
                
                class queue : public handle<cl_command_queue> {
                        typedef handle<cl_command_queue> base_type;
                public:
                        queue() : base_type(nullptr) {}
                        queue(cl_command_queue i) : base_type(i) {}
                        queue(const context& ctx,
                              const device& device,
                              cl_command_queue_properties props = 0);
                        event
                        write_to(buffer& buf, std::size_t offset,
                                 const void* src, std::size_t src_offset, 
                                 std::size_t size,
                                 std::vector<event>& wait_events,
                                 bool blocking);
                        event
                        read_from(void* dst, std::size_t dst_offset, 
                                  const buffer& buf, std::size_t offset,
                                  std::size_t size,
                                  std::vector<event>& wait_events,
                                  bool blocking);
                };


        }

        class plattform;
        class device;
        class queue;

        class mem {
        };

        class buffer : public mem {
        };

        class image : public mem {
        };

}

// Local variables:
// mode: c++
// end:
#endif
