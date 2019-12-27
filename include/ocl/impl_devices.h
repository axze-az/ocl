#if !defined (__OCL_IMPL_DEVICES_H__)
#define __OCL_IMPL_DEVICES_H__ 1

#include <ocl/config.h>
#define CL_TARGET_OPENCL_VERSION 110
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL 1
#define BOOST_COMPUTE_THREAD_SAFE 1
#define BOOST_DISABLE_ASSERTS 1
#include <boost/compute/core.hpp>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

namespace ocl {

    namespace impl {

        
        namespace bc= boost::compute;
        using const_str_ref = std::basic_string_view<char>;

        using error = bc::opencl_error;

        const_str_ref err2str(const error& e);
        const_str_ref err2str(int e);

        using program = bc::program;
        using context = bc::context;
        using device = bc::device;
        using queue = bc::command_queue;
        using buffer = bc::buffer;
        using kernel = bc::kernel;
        using event = bc::event;
        using wait_list = bc::wait_list;

        class argument_buffer {
            std::vector<char> _v;
        public:
            argument_buffer() : _v() {
                _v.reserve(4096);
            };
            // allow access to the stored data
            const char* data() const { return _v.data(); }
            // amount of data
            size_t size() const { return _v.size(); }
            // clear the buffer
            void clear() { _v.clear(); }
            // insert an argument into the buffer
            template <typename _T>
            void
            insert(const _T& t) {
                constexpr size_t st=sizeof(_T);
                constexpr size_t at=alignof(_T);
                constexpr size_t atm1=at;
                static_assert((at & atm1) == 0,
                              "type with non power of 2 alignment?");
                const size_t s=_v.size();
                // how many bytes are used from the last alignment?
                const size_t m=s&atm1;
                // const size_t pad = m ? at - m : 0;
                const size_t pad= (at - m) & atm1;
                const size_t ns=s+pad;
                const size_t nn=ns+st;
                _v.resize(nn, char(0xff));
                char* cd=_v.data() + ns;
                _T* d=reinterpret_cast<_T*>(cd);
                *d = t;
            }
            buffer create_buffer(const context& c) {
                return buffer(c, _v.size());
            }
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
}

// Local variables:
// mode: c++
// end:
#endif
