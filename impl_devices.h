#if !defined (__OCL_IMPL_DEVICES_H__)
#define __OCL_IMPL_DEVICES_H__ 1

#include <ocl/config.h>
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>
#include <experimental/string_view>

namespace ocl {

    namespace impl {

        using const_str_ref = std::experimental::basic_string_view<char>;
        
        using error = cl::Error;

        const_str_ref err2str(const error& e);
        const_str_ref err2str(int e);

        using program = cl::Program;
        using context = cl::Context;
        using device = cl::Device;
        using queue = cl::CommandQueue;
        using buffer = cl::Buffer;
        using kernel = cl::Kernel;
        using event = cl::Event;

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
