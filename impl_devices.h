#if !defined (__OCL_IMPL_DEVICES_H__)
#define __OCL_IMPL_DEVICES_H__ 1

#include <ocl/config.h>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace ocl {

        namespace impl {
                
                typedef cl::Error error;

                const char* err2str(const error& e);
                const char* err2str(int e);

                typedef cl::Program program;
                typedef cl::Context context;
                typedef cl::Device device;
                typedef cl::CommandQueue queue;
                typedef cl::Buffer buffer;
                typedef cl::Kernel kernel;
         
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
