//
// Copyright (C) 2010-2025 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#if !defined (__OCL_IMPL_DEVICES_H__)
#define __OCL_IMPL_DEVICES_H__ 1

#include <ocl/config.h>
#include <ocl/be/types.h>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

namespace ocl {

    namespace be {

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
        std::ostream&
        operator<<(std::ostream& s, const device_info& d);

        struct dev_info {
            size_t _max_compute_units;
            size_t _max_workgroup_size;
            size_t _max_local_memory;
            dev_info(const device& d);
        };
        // returns 0 if the request can not be satisfied
        size_t
        request_local_mem(const device& d, size_t lmem_req);
        size_t
        request_local_mem(const dev_info& di, size_t lmem_req);

        size_t
        calc_local_size(const dev_info& di,
                        size_t global_size,
                        size_t max_local_size,
                        size_t pref_local_size_multiple);

        // kernel execution info: calculates _local_size
        // and global_size with
        // _global_size >= s
        // _local_size == required_local_size from k
        // or
        // _global_size % _local_size == 0
        // and tries to utilize all compute units of d
        struct kexec_1d_info {
            size_t _local_size;
            size_t _global_size;
            size_t _size;
            kexec_1d_info(const device& d,
                          const kernel& k,
                          size_t s);
        };


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

        // determine the f32 gflops of the device
        float
        gflops_f32(const device& d);

        // determine the cores per unit
        float
        cores_per_unit(const device& d);

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
