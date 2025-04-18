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
#include "ocl/be/devices.h"
#include <boost/compute/system.hpp>
#include <iostream>
#include <iomanip>

std::ostream&
ocl::be::operator<<(std::ostream& s, const device_info& dd)
{
    const device& d= dd._d;
    std::string n=d.name();
    s << "device name: " << n << '\n';
    n = d.vendor();
    s << "device vendor: " << n << '\n';
    n = d.driver_version();
    s << "driver version: " << n << '\n';
    cl_device_type dt=d.get_info<cl_device_type>(CL_DEVICE_TYPE);
    s << "device type: ";
    switch (dt) {
    case CL_DEVICE_TYPE_CPU:
        s << "cpu\n" ;
        break;
    case CL_DEVICE_TYPE_GPU:
        s << "gpu\n";
        break;
    default:
        s << "unknown\n";
        break;
    }
    cl_uint t=d.get_info<cl_uint>(CL_DEVICE_VENDOR_ID);
    s << "vendor id: " << std::hex << t << std::dec << '\n';

    cl_command_queue_properties cqp(
        d.get_info<cl_command_queue_properties>(CL_DEVICE_QUEUE_PROPERTIES));
    s << "device queue properties:";
    bool comma(false);
    if ((cqp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)==
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        s << " out of order execution";
        comma = true;
    }
    if ((cqp & CL_QUEUE_PROFILING_ENABLE) ==
        CL_QUEUE_PROFILING_ENABLE) {
        if (comma)
            s << ',';
        s << " profiling";
    }
    s << '\n';
    auto ve = d.extensions();
    s << "device extensions:\n";
    for (const auto& ei : ve) {
        s << "   " << ei << '\n';
    }
    cl_device_local_mem_type lt=
        d.get_info<cl_device_local_mem_type>(CL_DEVICE_LOCAL_MEM_TYPE);
    s << "local memory type: "
      << (lt == CL_LOCAL ? "local" : "global" )
      << '\n';
    t=d.local_memory_size();
    s << "local memory size: " << t <<'\n';
    size_t gs=d.global_memory_size();
    s << "global memory size: " << gs
      <<" (" << double(gs)/(1024.0*1024.0) << " MB)\n";
    size_t cs=d.get_info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
    s << "global memory cache size: " << cs << '\n';
    cl_device_mem_cache_type ct=
        d.get_info<cl_device_mem_cache_type>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
    s << "global memory cache type: ";
    if (ct == CL_READ_ONLY_CACHE) {
        s << "read only";
    } else if (ct == CL_READ_WRITE_CACHE) {
        s << "read write";
    } else {
        s << "none";
    }
    s << '\n';
    cs=d.get_info<cl_uint>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
    s << "global memory cache line size: " << cs << '\n';
    t=d.get_info<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY);
    s << "max freq: " << t << " MHz\n";
    t=d.get_info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
    s << "max compute units: " << t << '\n';
    s << "estimated GFLOPS(float): " << std::fixed
      << std::setprecision(1) << gflops_f32(d) << '\n';
    return s;
}

ocl::be::dev_info::dev_info(const device& d)
    : _max_compute_units(d.compute_units()),
      _max_workgroup_size(d.max_work_group_size()),
      _max_local_memory(d.local_memory_size())
{
    cl_device_local_mem_type lt=
        d.get_info<cl_device_local_mem_type>(CL_DEVICE_LOCAL_MEM_TYPE);
    bool has_local_mem= lt == CL_LOCAL;
    if (has_local_mem == false)
        _max_local_memory=0;
}

ocl::be::
kexec_1d_info::kexec_1d_info(const device& d, const kernel& k, size_t s)
    : _local_size(), _global_size(), _size(s)
{
    auto k_req_local_size=
        k.get_work_group_info<std::array<size_t, 3> >
            (d, CL_KERNEL_COMPILE_WORK_GROUP_SIZE);
    std::size_t local_size=k_req_local_size[0];
    if (local_size == 0) {
        size_t k_local_size=k.get_work_group_info<size_t>
            (d, CL_KERNEL_WORK_GROUP_SIZE);
        size_t k_pref_local_size_multiple=k.get_work_group_info<size_t>
            (d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
        // group size must be larger than 1, otherwise reductions
        // do not work
        k_pref_local_size_multiple = std::max(k_pref_local_size_multiple,
                                              size_t(2));
        local_size=calc_local_size(dev_info(d),
                                   s,
                                   k_local_size,
                                   k_pref_local_size_multiple);
    }
    _local_size = local_size;
    size_t local_size_m_1 = local_size - 1;
    if ((local_size & (local_size_m_1))==0) {
        // how many bytes are used from the last alignment:
        size_t m = s & local_size_m_1;
        // pad bytes:
        size_t pad= (local_size - m) & local_size_m_1;
        _global_size = s + pad;
    } else {
        _global_size = ((s+local_size-1)/local_size)*local_size;
    }
}

std::size_t
ocl::be::request_local_mem(const device& d, size_t lmem_req)
{
    platform pf(d.get_info<cl_platform_id>(CL_DEVICE_PLATFORM));
    if (pf.name() != "rusticl") {
        cl_device_local_mem_type lt=
            d.get_info<cl_device_local_mem_type>(CL_DEVICE_LOCAL_MEM_TYPE);
        bool has_local_mem= lt == CL_LOCAL;
        if (has_local_mem == false)
            return 0;
    }
    size_t lmem_size=d.local_memory_size();
    // allow maximum of a 1/8 of the local device memory:
    return ((lmem_req << 3) < lmem_size) ? lmem_req : 0;
}

std::size_t
ocl::be::request_local_mem(const dev_info& di, size_t lmem_req)
{
    size_t lmem_size=di._max_local_memory;
    // allow maximum of a 1/8 of the local device memory:
    return ((lmem_req << 3) < lmem_size) ? lmem_req : 0;
}

std::size_t
ocl::be::calc_local_size(const dev_info& di,
                         size_t global_size,
                         size_t k_local_size,
                         size_t pref_local_size_multiple)
{
    size_t cu=di._max_compute_units;
    size_t local_size = std::min(k_local_size, di._max_workgroup_size);
    // we want to have work on all cu's:
    while (local_size * cu > global_size &&
           local_size > pref_local_size_multiple) {
        local_size >>= 1;
    }
    return local_size;
}

std::vector<ocl::be::device>
ocl::be::devices()
{
    return bc::system::devices();
}

std::vector<ocl::be::device>
ocl::be::filter_devices(const std::vector<device>& v,
                        device_type::type dt)
{
    std::vector<device> r;
    for (std::size_t i=0; i< v.size(); ++i) {
        const device& d= v[i];
        cl_device_type t=d.type();
        if ((t & static_cast<cl_device_type>(dt)) ==
            static_cast<cl_device_type>(dt))
            r.push_back(d);
    }
    return r;
}

std::vector<ocl::be::device>
ocl::be::gpu_devices(const std::vector<device>& v)
{
    return filter_devices(v, device_type::gpu);
}

std::vector<ocl::be::device>
ocl::be::gpu_devices()
{
    std::vector<device> all_devs(devices());
    return gpu_devices(all_devs);
}

std::vector<ocl::be::device>
ocl::be::cpu_devices(const std::vector<device>& v)
{
    return filter_devices(v, device_type::cpu);
}

std::vector<ocl::be::device>
ocl::be::cpu_devices()
{
    std::vector<device> all_devs(devices());
    return cpu_devices(all_devs);
}

float
ocl::be::gflops_f32(const device& d)
{
    cl_uint t=d.get_info<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY);
    float gflops=static_cast<float>(t)*0.001f;
    t=d.get_info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
    gflops *= static_cast<float>(t);
    gflops *= cores_per_unit(d);
    auto fpcfg=d.get_info<cl_device_fp_config>(CL_DEVICE_SINGLE_FP_CONFIG);
    if ((fpcfg & CL_FP_FMA) == CL_FP_FMA) {
        gflops *= 2.0f;
    } else {
        platform pf(d.get_info<cl_platform_id>(CL_DEVICE_PLATFORM));
        if (pf.name() == "rusticl" && d.type() == CL_DEVICE_TYPE_GPU &&
            d.get_info<cl_uint>(CL_DEVICE_VENDOR_ID)==0x1002) {
            gflops *= 2.0f;
        }
    }
    return gflops;
}

float
ocl::be::cores_per_unit(const device& d)
{
    float cores_per_unit=1;
    if (d.supports_extension(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXTENSION_NAME)) {
        cl_uint simd_per_unit=
            d.get_info<cl_uint>(CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD);
        cl_uint simd_instruction_width=
            d.get_info<cl_uint>(CL_DEVICE_SIMD_WIDTH_AMD);
        cores_per_unit=static_cast<float>(simd_per_unit)*
            static_cast<float>(simd_instruction_width);
        return cores_per_unit;
    }
    auto dt = d.type();
    if (dt == CL_DEVICE_TYPE_CPU) {

    } else if (dt == CL_DEVICE_TYPE_GPU) {
        auto vendor_id=d.get_info<cl_uint>(CL_DEVICE_VENDOR_ID);
        switch(vendor_id) {
        case 0x8086:
            cores_per_unit=8;
            break;
        case 0x1002:
            cores_per_unit=64;
            break;
        }
    }
    return cores_per_unit;
}

ocl::be::device
ocl::be::device_with_max_freq_x_units(const std::vector<device>& v)
{
    if (v.empty()) {
        throw error(CL_DEVICE_NOT_FOUND);
    }
    device r;
    float sp(-1.0f);
    for (std::size_t i=0; i<v.size(); ++i) {
        const device& d= v[i];
        cl_uint t= d.get_info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
        float spi(t);
        t= d.get_info<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY);
        spi *= float(t);
        spi *= cores_per_unit(d);
        if (spi> sp) {
            r=d;
            sp=spi;
        }
    }
    return r;
}

ocl::be::device
ocl::be::default_gpu_device()
{
    std::vector<device> gpu_devs(gpu_devices());
    if (gpu_devs.empty()) {
        throw error(CL_DEVICE_NOT_FOUND);
    }
    return device_with_max_freq_x_units(gpu_devs);
}

ocl::be::device
ocl::be::default_cpu_device()
{
    std::vector<device> cpu_devs(cpu_devices());
    if (cpu_devs.empty()) {
        throw error(CL_DEVICE_NOT_FOUND);
    }
    return device_with_max_freq_x_units(cpu_devs);
}

ocl::be::device
ocl::be::default_device()
{
    device r;
    try {
        // r= default_cpu_device();
        r= default_gpu_device();
    }
    catch (const error& e) {
        try {
            r= default_cpu_device();
        }
        catch (const error& e) {
            throw error(CL_DEVICE_NOT_FOUND);
        }
    }
    return r;
}
