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
#include "ocl/be/data.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <thread>

ocl::be::mutex
ocl::be::data::_instance_mutex;
std::atomic<bool>
ocl::be::data::_init;
std::shared_ptr<ocl::be::data>
ocl::be::data::_default;
ocl::be::mutex
ocl::be::data::_debug_mutex;

ocl::be::event
ocl::be::data::
enqueue_1d_kernel(const kernel& k, const kexec_1d_info& ki)
{
    if (_debug != 0) {
        const char nl='\n';
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": ";
        std::string tn=s.str() + kn;
        auto& d=dcq().d();
        std::uint64_t lms= k.get_work_group_info<std::uint64_t>(
            d, CL_KERNEL_LOCAL_MEM_SIZE);
        s.str("");
        s << tn << ": enqueue kernel " << nl
          << tn << ": size:           " << std::setw(8)
          << ki._size << nl
          << tn << ": global size:    " << std::setw(8)
          << ki._global_size << nl
          << tn << ": local size:     " << std::setw(8)
          << ki._local_size
          << nl
          << tn << ": local mem size: " << std::setw(8)
          << lms
          << nl;
        debug_print(s.str());
    }
    queue& q= dcq().q();
    auto& wl=dcq().wl();
    event ev=q.enqueue_1d_range_kernel(k,
                                       0,
                                       ki._global_size,
                                       ki._local_size,
                                       wl);
    q.flush();
    wl.clear();
    // TODO: figure out why we have memory leaks here:
    // wl.insert(ev);
    if (_debug != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": ";
        std::string tn=s.str() + kn;
        tn+=": enqueue done\n";
        debug_print(tn);
    }
    return ev;
}

ocl::be::event
ocl::be::data::
enqueue_1d_kernel(const kernel& k, size_t s)
{
    auto& d=dcq().d();
    kexec_1d_info ki(d, k, s);
    return enqueue_1d_kernel(k, ki);
}

bool
ocl::be::data::
supports(query_bool q)
{
    auto& d= dcq().d();
    bool r=false;
    switch (q) {
    case query_bool::fp16_fma: {
        cl_device_fp_config cfg=d.get_info<cl_device_fp_config>(
            CL_DEVICE_HALF_FP_CONFIG);
        r = (cfg & CL_FP_FMA) == CL_FP_FMA;
        break;
    }
    case query_bool::fp32_fma:{
        cl_device_fp_config cfg=d.get_info<cl_device_fp_config>(
            CL_DEVICE_SINGLE_FP_CONFIG);
        r = (cfg & CL_FP_FMA) == CL_FP_FMA;
        break;
    }
    case query_bool::fp64_fma: {
        cl_device_fp_config cfg=d.get_info<cl_device_fp_config>(
            CL_DEVICE_DOUBLE_FP_CONFIG);
        r = (cfg & CL_FP_FMA) == CL_FP_FMA;
        break;
    }}
    return r;
}

ocl::be::data_ptr
ocl::be::data::instance()
{
    if (_init == false) {
        scoped_lock _l(_instance_mutex);
        if (_init==false) {
            _default = std::make_shared<data>();
            _init= true;
        }
    }
    return _default;
}

std::shared_ptr<ocl::be::data>
ocl::be::data::create(const device& dev)
{
    return std::make_shared<data>(dev);
}

std::shared_ptr<ocl::be::data>
ocl::be::data::create(const device& dev, const context& ctx)
{
    return std::make_shared<data>(dev, ctx);
}

std::shared_ptr<ocl::be::data>
ocl::be::data::create(const device& dev, const context& ctx,
                           const queue& qe)
{
    return std::make_shared<data>(dev, ctx, qe);
}

void
ocl::be::data::debug_print(const std::string& m)
{
    std::scoped_lock _lck(_debug_mutex);
    std::cout << m;
}

std::uint32_t
ocl::be::data::read_debug_env()
{
    std::uint32_t r=0;
    const char* pe=std::getenv("OCL_DEBUG");
    if (pe != nullptr) {
        r = 1;
    }
    return r;
}

ocl::be::data::data()
    : _dcq(),
      _kcache(),
      _debug(read_debug_env())
{
}

ocl::be::data::data(const device& dev)
    : _dcq(dev),
      _kcache(),
      _debug(read_debug_env())
{
}

ocl::be::data::data(const device& dev, const context& ctx)
    : _dcq(dev, ctx),
      _kcache(),
      _debug(read_debug_env())
{
}

ocl::be::data::data(const device& dev, const context& ctx,
                            const queue& qe)
    : _dcq(dev, ctx, qe),
      _kcache(),
      _debug(read_debug_env())
{
}

ocl::be::data::~data()
{
}
