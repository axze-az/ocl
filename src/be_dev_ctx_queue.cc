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
#include <iomanip>
#include <thread>

namespace {

    cl_command_queue_properties
    enable_ooo_execution(const ocl::be::device& d) {
        cl_command_queue_properties p=0;
        cl_command_queue_properties cqp(
            d.get_info<cl_command_queue_properties>(CL_DEVICE_QUEUE_PROPERTIES));
        if ((cqp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)==
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            p |= ocl::cl::queue::enable_out_of_order_execution;
        }
        return p;
    }
}

ocl::be::dev_ctx_queue::
dev_ctx_queue()
    : _d(default_device()),
      _c(_d),
      _q(_c, _d, enable_ooo_execution(_d)),
      _wl(), _mtx()
{
}

ocl::be::dev_ctx_queue::
dev_ctx_queue(const device& dd)
    : _d(dd),
      _c(_d),
      _q(_c, _d, enable_ooo_execution(_d)),
      _wl(), _mtx()
{
}

ocl::be::dev_ctx_queue::
dev_ctx_queue(const device& dd, const context& ctx)
    : _d(dd), _c(ctx),
      _q(_c, _d, enable_ooo_execution(_d)),
      _wl(), _mtx()
{
}

ocl::be::dev_ctx_queue::
dev_ctx_queue(const device& dev, const context& ctx, const queue& qe)
    : _d(dev), _c(ctx), _q(qe), _wl(), _mtx()
{
}
