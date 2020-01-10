#include "ocl/be/data.h"
#include <cstdlib>
#include <iomanip>
#include <thread>

ocl::be::dev_ctx_queue::
dev_ctx_queue()
    : _d(default_device()),
      _c(_d),
      _q(_c, _d, queue::enable_out_of_order_execution),
      _wl(), _mtx()
{
}

ocl::be::dev_ctx_queue::
dev_ctx_queue(const device& dd)
    : _d(dd),
      _c(_d),
      _q(_c, _d, queue::enable_out_of_order_execution),
      _wl(), _mtx()
{
}

ocl::be::dev_ctx_queue::
dev_ctx_queue(const device& dd, const context& ctx)
    : _d(dd), _c(ctx),
      _q(_c, _d, queue::enable_out_of_order_execution),
      _wl(), _mtx()
{
}

ocl::be::dev_ctx_queue::
dev_ctx_queue(const device& dev, const context& ctx, const queue& qe)
    : _d(dev), _c(ctx), _q(qe), _wl(), _mtx()
{
}
