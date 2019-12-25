#include "impl_be_data.h"
#include <cstdlib>

ocl::impl::mutex
ocl::impl::be_data::_instance_mutex;
std::atomic<bool>
ocl::impl::be_data::_init;
std::shared_ptr<ocl::impl::be_data>
ocl::impl::be_data::_default;

void
ocl::impl::be_data::
enqueue_kernel(pgm_kernel_lock& pk, size_t s)
{
    queue& q= _q;
    if (debug() != 0) {
        std::cout << "executing kernel" << std::endl;
    }
    std::size_t local_size(
        pk._k.get_work_group_info<size_t>(_d, CL_KERNEL_WORK_GROUP_SIZE));
    std::size_t gs= ((s+local_size-1)/local_size)*local_size;
    if (_debug != 0) {
        std::cout << "kernel: size: " << s
                  << " global size: " << gs
                  << " local size: " << local_size
                  << std::endl;
    }
    // TODO: make access to backend_data::_evs thread safe
    auto& evs=_ev;
    event ev=q.enqueue_1d_range_kernel(pk._k,
                                       0,
                                       gs,
                                       local_size,
                                       evs);
    q.flush();
    evs.clear();
    // TODO: figure out why we have memory leaks here:
    // evs.insert(ev);
    ev.wait();
    if (_debug != 0) {
        std::cout << "execution done" << std::endl;
    }
    // q.flush();
    // q.finish();
}

ocl::impl::be_data_ptr
ocl::impl::be_data::instance()
{
    if (_init == false) {
        std::unique_lock<mutex> _l(_instance_mutex);
        if (_init==false) {
            _default = std::make_shared<be_data>();
            _init= true;
        }
    }
    return _default.get();
}

std::shared_ptr<ocl::impl::be_data>
ocl::impl::be_data::create(const device& dev)
{
    return std::make_shared<be_data>(dev);
}

std::shared_ptr<ocl::impl::be_data>
ocl::impl::be_data::create(const device& dev, const context& ctx)
{
    return std::make_shared<be_data>(dev, ctx);
}

std::shared_ptr<ocl::impl::be_data>
ocl::impl::be_data::create(const device& dev, const context& ctx,
                           const queue& qe)
{
    return std::make_shared<be_data>(dev, ctx, qe);
}

std::uint32_t
ocl::impl::be_data::read_debug_env()
{
    std::uint32_t r=0;
    const char* pe=std::getenv("OCL_DEBUG");
    if (pe != nullptr) {
        r = 1;
    }
    return r;
}

ocl::impl::be_data::be_data()
    : _d(default_device()), _c(_d),
      _q(_c, _d/*, queue::enable_out_of_order_execution*/),
      _debug(read_debug_env())
{
}

ocl::impl::be_data::be_data(const device& dev)
    : _d(dev), _c(_d),
      _q(_c, _d, queue::enable_out_of_order_execution),
      _debug(read_debug_env())
{
}

ocl::impl::be_data::be_data(const device& dev, const context& ctx)
    : _d(dev), _c(ctx),
      _q(_c, _d, queue::enable_out_of_order_execution),
      _debug(read_debug_env())
{
}

ocl::impl::be_data::be_data(const device& dev, const context& ctx,
                            const queue& qe)
    : _d(dev), _c(ctx), _q(qe), _debug(read_debug_env())
{
}

ocl::impl::be_data::~be_data()
{
    if (_q != nullptr) {
        _q.finish();
    }
}
