#include "impl_be_data.h"
#include <cstdlib>

ocl::impl::mutex
ocl::impl::be_data::_instance_mutex;
std::atomic<bool>
ocl::impl::be_data::_init;
std::shared_ptr<ocl::impl::be_data>
ocl::impl::be_data::_default;

std::shared_ptr<ocl::impl::be_data>
ocl::impl::be_data::instance()
{
    if (_init == false) {
        std::unique_lock<mutex> _l(_instance_mutex);
        if (_init==false) {
            _default = std::make_shared<be_data>();
            _init= true;
        }
    }
    return _default;
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
    : _d(default_device()), _c(), _q(), _debug(read_debug_env())
{
    // create context from device, command queue from context and
    // device
    std::vector<cl::Device> vd;
    vd.push_back(_d);
    _c= cl::Context(vd);
    _q= cl::CommandQueue(_c, _d);
}

ocl::impl::be_data::be_data(const device& dev)
    : _d(dev), _c(), _q(), _debug(read_debug_env())
{
    std::vector<cl::Device> vd;
    vd.push_back(_d);
    _c= cl::Context(vd);
    _q= cl::CommandQueue(_c, _d);
}

ocl::impl::be_data::be_data(const device& dev, const context& ctx)
    : _d(dev), _c(ctx), _q(cl::CommandQueue(_c, _d)), _debug(read_debug_env())
{
}

ocl::impl::be_data::be_data(const device& dev, const context& ctx,
                            const queue& qe)
    : _d(dev), _c(ctx), _q(qe)
{
}
