#include "impl_be_data.h"

std::mutex 
ocl::impl::be_data::_instance_mutex;
std::atomic<ocl::impl::be_data*>
ocl::impl::be_data::_instance(nullptr);

ocl::impl::be_data*
ocl::impl::be_data::instance()
{
        if (_instance == nullptr) {
                std::unique_lock<std::mutex> _l(_instance_mutex);
                if (_instance == nullptr)
                        _instance = new be_data();
        }
        return _instance;
}

#if 0                        
ocl::impl::be_data::be_data()
        : _d(default_device()), _c(_d), _q(_c, _d)
{
        // create context from device, command queue from context and
        // device
}
#else
ocl::impl::be_data::be_data()
        : _d(default_device()), _c(), _q()
{
        // create context from device, command queue from context and
        // device
        std::vector<cl::Device> vd;
        vd.push_back(_d);
        _c= cl::Context(vd);
        _q= cl::CommandQueue(_c, _d);
}
#endif

