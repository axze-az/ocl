#include "ocl/be/data.h"
#include <cstdlib>
#include <iomanip>
#include <thread>

ocl::be::kernel_cache::kernel_cache()
    : _kmap(), _mtx()
{
}

ocl::be::kernel_cache::iterator
ocl::be::kernel_cache::
find(const void* cookie)
{
    return _kmap.find(cookie);
}

void
ocl::be::kernel_cache::
erase(iterator f)
{
    _kmap.erase(f);
}

std::pair<ocl::be::kernel_cache::iterator, bool>
ocl::be::kernel_cache::
insert(const void* cookie, const pgm_kernel_lock& v)
{
    return _kmap.insert(std::make_pair(cookie, v));
}

void
ocl::be::kernel_cache::clear()
{
    _kmap.clear();
}

std::size_t
ocl::be::kernel_cache::size() const
{
    return _kmap.size();
}

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
    // evs.insert(ev);
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

ocl::be::data_ptr
ocl::be::data::instance()
{
    if (_init == false) {
        std::unique_lock<mutex> _l(_instance_mutex);
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
    std::unique_lock _lck(_debug_mutex);
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
