#include "ocl/dvec_base.h"
#include <iomanip>


ocl::dvec_base::~dvec_base()
{
}

ocl::dvec_base::dvec_base()
    : base_type(), _bed{}, _b{}
{
}


ocl::dvec_base::dvec_base(std::size_t s)
    : base_type(), _bed{be::data::instance()},
      _b{_bed->dcq().c(), s}
{
}

ocl::dvec_base::dvec_base(std::size_t s, const void* p)
    : base_type(), _bed{be::data::instance()},
      _b{_bed->dcq().c(), s}
{
    copy_from_host(p);
}

ocl::dvec_base::dvec_base(be::data_ptr pbe, std::size_t s)
    : base_type(), _bed{pbe},
      _b{_bed->dcq().c(), s}
{
}

ocl::dvec_base::dvec_base(be::data_ptr pbe, std::size_t s, const void* p)
    : base_type(), _bed{pbe},
      _b{_bed->dcq().c(), s}
{
    copy_from_host(p);
}

ocl::dvec_base::dvec_base(const dvec_base& r)
    : base_type(r),
      _bed{r._bed},
      _b(_bed != nullptr ?
         be::buffer(_bed->dcq().c(), r.buffer_size()) :
         be::buffer())
{
    copy_on_device(r);
}

ocl::dvec_base::dvec_base(dvec_base&& r)
    : base_type(std::move(r)), _bed(), _b()
{
    swap(r);
}

ocl::dvec_base&
ocl::dvec_base::operator=(const dvec_base& r)
{
    base_type::operator=(r);
    if (this != &r) {
        if (buffer_size() == r.buffer_size()) {
            copy_on_device(r);
        } else {
            dvec_base t(r);
            swap(t);
        }
    }
    return *this;
}

ocl::dvec_base&
ocl::dvec_base::operator=(dvec_base&& r)
{
    base_type::operator=(std::move(r));
    return swap(r);
}

ocl::dvec_base&
ocl::dvec_base::swap(dvec_base& r)
{
    std::swap(_bed, r._bed);
    std::swap(_b, r._b);
    return *this;
}

std::size_t
ocl::dvec_base::buffer_size()
    const
{
    size_t s=0;
    if (_b.get() != nullptr)
        s= _b.size();
    return s;
}

const ocl::be::buffer&
ocl::dvec_base::buf()
    const
{
    return _b;
}

ocl::be::data_ptr
ocl::dvec_base::backend_data()
{
    return _bed;
}

const ocl::be::data_ptr
ocl::dvec_base::backend_data()
    const
{
    return _bed;
}

void
ocl::dvec_base::copy_on_device(const dvec_base& r)
{
    size_t s =r.buffer_size();
    if (__likely(s)) {
        auto& dcq=_bed->dcq();
        auto& q= dcq.q();
        auto& wl=dcq.wl();
        {
            std::unique_lock<be::mutex> _ql(dcq.mtx());
            be::event ev= q.enqueue_copy_buffer(r._b, _b, 0, 0, s, wl);
            q.flush();
            wl.clear();
            wl.insert(ev);
        }
    }
}

void
ocl::dvec_base::copy_from_host(const void* p)
{
    std::size_t s=buffer_size();
    if (__likely(s)) {
        auto& dcq=_bed->dcq();
        auto& q= dcq.q();
        auto& wl=dcq.wl();
        be::event ev;
        {
            std::unique_lock<be::mutex> _ql(dcq.mtx());
            ev=q.enqueue_write_buffer_async(_b, 0, s, p, wl);
            q.flush();
            wl.clear();
        }
        ev.wait();
    }
}

void
ocl::dvec_base::copy_to_host(void* p)
    const
{
    std::size_t s=buffer_size();
    if (__likely(s)) {
        auto& dcq=_bed->dcq();
        auto& q= dcq.q();
        auto& wl=dcq.wl();
        be::event ev;
        {
            std::unique_lock<be::mutex> _ql(dcq.mtx());
            ev= q.enqueue_read_buffer_async(_b, 0, s, p, wl);
            q.flush();
            wl.clear();
        }
        ev.wait();
    }
}
