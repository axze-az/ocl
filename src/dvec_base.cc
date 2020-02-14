#include "ocl/dvec_base.h"
#include <iomanip>
#include "ocl/dvec.h"

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
      _b{_bed->dcq().c(), s,
         be::buffer::read_write|be::buffer::copy_host_ptr,
         const_cast<void*>(p)}
{
    // copy_from_host(p);
}

ocl::dvec_base::
dvec_base(const be::data_ptr& pbe, std::size_t s)
    : base_type(), _bed{pbe},
      _b{_bed->dcq().c(), s}
{
}

ocl::dvec_base::
dvec_base(const be::data_ptr& pbe, std::size_t s, const void* p)
    : base_type(), _bed{pbe},
      _b{_bed->dcq().c(), s,
         be::buffer::read_write|be::buffer::copy_host_ptr,
         const_cast<void*>(p)}
{
    // copy_from_host(p);
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

void
ocl::dvec_base::copy_on_device(const dvec_base& r)
{
    size_t s =r.buffer_size();
    // TODO: check for Mesa devices and use copy via
    // kernel, otherwise use enqueue_copy_buffer
#if 0
    if (s==0)
        return;
    if (s & 1) {
        dvec<char>& dst=static_cast<dvec<char>&>(*this);
        const dvec<char>& src=static_cast<const dvec<char>&>(r);
        execute(dst, src, this->backend_data(), s);
        return;
    }
    if (s & 2) {
        dvec<uint16_t>& dst=static_cast<dvec<uint16_t>&>(*this);
        const dvec<uint16_t>& src=static_cast<const dvec<uint16_t>&>(r);
        execute(dst, src, this->backend_data(), s>>1);
        return;
    }
#if 1
    dvec<float>& dst=static_cast<dvec<float>&>(*this);
    const dvec<float>& src=static_cast<const dvec<float>&>(r);
    execute(dst, src, this->backend_data(), s>>2);
#else
    if (s & 4) {
        dvec<uint32_t>& dst=static_cast<dvec<uint32_t>&>(*this);
        const dvec<uint32_t>& src=static_cast<const dvec<uint32_t>&>(r);
        execute(dst, src, this->backend_data(), s>>2);
        return;
    }
    if (s & 8) {
        dvec<cl_uint2>& dst=static_cast<dvec<cl_uint2>&>(*this);
        const dvec<cl_uint2>& src=static_cast<const dvec<cl_uint2>&>(r);
        execute(dst, src, this->backend_data(), s>>3);
        return;
    }
    if (s & 16) {
        dvec<cl_uint4>& dst=static_cast<dvec<cl_uint4>&>(*this);
        const dvec<cl_uint4>& src=static_cast<const dvec<cl_uint4>&>(r);
        execute(dst, src, this->backend_data(), s>>4);
        return;
    }
    if (s & 32) {
        dvec<cl_uint8>& dst=static_cast<dvec<cl_uint8>&>(*this);
        const dvec<cl_uint8>& src=static_cast<const dvec<cl_uint8>&>(r);
        execute(dst, src, this->backend_data(), s>>5);
        return;
    }
    dvec<cl_uint16>& dst=static_cast<dvec<cl_uint16>&>(*this);
    const dvec<cl_uint16>& src=static_cast<const dvec<cl_uint16>&>(r);
    execute(dst, src, this->backend_data(), s>>6);
#endif
#else
    if (__likely(s)) {
        auto& dcq=_bed->dcq();
        auto& q= dcq.q();
        auto& wl=dcq.wl();
        {
            be::scoped_lock _ql(dcq.mtx());
            be::event ev= q.enqueue_copy_buffer(r._b, _b, 0, 0, s, wl);
            q.flush();
            wl.clear();
            wl.insert(ev);
        }
    }
#endif
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
            be::scoped_lock _ql(dcq.mtx());
            ev=q.enqueue_write_buffer_async(_b, 0, s, p, wl);
            q.flush();
            wl.clear();
        }
        ev.wait();
    }
}

void
ocl::dvec_base::copy_from_host(const void* p, size_t buf_offs, size_t s)
{
    if (__likely(s)) {
        auto& dcq=_bed->dcq();
        auto& q= dcq.q();
        auto& wl=dcq.wl();
        be::event ev;
        {
            be::scoped_lock _ql(dcq.mtx());
            ev=q.enqueue_write_buffer_async(_b, buf_offs, s, p, wl);
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
            be::scoped_lock _ql(dcq.mtx());
            ev= q.enqueue_read_buffer_async(_b, 0, s, p, wl);
            q.flush();
            wl.clear();
        }
        ev.wait();
    }
}

void
ocl::dvec_base::copy_to_host(void* p, size_t buf_offs, size_t s)
    const
{
    if (__likely(s)) {
        auto& dcq=_bed->dcq();
        auto& q= dcq.q();
        auto& wl=dcq.wl();
        be::event ev;
        {
            be::scoped_lock _ql(dcq.mtx());
            ev= q.enqueue_read_buffer_async(_b, buf_offs, s, p, wl);
            q.flush();
            wl.clear();
        }
        ev.wait();
    }
}
