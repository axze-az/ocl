#include "vector.h"
#include <iomanip>

ocl::vector_base::~vector_base()
{
}

ocl::vector_base::vector_base()
    : _bed{}, _b{}
{
}

ocl::vector_base::vector_base(std::size_t s)
    : _bed{impl::be_data::instance()},
      _b{_bed->c(), s}
{
}

ocl::vector_base::vector_base(const vector_base& r)
    : _bed(r._bed), _b(r._b)
{
    copy_on_device(r);
}

ocl::vector_base::vector_base(vector_base&& r)
    : _bed(), _b()
{
    swap(r);
}

ocl::vector_base&
ocl::vector_base::operator=(const vector_base& r)
{
    if (this != &r) {
        if (buffer_size() == r.buffer_size()) {
            copy_on_device(r);
        } else {
            vector_base t(r);
            swap(t);
        }
    }
    return *this;
}

ocl::vector_base&
ocl::vector_base::operator=(vector_base&& r)
{
    return swap(r);
}

ocl::vector_base&
ocl::vector_base::swap(vector_base& r)
{
    std::swap(_bed, r._bed);
    std::swap(_b, r._b);
    return *this;
}

std::size_t
ocl::vector_base::buffer_size()
    const
{
    return _b.size();
}

const ocl::impl::buffer&
ocl::vector_base::buf()
    const
{
    return _b;
}

ocl::impl::be_data_ptr
ocl::vector_base::backend_data()
{
    return _bed;
}

const ocl::impl::be_data_ptr
ocl::vector_base::backend_data()
    const
{
    return _bed;
}

void
ocl::vector_base::copy_on_device(const vector_base& r)
{
    size_t s =r.buffer_size();
    if (__likely(s)) {
        impl::queue& q= _bed->q();
        std::vector<impl::event> next_evs(1);
        auto& evs=_bed->evs();
        impl::event ev= q.enqueue_copy_buffer(_b, r._b,0, 0, s, evs);
        q.flush();
        evs.clear();
        evs.insert(ev);
    }
}

void
ocl::vector_base::copy_from_host(const void* p)
{
    std::size_t s=buffer_size();
    if (__likely(s)) {
        impl::queue& q= _bed->q();
        auto& evs=_bed->evs();
        impl::event ev=q.enqueue_write_buffer(_b, 0, s, p, evs);
        q.flush();
        evs.clear();
        evs.insert(ev);
    }
}

void
ocl::vector_base::copy_to_host(void* p)
    const
{
    std::size_t s=buffer_size();
    if (__likely(s)) {
        impl::queue& q= _bed->q();
        auto& evs= _bed->evs();
        impl::event ev= q.enqueue_read_buffer(_b, 0, s, p, evs);
        q.flush();
        evs.clear();
        evs.insert(ev);
    }
}
