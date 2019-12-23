#include "vector.h"

ocl::vector_base::~vector_base()
{
}

ocl::vector_base::vector_base()
    : _bed{}, _b{}
{
}

ocl::vector_base::vector_base(std::size_t s)
    : _bed{impl::be_data::instance()},
      _b{_bed->c(), CL_MEM_READ_WRITE, s}
{
}

ocl::vector_base::vector_base(vector_base&& r)
    : _bed(std::move(r._bed)), _b(std::move(r._b))
{
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
    std::size_t r{_b() != nullptr ?
            _b.getInfo<CL_MEM_SIZE>(nullptr) : 0};
    return r;
}

const cl::Buffer&
ocl::vector_base::buf()
    const
{
    return _b;
}

std::shared_ptr<ocl::impl::be_data>&
ocl::vector_base::backend_data()
{
    return _bed;
}

const std::shared_ptr<ocl::impl::be_data>&
ocl::vector_base::backend_data()
    const
{
    return _bed;
}

void
ocl::vector_base::copy_on_device(const vector_base& r)
{
    if (r.buffer_size()) {
        impl::event ev;
        impl::queue& q= backend_data()->q();
        auto& evs=backend_data()->evs();
        const std::vector<impl::event>* pev= evs.empty() ? nullptr : &evs;
        q.enqueueCopyBuffer(this->buf(),
                            r.buf(),
                            0, 0,
                            r.buffer_size(),
                            pev,
                            &ev);
        q.flush();
        evs.clear();
        evs.emplace_back(ev);
    }
}

void
ocl::vector_base::copy_from_host(const void* p)
{
    std::size_t s=buffer_size();
    if (s) {
        impl::queue& q= backend_data()->q();
        auto& evs=backend_data()->evs();
        const std::vector<impl::event>* pev= evs.empty() ? nullptr : &evs;
        impl::event ev;
        q.enqueueWriteBuffer(this->buf(),
                             false,
                             0, s,
                             p,
                             pev,
                             &ev);
        q.flush();
        evs.clear();
        evs.emplace_back(ev);
    }
}

void
ocl::vector_base::copy_to_host(void* p)
    const
{
    std::size_t s=buffer_size();
    if (s) {
        std::shared_ptr<impl::be_data> bed(backend_data());
        impl::queue& q= bed->q();
        auto& evs=bed->evs();
        const std::vector<impl::event>* pev= evs.empty() ? nullptr : &evs;
        impl::event ev;
        q.enqueueReadBuffer(this->buf(),
                            false,
                            0, s,
                            p,
                            pev,
                            &ev);
        q.flush();
        ev.wait();
        evs.clear();
    }
}
