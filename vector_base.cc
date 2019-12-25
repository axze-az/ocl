#include "vector.h"
#include <iomanip>

#define DEBUG_VECTOR_BASE 0

namespace {

#if DEBUG_VECTOR_BASE>0
    void
    print_this(const void* p, const  ocl::impl::buffer& b) {
        auto bb=b.get();
        std::cout << "this: " << p << ' ';
        if (bb != nullptr) {
            std::size_t rc=b.reference_count();
            if (rc > 1) {
                std::cout << rc << " ---------" << std::endl;
            } else {
                std::cout << rc << std::endl;
            }
        } else {
            std::cout << 0 << std::endl;
        }
    }

    void
    print_r(const void* p, const  ocl::impl::buffer& b) {
        auto bb=b.get();
        std::cout << "   r: " << p << ' ';
        if (bb != nullptr) {
            std::size_t rc=b.reference_count();
            if (rc > 1) {
                std::cout << rc << " ---------" << std::endl;
            } else {
                std::cout << rc << std::endl;
            }
        } else {
            std::cout << 0 << std::endl;
        }
    }

    struct trace {
        const char* _f;
        const void* _p;
        void pr() {
            std::cout << _f  << ' ' << _p << std::endl;
        }
        trace (const char* f, void* p) : _f(f), _p(p) {
            std::cout << "enter "; pr();
        }
        ~trace() {
            std::cout << "leave "; pr();
        }
    };
#else
    void
    print_this(const void* p, const  ocl::impl::buffer& b) {
        static_cast<void>(p);
        static_cast<void>(b);
    }

    void
    print_r(const void* p, const  ocl::impl::buffer& b) {
        static_cast<void>(p);
        static_cast<void>(b);
    }

    struct trace {
        trace (const char* f, void* p) {
            static_cast<void>(f);
            static_cast<void>(p);
        }
    };
#endif
}

ocl::vector_base::~vector_base()
{
    trace t(__PRETTY_FUNCTION__, this);
    print_this(this, _b);
}

ocl::vector_base::vector_base()
    : _bed{}, _b{}
{
    trace t(__PRETTY_FUNCTION__, this);
    print_this(this, _b);
}

ocl::vector_base::vector_base(std::size_t s)
    : _bed{impl::be_data::instance()},
      _b{_bed->c(), s}
{
    trace t(__PRETTY_FUNCTION__, this);
    print_this(this, _b);
}

ocl::vector_base::vector_base(const vector_base& r)
    : _bed{impl::be_data::instance()},
      _b{_bed->c(), r.buffer_size()}
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    copy_on_device(r);
    print_this(this, _b);
}

ocl::vector_base::vector_base(vector_base&& r)
    : _bed(), _b()
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    swap(r);
    print_this(this, _b);
}

ocl::vector_base&
ocl::vector_base::operator=(const vector_base& r)
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    if (this != &r) {
        if (buffer_size() == r.buffer_size()) {
            copy_on_device(r);
        } else {
            vector_base t(r);
            swap(t);
        }
    }
    print_this(this, _b);
    return *this;
}

ocl::vector_base&
ocl::vector_base::operator=(vector_base&& r)
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    print_this(this, _b);
    return swap(r);
}

ocl::vector_base&
ocl::vector_base::swap(vector_base& r)
{
    trace t(__PRETTY_FUNCTION__, this);
    std::swap(_bed, r._bed);
    std::swap(_b, r._b);
    print_r(&r, r._b);
    print_this(this, _b);
    return *this;
}

std::size_t
ocl::vector_base::buffer_size()
    const
{
    size_t s=0;
    if (_b.get() != nullptr)
        s= _b.size();
    return s;
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
        ev.wait();
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
        ev.wait();
    }
}
