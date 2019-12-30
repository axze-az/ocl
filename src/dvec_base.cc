#include "ocl/dvec.h"
#include <iomanip>

#define DEBUG_dvec_base 0

namespace {

#if DEBUG_dvec_base>0
    void
    print_this(const void* p, const  ocl::be::buffer& b) {
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
    print_r(const void* p, const  ocl::be::buffer& b) {
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
    print_this(const void* p, const  ocl::be::buffer& b) {
        static_cast<void>(p);
        static_cast<void>(b);
    }

    void
    print_r(const void* p, const  ocl::be::buffer& b) {
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

ocl::dvec_base::~dvec_base()
{
    trace t(__PRETTY_FUNCTION__, this);
    print_this(this, _b);
}

ocl::dvec_base::dvec_base()
    : _bed{}, _b{}
{
    trace t(__PRETTY_FUNCTION__, this);
    print_this(this, _b);
}

ocl::dvec_base::dvec_base(be::data_ptr pbe, std::size_t s)
    : _bed{pbe},
      _b{_bed->dcq().c(), s}
{
    trace t(__PRETTY_FUNCTION__, this);
    print_this(this, _b);
}

ocl::dvec_base::dvec_base(std::size_t s)
    : _bed{be::data::instance()},
      _b{_bed->dcq().c(), s}
{
    trace t(__PRETTY_FUNCTION__, this);
    print_this(this, _b);
}

ocl::dvec_base::dvec_base(const dvec_base& r)
    : _bed{r._bed},
      _b(_bed != nullptr ?
         be::buffer(_bed->dcq().c(), r.buffer_size()) :
         be::buffer())
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    copy_on_device(r);
    print_this(this, _b);
}

ocl::dvec_base::dvec_base(dvec_base&& r)
    : _bed(), _b()
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    swap(r);
    print_this(this, _b);
}

ocl::dvec_base&
ocl::dvec_base::operator=(const dvec_base& r)
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    if (this != &r) {
        if (buffer_size() == r.buffer_size()) {
            copy_on_device(r);
        } else {
            dvec_base t(r);
            swap(t);
        }
    }
    print_this(this, _b);
    return *this;
}

ocl::dvec_base&
ocl::dvec_base::operator=(dvec_base&& r)
{
    trace t(__PRETTY_FUNCTION__, this);
    print_r(&r, r._b);
    print_this(this, _b);
    return swap(r);
}

ocl::dvec_base&
ocl::dvec_base::swap(dvec_base& r)
{
    trace t(__PRETTY_FUNCTION__, this);
    std::swap(_bed, r._bed);
    std::swap(_b, r._b);
    print_r(&r, r._b);
    print_this(this, _b);
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
            be::event ev= q.enqueue_copy_buffer(_b, r._b,0, 0, s, wl);
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
            ev=q.enqueue_write_buffer(_b, 0, s, p, wl);
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
            ev= q.enqueue_read_buffer(_b, 0, s, p, wl);
            q.flush();
            wl.clear();
        }
        ev.wait();
    }
}
