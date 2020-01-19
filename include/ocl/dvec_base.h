#if !defined (__OCL_DVEC_BASE_H__)
#define __OCL_DVEC_BASE_H__ 1

#include <ocl/config.h>
#include <ocl/be/data.h>
#include <memory>
#include <atomic>
#include <iostream>
#include <iomanip>

#define DEBUG_DVEC_BASE 0

namespace ocl {

    namespace impl {

#if DEBUG_DVEC_BASE>0
        template <typename _I>
        struct _counter_state {
            enum es {
                construct,
                copy_construct,
                move_construct,
                copy_assign,
                move_assign,
                destruct,
                objects,
                LAST
            };
            _I _v[LAST];
            _counter_state() : _v{} {}
            void
            inc(std::initializer_list<es> l) {
                for (auto b=l.begin(), e=l.end(); b!=e; ++b) {
                    auto i=*b;
                    ++_v[i];
                }
            }
            void
            dec(es i) {
                --_v[i];
            }
        };

        template <typename _I>
        std::ostream&
        operator<<(std::ostream& s, const _counter_state<_I>& i) {
            using es=typename _counter_state<_I>::es;
            s << "construct:      " << std::setw(4)
              << i._v[es::construct] << '\n'
              << "copy construct: " << std::setw(4)
              << i._v[es::copy_construct] << '\n'
              << "move construct: " << std::setw(4)
              << i._v[es::move_construct] << '\n'
              << "copy assign:    " << std::setw(4)
              << i._v[es::copy_assign] << '\n'
              << "move assign:    " << std::setw(4)
              << i._v[es::move_assign] << '\n'
              << "destruct:       " << std::setw(4)
              << i._v[es::destruct] << '\n'
              << "objects:        " << std::setw(4)
              << i._v[es::objects] << '\n';
            return s;
        }


        template <typename _TAG>
        struct _counter {
            using st_t = _counter_state<std::atomic<int64_t> >;
            static
            std::unique_ptr<st_t> _instance;
        public:
            _counter() {
                _instance->inc({st_t::construct, st_t::objects});
            }
            _counter(const _counter& ) {
                _instance->inc({st_t::construct,
                                st_t::copy_construct,
                                st_t::objects});
            }
            _counter(_counter&& ) {
                _instance->inc({st_t::construct,
                                st_t::move_construct,
                                st_t::objects});
            }
            _counter& operator=(const _counter& ) {
                _instance->inc({st_t::copy_assign});
                return *this;
            }
            _counter& operator=(_counter&& ) {
                _instance->inc({st_t::move_assign});
                return *this;
            }
            ~_counter() {
                _instance->inc({st_t::destruct});
                _instance->dec(st_t::objects);
            }
            static
            _counter_state<int64_t> state() {
                _counter_state<int64_t> d;
                const st_t& s=*_instance;
                for (size_t i=0; i< st_t::LAST; ++i)
                    d._v[i] = s._v[i];
                return d;
            }
        };

        template <typename _TAG>
        std::unique_ptr<typename _counter<_TAG>::st_t>
        _counter<_TAG>::_instance=std::make_unique<_counter<_TAG>::st_t>();
#else
        template <typename _TAG>
        struct _counter {
            static
            const char* state() {
                return "object statistics are unavailable\n";
            }
        };
#endif
    }

    // dvec base class wrapping an opencl buffer and a
    // (shared) pointer to opencl backend data
    class dvec_base : public impl::_counter<dvec_base> {
        using base_type = impl::_counter<dvec_base>;
        // shared pointer to the backend data
        be::data_ptr _bed;
        // backend buffer object
        be::buffer _b;
    protected:
        // destructor
        ~dvec_base();
        // default constructor
        dvec_base();
        // constructor, with size
        explicit dvec_base(std::size_t s);
        // constructor, with size and source
        dvec_base(std::size_t s, const void* p);
        // constructor with backend data ptr
        dvec_base(be::data_ptr pbe, std::size_t s);
        // constructor with backend data ptr, size and source
        dvec_base(be::data_ptr pbe, std::size_t s, const void* p);
        // copy constructor
        dvec_base(const dvec_base& r);
        // move constructor
        dvec_base(dvec_base&& r);
        // assignment operator
        dvec_base& operator=(const dvec_base& r);
        // move assignment operator
        dvec_base& operator=(dvec_base&& r);
        // swap two dvec base objects
        dvec_base& swap(dvec_base& r);
        // fill pattern p with pattern length into this (OPENCL 1.2)
        // void fill_on_device(const void* p, size_t ps);
        // device device copy
        void copy_on_device(const dvec_base& r);
        // host device copy
        void copy_from_host(const void* src);
        // host device copy
        void copy_from_host(const void* src, size_t buf_offs, size_t len);
        // device host copy
        void copy_to_host(void* dst)
            const;
        // device host copy
        void copy_to_host(void* dst, size_t buf_offs, size_t len)
            const;
    public:
        // return the size of the dvec in bytes
        std::size_t buffer_size() const;
        // return the underlying opencl buffer
        const be::buffer& buf() const;
        // return the opencl backend information
        be::data_ptr
        backend_data();
        // return the opencl backend information
        const be::data_ptr
        backend_data() const;
    };
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_DVEC_BASE_H__
