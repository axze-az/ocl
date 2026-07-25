//
// Copyright (C) 2010-2026 Axel Zeuner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
//
#if !defined (__OCL_DVEC_BASE_H__)
#define __OCL_DVEC_BASE_H__ 1

#include <ocl/config.h>
#include <ocl/be/data.h>

#define DEBUG_DVEC_BASE 0

#if DEBUG_DVEC_BASE != 0
#include <iostream>
#include <iomanip>
#endif

namespace ocl {

    namespace impl {

#if DEBUG_DVEC_BASE>0
	// template class containing number about objects, object
	// states and object events
        template <typename _I>
        struct _counter_state {
            enum class es {
                construct,
                copy_construct,
                move_construct,
                copy_assign,
                move_assign,
                destruct,
                objects,
                LAST
            };
	    // date members
            _I _v[static_cast<int>(es::LAST)];
	    // default constructor
            _counter_state() : _v{} {}
	    // increment a list of entries
            void
            inc(std::initializer_list<es> l) {
                for (auto b=l.begin(), e=l.end(); b!=e; ++b) {
                    auto i=static_cast<size_t>(*b);
                    ++_v[i];
                }
            }
	    // decrement an entry
            void
            dec(es i) {
		auto si=static_cast<size_t>(i);
                --_v[si];
            }
        };

	// output operator for _counter_state
        template <typename _I>
        std::ostream&
        operator<<(std::ostream& s, const _counter_state<_I>& i) {
            using es=typename _counter_state<_I>::es;
            s << "construct:      " << std::setw(4)
              << i._v[size_t(es::construct)] << '\n'
              << "copy construct: " << std::setw(4)
              << i._v[size_t(es::copy_construct)] << '\n'
              << "move construct: " << std::setw(4)
              << i._v[size_t(es::move_construct)] << '\n'
              << "copy assign:    " << std::setw(4)
	      << i._v[size_t(es::copy_assign)] << '\n'
              << "move assign:    " << std::setw(4)
	      << i._v[size_t(es::move_assign)] << '\n'
              << "destruct:       " << std::setw(4)
	      << i._v[size_t(es::destruct)] << '\n'
              << "objects:        " << std::setw(4)
	      << i._v[size_t(es::objects)] << '\n';
            return s;
        }

	// template class managing a single shared _counter_state
        template <typename _TAG>
        struct _counter {
            using st_t = _counter_state<std::atomic<int64_t> >;
            static
            std::unique_ptr<st_t> _instance;
        public:
	    // constructor. increments number of constructions and objects
            _counter() {
                _instance->inc({st_t::es::construct, st_t::es::objects});
            }
	    // copy constructor. increments number of constructions,
	    // of copy constructions and objects
            _counter(const _counter& ) {
                _instance->inc({st_t::es::construct,
			st_t::es::copy_construct,
			st_t::es::objects});
            }
	    // move constructr.  increments number of constructions,
	    // of move constructions and objects
            _counter(_counter&& ) {
                _instance->inc({st_t::es::construct,
                                st_t::es::move_construct,
                                st_t::es::objects});
            }
	    // assignment operator. increments number of copy assignments
            _counter& operator=(const _counter& ) {
                _instance->inc({st_t::es::copy_assign});
                return *this;
            }
	    // move assignment operator. increments number of move assignments
            _counter& operator=(_counter&& ) {
                _instance->inc({st_t::es::move_assign});
                return *this;
            }
	    // move assignment operator. increments number of
	    // destructions and decements number of objects
            ~_counter() {
                _instance->inc({st_t::es::destruct});
                _instance->dec(st_t::es::objects);
            }
	    // return the _counter_state of the single instance
            static
            _counter_state<int64_t> state() {
                _counter_state<int64_t> d;
                const st_t& s=*_instance;
		constexpr auto e=static_cast<size_t>(st_t::es::LAST);
                for (size_t i=0; i< e; ++i)
                    d._v[i] = s._v[i];
                return d;
            }
        };

	// _instance poiner of _counter<_TAG>
        template <typename _TAG>
        std::unique_ptr<typename _counter<_TAG>::st_t>
        _counter<_TAG>::_instance=std::make_unique<_counter<_TAG>::st_t>();
#else
	// non debug _counter implementation
        template <typename _TAG>
        struct _counter {
	    // return an error message 
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
    public:
        // destructor
        ~dvec_base();
        // default constructor
        dvec_base();
        // constructor, with size
        explicit dvec_base(size_t s);
        // constructor, with size and source
        dvec_base(size_t s, const void* p);
        // constructor with backend data ptr
        dvec_base(const be::data_ptr& pbe, size_t s);
        // constructor with backend data ptr, size and source
        dvec_base(const be::data_ptr& pbe, size_t s, const void* p);
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
        // return the size of the underlying opencl buffer in bytes
        size_t buffer_size() const;
        // return the underlying opencl buffer
        const be::buffer& buf() const;
        // return the opencl backend information
        be::data_ptr&
        backend_data();
        // return the opencl backend information
        const be::data_ptr&
        backend_data() const;
    };
}

inline
const ocl::be::buffer&
ocl::dvec_base::buf()
    const
{
    return _b;
}

inline
ocl::size_t
ocl::dvec_base::buffer_size()
    const
{
    size_t s=0;
    if (_b.get() != nullptr)
        s= _b.size();
    return s;
}

inline
ocl::be::data_ptr&
ocl::dvec_base::backend_data()
{
    return _bed;
}

inline
const ocl::be::data_ptr&
ocl::dvec_base::backend_data()
    const
{
    return _bed;
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_DVEC_BASE_H__
