//
// Copyright (C) 2010-2025 Axel Zeuner
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
#if !defined (__OCL_BE_KERNEL_CACHE_H__)
#define __OCL_BE_KERNEL_CACHE_H__ 1

#include <ocl/config.h>
#include <ocl/be/devices.h>
#include <ocl/be/kernel_key.h>
#include <memory>
#include <mutex>
#include <map>

namespace ocl {

    namespace be {

        using mutex = std::mutex;

        class scoped_lock {
            mutex& _mtx;
        public:
            scoped_lock(mutex& m);
            ~scoped_lock();
        };

        struct kernel_with_lock {
            program _p;
            kernel _k;
            mutex _m;
            kernel_with_lock(const program& p, const kernel& k)
                : _p(p), _k(k), _m() {}
        };

        class kernel_handle {
            std::shared_ptr<kernel_with_lock> _h;
        public:
            kernel_handle(const program& p, const kernel& k)
                : _h(std::make_shared<kernel_with_lock>(p, k)) {
            }
            kernel& k() { return _h->_k; }
            mutex& mtx() { return _h->_m; }
        };

        class kernel_cache {
            using kmap_t = std::map<kernel_key, kernel_handle>;
            kmap_t _kmap;
            mutex _mtx;
        public:
            using iterator = typename kmap_t::iterator;
            kernel_cache();
            // return the mutex
            mutex& mtx() { return _mtx; }
            // begin
            iterator
            begin() { return _kmap.begin(); }
            // end
            iterator
            end() { return _kmap.end(); }
            // find a kernel
            iterator
            find(const kernel_key& cookie);
            // erase an entry
            void
            erase(iterator f);
            // insert an entry
            std::pair<iterator, bool>
            insert(const kernel_key& cookie, const kernel_handle& v);
            // clear the cache
            void clear();
            // return the size
            size_t size() const;
        };
    }
}

// Local variables:
// mode: c++
// end:
#endif
