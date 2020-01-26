#if !defined (__OCL_BE_KERNEL_CACHE_H__)
#define __OCL_BE_KERNEL_CACHE_H__ 1

#include <ocl/config.h>
#include <ocl/be/devices.h>
#include <ocl/be/kernel_key.h>
#include <mutex>
#include <map>

namespace ocl {

    namespace be {

        using mutex = std::mutex;

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
            void lock() { _h->_m.lock(); }
            void unlock() { _h->_m.unlock(); }
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
