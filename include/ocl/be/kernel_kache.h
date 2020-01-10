#if !defined (__OCL_BE_KERNEL_CACHE_H__)
#define __OCL_BE_KERNEL_CACHE_H__ 1

#include <ocl/config.h>
#include <ocl/be/devices.h>
#include <ocl/be/kernel_key.h>
#include <map>

namespace ocl {

    namespace be {

        using mutex = std::mutex;
        
        // for mesa we need the keep the programs
        struct pgm_kernel_lock {
            program _p;
            kernel _k;
            std::shared_ptr<mutex> _m;
            pgm_kernel_lock(const program& p,
                            const kernel& k) :
                _p(p), _k(k), _m(new mutex()) {}
            void lock() { _m->lock(); }
            void unlock() { _m->unlock(); }
        };

        class kernel_cache {
            using kmap_t = std::map<kernel_key, pgm_kernel_lock>;
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
            insert(const kernel_key& cookie, const pgm_kernel_lock& v);
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
