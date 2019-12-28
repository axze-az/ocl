#if !defined (__OCL_BE_DATA__)
#define __OCL_BE_DATA__ 1

#include <ocl/config.h>
#include <ocl/be/devices.h>
#include <map>
#include <mutex>
#include <memory> // for shared_ptr
#include <atomic>
#include <iostream>

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
            using kmap_t = std::map<const void*, pgm_kernel_lock>;
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
            find(const void* cookie);
            // erase an entry
            void
            erase(iterator f);
            // insert an entry
            std::pair<iterator, bool>
            insert(const void* cookie, const pgm_kernel_lock& v);
            // clear the cache
            void clear();
            // return the size
            size_t size() const;
        };

        class dev_ctx_queue {
            // backend device
            device _d;
            // backend context
            context _c;
            // backend queue
            queue _q;
            // list with pending events
            wait_list _wl;
            // mutex protecting access to _q and _wl
            mutex _mtx;
        public:
            dev_ctx_queue();
            dev_ctx_queue(const dev_ctx_queue&) = delete;
            dev_ctx_queue& operator=(const dev_ctx_queue&) = delete;
            dev_ctx_queue(const device& dd);
            dev_ctx_queue(const device& dd, const context& ctx);
            dev_ctx_queue(const device& dd, const context& ctx,
                          const queue& q);
            device& d() { return _d; }
            context& c() { return _c; }
            queue& q() { return _q; }
            wait_list& wl() { return _wl; }
            mutex& mtx() { return _mtx; }
        };

        class data {
            dev_ctx_queue _dcq;
            kernel_cache _kcache;
            uint32_t _debug;

            static
            uint32_t read_debug_env();
            static mutex _instance_mutex;
            static std::atomic<bool> _init;
            static std::shared_ptr<data> _default;
        public:
            data(const data&) = delete;
            data& operator=(const data&) = delete;
            data();
            data(const device& dev);
            data(const device& dev, const context& ctx);
            data(const device& dev, const context& ctx,
                 const queue& qe);
            ~data();
            dev_ctx_queue& dcq() { return _dcq; }
            kernel_cache& kcache() { return _kcache; }
            uint32_t debug() const { return _debug; }
            // enqueue a kernel with already bound arguments with
            // size s
            event
            enqueue_kernel(pgm_kernel_lock& pk, size_t s);
            // shared, default backend data
            static
            std::shared_ptr<data>
            instance();

            static
            std::shared_ptr<data>
            create(const device& d);

            static
            std::shared_ptr<data>
            create(const device& dev, const context& ctx);

            static
            std::shared_ptr<data>
            create(const device& dev, const context& ctx,
                   const queue& qe);
        };
        typedef std::shared_ptr<data> data_ptr;


    }

}

// Local variables:
// mode: c++
// end:
#endif
