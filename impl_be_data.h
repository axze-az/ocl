#if !defined (__OCL_IMPL_BE_DATA__)
#define __OCL_IMPL_BE_DATA__ 1

#include <ocl/config.h>
#include <ocl/impl_devices.h>
#include <map>
#include <mutex>
#include <memory> // for shared_ptr
#include <atomic>
#include <iostream>

namespace ocl {

    namespace impl {

        struct mutex : public std::mutex {
            using base_type = std::mutex;
            using base_type::base_type;
#if 0
            void lock() {
                try {
                    base_type::lock();
                }
                catch (...) {
                    std::cout << "lock failed\n";
                }
            }
            void unlock() {
                try {
                    base_type::unlock();
                }
                catch (...) {
                    std::cout << "unlock failed\n";
                }
            }
#endif
        };


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

        class be_data {
        public:
            be_data(const be_data&) = delete;
            be_data& operator=(const be_data&) = delete;

            void lock() {
                _m.lock();
            }

            void unlock() {
                _m.unlock();
            }

            device& d() {
                return _d;
            }
            queue& q() {
                return _q;
            }
            context& c() {
                return _c;
            }

            wait_list& evs() {
                return _ev;
            }


            uint32_t debug() const {
                return _debug;
            }

            typedef std::map<const void*, pgm_kernel_lock>
            kernel_map_type;
            typedef kernel_map_type::iterator iterator;

            iterator
            find(const void* cookie) {
                return _kmap.find(cookie);
            }

            void
            erase(iterator f) {
                _kmap.erase(f);
            }

            std::pair<iterator, bool>
            insert(const void* cookie, const pgm_kernel_lock& v) {
                return _kmap.insert(std::make_pair(cookie, v));
            }

            void clear() {
                _kmap.clear();
            }

            iterator begin() {
                return _kmap.begin();
            }

            iterator end() {
                return _kmap.end();
            }

            // enqueue a kernel with already bound arguments with
            // size s
            void
            enqueue_kernel(pgm_kernel_lock& pk, size_t s);

            // shared, default backend data
            static
            be_data*
            instance();

            static
            std::shared_ptr<be_data>
            create(const device& d);

            static
            std::shared_ptr<be_data>
            create(const device& dev, const context& ctx);

            static
            std::shared_ptr<be_data>
            create(const device& dev, const context& ctx,
                   const queue& qe);

            be_data();
            be_data(const device& dev);
            be_data(const device& dev, const context& ctx);
            be_data(const device& dev, const context& ctx,
                    const queue& qe);

            ~be_data();
        private:
            mutex _m;
            device _d;
            context _c;
            queue _q;
            kernel_map_type _kmap;
            wait_list _ev;
            uint32_t _debug;

            static
            uint32_t read_debug_env();

            static mutex _instance_mutex;
            static std::atomic<bool> _init;
            static std::shared_ptr<be_data> _default;
        };

        typedef be_data* be_data_ptr;


    }

}

// Local variables:
// mode: c++
// end:
#endif
