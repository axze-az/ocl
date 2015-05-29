#if !defined (__OCL_IMPL_BE_DATA__)
#define __OCL_IMPL_BE_DATA__ 1

#include <ocl/config.h>
#include <ocl/impl_devices.h>
#include <map>
#include <mutex>
#include <memory> // for shared_ptr
#include <atomic>

namespace ocl {

    namespace impl {
        // for mesa we need the keep the programs
        struct pgm_kernel_lock {
            program _p;
            kernel _k;
            std::shared_ptr<std::mutex> _m;
            pgm_kernel_lock(const program& p,
                            const kernel& k) :
                _p(p), _k(k), _m(new std::mutex()) {}
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

            bool try_lock() {
                return _m.try_lock();
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

            typedef std::map<const void*, pgm_kernel_lock>
            kernel_map_type;
            typedef kernel_map_type::iterator iterator;

            iterator
            find(const void* cookie) {
                return _kmap.find(cookie);
            }

            std::pair<iterator, bool>
            insert(const void* cookie, const pgm_kernel_lock& v) {
                return _kmap.insert(std::make_pair(cookie,
                                                   v));
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

            // shared, default backend data
            static
            std::shared_ptr<be_data>
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

        private:
            std::mutex _m;
            device _d;
            context _c;
            queue _q;
            kernel_map_type _kmap;


            static std::mutex _instance_mutex;
            static std::atomic<bool> _init;
            static std::shared_ptr<be_data> _default;
        };

        typedef std::shared_ptr<be_data> be_data_ptr;


    }

}

// Local variables:
// mode: c++
// end:
#endif
