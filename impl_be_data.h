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
                        
                        iterator begin() {
                                return _kmap.begin();
                        }

                        iterator end() {
                                return _kmap.end();
                        }

                        static
                        be_data* instance();
                private:
                        std::mutex _m;
                        device _d;
                        context _c;
                        queue _q;
                        kernel_map_type _kmap;
                        
                        be_data();
                        
                        static std::mutex _instance_mutex;
                        static std::atomic<be_data*> _instance;
                };

                inline context& be_context() {
                        return be_data::instance()->c();
                }
                
                inline device& be_device() {
                        return be_data::instance()->d();
                }

                inline queue& be_queue() {
                        return be_data::instance()->q();
                }

        }

}

// Local variables:
// mode: c++
// end:
#endif
