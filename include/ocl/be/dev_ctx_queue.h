#if !defined (__OCL_BE_DEV_CTX_QUEUE_H__)
#define __OCL_BE_DEV_CTX_QUEUE_H__ 1

#include <ocl/config.h>
#include <ocl/be/devices.h>
#include <mutex>

namespace ocl {

    namespace be {

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
    }

}

// Local variables:
// mode: c++
// end:
#endif
