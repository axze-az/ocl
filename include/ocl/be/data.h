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
#if !defined (__OCL_BE_DATA_H__)
#define __OCL_BE_DATA_H__ 1

#include <ocl/config.h>
#include <ocl/types.h>
#include <ocl/be/kernel_cache.h>
#include <ocl/be/dev_ctx_queue.h>
#include <memory> // for shared_ptr
#include <atomic>

namespace ocl {

    namespace be {

        class data {
            // backend device, context and queue
            dev_ctx_queue _dcq;
            // the kernel cache
            kernel_cache _kcache;
            // debug flags
            uint32_t _debug;
            // read the debug flags from the environment
            static
            uint32_t read_debug_env();
            // singleton support for data:
            static mutex _instance_mutex;
            static std::atomic<bool> _init;
            static std::shared_ptr<data> _default;
            // mutex protecting the debug output
            static mutex _debug_mutex;
        public:
            data(const data&) = delete;
            data& operator=(const data&) = delete;
            // default constructor
            data();
            // construction from device
            data(const device& dev);
            // construction from device and context
            data(const device& dev, const context& ctx);
            // construction from device, context and queue
            data(const device& dev, const context& ctx,
                 const queue& qe);
            // destructor
            ~data();
            // return device, context and queue
            dev_ctx_queue& dcq() { return _dcq; }
            // return the kernel cache
            kernel_cache& kcache() { return _kcache; }
            // return the debug flags
            const uint32_t& debug() const { return _debug; }
            // enqueue a kernel with already bound arguments with
            // size s
            event
            enqueue_1d_kernel(const kernel& k, size_t s);
            // enqueue a kernel with already bound arguments using
            // ki
            event
            enqueue_1d_kernel(const kernel& k,
                              const kexec_1d_info& ki);

            enum class query_bool {
                fp16_fma,
                fp32_fma,
                fp64_fma
            };

            bool supports(query_bool q);

            // shared, default backend data
            static
            std::shared_ptr<data>
            instance();
            // create a shared ptr
            static
            std::shared_ptr<data>
            create(const device& d);
            // create a shared ptr
            static
            std::shared_ptr<data>
            create(const device& dev, const context& ctx);
            // create a shared ptr
            static
            std::shared_ptr<data>
            create(const device& dev, const context& ctx,
                   const queue& qe);
            // debug output
            static
            void
            debug_print(const std::string& m);
        };
        // backend pointer definition
        using data_ptr = std::shared_ptr<data>;


    }

}

// Local variables:
// mode: c++
// end:
#endif
