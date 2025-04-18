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
#if !defined (__OCL_BE_ARGUMENT_BUFFER_H__)
#define __OCL_BE_ARGUMENT_BUFFER_H__ 1

#include <ocl/config.h>
#include <vector>

namespace ocl {
    namespace be {

        class argument_buffer {
            std::vector<char> _v;
            size_t _max_alignment;
        public:
            argument_buffer() : _v(), _max_alignment(0) {
                _v.reserve(4096);
            };
            // allow access to the stored data
            const char* data() const { return _v.data(); }
            // allow access to the stored data
            char* data() { return _v.data(); }
            // amount of data
            size_t size() const { return _v.size(); }
            // clear the buffer
            void clear() { _v.clear(); _max_alignment=0; }
            // insert an argument into the buffer
            template <typename _T>
            void
            insert(const _T& t) {
                constexpr size_t st=sizeof(_T);
                constexpr size_t at=alignof(_T);
                constexpr size_t atm1=at-1;
                static_assert((at & atm1) == 0,
                              "type with non power of 2 alignment?");
                const size_t s=_v.size();
                // how many bytes are used from the last alignment?
                const size_t m=s&atm1;
                // const size_t pad = m ? at - m : 0;
                const size_t pad= (at - m) & atm1;
                const size_t ns=s+pad;
                const size_t nn=ns+st;
                _v.resize(nn, char(0xff));
                char* cd=_v.data() + ns;
                _T* d=reinterpret_cast<_T*>(cd);
                *d = t;
                _max_alignment = std::max(_max_alignment, at);
            }
            // insert a constant array into to the buffer
            template <typename _T, size_t _N>
            void
            insert(const _T(&t)[_N]) {
                for (size_t i=0; i<_N; ++i) {
                    insert(t[i]);
                }
            }
            // insert a memory area into the buffer
            template <typename _T>
            void
            insert(const _T* p, size_t n) {
                for (size_t i=0; i<n; ++i) {
                    insert(p[i]);
                }
            }
            // return the maximum alignment seen so far
            size_t max_alignment() const { return _max_alignment; }
            // pad the buffer to a multiple of _max_alignment
            void
            pad_to_max_alignment() {
                size_t s= _v.size();
                const size_t max_alignment_m_1 = _max_alignment - 1;
                const size_t m= s & max_alignment_m_1;
                const size_t pad= (_max_alignment - m) & max_alignment_m_1;
                if (pad != 0) {
                    const size_t ns=s+pad;
                    _v.resize(ns, char(0xff));
                }
            }
            // align the end of the buffer to an multiple of x
            template <std::size_t _N>
            void
            pad_to_multiple_of() {
                constexpr size_t at=_N;
                constexpr size_t atm1=at-1;
                static_assert((at & atm1) == 0,
                              "type with non power of 2 alignment?");
                const size_t s=_v.size();
                // how many bytes are used from the last alignment?
                const size_t m=s&atm1;
                // const size_t pad = m ? at - m : 0;
                const size_t pad= (at - m) & atm1;
                if (pad != 0) {
                    const size_t ns=s+pad;
                    _v.resize(ns, char(0xff));
                }
            }
        };
    }
}

// Local variables:
// mode: c++
// end:
#endif
