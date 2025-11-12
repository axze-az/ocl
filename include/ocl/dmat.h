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
#if !defined (__OCL_DMAT_H__)
#define __OCL_DMAT_H__ 1

#include <ocl/config.h>
#include <ocl/dvec_base.h>
#include <clblast.h>

namespace ocl {

    template <typename _T>
    class dmat : public dvec_base {
	size_t _rows;
    public:
	dmat(size_t rows, size_t cols);
	dmat(const dmat& r);
	dmat(dmat&& r);
	dmat& operator=(const dmat& r);
	dmat& operator=(dmat&& r);
	~dmat();

	const size_t& rows() const;
	const size_t& cols() const;
	enum class layout {
	    row_major,
	    column_major
	};
	const layout mem_layout() const;
	dmat& change_mem_layout(layout n);
    private:
	size_t _rows;
	size_t _cols;
	layout _layout;
    };

    template <typename _T>
    dmat<_T>
    transpose(const dmat<_T>& a);

    template <typename _T>
    dmat<_T>
    operator+(const dmat<_T>& a, const dmat<_T>& b);
    
    
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_DMAT_H__
