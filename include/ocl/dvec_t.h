#if !defined (__OCL_DVEC_T_H__)
#define __OCL_DVEC_T_H__ 1

#include <ocl/config.h>
#include <ocl/dvec_base.h>
#include <ocl/expr.h>
#include <ocl/expr_kernel.h>
#include <initializer_list>
#include <cstdint>
#include <vector>

namespace ocl {

    namespace impl {

        template <typename _T>
        struct dvec_select_mask_value {
            using type = _T;
        };

        template <>
        struct dvec_select_mask_value<double> {
            using type = std::int64_t;
        };

        template <>
        struct dvec_select_mask_value<float> {
            using type = std::int32_t;
        };

        template <typename _T>
        using dvec_select_mask_value_t =
            typename dvec_select_mask_value<_T>::type;

        template <size_t _N>
        constexpr inline size_t const_log2() {
            static_assert(_N != 0 && (_N & (_N-1))==0,
                         "Value is not a valid power of 2");
            return 1 + const_log2<_N/2>();
        }
        template <>
        constexpr inline size_t const_log2<1>() {
            return 0;
        }
    }
    // dvec: representation of data on the acceleration device
    template <class _T>
    class dvec : public dvec_base {
        using base_type = dvec_base;
    public:
        using value_type = _T;
        using mask_value_type = impl::dvec_select_mask_value_t<_T>;
        using mask_type = dvec<mask_value_type>;
        // using base_type::backend_data;
        // using base_type::buf;
        ~dvec() {}
        // size of the dvec
        std::size_t size() const;
        // default constructor.
        dvec() : base_type{} {}
        // constructor from backend data and size.
        dvec(be::data_ptr pbe, std::size_t s);
        // constructor from memory buffer
        dvec(std::size_t n, const _T* s);
        // constructor with size and initializer
        dvec(std::size_t n, const _T& i);
        // constructor from initializer list
        dvec(std::initializer_list<_T> l);
        // constructor with size and initializer
        template <typename _U>
        dvec(std::size_t n, const _U& i);
        // copy constructor
        dvec(const dvec& v);
        // move constructor
        dvec(dvec&& v);
        // construction from std::vector, forces move of data
        // from host to device
        dvec(const std::vector<_T>& v);
        // assignment operator from dvec
        dvec& operator=(const dvec& v);
        // move assignment
        dvec& operator=(dvec&& v);
        // assignment from scalar
        dvec& operator=(const _T& i);
        // template constructor for evaluation of expressions
        template <template <class _V> class _OP,
                  class _L, class _R>
        dvec(const expr<_OP<dvec<_T> >, _L, _R>& r);
        // assignment from scalar
        template <template <class _V> class _OP,
                  class _L, class _R>
        dvec& operator=(const expr<_OP<dvec<_T> >, _L, _R>& r);
        // conversion operator to std::vector, forces move of
        // data to host
        explicit operator std::vector<_T> () const;
    };

    template <class _T>
    struct expr_traits<dvec<_T> > {
        using type = const dvec<_T>&;
    };

}

template <class _T>
inline
ocl::dvec<_T>::dvec(be::data_ptr pbe, std::size_t n)
    : base_type{pbe, n*sizeof(_T)}
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(std::size_t n, const _T* p)
    : base_type{n*sizeof(_T)}
{
    copy_from_host(p);
}

template <class _T>
inline
ocl::dvec<_T>::dvec(std::size_t s, const _T& i)
    : base_type{s * sizeof(_T)}
{
    if (s) {
        execute(*this, i, backend_data(), s);
    }
}

template <class _T>
template <typename _U>
inline
ocl::dvec<_T>::dvec(std::size_t s, const _U& i)
    : base_type{s * sizeof(_T)}
{
    if (s) {
        execute(*this, i, backend_data(), s);
    }
}

template <class _T>
inline
ocl::dvec<_T>::dvec(const dvec& r)
    : base_type(r)
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(dvec&& r)
    : base_type(std::move(r))
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(const std::vector<_T>& r)
    : base_type{sizeof(_T) * r.size()}
{
    copy_from_host(&r[0]);
}

template <class _T>
inline
ocl::dvec<_T>::dvec(std::initializer_list<_T> l)
    : base_type{sizeof(_T) * l.size()}
{
    copy_from_host(l.begin());
}

template <class _T>
template <template <class _V> class _OP, class _L, class _R>
inline
ocl::
dvec<_T>::dvec(const expr<_OP<dvec<_T> >, _L, _R>& r)
    : base_type{ocl::backend_data(r), eval_size(r)*sizeof(_T)}
{
    size_t s=size();
    if (s) {
        execute(*this, r, backend_data(), s);
    }
}

template <class _T>
template <template <class _V> class _OP, class _L, class _R>
inline
ocl::dvec<_T>&
ocl::dvec<_T>::operator=(const expr<_OP<dvec<_T> >, _L, _R>& r)
{
    size_t s=eval_size(r);
    if (s == size()) {
        execute(*this, r, backend_data(), s);
    } else {
        be::data_ptr p=backend_data();
        if (p==nullptr)
            p=ocl::backend_data(r);
        dvec t(p, s);
        execute(t, r, p, s);
        swap(t);
    }
    return *this;
}

template <class _T>
inline
ocl::dvec<_T>&
ocl::dvec<_T>::operator=(const dvec& r)
{
    base_type::operator=(r);
    return *this;
}

template <class _T>
inline
ocl::dvec<_T>&
ocl::dvec<_T>::operator=(dvec&& r)
{
    base_type::operator=(std::move(r));
    return *this;
}

template <class _T>
inline
ocl::dvec<_T>::operator std::vector<_T> ()
    const
{
    std::size_t n(this->size());
    std::vector<_T> v(n);
    copy_to_host(&v[0]);
    return v;
}

template <class _T>
inline
std::size_t
ocl::dvec<_T>::size() const
{
    constexpr const size_t st=sizeof(_T);
    std::size_t s=buffer_size();
    if constexpr ((st & (st-1))==0) {
        constexpr size_t shift=impl::const_log2<st>();
        s >>= shift;
    } else {
        s /= st;
    }
    return s;
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_VECTOR_H__
