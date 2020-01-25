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
        ~dvec();
        // default constructor.
        dvec();
        // copy constructor
        dvec(const dvec& v);
        // move constructor
        dvec(dvec&& v);
        // assignment operator from dvec
        dvec& operator=(const dvec& v);
        // move assignment
        dvec& operator=(dvec&& v);
        // constructor with size and uninitialized memory
        explicit dvec(std::size_t s);
        // constructor from backend data and size with uninitialized
        // memory
        dvec(be::data_ptr pbe, std::size_t s);
        // constructor from memory buffer
        dvec(std::size_t n, const _T* s);
        // constructor from memory buffer
        dvec(be::data_ptr pbe, std::size_t n, const _T* s);
        // constructor with size and initializer
        dvec(std::size_t n, const _T& i);
        // constructor with size and initializer
        dvec(be::data_ptr pbe, std::size_t n, const _T& i);
        // constructor with size and initializer
        template <typename _U>
        dvec(std::size_t n, const _U& i);
        // constructor with size and initializer
        template <typename _U>
        dvec(be::data_ptr pbe, std::size_t n, const _U& i);
        // constructor from initializer list
        dvec(std::initializer_list<_T> l);
        // constructor from initializer list
        dvec(be::data_ptr pbe, std::initializer_list<_T> l);
        // construction from std::vector, forces move of data
        // from host to device
        dvec(const std::vector<_T>& v);
        // construction from std::vector, forces move of data
        // from host to device
        dvec(be::data_ptr pbe, const std::vector<_T>& v);
        // assignment from scalar
        dvec& operator=(const _T& i);
        // assignment from scalar
        template <typename _U>
        dvec& operator=(const _U& i);

        // template constructor for evaluation of expressions
        template <template <class _V> class _OP,
                  class _L, class _R>
        dvec(const expr<_OP<dvec<_T> >, _L, _R>& r);
        // assignment from scalar
        template <template <class _V> class _OP,
                  class _L, class _R>
        dvec& operator=(const expr<_OP<dvec<_T> >, _L, _R>& r);
        // copy the contents of the device buffer into p[0, size())
        void
        copy_to_host(_T* p) const;
        // copy the contents of the device buffer [offs, offs+cnt)
        // into p[0, cnt)
        void
        copy_to_host(_T* p, size_t offs, size_t cnt) const;
        // copy the contents of p into the device buffer
        void
        copy_from_host(const _T* p);
        // copy the contents of p[0, cnt) into the device buffer
        // [offs, offs+cnt)
        void
        copy_from_host(const _T* p, size_t offs, size_t cnt);
        // conversion operator to std::vector, forces move of
        // data to host
        explicit operator std::vector<_T> () const;
        // size of the dvec
        std::size_t size() const;
    };

    template <class _T>
    struct expr_traits<dvec<_T> > {
        using type = const dvec<_T>&;
    };

    // generate a custom function for dvec<_T> expressions
    template <typename _T, typename ... _AX>
    auto
    custom_func(const std::string& name,
                const std::string& body,
                _AX&&... ax);

    // generate a custom kernel with predefined vector size
    // for dvec<_T> expressions
    template <typename _T, typename ... _AX>
    auto
    custom_kernel_with_size(const std::string& name,
                            const std::string& body,
                            std::size_t s,
                            _AX&&... ax);

    // generate a custom kernel with for dvec<_T> expressions
    template <typename _T, typename ... _AX>
    auto
    custom_kernel(const std::string& name,
                  const std::string& body,
                  _AX&&... ax);
#if 0
    extern template class dvec<double>;
    extern template class dvec<float>;
    extern template class dvec<int64_t>;
    extern template class dvec<uint64_t>;
    extern template class dvec<int32_t>;
    extern template class dvec<uint32_t>;
    extern template class dvec<int16_t>;
    extern template class dvec<uint16_t>;
    extern template class dvec<char>;
    extern template class dvec<unsigned char>;
    extern template class dvec<signed char>;
#endif
    namespace debug {
        template <typename _T>
        void
        dump(const dvec<_T>& v, const std::string& pfx="") {
            std::cout << &v << ' ' << pfx << " (\n";
            std::vector<_T> vh(v);
            for (std::size_t i=0; i<vh.size(); ++i) {
                std::cout << vh[i];
                if ((i&7)==7) {
                    std::cout << '\n';
                } else if (i+1 < vh.size()){
                    std::cout << ", ";
                }
            }
            std::cout << " )";
            if ((vh.size() & 7) != 0) {
                std::cout << '\n';
            }
        }
    }
}

template <class _T>
inline
ocl::dvec<_T>::~dvec()
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec() : base_type()
{
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
ocl::dvec<_T>::dvec(std::size_t n)
    : base_type{n*sizeof(_T)}
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(be::data_ptr pbe, std::size_t n)
    : base_type{pbe, n*sizeof(_T)}
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(std::size_t n, const _T* s)
    : base_type{n*sizeof(_T), s}
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(be::data_ptr pbe, std::size_t n, const _T* s)
    : base_type{pbe, n*sizeof(_T), s}
{
}

template <class _T>
ocl::dvec<_T>::dvec(std::size_t s, const _T& i)
    : base_type{s * sizeof(_T)}
{
    if (s) {
        execute(*this, i, backend_data(), s);
    }
}

template <class _T>
ocl::dvec<_T>::dvec(be::data_ptr pbe, std::size_t s, const _T& i)
    : base_type{pbe, s * sizeof(_T)}
{
    if (s) {
        execute(*this, i, pbe, s);
    }
}

template <class _T>
template <typename _U>
ocl::dvec<_T>::dvec(std::size_t s, const _U& i)
    : base_type{s * sizeof(_T)}
{
    if (s) {
        execute(*this, i, backend_data(), s);
    }
}

template <class _T>
template <typename _U>
ocl::dvec<_T>::dvec(be::data_ptr pbe, std::size_t s, const _U& i)
    : base_type{pbe, s * sizeof(_T)}
{
    if (s) {
        execute(*this, i, pbe, s);
    }
}

template <class _T>
inline
ocl::dvec<_T>::dvec(std::initializer_list<_T> l)
    : base_type{sizeof(_T) * l.size(), l.begin()}
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(be::data_ptr pbe, std::initializer_list<_T> l)
    : base_type{pbe, sizeof(_T) * l.size(), l.begin()}
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(const std::vector<_T>& r)
    : base_type{sizeof(_T) * r.size(), &r[0]}
{
}

template <class _T>
inline
ocl::dvec<_T>::dvec(be::data_ptr pbe, const std::vector<_T>& r)
    : base_type{pbe, sizeof(_T) * r.size(), &r[0]}
{
}

template <class _T>
ocl::dvec<_T>&
ocl::dvec<_T>::operator=(const _T& i)
{
    size_t s=size();
    if (s) {
        execute(*this, i, backend_data(), s);
    }
    return *this;
}

template <class _T>
template <class _U>
ocl::dvec<_T>&
ocl::dvec<_T>::operator=(const _U& i)
{
    size_t s=size();
    if (s) {
        execute(*this, i, backend_data(), s);
    }
    return *this;
}

template <class _T>
template <template <class _V> class _OP, class _L, class _R>
ocl::dvec<_T>::dvec(const expr<_OP<dvec<_T> >, _L, _R>& r)
    : base_type{ocl::backend_data(r),
                eval_size(r)*sizeof(_T)}
{
    size_t s=size();
    if (s) {
        execute(*this, r, backend_data(), s);
    }
}

template <class _T>
template <template <class _V> class _OP, class _L, class _R>
ocl::dvec<_T>&
ocl::dvec<_T>::operator=(const expr<_OP<dvec<_T> >, _L, _R>& r)
{
    size_t s=eval_size(r);
    if (s) {
        be::data_ptr p=ocl::backend_data(r);
        be::data_ptr pm=backend_data();
        if (s <= size() && pm==p) {
            execute(*this, r, p, s);
        } else {
            if (p == nullptr)
                p = pm;
            dvec t(p, s);
            execute(t, r, p, s);
            swap(t);
        }
    }
    return *this;
}

template <class _T>
inline
void
ocl::dvec<_T>::copy_to_host(_T* p)
    const
{
    dvec_base::copy_to_host(p);
}

template <class _T>
inline
void
ocl::dvec<_T>::copy_to_host(_T* p, size_t offs, size_t cnt)
    const
{
    dvec_base::copy_to_host(p, offs*sizeof(_T), cnt*sizeof(_T));
}


template <class _T>
inline
void
ocl::dvec<_T>::copy_from_host(const _T* p)
{
    dvec_base::copy_from_host(p);
}

template <class _T>
inline
void
ocl::dvec<_T>::copy_from_host(const _T* p, size_t offs, size_t cnt)
{
    dvec_base::copy_from_host(p, offs*sizeof(_T), cnt*sizeof(_T));
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

template <typename _T, typename ... _AX>
auto
ocl::
custom_func(const std::string& name,  const std::string& body,
            _AX&&... ax)
{
    return make_expr<dop::custom_f<dvec<_T> > >(
        impl::cf_body(name, body),
        impl::custom_args<dvec<_T>>(
            std::forward<_AX&&>(ax) ...));
}

template <typename _T, typename ... _AX>
auto
ocl::
custom_kernel_with_size(const std::string& name,
                        const std::string& body,
                        std::size_t s,
                        _AX&&... ax)
{
    return make_expr<dop::custom_k<dvec<_T> > >(
        impl::ck_body(name, body, s),
        impl::custom_args<dvec<_T>>(
            std::forward<_AX&&>(ax) ...));
}

template <typename _T, typename ... _AX>
auto
ocl::
custom_kernel(const std::string& name,
              const std::string& body,
              _AX&&... ax)
{
    return make_expr<dop::custom_k<dvec<_T> > >(
        impl::ck_body(name, body),
        impl::custom_args<dvec<_T>>(
            std::forward<_AX&&>(ax) ...));
}

// Local variables:
// mode: c++
// end:
#endif // __OCL_VECTOR_H__
