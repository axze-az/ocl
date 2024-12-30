#include "ocl/ocl.h"
#include "ocl/test/tools.h"

namespace ocl {

    namespace impl {
        struct variable_dvec_lookup_table {
            static
            __ck_body
            gen_body(const std::string_view& tname,
                     const std::string_view& iname);
        };
    }

    template <typename _T, typename _I>
    class variable_dvec_lookup_table
        : private impl::variable_dvec_lookup_table {
        using base_type = impl::variable_dvec_lookup_table;
        const dvec<_I>* _idx;
    public:
        variable_dvec_lookup_table(const dvec<_I>& idx) : _idx(&idx) {}
        template <std::size_t _N>
        auto
        from(const _T(&tbl)[_N]) const;

        template <std::size_t _N>
        auto
        from(const _T(&tbl)[_N], size_t zero_offset) const;
    };

    template <typename _T, typename _I>
    variable_dvec_lookup_table<_T, _I>
    make_variable_lookup_table(const dvec<_I>& idx) {
        return variable_dvec_lookup_table<_T, _I>(idx);
    };

    namespace test {
        void
        lookup();
    }
}

ocl::impl::__ck_body
ocl::impl::variable_dvec_lookup_table::
gen_body(const std::string_view& tname, const std::string_view& iname)
{
    std::ostringstream s;
    s << "lookup_" << tname << '_' << iname;
    const std::string kname=s.str();
    s.str("");
    s <<"void " << kname << "(\n"
        "ulong n,\n"
        "__global " << tname << "* res,\n"
        "__global const " << iname << "* idx,\n"
        "__arg_local const " << tname << "* s,\n"
	"ulong zero_offset\n"
        ")\n"
        "{\n"
        "    ulong gid=get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        res[gid] = s[zero_offset+idx[gid]];\n"
        "    }\n"
        "}\n";
    const std::string ksrc=s.str();
    return __ck_body(kname, ksrc);
}

template <typename _T, typename _I>
template <std::size_t _N>
auto
ocl::variable_dvec_lookup_table<_T, _I>::
from(const _T(&tbl)[_N], size_t zero_offset)
    const
{
    const auto tname=be::type_2_name<_T>::v();
    const auto iname=be::type_2_name<_I>::v();
    impl::__ck_body ckb=base_type::gen_body(tname, iname);
    return custom_kernel<_T>(ckb.name(), ckb.body(), *_idx, tbl, zero_offset);
}

template <typename _T, typename _I>
template <std::size_t _N>
auto
ocl::variable_dvec_lookup_table<_T, _I>::
from(const _T(&tbl)[_N])
    const
{
    return from(tbl, 0);
}

void
ocl::test::lookup()
{
    static const float tbl[]={
        8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };
    dvec<float> r({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                   -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f});
    dump(tbl, "lookup_table: ");
    dvec<int> i=(cvt<dvec<int>>(r)-1) & 7;
    dump(i, "index= (r-1) & 7 ");
    auto lck=make_variable_lookup_table<float>(i);
    dvec<float> rs=lck.from(tbl);
    dump(rs, "result after lookup: ");

    static const float tblnz[]={
        -3.0f, -2.0f, -1.0f, +0.0f, +1.0f, +2.0f, +3.0f
    };
    dvec<int> inz{-3, -2, -1, +0, +1, +2, +3,
                  +3, +2, +1, +0, -1, -2, -3};
    dump(tblnz, "lookup_table: ");
    dump(inz, "index=");
    auto lcknz=make_variable_lookup_table<float>(inz);
    dvec<float> rsnz=lcknz.from(tblnz, 3);
    dump(rsnz, "result after lookup: ");
}

int main()
{
    ocl::test::lookup();
    return 0;
}
