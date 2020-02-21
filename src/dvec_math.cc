#include "ocl/dvec_math.h"

ocl::impl::__cf_body
ocl::impl::
gen_horner(const std::string_view& tname,
           const std::string_view& cname,
           size_t n)
{
    std::ostringstream s;
    s << "horner_" << n << '_' << tname << '_' << cname;
    const std::string hname=s.str();
    s.str("");
    s << tname << " " << hname << "("
      << tname << " x, "
        "__arg_local const " << cname << "* c)\n"
        "{\n"
        "    "<< tname << " r=c[0];\n";
    for (size_t i=1; i<n; ++i) {
        s << "    r=x*r+c["<< i<<"];\n";
    }
    s << "    return r;\n"
         "}\n";
    return __cf_body(hname, s.str());
}

ocl::impl::__cf_body
ocl::impl::
gen_horner2(const std::string_view& tname,
            const std::string_view& cname,
            size_t n)
{
    std::ostringstream s;
    s << "horner2_" << n << '_' << tname << '_' << cname;
    const std::string hname=s.str();
    s.str("");
    s << tname << " " << hname << "("
      << tname << " x, "
      << tname << " x2, "
        "__arg_local const " << cname << "* c)\n"
        "{\n"
        "    " << tname << " r0=c[0];\n"
        "    " << tname << " r1=c[1];\n";
    const std::size_t _NE= n & ~(std::size_t(1));
    for (size_t i=2; i<_NE; i+=2) {
        s << "    r0=x2*r0+c["<< i<<"];\n";
        s << "    r1=x2*r1+c["<< i+1<<"];\n";
    }
    s << "    " << tname << " r= x*r0+r1;\n";
    if ( n & 1) {
        s << "    r = x*r + c[" << n-1 << "];\n";
    }
    s << "    return r;\n"
         "}\n";
    return __cf_body(hname, s.str());
}

ocl::impl::__cf_body
ocl::impl::
gen_horner4(const std::string_view& tname,
            const std::string_view& cname,
            size_t n)
{
    std::ostringstream s;
    s << "horner4_" << n << '_' << tname << '_' << cname;
    const std::string hname=s.str();
    s.str("");
    s << tname << " " << hname << "("
      << tname << " x, "
      << tname << " x2, "
      << tname << " x4, "
        "__arg_local const " << cname << "* c)\n"
        "{\n"
        "    " << tname << " r0=c[0];\n"
        "    " << tname << " r1=c[1];\n"
        "    " << tname << " r2=c[2];\n"
        "    " << tname << " r3=c[3];\n";
    const std::size_t _NE= n & ~(std::size_t(3));
    for (size_t i=4; i<_NE; i+=4) {
        s << "    r0=x4*r0+c["<< i<<"];\n";
        s << "    r1=x4*r1+c["<< i+1<<"];\n";
        s << "    r2=x4*r2+c["<< i+2<<"];\n";
        s << "    r3=x4*r3+c["<< i+3<<"];\n";
    }
    s << '\n';
    s << "    " << tname << " r02 = x2*r0 + r2;\n";
    s << "    " << tname << " r13 = x2*r1 + r3;\n";
    s << "    " << tname << " r = x*r02 + r13;\n";
    const std::size_t _NR= n & std::size_t(3);
    switch (_NR) {
    default:
        break;
    case 1:
        s << "    r= x*r + c[" << n-1 <<"];\n";
        break;
    case 2:
        s << "    " << tname << " a= x2*r + c["<< n-1 << "];\n";
        s << "    r = x*c[" << n-2 << "] + a;\n";
        break;
    case 3:
        s << "    " << tname << " a= x2*r + c["<< n-2 << "];\n";
        s << "    " << tname
          << " b= x2*c["<< n-3 << "] + c["<< n - 1 << "];\n";
        s << "    r = x*a +b;\n";
        break;
    }
    s << "    return r;\n"
         "}\n";
    return __cf_body(hname, s.str());
}

