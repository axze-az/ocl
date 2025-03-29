#include "ocl/dvec_math.h"

ocl::impl::__cf_body
ocl::impl::
gen_horner(const std::string_view& tname,
           const std::string_view& cname,
           size_t n, bool use_fma, bool use_mad)
{
    std::ostringstream s;
    s << "horner_" << n << '_' << tname << '_' << cname;
    if (use_fma) {
        s << "_fma";
    } else if (use_mad) {
        s << "_mad";
    }
    const std::string hname=s.str();
    s.str("");
    s << tname << " " << hname << "("
      << tname << " x, "
        "__arg_local const " << cname << "* c)\n"
        "{\n"
        "    "<< tname << " r=c[0];\n";
#if 1
#if 0
    // this code is terrrible
    s << "#pragma unroll ("<< std::min(n, size_t(256)) << ")\n";
    s << "    for (size_t i=1; i<" << n << "; ++i) {\n"
      << "        " << tname << " ci=c[i];\n";
    if (use_fma) {
        s << "        r=fma(x, r, ci);\n";
    } else if (use_mad) {
        s << "        r=mad(x, r, ci);\n";
    } else {
        s << "        r=x*r+ci;\n";
    }
    s << "    };\n";
#else
    s << "    "<< tname << " ci;\n";
    for (size_t i=1; i<n; ++i) {
        s << "    ci=c[" << i<< "];\n";
        if (use_fma) {
            s << "    r=fma(x, r, ci);\n";
        } else if (use_mad) {
            s << "    r=mad(x, r, ci);\n";
        } else {
            s << "    r=x*r+ci;\n";
        }
    }
#endif
#else
     for (size_t i=1; i<n; ++i) {
         if (use_fma) {
             s << "    r=fma(x, r, c["<< i<<"]);\n";
         } else if (use_mad) {
             s << "    r=mad(x, r, c["<< i<<"]);\n";
         } else {
             s << "    r=x*r+c["<< i<<"];\n";
         }
     }
#endif
    s << "    return r;\n"
         "}\n";
    return __cf_body(hname, s.str());
}

ocl::impl::__cf_body
ocl::impl::
gen_horner2(const std::string_view& tname,
            const std::string_view& cname,
            size_t n,
            bool x2_internal, bool use_fma, bool use_mad)
{
    std::ostringstream s;
    s << "horner2_" << n << '_' << tname << '_' << cname;
    if (use_fma) {
        s << "_fma";
    } else if (use_mad) {
        s << "_mad";
    }
    if (x2_internal) {
        s << "_x2";
    }
    const std::string hname=s.str();
    s.str("");
    s << tname << " " << hname << "("
      << tname << " x, ";
    if (x2_internal==false) {
        s << tname << " x2, ";
    }
    s << "__arg_local const " << cname << "* c)\n"
         "{\n";
    if (x2_internal) {
        s <<"    " << tname << " x2=x*x;\n";
    }
    s << "    " << tname << " r0=c[0];\n"
         "    " << tname << " r1=c[1];\n";
    const std::size_t _NE= n & ~(std::size_t(1));
    for (size_t i=2; i<_NE; i+=2) {
        if (use_fma) {
            s << "    r0=fma(x2, r0, c["<<i<<"]);\n";
            s << "    r1=fma(x2, r1, c["<<i+1<<"]);\n";
        } else if (use_mad) {
            s << "    r0=mad(x2, r0, c["<<i<<"]);\n";
            s << "    r1=mad(x2, r1, c["<<i+1<<"]);\n";
        } else {
            s << "    r0=x2*r0+c["<< i<<"];\n";
            s << "    r1=x2*r1+c["<< i+1<<"];\n";
        }
    }
    s << "    " << tname << " r=";
    if (use_fma) {
        s <<  "fma(x, r0, r1);\n";
    } else if (use_mad) {
        s <<  "mad(x, r0, r1);\n";
    } else {
        s <<  "x*r0+r1;\n";
    }
    if ( n & 1) {
        if (use_fma) {
            s << "    r=fma(x, r, c[" << n-1 << "]);\n";
        } else if (use_mad){
            s << "    r=mad(x, r, c[" << n-1 << "]);\n";
        } else {
            s << "    r=x*r+c[" << n-1 << "];\n";
        }
    }
    s << "    return r;\n"
         "}\n";
    return __cf_body(hname, s.str());
}

ocl::impl::__cf_body
ocl::impl::
gen_horner4(const std::string_view& tname,
            const std::string_view& cname,
            size_t n,
            bool x2x4_internal, bool use_fma, bool use_mad)
{
    std::ostringstream s;
    s << "horner4_" << n << '_' << tname << '_' << cname;
    if (use_fma) {
        s << "_fma";
    } else if (use_mad) {
        s << "_mad";
    }
    if (x2x4_internal) {
        s << "_x2x4";
    }
    const std::string hname=s.str();
    s.str("");
    s << tname << " " << hname << "("
      << tname << " x, ";
    if (x2x4_internal==false) {
      s << tname << " x2, "
        << tname << " x4, ";
    }
    s<< "__arg_local const " << cname << "* c)\n"
        "{\n";
    if (x2x4_internal==true) {
        s <<"    " << tname << " x2=x*x;\n";
        s <<"    " << tname << " x4=x2*x2;\n";
    }
    s<< "    " << tname << " r0=c[0];\n"
        "    " << tname << " r1=c[1];\n"
        "    " << tname << " r2=c[2];\n"
        "    " << tname << " r3=c[3];\n";
    const std::size_t _NE= n & ~(std::size_t(3));
    for (size_t i=4; i<_NE; i+=4) {
        if (use_fma) {
            s << "    r0=fma(x4, r0, c["<< i<<"]);\n";
            s << "    r1=fma(x4, r1, c["<< i+1<<"]);\n";
            s << "    r2=fma(x4, r2, c["<< i+2<<"]);\n";
            s << "    r3=fma(x4, r3, c["<< i+3<<"]);\n";
        } else if (use_mad) {
            s << "    r0=mad(x4, r0, c["<< i<<"]);\n";
            s << "    r1=mad(x4, r1, c["<< i+1<<"]);\n";
            s << "    r2=mad(x4, r2, c["<< i+2<<"]);\n";
            s << "    r3=mad(x4, r3, c["<< i+3<<"]);\n";
        } else {
            s << "    r0=x4*r0+c["<< i<<"];\n";
            s << "    r1=x4*r1+c["<< i+1<<"];\n";
            s << "    r2=x4*r2+c["<< i+2<<"];\n";
            s << "    r3=x4*r3+c["<< i+3<<"];\n";
        }
    }
    s << '\n';
    if (use_fma) {
        s << "    " << tname << " r02=fma(x2, r0, r2);\n";
        s << "    " << tname << " r13=fma(x2, r1, r3);\n";
        s << "    " << tname << " r=fma(x, r02, r13);\n";
    } else if (use_mad) {
        s << "    " << tname << " r02=mad(x2, r0, r2);\n";
        s << "    " << tname << " r13=mad(x2, r1, r3);\n";
        s << "    " << tname << " r=mad(x, r02, r13);\n";
    } else {
        s << "    " << tname << " r02=x2*r0+r2;\n";
        s << "    " << tname << " r13=x2*r1+r3;\n";
        s << "    " << tname << " r=x*r02+r13;\n";
    }
    const std::size_t _NR= n & std::size_t(3);
    switch (_NR) {
    default:
        break;
    case 1:
        if (use_fma) {
            s << "    r=fma(x, r, c[" << n-1 <<"]);\n";
        } else if (use_mad) {
            s << "    r=mad(x, r, c[" << n-1 <<"]);\n";
        } else {
            s << "    r=x*r+c[" << n-1 <<"];\n";
        }
        break;
    case 2:
        if (use_fma) {
            s << "    " << tname << " a=fma(x2, r, c["<< n-1 << "]);\n";
            s << "    r=fma(x, c[" << n-2 << "], a);\n";
        } else if (use_mad) {
            s << "    " << tname << " a=mad(x2, r, c["<< n-1 << "]);\n";
            s << "    r=mad(x, c[" << n-2 << "], a);\n";
        } else {
            s << "    " << tname << " a=x2*r+c["<< n-1 << "];\n";
            s << "    r=x*c[" << n-2 << "]+a;\n";
        }
        break;
    case 3:
        if (use_fma) {
            s << "    " << tname << " a=fma(x2, r, c["<< n-2 << "]);\n";
            s << "    " << tname
            << " b=fma(x2, c["<< n-3 << "], c["<< n - 1 << "]);\n";
            s << "    r=fma(x, a, b);\n";
        } else if (use_mad) {
            s << "    " << tname << " a=mad(x2, r, c["<< n-2 << "]);\n";
            s << "    " << tname
            << " b=mad(x2, c["<< n-3 << "], c["<< n - 1 << "]);\n";
            s << "    r=mad(x, a, b);\n";
        } else {
            s << "    " << tname << " a=x2*r+c["<< n-2 << "];\n";
            s << "    " << tname
            << " b=x2*c["<< n-3 << "]+c["<< n - 1 << "];\n";
            s << "    r=x*a+b;\n";
        }
        break;
    }
    s << "    return r;\n"
         "}\n";
    return __cf_body(hname, s.str());
}

