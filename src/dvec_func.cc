#include "ocl/dvec_func.h"

std::string
ocl::dop::names::f_sqrt_base::name(const char* tname)
{
    return name(std::string(tname));
}

std::string
ocl::dop::names::f_sqrt_base::name(const std::string& tname)
{
    return std::string("__sqrt_") + tname;
}

std::string
ocl::dop::names::f_sqrt_base::body(const char* tname)
{
    return body(std::string(tname));
}

std::string
ocl::dop::names::f_sqrt_base::body(const std::string& tname)
{
    std::string inl="inline ";
    std::string fbody =
        inl +
        tname + " __sqrt_" + tname + "(" +
        tname + " a)\n"
        "{\n"
        "    " + tname + " r=rsqrt(a);\n"
        "    " + tname + " rah, ral;\n"
        "    rah=a*r;\n"
        "    ral=fma(-a, r, rah);\n"
        "    " + tname + " th= fma(r, rah, -1.0f);\n"
        "    th=fma(r, ral, th);\n"
        "    r= fma(-0.5f*r*a, th, r*a);\n"
        "    r= isnan(r) ? a*r : r;\n"
        "    r= a==0 ? a : r;\n"
        "    return r;\n"
        "}\n";
    return fbody;
}

std::string
ocl::dop::names::f_sel_base::
body(const std::string& s, const std::string& on_true,
     const std::string& on_false)
{
    std::string r="(( ";
    r += s + ") ? (" + on_true + ") : (" + on_false + "))";
    return r;
}

