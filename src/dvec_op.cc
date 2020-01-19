#include "ocl/dvec_op.h"

std::string
ocl::dop::unary_func_base::body(const std::string& l, bool is_operator,
                                const char* name)
{
    return body(l, is_operator, std::string(name));
}

std::string
ocl::dop::unary_func_base::body(const std::string& l, bool is_operator,
                                const std::string& name)
{
    std::string res=name;
    if (is_operator == false)
        res += '(';
    res += l;
    if (is_operator == false)
        res += ')';
    return res;
}

// generate the body of an binary_func object
std::string
ocl::dop::binary_func_base::body(const std::string& l, const std::string& r,
                                 bool is_operator, const char* name)
{
    return body(l, r, is_operator, std::string(name));
}
// generate the body of an binary_func object
std::string
ocl::dop::binary_func_base::body(const std::string& l, const std::string& r,
                                 bool is_operator, const std::string& name)
{
    std::string res;
    if (is_operator == false) {
        res = name;
        res += "(";
    }
    res += l;
    if (is_operator == false) {
        res += ", ";
    } else {
        res += name;
    }
    res += r;
    if (is_operator == false)
        res += ")";
    return res;
}

std::string
ocl::dop::names::div_base::name(const char* tname)
{
    return name(std::string(tname));
}

std::string
ocl::dop::names::div_base::name(const std::string& tname)
{
    return std::string("__div_") + tname;
}

std::string
ocl::dop::names::div_base::body(const char* tname)
{
    return body(std::string(tname));
}

std::string
ocl::dop::names::div_base::body(const std::string& tname)
{
    std::string inl="inline ";
#if 0
    // not precise enough
    // -0x1.9d9e4cp+0/-0x1.afc97ep+0=0x1.ea74c2p-1 != 0x1.ea74cp-1 0x1p-24
    std::string fbody =
        inl +
        tname + " __div_" + tname + "(" +
        tname + " a, " + tname + " b)\n"
        "{\n"
        "    " + tname + " q0=a/b;\n"
        "    " + tname + " r= fma(q0, -b, a);\n"
        "    " + tname + " q1= r/b;\n"
        "    " + tname + " q = q0+q1; \n"
        "    q1= isnan(q1) ? q0 : q;\n"
        "    return q1;\n"
        "}\n";
#else
    std::string fbody =
        inl +
        tname + " __div_" + tname + "(" +
        tname + " a, " + tname + " b)\n"
        "{\n"
        "    " + tname + " xn=1.0f/b;\n"
        "    xn = fma(xn, fma(xn, -b, 1.0f), xn);\n"
        "    " + tname + " yn= a*xn;\n"
        "    yn= fma(xn, fma(yn, -b, a), yn);\n"
        "    yn= isnan(yn) ? a/b : yn;\n"
        "    return yn;\n"
        "}\n";
#endif
    return fbody;
}


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
