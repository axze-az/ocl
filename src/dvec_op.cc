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
    std::string fbody =
        inl +
        tname + " __div_" + tname + "(" +
        tname + " a, " + tname + " b)\n"
        "{\n"
        "    " + tname + " xn=1.0f/b;\n"
        "    xn = fma(xn, fma(xn, -b, 1.0f), xn);\n"
        "    " + tname + " yn= a*xn;\n"
        "    yn= fma(xn, fma(yn, -b, a), yn);\n"
        "    return yn;\n"
        "}\n";
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
        "    return r;\n"
        "}\n";
    return fbody;
}
