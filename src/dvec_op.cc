#include "ocl/dvec_op.h"

std::string
ocl::impl::
decl_buffer_args_dvec_t(const std::string_view& tname,
                        unsigned& arg_num,
                        bool ro)
{
    std::ostringstream s;
    s << spaces(4) << "__global ";
    if (ro) {
        s << "const ";
    }
    s << tname
      << "* arg" << arg_num << ",\n";
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
decl_buffer_args_dvec_t(const char* tname,
                        unsigned& arg_num,
                        bool ro)
{
    return decl_buffer_args_dvec_t(std::string_view(tname),
                                   arg_num, ro);
}

std::string
ocl::impl::
decl_buffer_args_dvec_t(const std::string& tname,
                        unsigned& arg_num,
                        bool ro)
{
    return decl_buffer_args_dvec_t(std::string_view(tname),
                                   arg_num, ro);
}

std::string
ocl::impl::
concat_args_dvec_t(var_counters& c)
{
    std::ostringstream s;
    s << "arg" << c._buf_num;
    ++c._var_num;
    ++c._buf_num;
    return s.str();
}

void
ocl::impl::
bind_buffer_args_dvec_t(const dvec_base& r,
                        const std::string_view& tname,
                        unsigned& buf_num,
                        be::kernel& k,
                        bool const_val,
                        size_t elements)
{
    if (r.backend_data()->debug() != 0) {
        std::string kn=k.name();
        std::ostringstream s;
        s << std::this_thread::get_id() << ": "
          << kn << ": " << &r << ": binding ";
        if (const_val)
            s << "const ";
        s << "dvec<"
          << tname << "> with "
          << elements
          << " elements to arg " << buf_num << '\n';
        be::data::debug_print(s.str());
    }
    k.set_arg(buf_num, r.buf());
    ++buf_num;
}

void
ocl::impl::
bind_buffer_args_dvec_t(const dvec_base& r,
                        const char* tname,
                        unsigned& buf_num,
                        be::kernel& k,
                        bool const_val,
                        size_t elements)
{
    return bind_buffer_args_dvec_t(r, std::string_view(tname),
                                   buf_num, k, const_val, elements);
}
 
void
ocl::impl::
bind_buffer_args_dvec_t(const dvec_base& r,
                        const std::string& tname,
                        unsigned& buf_num,
                        be::kernel& k,
                        bool const_val,
                        size_t elements)
{
    return bind_buffer_args_dvec_t(r, std::string_view(tname),
                                   buf_num, k, const_val, elements);
}

std::string
ocl::impl::
store_result_dvec_t(var_counters& c)
{
    std::ostringstream s;
    s << spaces(8)
      << "arg" << c._buf_num
      << "[gid] =";
    ++c._var_num;
    ++c._buf_num;
    return s.str();
}

std::string
ocl::impl::
eval_args_dvec_t(const std::string_view& tname,
                 unsigned& arg_num,
                 bool ro)
{
    std::ostringstream s;
    s << spaces(4) << "__global " ;
    if (ro) {
        s<< "const ";
    }
    s << tname
      << "* arg"  << arg_num;
    ++arg_num;
    return s.str();
}

std::string
ocl::impl::
eval_args_dvec_t(const char* tname,
                 unsigned& arg_num,
                 bool ro)
{
    return eval_args_dvec_t(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_args_dvec_t(const std::string& tname,
                 unsigned& arg_num,
                 bool ro)
{
    return eval_args_dvec_t(std::string_view(tname), arg_num, ro);
}


std::string
ocl::impl::
eval_vars_dvec_t(const std::string_view& tname,
                 unsigned& arg_num,
                 bool ro)
{
    std::ostringstream s;
    s << spaces(8) << tname
      << " v" << arg_num;
    if (ro== true) {
        s << " = arg"
          << arg_num << "[gid];";
    }
    std::string a(s.str());
    ++arg_num;
    return a;
}

std::string
ocl::impl::
eval_vars_dvec_t(const char* tname,
                 unsigned& arg_num,
                 bool ro)
{
    return eval_vars_dvec_t(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_vars_dvec_t(const std::string& tname,
                 unsigned& arg_num,
                 bool ro)
{
    return eval_vars_dvec_t(std::string_view(tname), arg_num, ro);
}

std::string
ocl::impl::
eval_results_dvec_t(unsigned& res_num)
{
    std::ostringstream s;
    s << spaces(8) << "arg" << res_num << "[gid]="
      << " v" << res_num << ';';
    ++res_num;
    return s.str();
}

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
